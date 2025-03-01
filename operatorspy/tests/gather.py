from ctypes import POINTER, Structure, c_int32, c_void_p
import ctypes
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from operatorspy import (
    open_lib,
    to_tensor,
    DeviceEnum,
    infiniopHandle_t,
    infiniopTensorDescriptor_t,
    create_handle,
    destroy_handle,
    check_error,
)

from operatorspy.tests.test_utils import get_args
from enum import Enum, auto
import torch

# constant for control whether profile the pytorch and lib functions
# NOTE: need to manually add synchronization function to the lib function,
#       e.g., cudaDeviceSynchronize() for CUDA
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000

class Inplace(Enum):
    OUT_OF_PLACE = auto()

class GatherDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopGatherDescriptor_t = POINTER(GatherDescriptor)

def gather(rank, axis, inputTensor, indexTensor):
    if PROFILE:
        indices = [slice(None)] * rank
        indices[axis] = indexTensor
        outTensor = inputTensor[tuple(indices)]
        torch.cuda.synchronize()
        return outTensor
    indices = [slice(None)] * rank
    indices[axis] = indexTensor
    outTensor = inputTensor[tuple(indices)]
    return outTensor

def test(
    lib,
    handle,
    torch_device,
    input_shape, 
    indices_shape,
    axis,
    tensor_dtype=torch.float16,
    inplace=Inplace.OUT_OF_PLACE,
):
    print(
        f"Testing Gather on {torch_device} with input_shape:{input_shape} indices_shape:{indices_shape} axis:{axis} dtype:{tensor_dtype} inplace: {inplace.name}"
    )
    input_ = torch.rand(input_shape, dtype=tensor_dtype).to(torch_device)
    indices_ = torch.randint(0, input_shape[axis], indices_shape, dtype=torch.int64).to(torch_device)
    rank =len(input_shape)
    for i in range(NUM_PRERUN if PROFILE else 1):
        ans = gather(rank, axis, input_, indices_)
    if PROFILE:
        start_time = time.time()
        for i in range(NUM_ITERATIONS):
            _ = gather(rank, axis, input_, indices_)
        elapsed = (time.time() - start_time)
        print(f"pytorch time: {elapsed :6f}")

    output_ = torch.rand(ans.shape, dtype=tensor_dtype).to(torch_device)
    input_tensor = to_tensor(input_, lib)
    indices_tensor = to_tensor(indices_, lib)
    output_tensor = to_tensor(output_, lib)
    descriptor = infiniopGatherDescriptor_t()

    check_error(
        lib.infiniopCreateGatherDescriptor(
            handle,
            ctypes.byref(descriptor),
            output_tensor.descriptor,
            input_tensor.descriptor,
            indices_tensor.descriptor,
	    ctypes.c_int32(axis)
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    output_tensor.descriptor.contents.invalidate()
    input_tensor.descriptor.contents.invalidate()
    indices_tensor.descriptor.contents.invalidate()

    for i in range(NUM_PRERUN if PROFILE else 1):
        check_error(
            lib.infiniopGather(descriptor, output_tensor.data, input_tensor.data, indices_tensor.data, None)
        )
    if PROFILE:
        start_time = time.time()
        for i in range(NUM_ITERATIONS):
            check_error(
                lib.infiniopGather(descriptor, output_tensor.data, input_tensor.data, indices_tensor.data, None)
            )
        elapsed = (time.time() - start_time)
        print(f"    lib time: {elapsed :6f}")
    assert torch.allclose(output_, ans, atol=0, rtol=1e-3)
    check_error(lib.infiniopDestroyGatherDescriptor(descriptor))

def test_cpu(lib, test_cases):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)
    for input_shape, indices_shape, axis, inplace in test_cases:
        test(lib, handle, "cpu", input_shape, indices_shape, axis, tensor_dtype=torch.float16, inplace=inplace)
        test(lib, handle, "cpu", input_shape, indices_shape, axis, tensor_dtype=torch.float32, inplace=inplace)
    destroy_handle(lib, handle)


def test_cuda(lib, test_cases):
    device = DeviceEnum.DEVICE_CUDA
    handle = create_handle(lib, device)
    for input_shape, indices_shape, axis, inplace in test_cases:
        # test(lib, handle, "cuda", input_shape, indices_shape, axis, tensor_dtype=torch.float16, inplace=inplace)
        test(lib, handle, "cuda", input_shape, indices_shape, axis, tensor_dtype=torch.float32, inplace=inplace)
    destroy_handle(lib, handle)


def test_bang(lib, test_cases):
    import torch_mlu

    device = DeviceEnum.DEVICE_BANG
    handle = create_handle(lib, device)
    for input_shape, indices_shape, axis, inplace in test_cases:
        test(lib, handle, "mlu", input_shape, indices_shape, axis, tensor_dtype=torch.float16, inplace=inplace)
        test(lib, handle, "mlu", input_shape, indices_shape, axis, tensor_dtype=torch.float32, inplace=inplace)
    destroy_handle(lib, handle)


if __name__ == "__main__":
    test_cases = [
        # input_shape, indices_shape, axis, inplace
        ((3, 2), (2, 2), 0, Inplace.OUT_OF_PLACE),
        ((3, 2), (1, 2), 1, Inplace.OUT_OF_PLACE),
        ((50257, 768), (16, 1024), 0, Inplace.OUT_OF_PLACE),
        ((3, 2), (2, 2), 0, Inplace.OUT_OF_PLACE),
        ((3, 2), (1, 2), 1, Inplace.OUT_OF_PLACE),
        ((50257, 768), (16, 1024), 0, Inplace.OUT_OF_PLACE),
    ]
    args = get_args()
    lib = open_lib()
    lib.infiniopCreateGatherDescriptor.restype = c_int32
    lib.infiniopCreateGatherDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopGatherDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
	c_int32,
    ]
    lib.infiniopGather.restype = c_int32
    lib.infiniopGather.argtypes = [
        infiniopGatherDescriptor_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroyGatherDescriptor.restype = c_int32
    lib.infiniopDestroyGatherDescriptor.argtypes = [
        infiniopGatherDescriptor_t,
    ]

    if args.cpu:
        test_cpu(lib, test_cases)
    if args.cuda:
        test_cuda(lib, test_cases)
    if args.bang:
        test_bang(lib, test_cases)
    if not (args.cpu or args.cuda or args.bang):
        test_cpu(lib, test_cases)
    print("\033[92mTest passed!\033[0m")
