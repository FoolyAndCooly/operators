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

class ReduceMinDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopReduceMinDescriptor_t = POINTER(ReduceMinDescriptor)

def reduce_min(data, axes):
    result = torch.amin(data, dim=axes)
    return result

def test(
    lib,
    handle,
    torch_device,
    data_shape, 
    tensor_dtype=torch.float16,
    inplace=Inplace.OUT_OF_PLACE,
):
    print(
        f"Testing ReduceMin on {torch_device} with data_shape:{data_shape} dtype:{tensor_dtype} inplace: {inplace.name}"
    )
    data = torch.rand(data_shape, dtype=tensor_dtype).to(torch_device)
    rank = len(data_shape)
    axes = torch.randint(low=0, high=rank, size=(1,))
    for i in range(NUM_PRERUN if PROFILE else 1):
        ans = reduce_min(data, axes[0].item())
    if PROFILE:
        start_time = time.time()
        for i in range(NUM_ITERATIONS):
            _ = reduce_min(data, axes)
        elapsed = (time.time() - start_time)
        print(f"pytorch time: {elapsed :6f}")

    reduced = torch.rand(ans.shape, dtype=tensor_dtype).to(torch_device)
    data_tensor = to_tensor(data, lib)
    axes_tensor = to_tensor(axes, lib)
    reduced_tensor = to_tensor(reduced, lib)
    keepdims = 1;
    noop_with_empty_axes = 0
    descriptor = infiniopReduceMinDescriptor_t()

    check_error(
        lib.infiniopCreateReduceMinDescriptor(
            handle,
            ctypes.byref(descriptor),
            reduced_tensor.descriptor,
            data_tensor.descriptor,
            axes_tensor.descriptor,
	    ctypes.c_int32(keepdims),
            ctypes.c_int32(noop_with_empty_axes)
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    # reduced_tensor.descriptor.contents.invalidate()
    # data_tensor.descriptor.contents.invalidate()
    # axes_tensor.descriptor.contents.invalidate()

    for i in range(NUM_PRERUN if PROFILE else 1):
        check_error(
            lib.infiniopReduceMin(descriptor, reduced_tensor.data, data_tensor.data, axes_tensor.data, None)
        )
    if PROFILE:
        start_time = time.time()
        for i in range(NUM_ITERATIONS):
            check_error(
                lib.infiniopReduceMin(descriptor, reduced_tensor.data, data_tensor.data, axes_tensor.data, None)
            )
        elapsed = (time.time() - start_time)
        print(f"    lib time: {elapsed :6f}")
    assert torch.allclose(reduced, ans, atol=1e-6, rtol=1e-6)
    check_error(lib.infiniopDestroyReduceMinDescriptor(descriptor))

def test_cpu(lib, test_cases):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)
    for data_shape, inplace in test_cases:
        test(lib, handle, "cpu", data_shape, tensor_dtype=torch.float16, inplace=inplace)
        test(lib, handle, "cpu", data_shape, tensor_dtype=torch.float32, inplace=inplace)
    destroy_handle(lib, handle)


def test_cuda(lib, test_cases):
    device = DeviceEnum.DEVICE_CUDA
    handle = create_handle(lib, device)
    for data_shape, inplace in test_cases:
        test(lib, handle, "cuda", data_shape, tensor_dtype=torch.float16, inplace=inplace)
        test(lib, handle, "cuda", data_shape, tensor_dtype=torch.float32, inplace=inplace)
    destroy_handle(lib, handle)


def test_bang(lib, test_cases):
    import torch_mlu

    device = DeviceEnum.DEVICE_BANG
    handle = create_handle(lib, device)
    for data_shape, inplace in test_cases:
        test(lib, handle, "mlu", data_shape, tensor_dtype=torch.float16, inplace=inplace)
        test(lib, handle, "mlu", data_shape, tensor_dtype=torch.float32, inplace=inplace)
    destroy_handle(lib, handle)


if __name__ == "__main__":
    test_cases = [
        # ((3, 2), Inplace.OUT_OF_PLACE),
	# ((16, 4), Inplace.OUT_OF_PLACE),
        # ((31, 15), Inplace.OUT_OF_PLACE),
	((1, 129), Inplace.OUT_OF_PLACE),
        ((129, 129), Inplace.OUT_OF_PLACE),
        ((50257, 768), Inplace.OUT_OF_PLACE),
        ((52, 32, 72), Inplace.OUT_OF_PLACE),
    ]
    args = get_args()
    lib = open_lib()
    lib.infiniopCreateReduceMinDescriptor.restype = c_int32
    lib.infiniopCreateReduceMinDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopReduceMinDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
	c_int32,
	c_int32
    ]
    lib.infiniopReduceMin.restype = c_int32
    lib.infiniopReduceMin.argtypes = [
        infiniopReduceMinDescriptor_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroyReduceMinDescriptor.restype = c_int32
    lib.infiniopDestroyReduceMinDescriptor.argtypes = [
        infiniopReduceMinDescriptor_t,
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
