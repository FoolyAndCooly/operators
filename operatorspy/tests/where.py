from ctypes import POINTER, Structure, c_int32, c_void_p
import ctypes
import sys
import os
import time

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
PROFILE = True
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE_A = auto()
    INPLACE_B = auto()


class WhereDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopWhereDescriptor_t = POINTER(WhereDescriptor)


def where(condition, x, y):
    if PROFILE:
        ans = torch.where(condition, x, y)
        torch.cuda.synchronize()
        return ans
    return torch.where(condition, x, y)


def test(
    lib,
    handle,
    torch_device,
    condition_shape,
    c_shape, 
    a_shape, 
    b_shape,
    tensor_dtype=torch.float16,
    inplace=Inplace.OUT_OF_PLACE,
):
    print(
        f"Testing Where on {torch_device} with c_shape:{c_shape} a_shape:{a_shape} b_shape:{b_shape} dtype:{tensor_dtype} inplace: {inplace.name}"
    )
    if a_shape != b_shape and inplace != Inplace.OUT_OF_PLACE:
        print("Unsupported test: broadcasting does not support in-place")
        return

    a = torch.rand(a_shape, dtype=tensor_dtype).to(torch_device)
    b = torch.rand(b_shape, dtype=tensor_dtype).to(torch_device)
    c = torch.rand(c_shape, dtype=tensor_dtype).to(torch_device) if inplace == Inplace.OUT_OF_PLACE else (a if inplace == Inplace.INPLACE_A else b)
    condition = torch.randint(0, 2, condition_shape, dtype=torch.uint8).to(torch_device)
    for i in range(NUM_PRERUN if PROFILE else 1):
        ans = where(condition, a, b)
    if PROFILE:
        start_time = time.time()
        for i in range(NUM_ITERATIONS):
            _ = where(condition, a, b)
        elapsed = (time.time() - start_time)
        print(f"pytorch time: {elapsed :6f}")


    a_tensor = to_tensor(a, lib)
    b_tensor = to_tensor(b, lib)
    c_tensor = to_tensor(c, lib) if inplace == Inplace.OUT_OF_PLACE else (a_tensor if inplace == Inplace.INPLACE_A else b_tensor)
    condition_tensor = to_tensor(condition, lib)

    descriptor = infiniopWhereDescriptor_t()

    check_error(
        lib.infiniopCreateWhereDescriptor(
            handle,
            ctypes.byref(descriptor),
	    condition_tensor.descriptor,
            c_tensor.descriptor,
            a_tensor.descriptor,
            b_tensor.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    condition_tensor.descriptor.contents.invalidate()
    c_tensor.descriptor.contents.invalidate()
    a_tensor.descriptor.contents.invalidate()
    b_tensor.descriptor.contents.invalidate()

    for i in range(NUM_PRERUN if PROFILE else 1):
        check_error(
            lib.infiniopWhere(descriptor, condition_tensor.data, c_tensor.data, a_tensor.data, b_tensor.data, None)
        )
    if PROFILE:
        start_time = time.time()
        for i in range(NUM_ITERATIONS):
            check_error(
                lib.infiniopWhere(descriptor, condition_tensor.data, c_tensor.data, a_tensor.data, b_tensor.data, None)
            )
        elapsed = (time.time() - start_time)
        print(f"    lib time: {elapsed :6f}")
    assert torch.allclose(c, ans, atol=0, rtol=1e-3)
    check_error(lib.infiniopDestroyWhereDescriptor(descriptor))


def test_cpu(lib, test_cases):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)
    for condition_shape, c_shape, a_shape, b_shape, inplace in test_cases:
        test(lib, handle, "cpu", condition_shape, c_shape, a_shape, b_shape, tensor_dtype=torch.float16, inplace=inplace)
        test(lib, handle, "cpu", condition_shape, c_shape, a_shape, b_shape, tensor_dtype=torch.float32, inplace=inplace)
    destroy_handle(lib, handle)


def test_cuda(lib, test_cases):
    device = DeviceEnum.DEVICE_CUDA
    handle = create_handle(lib, device)
    for condition_shape, c_shape, a_shape, b_shape, inplace in test_cases:
        test(lib, handle, "cuda", condition_shape, c_shape, a_shape, b_shape, tensor_dtype=torch.float16, inplace=inplace)
        test(lib, handle, "cuda", condition_shape, c_shape, a_shape, b_shape, tensor_dtype=torch.float32, inplace=inplace)
    destroy_handle(lib, handle)


def test_bang(lib, test_cases):
    import torch_mlu

    device = DeviceEnum.DEVICE_BANG
    handle = create_handle(lib, device)
    for condition_shape, c_shape, a_shape, b_shape, inplace in test_cases:
        test(lib, handle, "mlu", condition_shape, c_shape, a_shape, b_shape, tensor_dtype=torch.float16, inplace=inplace)
        test(lib, handle, "mlu", condition_shape, c_shape, a_shape, b_shape, tensor_dtype=torch.float32, inplace=inplace)
    destroy_handle(lib, handle)


if __name__ == "__main__":
    test_cases = [
        # condition_shape, c_shape, a_shape, b_shape, inplace
        # ((32, 150, 512000), (32, 150, 512000), (32, 150, 512000), (32, 150, 512000), Inplace.OUT_OF_PLACE),
        # ((32, 150, 51200), (32, 150, 51200), (32, 150, 51200), (32, 150, 1), Inplace.OUT_OF_PLACE),
        # ((32, 150, 51200), (32, 150, 51200), (32, 150, 51200), (32, 150, 51200), Inplace.OUT_OF_PLACE),
        ((1, 3), (1, 3), (1, 3), (1, 3), Inplace.OUT_OF_PLACE),
        ((), (), (), (), Inplace.OUT_OF_PLACE),
        ((2, 2), (2, 2), (2, 2), (1, 2), Inplace.OUT_OF_PLACE),
        ((2, 1, 3), (2, 2, 3), (2, 1, 3), (2, 2, 3), Inplace.OUT_OF_PLACE),
        ((32, 20, 512), (32, 20, 512), (32, 20, 512), (32, 20, 512), Inplace.INPLACE_A),
        ((32, 20, 512), (32, 20, 512), (32, 20, 512), (32, 20, 512), Inplace.INPLACE_B),
        ((1, 256, 112, 112), (32, 256, 112, 112), (32, 256, 112, 1), (32, 256, 112, 112), Inplace.OUT_OF_PLACE),
        ((32, 256, 112, 112), (32, 256, 112, 112), (32, 256, 112, 112), (32, 256, 112, 112), Inplace.OUT_OF_PLACE),
        ((2, 1, 3), (2, 4, 3), (2, 1, 3), (4, 3), Inplace.OUT_OF_PLACE),
        ((2, 3, 4, 5), (2, 3, 4, 5), (2, 3, 4, 5), (5,), Inplace.OUT_OF_PLACE),
        ((4, 5), (3, 2, 4, 5), (4, 5), (3, 2, 1, 1), Inplace.OUT_OF_PLACE),
    ]
    args = get_args()
    lib = open_lib()
    lib.infiniopCreateWhereDescriptor.restype = c_int32
    lib.infiniopCreateWhereDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopWhereDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]
    lib.infiniopWhere.restype = c_int32
    lib.infiniopWhere.argtypes = [
        infiniopWhereDescriptor_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroyWhereDescriptor.restype = c_int32
    lib.infiniopDestroyWhereDescriptor.argtypes = [
        infiniopWhereDescriptor_t,
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
