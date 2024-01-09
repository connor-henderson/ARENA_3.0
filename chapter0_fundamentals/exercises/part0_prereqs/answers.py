# %%

import sys
import numpy as np
import einops
from pathlib import Path

# Make sure exercises are in the path
section_dir = Path(__file__).parent
exercises_dir = section_dir.parent
assert exercises_dir.name == "exercises", f"This file should be run inside 'exercises/part2_cnns', not '{section_dir}'"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow, line, bar
from part0_prereqs.utils import display_array_as_img, display_soln_array_as_img
import part0_prereqs.tests as tests

MAIN = __name__ == "__main__"

# %% 1️⃣ EINOPS AND EINSUM

arr = np.load(section_dir / "numbers.npy")


# %%
display_array_as_img(arr[0])

# %%
display_soln_array_as_img(1)
# %%
print(arr.shape)
arr1 = einops.rearrange(arr, 'b c h w -> c h (b w)')
print(arr1.shape)

display_array_as_img(arr1)
# %%
display_soln_array_as_img(2)
# %%
arr2 = einops.repeat(arr[0], 'c h w -> c (repeat h) w', repeat=2)
display_array_as_img(arr2)
# %%
display_soln_array_as_img(3)
# %%
test = np.array([arr[0], arr[1]])
arr3 = einops.repeat(test, 'b c h w -> c (b h) (repeat w)', repeat=2)
print(arr3.shape)
display_array_as_img(arr3)
# %%
display_soln_array_as_img(4)

# %%
arr4 = einops.repeat(arr[0], 'c h w -> c (h repeat) w', repeat=2)
print(arr4.shape)
display_array_as_img(arr4)
# %%
display_soln_array_as_img(5)

# %%
# arr5 = einops.repeat(arr[0], 'c h w -> c h (repeat w)', repeat=3)
# arr5 = einops.reduce(arr5, 'c h w -> h w', 'prod')
arr5 = einops.rearrange(arr[0], "c h w -> h (c w)")
print(arr5.shape)
display_array_as_img(arr5)

# %%
display_soln_array_as_img(6)
# %%
arr6 = einops.rearrange(arr, "(b1 b2) c h w -> c (b1 h) (b2 w)", b1=2)
print(arr6.shape)
display_array_as_img(arr6)
# %%
import torch as t

def assert_all_equal(actual: t.Tensor, expected: t.Tensor) -> None:
    assert actual.shape == expected.shape, f"Shape mismatch, got: {actual.shape}"
    assert (actual == expected).all(), f"Value mismatch, got: {actual}"
    print("Passed!")

def assert_all_close(actual: t.Tensor, expected: t.Tensor, rtol=1e-05, atol=0.0001) -> None:
    assert actual.shape == expected.shape, f"Shape mismatch, got: {actual.shape}"
    assert t.allclose(actual, expected, rtol=rtol, atol=atol)
    print("Passed!")

# %%
def rearrange_1() -> t.Tensor:
    '''Return the following tensor using only torch.arange and einops.rearrange:

    [[3, 4],
     [5, 6],
     [7, 8]]
    '''
    tensor = t.arange(3, 9)
    tensor = einops.rearrange(tensor, '(b1 b2) -> b1 b2', b1=3)
    return tensor

expected = t.tensor([[3, 4], [5, 6], [7, 8]])
assert_all_equal(rearrange_1(), expected)
# %%
def rearrange_2() -> t.Tensor:
    '''Return the following tensor using only torch.arange and einops.rearrange:

    [[1, 2, 3],
     [4, 5, 6]]
    '''
    tensor = t.arange(1, 7)
    tensor = einops.rearrange(tensor, '(b1 b2) -> b1 b2', b1=2)
    return tensor


assert_all_equal(rearrange_2(), t.tensor([[1, 2, 3], [4, 5, 6]]))

# %%
def rearrange_3() -> t.Tensor:
    '''Return the following tensor using only torch.arange and einops.rearrange:

    [[[1], [2], [3], [4], [5], [6]]]
    '''
    tensor = t.arange(1, 7)
    tensor = einops.rearrange(tensor, '(b1 b2) -> b1 b2', b1=6)
    tensor = einops.rearrange(tensor, '(b1 b2) h -> b1 b2 h', b1=1)
    # tensor = einops.rearrange(tensor, 'a -> 1 a 1') # better implementation
    print(tensor.shape)
    return tensor


assert_all_equal(rearrange_3(), t.tensor([[[1], [2], [3], [4], [5], [6]]]))

# %%
def temperatures_average(temps: t.Tensor) -> t.Tensor:
    '''Return the average temperature for each week.

    temps: a 1D temperature containing temperatures for each day.
    Length will be a multiple of 7 and the first 7 days are for the first week, second 7 days for the second week, etc.

    You can do this with a single call to reduce.
    '''
    assert len(temps) % 7 == 0
    temps = einops.rearrange(temps, '(b1 b2) -> b1 b2', b1=3)
    avg_temps = einops.reduce(temps, 'a b -> a', 'mean')
    return avg_temps


temps = t.Tensor([71, 72, 70, 75, 71, 72, 70, 68, 65, 60, 68, 60, 55, 59, 75, 80, 85, 80, 78, 72, 83])
expected = t.tensor([71.5714, 62.1429, 79.0])
assert_all_close(temperatures_average(temps), expected)
# %%
def temperatures_differences(temps: t.Tensor) -> t.Tensor:
    '''For each day, subtract the average for the week the day belongs to.

    temps: as above
    '''
    assert len(temps) % 7 == 0
    avgs = temperatures_average(temps)
    avgs = einops.repeat(avgs, 'a -> a repeat', repeat=7)
    temps = einops.rearrange(temps, '(b1 b2) -> b1 b2', b1=3)

    minus_avg = temps - avgs
    temps = einops.rearrange(minus_avg, 'w d -> (w d)')
    return temps


expected = t.tensor(
    [
        -0.5714,
        0.4286,
        -1.5714,
        3.4286,
        -0.5714,
        0.4286,
        -1.5714,
        5.8571,
        2.8571,
        -2.1429,
        5.8571,
        -2.1429,
        -7.1429,
        -3.1429,
        -4.0,
        1.0,
        6.0,
        1.0,
        -1.0,
        -7.0,
        4.0,
    ]
)
actual = temperatures_differences(temps)
assert_all_close(actual, expected)

# %%
def temperatures_normalized(temps: t.Tensor) -> t.Tensor:
    '''For each day, subtract the weekly average and divide by the weekly standard deviation.

    temps: as above

    Pass torch.std to reduce.
    '''
    avg = einops.repeat(temperatures_average(temps), "w -> (w 7)")
    std = einops.repeat(einops.reduce(temps, "(h 7) -> h", t.std), "w -> (w 7)")
    return (temps - avg) / std


expected = t.tensor(
    [
        -0.3326,
        0.2494,
        -0.9146,
        1.9954,
        -0.3326,
        0.2494,
        -0.9146,
        1.1839,
        0.5775,
        -0.4331,
        1.1839,
        -0.4331,
        -1.4438,
        -0.6353,
        -0.8944,
        0.2236,
        1.3416,
        0.2236,
        -0.2236,
        -1.5652,
        0.8944,
    ]
)
actual = temperatures_normalized(temps)
assert_all_close(actual, expected)
# %%
def einsum_trace(mat: np.ndarray):
    '''
    Returns the same as `np.trace`.
    '''
    return einops.einsum(mat, 'i i ->')

def einsum_mv(mat: np.ndarray, vec: np.ndarray):
    '''
    Returns the same as `np.matmul`, when `mat` is a 2D array and `vec` is 1D.
    '''
    return einops.einsum(mat, vec, 'i j, j -> i')

def einsum_mm(mat1: np.ndarray, mat2: np.ndarray):
    '''
    Returns the same as `np.matmul`, when `mat1` and `mat2` are both 2D arrays.
    '''
    return einops.einsum(mat1, mat2, 'i j, j k -> i k')

def einsum_inner(vec1: np.ndarray, vec2: np.ndarray):
    '''
    Returns the same as `np.inner`.
    '''
    return einops.einsum(vec1, vec2, 'i,i->')

def einsum_outer(vec1: np.ndarray, vec2: np.ndarray):
    '''
    Returns the same as `np.outer`.
    '''
    return einops.einsum(vec1, vec2, 'i , j -> i j')

tests.test_einsum_trace(einsum_trace)
tests.test_einsum_mv(einsum_mv)
tests.test_einsum_mm(einsum_mm)
tests.test_einsum_inner(einsum_inner)
tests.test_einsum_outer(einsum_outer)
# %%
