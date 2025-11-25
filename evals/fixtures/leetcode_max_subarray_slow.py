"""
LeetCode #53: Maximum Subarray
Inefficient O(n³) brute force solution
"""


def max_subarray_slow(nums):
    """
    Find contiguous subarray with largest sum.

    This is an intentionally inefficient implementation for testing.
    """
    if not nums:
        return 0

    max_sum = float('-inf')

    # Inefficient O(n³) approach - three nested loops
    for i in range(len(nums)):
        for j in range(i, len(nums)):
            # Calculate sum of subarray from i to j
            current_sum = 0
            for k in range(i, j + 1):
                current_sum += nums[k]

            if current_sum > max_sum:
                max_sum = current_sum

    return max_sum


# Test cases
if __name__ == "__main__":
    assert max_subarray_slow([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == 6
    assert max_subarray_slow([1]) == 1
    assert max_subarray_slow([5, 4, -1, 7, 8]) == 23
    print("All test cases passed!")
