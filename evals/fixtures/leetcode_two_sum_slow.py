"""
LeetCode #1: Two Sum
Inefficient O(n²) solution using nested loops
"""


def two_sum_slow(nums, target):
    """
    Given an array of integers nums and an integer target,
    return indices of the two numbers such that they add up to target.

    This is an intentionally inefficient implementation for testing.
    """
    # Inefficient O(n²) approach - checking every pair
    for i in range(len(nums)):
        for j in range(len(nums)):
            if i != j:
                if nums[i] + nums[j] == target:
                    return [i, j]
    return []


# Test cases
if __name__ == "__main__":
    assert two_sum_slow([2, 7, 11, 15], 9) == [0, 1]
    assert two_sum_slow([3, 2, 4], 6) == [1, 2]
    assert two_sum_slow([3, 3], 6) == [0, 1]
    print("All test cases passed!")
