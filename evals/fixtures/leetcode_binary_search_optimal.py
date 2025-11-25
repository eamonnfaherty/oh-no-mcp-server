"""
LeetCode #704: Binary Search
Optimal O(log n) solution - NO PERFORMANCE ISSUES
"""


def binary_search(nums, target):
    """
    Search for target in sorted array using binary search.

    This is an optimal implementation with no performance issues.
    """
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = left + (right - left) // 2  # Prevents overflow

        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1


# Test cases
if __name__ == "__main__":
    assert binary_search([-1, 0, 3, 5, 9, 12], 9) == 4
    assert binary_search([-1, 0, 3, 5, 9, 12], 2) == -1
    assert binary_search([5], 5) == 0
    print("All test cases passed!")
