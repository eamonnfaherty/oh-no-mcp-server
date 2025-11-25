"""
LeetCode #217: Contains Duplicate
Inefficient O(n²) solution using nested loops
"""


def contains_duplicate_slow(nums):
    """
    Check if array contains any duplicates.

    This is an intentionally inefficient implementation for testing.
    """
    # Inefficient O(n²) approach - comparing every element with every other
    for i in range(len(nums)):
        for j in range(len(nums)):
            if i != j and nums[i] == nums[j]:
                return True
    return False


# Test cases
if __name__ == "__main__":
    assert contains_duplicate_slow([1, 2, 3, 1]) == True
    assert contains_duplicate_slow([1, 2, 3, 4]) == False
    assert contains_duplicate_slow([1, 1, 1, 3, 3, 4, 3, 2, 4, 2]) == True
    print("All test cases passed!")
