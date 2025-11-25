"""
LeetCode #238: Product of Array Except Self
Inefficient solution with excessive space usage and multiple passes
"""


def product_except_self_slow(nums):
    """
    Return array where each element is product of all other elements.

    This is an intentionally inefficient implementation for testing.
    """
    n = len(nums)
    result = []

    # Inefficient: Creating multiple intermediate arrays
    for i in range(n):
        # Create left array for each position
        left_products = []
        for j in range(i):
            left_products.append(nums[j])

        # Create right array for each position
        right_products = []
        for j in range(i + 1, n):
            right_products.append(nums[j])

        # Calculate product of left side
        left_product = 1
        for val in left_products:
            left_product *= val

        # Calculate product of right side
        right_product = 1
        for val in right_products:
            right_product *= val

        result.append(left_product * right_product)

    return result


# Test cases
if __name__ == "__main__":
    assert product_except_self_slow([1, 2, 3, 4]) == [24, 12, 8, 6]
    assert product_except_self_slow([-1, 1, 0, -3, 3]) == [0, 0, 9, 0, 0]
    print("All test cases passed!")
