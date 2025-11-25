"""
LeetCode #125: Valid Palindrome
Inefficient solution with excessive string operations
"""


def is_palindrome_slow(s):
    """
    Check if string is a palindrome after removing non-alphanumeric chars.

    This is an intentionally inefficient implementation for testing.
    """
    # Inefficient: Multiple passes and string concatenations
    cleaned = ""
    for char in s:
        if char.isalnum():
            cleaned = cleaned + char.lower()  # String concatenation in loop - O(n²)

    # Inefficient: Creating reversed string instead of two-pointer
    reversed_str = ""
    for i in range(len(cleaned) - 1, -1, -1):
        reversed_str = reversed_str + cleaned[i]  # More O(n²) concatenation

    return cleaned == reversed_str


# Test cases
if __name__ == "__main__":
    assert is_palindrome_slow("A man, a plan, a canal: Panama") == True
    assert is_palindrome_slow("race a car") == False
    assert is_palindrome_slow(" ") == True
    print("All test cases passed!")
