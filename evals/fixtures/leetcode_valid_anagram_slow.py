"""
LeetCode #242: Valid Anagram
Inefficient solution with repeated sorting operations
"""


def is_anagram_slow(s, t):
    """
    Check if t is an anagram of s.

    This is an intentionally inefficient implementation for testing.
    """
    if len(s) != len(t):
        return False

    # Inefficient: Sorting inside loops and character-by-character comparison
    for char in s:
        # Sort both strings repeatedly for each character check - O(n² log n)
        s_sorted = sorted(s)
        t_sorted = sorted(t)

        # Inefficient character counting
        s_count = 0
        t_count = 0
        for c in s_sorted:
            if c == char:
                s_count += 1
        for c in t_sorted:
            if c == char:
                t_count += 1

        if s_count != t_count:
            return False

    return True


# Test cases
if __name__ == "__main__":
    assert is_anagram_slow("anagram", "nagaram") == True
    assert is_anagram_slow("rat", "car") == False
    assert is_anagram_slow("listen", "silent") == True
    print("All test cases passed!")
