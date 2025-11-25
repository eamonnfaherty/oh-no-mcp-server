"""
LeetCode #3: Longest Substring Without Repeating Characters
Inefficient sliding window with set recreation
"""


def length_longest_substring_slow(s):
    """
    Find length of longest substring without repeating characters.

    This is an intentionally inefficient implementation for testing.
    """
    if not s:
        return 0

    max_length = 0

    # Sliding window but with inefficient set operations
    for i in range(len(s)):
        for j in range(i, len(s)):
            # Inefficient: Recreating set and checking entire substring each time
            substring = s[i:j + 1]

            # Rebuild character set from scratch each iteration
            char_set = set()
            has_duplicate = False

            for char in substring:
                if char in char_set:
                    has_duplicate = True
                    break
                # Inefficient: Adding one at a time instead of using set(substring)
                char_set.add(char)

            if not has_duplicate:
                current_length = j - i + 1
                if current_length > max_length:
                    max_length = current_length
            else:
                break

    return max_length


# Test cases
if __name__ == "__main__":
    assert length_longest_substring_slow("abcabcbb") == 3
    assert length_longest_substring_slow("bbbbb") == 1
    assert length_longest_substring_slow("pwwkew") == 3
    assert length_longest_substring_slow("") == 0
    print("All test cases passed!")
