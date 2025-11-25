"""
LeetCode #49: Group Anagrams
Inefficient solution with repeated sorting in nested loops
"""


def group_anagrams_slow(strs):
    """
    Group strings that are anagrams of each other.

    This is an intentionally inefficient implementation for testing.
    """
    if not strs:
        return []

    result = []
    used = [False] * len(strs)

    # Inefficient nested loop with repeated sorting
    for i in range(len(strs)):
        if used[i]:
            continue

        group = [strs[i]]
        used[i] = True

        for j in range(i + 1, len(strs)):
            if used[j]:
                continue

            # Inefficient: Sorting strings repeatedly for comparison
            sorted_i = ''.join(sorted(strs[i]))
            sorted_j = ''.join(sorted(strs[j]))

            # Checking character by character instead of direct comparison
            is_anagram = True
            if len(sorted_i) != len(sorted_j):
                is_anagram = False
            else:
                for k in range(len(sorted_i)):
                    if sorted_i[k] != sorted_j[k]:
                        is_anagram = False
                        break

            if is_anagram:
                group.append(strs[j])
                used[j] = True

        result.append(group)

    return result


# Test cases
if __name__ == "__main__":
    result = group_anagrams_slow(["eat", "tea", "tan", "ate", "nat", "bat"])
    # Sort for comparison since order doesn't matter
    result = [sorted(group) for group in result]
    result = sorted(result)
    expected = [sorted(group) for group in [["bat"], ["nat", "tan"], ["ate", "eat", "tea"]]]
    expected = sorted(expected)
    assert result == expected
    print("All test cases passed!")
