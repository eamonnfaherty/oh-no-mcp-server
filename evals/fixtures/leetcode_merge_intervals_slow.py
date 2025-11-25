"""
LeetCode #56: Merge Intervals
Inefficient solution with repeated passes and list operations
"""


def merge_intervals_slow(intervals):
    """
    Merge overlapping intervals.

    This is an intentionally inefficient implementation for testing.
    """
    if not intervals:
        return []

    # Inefficient: Repeatedly checking and merging in nested loops
    result = intervals.copy()
    merged = True

    while merged:
        merged = False
        new_result = []

        # Inefficient nested loop checking all pairs
        for i in range(len(result)):
            current = result[i]
            was_merged = False

            for j in range(len(result)):
                if i != j:
                    other = result[j]
                    # Check if intervals overlap
                    if current[0] <= other[1] and other[0] <= current[1]:
                        # Merge intervals
                        merged_interval = [
                            min(current[0], other[0]),
                            max(current[1], other[1])
                        ]
                        if merged_interval not in new_result:
                            new_result.append(merged_interval)
                            merged = True
                            was_merged = True
                            break

            if not was_merged and current not in new_result:
                new_result.append(current)

        result = new_result

    return sorted(result)


# Test cases
if __name__ == "__main__":
    assert merge_intervals_slow([[1, 3], [2, 6], [8, 10], [15, 18]]) == [[1, 6], [8, 10], [15, 18]]
    assert merge_intervals_slow([[1, 4], [4, 5]]) == [[1, 5]]
    print("All test cases passed!")
