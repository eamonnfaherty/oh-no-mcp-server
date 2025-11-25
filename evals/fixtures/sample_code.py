"""Sample code for testing performance review."""


def slow_function():
    """Example of inefficient code."""
    result = []
    for i in range(1000):
        result.append(i * 2)
    return result


def inefficient_search(items, target):
    """Linear search example."""
    for i in range(len(items)):
        if items[i] == target:
            return i
    return -1
