"""
LeetCode #121: Best Time to Buy and Sell Stock
Inefficient O(n²) solution using nested loops
"""


def max_profit_slow(prices):
    """
    Find maximum profit from buying and selling stock once.

    This is an intentionally inefficient implementation for testing.
    """
    if not prices:
        return 0

    # Inefficient O(n²) approach - checking every buy/sell combination
    max_profit = 0
    for i in range(len(prices)):
        for j in range(i + 1, len(prices)):
            profit = prices[j] - prices[i]
            if profit > max_profit:
                max_profit = profit

    return max_profit


# Test cases
if __name__ == "__main__":
    assert max_profit_slow([7, 1, 5, 3, 6, 4]) == 5
    assert max_profit_slow([7, 6, 4, 3, 1]) == 0
    assert max_profit_slow([2, 4, 1]) == 2
    print("All test cases passed!")
