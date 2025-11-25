"""
Evaluation tests using Claude Code CLI to get actual performance reviews.

This test:
1. Reads a LeetCode solution with performance issues
2. Calls the MCP server via Claude Code CLI to get a review prompt
3. Sends that prompt to Claude via CLI to get actual feedback
4. Compares Claude's feedback against exemplary expected feedback
5. Uses semantic similarity metrics (BERT, BLEU, Cosine) to score
"""

import subprocess
import json
from pathlib import Path
import pytest
import os

from evals.similarity_metrics import SimilarityEvaluator


# Test configuration
SIMILARITY_THRESHOLD = 0.6  # Minimum average similarity score to pass
FIXTURES_DIR = Path(__file__).parent / "fixtures"
OUTPUTS_DIR = Path(__file__).parent / "outputs"
PROJECT_ROOT = Path(__file__).parent.parent
MCP_CONFIG_FILE = Path(__file__).parent / "mcp_config.json"


@pytest.fixture(scope="session")
def evaluator():
    """Create a similarity evaluator instance."""
    return SimilarityEvaluator(threshold=SIMILARITY_THRESHOLD)


@pytest.fixture(scope="session", autouse=True)
def setup_outputs_dir():
    """Ensure outputs directory exists."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    yield


def save_output(filename: str, content: str) -> Path:
    """Save output to a file."""
    output_path = OUTPUTS_DIR / filename
    output_path.write_text(content)
    return output_path


def run_claude_code_cli(args: list[str], input_text: str = None, timeout: int = 120, use_mcp_config: bool = False) -> dict:
    """
    Run claude CLI command and return the result.

    Args:
        args: List of command arguments
        input_text: Optional input to send to stdin
        timeout: Timeout in seconds
        use_mcp_config: If True, add --mcp-config with path to config file

    Returns:
        dict with 'stdout', 'stderr', 'returncode'
    """
    cmd = ["claude"]

    # Add MCP config if requested (must use = syntax)
    if use_mcp_config:
        cmd.append(f"--mcp-config={str(MCP_CONFIG_FILE.absolute())}")

    cmd.extend(args)

    try:
        result = subprocess.run(
            cmd,
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    except subprocess.TimeoutExpired as e:
        return {
            "stdout": e.stdout.decode() if e.stdout else "",
            "stderr": e.stderr.decode() if e.stderr else "",
            "returncode": -1,
            "error": "Timeout"
        }
    except FileNotFoundError:
        pytest.skip("claude CLI not found. Install Claude Code to run these tests.")
    except Exception as e:
        return {
            "stdout": "",
            "stderr": str(e),
            "returncode": -1,
            "error": str(e)
        }


class TestLeetCodeReview:
    """Test performance reviews of LeetCode solutions via Claude Code CLI."""

    def _run_performance_review_test(
        self,
        evaluator,
        code_filename: str,
        expected_filename: str,
        output_filename: str,
        test_name: str,
        optimization_hint: str = ""
    ):
        """
        Helper method to run a performance review test.

        Args:
            evaluator: SimilarityEvaluator instance
            code_filename: Name of the code file in fixtures/
            expected_filename: Name of expected feedback file in fixtures/
            output_filename: Name for actual feedback output file
            test_name: Display name for the test (e.g., "TWO SUM")
            optimization_hint: Optional hint about the expected optimization
        """
        # Step 1: Read the inefficient code
        code_file = FIXTURES_DIR / code_filename
        inefficient_code = code_file.read_text()

        print("\n" + "="*80)
        print(f"STEP 1: Read {test_name} solution")
        print("="*80)
        print(f"Code length: {len(inefficient_code)} characters")

        # Step 2: Ask Claude to use the MCP server to review the code
        print("\n" + "="*80)
        print("STEP 2: Send code to Claude with MCP server for review")
        print("="*80)

        # Create a prompt asking Claude to use the oh-no MCP server
        user_prompt = f"""Please use the oh-no MCP server to review this code for performance issues.

```python
{inefficient_code}
```

Use the oh_no tool with scope="text" to get a review prompt, then provide your analysis following this structure:

1. **Performance Bottlenecks** - Identify the main performance issues and their impact
2. **Memory Usage** - Analyze memory consumption patterns
3. **Algorithm Complexity** - Provide time and space complexity analysis
4. **Optimization Suggestions** - Focus on the PRIMARY recommended optimization{optimization_hint}
5. **Best Practices** - List key coding best practices relevant to this code
6. **Performance Impact** - Quantify the improvement with concrete examples

Keep the response focused and concise. For optimization suggestions, provide ONE main recommended approach with code example, not multiple alternatives."""

        # Send to Claude with MCP config and print mode
        # Use --dangerously-skip-permissions to bypass tool permission prompts
        claude_result = run_claude_code_cli(
            ["--print", "--dangerously-skip-permissions", user_prompt],
            timeout=120,
            use_mcp_config=True
        )

        if claude_result["returncode"] != 0:
            pytest.fail(
                f"Claude failed.\n"
                f"Error: {claude_result['stderr']}"
            )

        actual_feedback = claude_result["stdout"].strip()
        save_output(output_filename, actual_feedback)

        print(f"Feedback received: {len(actual_feedback)} characters")
        print(f"Preview: {actual_feedback[:300]}...")

        # Step 3: Load exemplary expected feedback
        print("\n" + "="*80)
        print("STEP 3: Load exemplary expected feedback")
        print("="*80)

        expected_feedback = (FIXTURES_DIR / expected_filename).read_text()
        print(f"Expected feedback: {len(expected_feedback)} characters")

        # Step 4: Compare using semantic similarity
        print("\n" + "="*80)
        print("STEP 4: Compare feedback using semantic similarity")
        print("="*80)

        result = evaluator.evaluate(expected_feedback, actual_feedback)

        # Print detailed results
        print("\n" + "="*80)
        print(f"{test_name} PERFORMANCE REVIEW EVALUATION")
        print("="*80)
        print(f"BERT Score:      {result.bert_score:.4f}")
        print(f"Self-BLEU Score: {result.self_bleu_score:.4f}")
        print(f"Cosine Score:    {result.cosine_score:.4f}")
        print(f"Average Score:   {result.average_score:.4f}")
        print(f"Threshold:       {result.threshold:.4f}")
        print(f"Result:          {'✓ PASS' if result.passed else '✗ FAIL'}")
        print("="*80 + "\n")

        # If test fails, show both responses for comparison
        if not result.passed:
            print("\n" + "="*80)
            print("TEST FAILED - RESPONSE COMPARISON")
            print("="*80)

            # Identify which metrics failed
            failed_metrics = []
            if result.bert_score < result.threshold:
                failed_metrics.append(("BERT", result.bert_score))
            if result.self_bleu_score < result.threshold:
                failed_metrics.append(("Self-BLEU", result.self_bleu_score))
            if result.cosine_score < result.threshold:
                failed_metrics.append(("Cosine", result.cosine_score))

            # Plain English explanation
            print("\n" + "-"*80)
            print("WHY THIS TEST FAILED:")
            print("-"*80)
            print(f"The test measures how similar Claude's actual performance review is to")
            print(f"an exemplary expected review using three metrics:")
            print(f"  • BERT Score: {result.bert_score:.4f} - measures semantic meaning similarity")
            print(f"  • Self-BLEU: {result.self_bleu_score:.4f} - measures word/phrase overlap")
            print(f"  • Cosine Similarity: {result.cosine_score:.4f} - measures vector similarity")
            print(f"\nAverage similarity: {result.average_score:.4f}")
            print(f"Required threshold: {result.threshold:.4f}")

            # Show which specific metrics failed
            if failed_metrics:
                print(f"\n⚠️  METRICS BELOW THRESHOLD ({result.threshold:.4f}):")
                for metric_name, score in failed_metrics:
                    print(f"  ✗ {metric_name}: {score:.4f}")
                    if metric_name == "BERT":
                        print(f"    → BERT measures semantic meaning using AI embeddings")
                        print(f"    → Low score means the actual feedback has different conceptual content")
                        print(f"    → Example: Talking about memory instead of time complexity")
                    elif metric_name == "Self-BLEU":
                        print(f"    → Self-BLEU measures exact word and phrase overlap")
                        print(f"    → Low score means different vocabulary/phrasing was used")
                        print(f"    → Example: 'hash map' vs 'dictionary', 'O(n)' vs 'linear time'")
                    elif metric_name == "Cosine":
                        print(f"    → Cosine measures directional similarity of AI embeddings")
                        print(f"    → Low score means the responses focus on different aspects")
                        print(f"    → Example: Expected focuses on algorithms, actual on style")

            print(f"\nThe average similarity score is below the threshold, meaning Claude's")
            print(f"review doesn't closely enough match the expected exemplary feedback.")
            print(f"\nThis could indicate:")
            print(f"  • The MCP server isn't providing the right guidance")
            print(f"  • Claude isn't following the MCP prompt effectively")
            print(f"  • The performance issues weren't identified correctly")
            print("-"*80)

            print("\n" + "-"*80)
            print("EXEMPLARY EXPECTED FEEDBACK:")
            print("-"*80)
            print(expected_feedback)
            print("\n" + "-"*80)
            print("ACTUAL CLAUDE FEEDBACK:")
            print("-"*80)
            print(actual_feedback)
            print("\n" + "="*80 + "\n")

        # Assert the test passes
        assert result.passed, (
            f"Similarity check failed. Average score {result.average_score:.4f} "
            f"is below threshold {result.threshold:.4f}\n"
            f"This means Claude's feedback doesn't match the expected quality.\n"
            f"See above for detailed explanation and comparison of expected vs actual feedback."
        )

        # Also verify individual metrics aren't too low
        if result.bert_score <= 0.4:
            print("\n" + "="*80)
            print("BERT SCORE TOO LOW - RESPONSE COMPARISON")
            print("="*80)

            # Plain English explanation for BERT failure
            print("\n" + "-"*80)
            print("WHY BERT SCORE IS TOO LOW:")
            print("-"*80)
            print(f"BERT score: {result.bert_score:.4f} (minimum required: 0.40)")
            print(f"\nBERT (Bidirectional Encoder Representations from Transformers) measures")
            print(f"the semantic meaning similarity between two texts. A score below 0.4")
            print(f"indicates the actual feedback has fundamentally different meaning from")
            print(f"the expected feedback.")
            print(f"\nPossible reasons:")
            print(f"  • Claude identified completely different performance issues")
            print(f"  • The feedback focuses on wrong aspects of the code")
            print(f"  • Claude didn't use the MCP server's review prompt at all")
            print(f"  • The response is asking for permissions or is an error message")
            print("-"*80)

            print("\n" + "-"*80)
            print("EXEMPLARY EXPECTED FEEDBACK:")
            print("-"*80)
            print(expected_feedback)
            print("\n" + "-"*80)
            print("ACTUAL CLAUDE FEEDBACK:")
            print("-"*80)
            print(actual_feedback)
            print("\n" + "="*80 + "\n")

        assert result.bert_score > 0.4, (
            f"BERT score too low: {result.bert_score:.4f}\n"
            f"Semantic meaning is too different from expected feedback.\n"
            f"See above for detailed explanation and comparison of expected vs actual feedback."
        )

    def test_two_sum_slow_solution(self, evaluator):
        """Test reviewing the inefficient Two Sum solution (O(n²) nested loops)."""
        self._run_performance_review_test(
            evaluator,
            "leetcode_two_sum_slow.py",
            "leetcode_two_sum_expected_feedback.txt",
            "two_sum_actual_feedback.txt",
            "TWO SUM",
            " (hash map approach for O(n) solution)"
        )

    def test_valid_palindrome_slow_solution(self, evaluator):
        """Test reviewing Valid Palindrome with string concatenation issues."""
        self._run_performance_review_test(
            evaluator,
            "leetcode_valid_palindrome_slow.py",
            "leetcode_valid_palindrome_expected_feedback.txt",
            "valid_palindrome_actual_feedback.txt",
            "VALID PALINDROME",
            " (list comprehension and two-pointer approach)"
        )

    def test_contains_duplicate_slow_solution(self, evaluator):
        """Test reviewing Contains Duplicate with nested loops."""
        self._run_performance_review_test(
            evaluator,
            "leetcode_contains_duplicate_slow.py",
            "leetcode_contains_duplicate_expected_feedback.txt",
            "contains_duplicate_actual_feedback.txt",
            "CONTAINS DUPLICATE",
            " (hash set for O(n) solution)"
        )

    def test_stock_slow_solution(self, evaluator):
        """Test reviewing Best Time to Buy/Sell Stock with nested loops."""
        self._run_performance_review_test(
            evaluator,
            "leetcode_stock_slow.py",
            "leetcode_stock_expected_feedback.txt",
            "stock_actual_feedback.txt",
            "BEST TIME TO BUY/SELL STOCK",
            " (single pass with min tracking)"
        )

    def test_valid_anagram_slow_solution(self, evaluator):
        """Test reviewing Valid Anagram with repeated sorting."""
        self._run_performance_review_test(
            evaluator,
            "leetcode_valid_anagram_slow.py",
            "leetcode_valid_anagram_expected_feedback.txt",
            "valid_anagram_actual_feedback.txt",
            "VALID ANAGRAM",
            " (hash map for character frequency counting)"
        )

    def test_product_array_slow_solution(self, evaluator):
        """Test reviewing Product of Array Except Self with excessive arrays."""
        self._run_performance_review_test(
            evaluator,
            "leetcode_product_array_slow.py",
            "leetcode_product_array_expected_feedback.txt",
            "product_array_actual_feedback.txt",
            "PRODUCT OF ARRAY EXCEPT SELF",
            " (two-pass prefix/suffix product approach)"
        )

    def test_max_subarray_slow_solution(self, evaluator):
        """Test reviewing Maximum Subarray with O(n³) brute force."""
        self._run_performance_review_test(
            evaluator,
            "leetcode_max_subarray_slow.py",
            "leetcode_max_subarray_expected_feedback.txt",
            "max_subarray_actual_feedback.txt",
            "MAXIMUM SUBARRAY",
            " (Kadane's algorithm for O(n) solution)"
        )

    def test_merge_intervals_slow_solution(self, evaluator):
        """Test reviewing Merge Intervals with repeated passes."""
        self._run_performance_review_test(
            evaluator,
            "leetcode_merge_intervals_slow.py",
            "leetcode_merge_intervals_expected_feedback.txt",
            "merge_intervals_actual_feedback.txt",
            "MERGE INTERVALS",
            " (sort once then single merge pass)"
        )

    def test_group_anagrams_slow_solution(self, evaluator):
        """Test reviewing Group Anagrams with repeated sorting."""
        self._run_performance_review_test(
            evaluator,
            "leetcode_group_anagrams_slow.py",
            "leetcode_group_anagrams_expected_feedback.txt",
            "group_anagrams_actual_feedback.txt",
            "GROUP ANAGRAMS",
            " (hash map with sorted keys)"
        )

    def test_longest_substring_slow_solution(self, evaluator):
        """Test reviewing Longest Substring with substring recreation."""
        self._run_performance_review_test(
            evaluator,
            "leetcode_longest_substring_slow.py",
            "leetcode_longest_substring_expected_feedback.txt",
            "longest_substring_actual_feedback.txt",
            "LONGEST SUBSTRING WITHOUT REPEATING",
            " (sliding window with hash set)"
        )

    def test_binary_search_optimal_solution(self, evaluator):
        """Test reviewing optimal Binary Search (should find NO issues)."""
        self._run_performance_review_test(
            evaluator,
            "leetcode_binary_search_optimal.py",
            "leetcode_binary_search_expected_feedback.txt",
            "binary_search_actual_feedback.txt",
            "BINARY SEARCH (OPTIMAL)",
            " - this code is already optimal"
        )


def test_claude_code_cli_available():
    """Test that claude-code CLI is available."""
    result = run_claude_code_cli(["--version"])
    if result["returncode"] != 0:
        pytest.skip("claude-code CLI not found or not working")

    print(f"\nClaude Code version: {result['stdout'].strip()}")


def test_mcp_server_configured():
    """Test that oh-no MCP server can be accessed via --mcp-config."""
    result = run_claude_code_cli(["mcp", "list"], use_mcp_config=True)
    if result["returncode"] != 0:
        pytest.skip(f"Cannot list MCP servers: {result['stderr']}")

    if "oh-no" not in result["stdout"]:
        pytest.skip(
            "oh-no MCP server not found in MCP list. "
            f"Output: {result['stdout']}"
        )

    print(f"\noh-no MCP server is accessible via --mcp-config")
