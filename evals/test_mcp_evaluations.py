"""Evaluation tests for oh-no-mcp-server using CLI and semantic similarity."""

import json
import subprocess
import pytest
from pathlib import Path
from typing import Dict, Any
from evals.similarity_metrics import SimilarityEvaluator


# Test configuration
SIMILARITY_THRESHOLD = 0.6  # Minimum average similarity score to pass
FIXTURES_DIR = Path(__file__).parent / "fixtures"
OUTPUTS_DIR = Path(__file__).parent / "outputs"


@pytest.fixture(scope="session")
def evaluator():
    """Create a similarity evaluator instance."""
    return SimilarityEvaluator(threshold=SIMILARITY_THRESHOLD)


@pytest.fixture(scope="session", autouse=True)
def setup_outputs_dir():
    """Ensure outputs directory exists."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    yield
    # Optionally clean up after tests
    # for file in OUTPUTS_DIR.glob("*.txt"):
    #     file.unlink()


def call_mcp_tool(tool_name: str, arguments: Dict[str, Any]) -> str:
    """
    Call the MCP server tool via CLI and return the output.

    This simulates calling the MCP server through Claude Code CLI.

    Args:
        tool_name: Name of the MCP tool to call
        arguments: Arguments to pass to the tool

    Returns:
        The text output from the tool
    """
    # For now, we'll use Python to call the tool directly
    # In production, this would use the actual MCP CLI
    import asyncio
    from oh_no_mcp_server.server import call_tool

    async def run_tool():
        result = await call_tool(tool_name, arguments)
        return result[0].text if result else ""

    return asyncio.run(run_tool())


def save_output(filename: str, content: str) -> Path:
    """
    Save output to a file.

    Args:
        filename: Name of the output file
        content: Content to save

    Returns:
        Path to the saved file
    """
    output_path = OUTPUTS_DIR / filename
    output_path.write_text(content)
    return output_path


class TestTextScopeReview:
    """Test performance review of text/code snippets."""

    def test_simple_code_snippet_review(self, evaluator):
        """Test reviewing a simple inefficient code snippet."""
        # Read the sample code
        sample_code = (FIXTURES_DIR / "sample_code.py").read_text()

        # Extract just the slow_function for testing
        code_snippet = """def slow_function():
    result = []
    for i in range(1000):
        result.append(i * 2)
    return result"""

        # Call the MCP tool
        actual_output = call_mcp_tool(
            "oh_no",
            {
                "scope": "text",
                "content": code_snippet
            }
        )

        # Save the actual output
        save_output("text_review_actual.txt", actual_output)

        # Load expected output
        expected_output = (FIXTURES_DIR / "expected_text_review.txt").read_text()

        # Evaluate similarity
        result = evaluator.evaluate(expected_output, actual_output)

        # Print detailed results
        print("\n" + "="*80)
        print("TEXT SCOPE REVIEW EVALUATION")
        print("="*80)
        print(f"BERT Score:      {result.bert_score:.4f}")
        print(f"Self-BLEU Score: {result.self_bleu_score:.4f}")
        print(f"Cosine Score:    {result.cosine_score:.4f}")
        print(f"Average Score:   {result.average_score:.4f}")
        print(f"Threshold:       {result.threshold:.4f}")
        print(f"Result:          {'✓ PASS' if result.passed else '✗ FAIL'}")
        print("="*80 + "\n")

        # Assert the test passes
        assert result.passed, (
            f"Similarity check failed. Average score {result.average_score:.4f} "
            f"is below threshold {result.threshold:.4f}"
        )

        # Also check individual metrics aren't too low
        assert result.bert_score > 0.5, f"BERT score too low: {result.bert_score:.4f}"
        assert result.cosine_score > 0.5, f"Cosine score too low: {result.cosine_score:.4f}"


class TestFileScopeReview:
    """Test performance review of individual files."""

    def test_file_review(self, evaluator):
        """Test reviewing a Python file."""
        # Use the sample code file
        file_path = FIXTURES_DIR / "sample_code.py"

        # Call the MCP tool
        actual_output = call_mcp_tool(
            "oh_no",
            {
                "scope": "file",
                "content": str(file_path)
            }
        )

        # Save the actual output
        save_output("file_review_actual.txt", actual_output)

        # Expected prompt format for file review - read from actual file content
        file_content = (FIXTURES_DIR / "sample_code.py").read_text()
        expected_prompt = (
            "Please review the following code for performance issues and potential optimizations:\n\n"
            f"File: {str(file_path)}\n\n"
            "```\n"
            f"{file_content}"
            "```\n\n"
            "Provide a detailed analysis including:\n"
            "1. Performance bottlenecks or inefficiencies\n"
            "2. Memory usage concerns\n"
            "3. Algorithm complexity issues\n"
            "4. Suggestions for optimization\n"
            "5. Best practices recommendations\n\n"
            "Please provide the review as text output."
        )

        # Evaluate similarity
        result = evaluator.evaluate(expected_prompt, actual_output)

        # Print detailed results
        print("\n" + "="*80)
        print("FILE SCOPE REVIEW EVALUATION")
        print("="*80)
        print(f"BERT Score:      {result.bert_score:.4f}")
        print(f"Self-BLEU Score: {result.self_bleu_score:.4f}")
        print(f"Cosine Score:    {result.cosine_score:.4f}")
        print(f"Average Score:   {result.average_score:.4f}")
        print(f"Threshold:       {result.threshold:.4f}")
        print(f"Result:          {'✓ PASS' if result.passed else '✗ FAIL'}")
        print("="*80 + "\n")

        # Assert the test passes
        assert result.passed, (
            f"Similarity check failed. Average score {result.average_score:.4f} "
            f"is below threshold {result.threshold:.4f}"
        )


class TestPromptQuality:
    """Test that the prompts generated are of high quality."""

    def test_prompt_contains_key_elements(self, evaluator):
        """Test that generated prompts contain essential elements."""
        code_snippet = "x = [i for i in range(1000000)]"

        actual_output = call_mcp_tool(
            "oh_no",
            {
                "scope": "text",
                "content": code_snippet
            }
        )

        save_output("prompt_quality_actual.txt", actual_output)

        # Expected prompt format
        expected_prompt = """Please review the following code for performance issues and potential optimizations:

```
x = [i for i in range(1000000)]
```

Provide a detailed analysis including:
1. Performance bottlenecks or inefficiencies
2. Memory usage concerns
3. Algorithm complexity issues
4. Suggestions for optimization
5. Best practices recommendations

Please provide the review as text output."""

        result = evaluator.evaluate(expected_prompt, actual_output)

        print("\n" + "="*80)
        print("PROMPT QUALITY EVALUATION")
        print("="*80)
        print(f"BERT Score:      {result.bert_score:.4f}")
        print(f"Self-BLEU Score: {result.self_bleu_score:.4f}")
        print(f"Cosine Score:    {result.cosine_score:.4f}")
        print(f"Average Score:   {result.average_score:.4f}")
        print(f"Threshold:       {result.threshold:.4f}")
        print(f"Result:          {'✓ PASS' if result.passed else '✗ FAIL'}")
        print("="*80 + "\n")

        assert result.passed, (
            f"Prompt quality check failed. Average score {result.average_score:.4f} "
            f"is below threshold {result.threshold:.4f}"
        )


class TestOutputConsistency:
    """Test that outputs are consistent across multiple runs."""

    def test_consistency_across_runs(self, evaluator):
        """Test that the same input produces similar outputs."""
        code_snippet = "def test(): return sum([i**2 for i in range(100)])"

        # Run the tool twice
        output1 = call_mcp_tool(
            "oh_no",
            {"scope": "text", "content": code_snippet}
        )

        output2 = call_mcp_tool(
            "oh_no",
            {"scope": "text", "content": code_snippet}
        )

        save_output("consistency_run1.txt", output1)
        save_output("consistency_run2.txt", output2)

        # Outputs should be identical or very similar
        result = evaluator.evaluate(output1, output2)

        print("\n" + "="*80)
        print("OUTPUT CONSISTENCY EVALUATION")
        print("="*80)
        print(f"BERT Score:      {result.bert_score:.4f}")
        print(f"Self-BLEU Score: {result.self_bleu_score:.4f}")
        print(f"Cosine Score:    {result.cosine_score:.4f}")
        print(f"Average Score:   {result.average_score:.4f}")
        print(f"Threshold:       {result.threshold:.4f}")
        print(f"Result:          {'✓ PASS' if result.passed else '✗ FAIL'}")
        print("="*80 + "\n")

        # For consistency, we expect very high similarity (0.95+)
        assert result.average_score > 0.95, (
            f"Output consistency check failed. Average score {result.average_score:.4f} "
            f"should be > 0.95 for identical inputs"
        )


def test_evaluator_metrics_range(evaluator):
    """Test that all metrics return values in valid range [0, 1]."""
    text1 = "This is a test string for evaluation."
    text2 = "This is another test string for comparison."

    result = evaluator.evaluate(text1, text2)

    assert 0.0 <= result.bert_score <= 1.0, f"BERT score out of range: {result.bert_score}"
    assert 0.0 <= result.self_bleu_score <= 1.0, f"Self-BLEU score out of range: {result.self_bleu_score}"
    assert 0.0 <= result.cosine_score <= 1.0, f"Cosine score out of range: {result.cosine_score}"
    assert 0.0 <= result.average_score <= 1.0, f"Average score out of range: {result.average_score}"
