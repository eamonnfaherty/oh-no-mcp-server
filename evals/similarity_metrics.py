"""Semantic similarity metrics for evaluating MCP server outputs."""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk


# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


@dataclass
class SimilarityResult:
    """Result of similarity comparison."""
    bert_score: float
    self_bleu_score: float
    cosine_score: float
    average_score: float
    passed: bool
    threshold: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'bert_score': self.bert_score,
            'self_bleu_score': self.self_bleu_score,
            'cosine_score': self.cosine_score,
            'average_score': self.average_score,
            'passed': self.passed,
            'threshold': self.threshold,
        }


class SimilarityEvaluator:
    """Evaluates similarity between expected and actual outputs."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', threshold: float = 0.7):
        """
        Initialize the evaluator.

        Args:
            model_name: Name of the sentence transformer model to use
            threshold: Minimum average score required to pass (0-1)
        """
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold
        self.smoothing = SmoothingFunction()

    def compute_bert_similarity(self, text1: str, text2: str) -> float:
        """
        Compute BERT-based semantic similarity using sentence transformers.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score between 0 and 1
        """
        embeddings = self.model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)

    def compute_self_bleu(self, text1: str, text2: str) -> float:
        """
        Compute Self-BLEU score between two texts.

        Self-BLEU measures n-gram overlap between texts.

        Args:
            text1: Reference text
            text2: Candidate text

        Returns:
            BLEU score between 0 and 1
        """
        # Tokenize
        reference = nltk.word_tokenize(text1.lower())
        candidate = nltk.word_tokenize(text2.lower())

        # Compute BLEU with smoothing to handle edge cases
        score = sentence_bleu(
            [reference],
            candidate,
            smoothing_function=self.smoothing.method1
        )
        return float(score)

    def compute_cosine_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between text embeddings.

        This uses the same BERT embeddings as compute_bert_similarity
        but is kept separate for clarity in reporting.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Cosine similarity score between 0 and 1
        """
        embeddings = self.model.encode([text1, text2])
        # Normalize embeddings
        norm1 = embeddings[0] / np.linalg.norm(embeddings[0])
        norm2 = embeddings[1] / np.linalg.norm(embeddings[1])
        # Compute cosine similarity
        similarity = np.dot(norm1, norm2)
        return float(similarity)

    def evaluate(self, expected: str, actual: str) -> SimilarityResult:
        """
        Evaluate similarity between expected and actual outputs.

        Args:
            expected: Expected output text
            actual: Actual output text

        Returns:
            SimilarityResult with all metrics and pass/fail
        """
        # Compute all metrics
        bert_score = self.compute_bert_similarity(expected, actual)
        self_bleu = self.compute_self_bleu(expected, actual)
        cosine_score = self.compute_cosine_similarity(expected, actual)

        # Calculate average score
        average_score = (bert_score + self_bleu + cosine_score) / 3.0

        # Determine pass/fail
        passed = average_score >= self.threshold

        return SimilarityResult(
            bert_score=bert_score,
            self_bleu_score=self_bleu,
            cosine_score=cosine_score,
            average_score=average_score,
            passed=passed,
            threshold=self.threshold
        )

    def batch_evaluate(
        self,
        comparisons: List[Tuple[str, str]]
    ) -> List[SimilarityResult]:
        """
        Evaluate multiple expected/actual pairs.

        Args:
            comparisons: List of (expected, actual) tuples

        Returns:
            List of SimilarityResult objects
        """
        results = []
        for expected, actual in comparisons:
            result = self.evaluate(expected, actual)
            results.append(result)
        return results
