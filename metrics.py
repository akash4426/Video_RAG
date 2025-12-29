"""
Metrics Module for Video RAG Research Evaluation
================================================

This module provides comprehensive metrics for evaluating retrieval performance
in the Video RAG system, suitable for research papers and academic evaluation.

Metrics included:
- Top-K Chunks Retrieval Percentage
- Precision@K and Recall@K
- Mean Average Precision (MAP)
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG@K)
- Average Similarity Score
- Coverage and Diversity metrics
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import json
import logging
from datetime import datetime

logger = logging.getLogger("VideoRAG.Metrics")


class RetrievalMetrics:
    """
    Compute and track retrieval metrics for Video RAG evaluation.
    
    This class provides methods to calculate various information retrieval
    metrics commonly used in research papers.
    """
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.queries_evaluated = 0
        self.accumulated_metrics = {
            "top_k_retrieval_percentage": [],
            "precision_at_k": [],
            "recall_at_k": [],
            "average_precision": [],
            "reciprocal_rank": [],
            "ndcg_at_k": [],
            "avg_similarity_score": [],
        }
    
    def compute_top_k_retrieval_percentage(
        self,
        retrieved_indices: np.ndarray,
        relevant_indices: List[int],
        k: int
    ) -> float:
        """
        Calculate Top-K Chunks Retrieval Percentage.
        
        This metric measures what percentage of relevant chunks were retrieved
        in the top-K results.
        
        Formula:
            Top-K Retrieval % = (# relevant chunks in top-K) / (total # relevant chunks) × 100
        
        Args:
            retrieved_indices: Array of indices for top-K retrieved results
            relevant_indices: List of ground truth relevant indices
            k: Number of top results to consider
        
        Returns:
            Percentage of relevant chunks retrieved in top-K (0-100)
        
        Example:
            >>> metrics = RetrievalMetrics()
            >>> retrieved = np.array([5, 12, 8, 20])
            >>> relevant = [5, 8, 15, 30]
            >>> pct = metrics.compute_top_k_retrieval_percentage(retrieved[:3], relevant, k=3)
            >>> print(f"Top-3 Retrieval: {pct}%")
            # Top-3 Retrieval: 50.0%  (2 out of 4 relevant chunks found)
        """
        if len(relevant_indices) == 0:
            logger.warning("No relevant indices provided for Top-K calculation")
            return 0.0
        
        # Take only top-K results
        top_k_retrieved = set(retrieved_indices[:k])
        relevant_set = set(relevant_indices)
        
        # Count how many relevant chunks are in top-K
        relevant_in_top_k = len(top_k_retrieved.intersection(relevant_set))
        
        # Calculate percentage
        percentage = (relevant_in_top_k / len(relevant_indices)) * 100
        
        return percentage
    
    def compute_precision_at_k(
        self,
        retrieved_indices: np.ndarray,
        relevant_indices: List[int],
        k: int
    ) -> float:
        """
        Calculate Precision@K.
        
        Precision@K measures what fraction of the top-K retrieved results
        are actually relevant.
        
        Formula:
            Precision@K = (# relevant items in top-K) / K
        
        Args:
            retrieved_indices: Array of retrieved result indices
            relevant_indices: List of ground truth relevant indices
            k: Number of top results to consider
        
        Returns:
            Precision score (0.0 to 1.0)
        
        Example:
            If top-5 results contain 3 relevant items: Precision@5 = 3/5 = 0.6
        """
        if k == 0:
            return 0.0
        
        top_k_retrieved = set(retrieved_indices[:k])
        relevant_set = set(relevant_indices)
        
        relevant_in_top_k = len(top_k_retrieved.intersection(relevant_set))
        
        return relevant_in_top_k / k
    
    def compute_recall_at_k(
        self,
        retrieved_indices: np.ndarray,
        relevant_indices: List[int],
        k: int
    ) -> float:
        """
        Calculate Recall@K.
        
        Recall@K measures what fraction of all relevant items were found
        in the top-K results.
        
        Formula:
            Recall@K = (# relevant items in top-K) / (total # relevant items)
        
        Args:
            retrieved_indices: Array of retrieved result indices
            relevant_indices: List of ground truth relevant indices
            k: Number of top results to consider
        
        Returns:
            Recall score (0.0 to 1.0)
        
        Example:
            If there are 10 relevant items total and top-5 contains 3 of them:
            Recall@5 = 3/10 = 0.3
        """
        if len(relevant_indices) == 0:
            return 0.0
        
        top_k_retrieved = set(retrieved_indices[:k])
        relevant_set = set(relevant_indices)
        
        relevant_in_top_k = len(top_k_retrieved.intersection(relevant_set))
        
        return relevant_in_top_k / len(relevant_indices)
    
    def compute_average_precision(
        self,
        retrieved_indices: np.ndarray,
        relevant_indices: List[int],
        k: Optional[int] = None
    ) -> float:
        """
        Calculate Average Precision (AP).
        
        AP is the average of precision values at each position where
        a relevant item is retrieved.
        
        Formula:
            AP = (Σ Precision@k × rel(k)) / (# relevant items)
            where rel(k) = 1 if item at position k is relevant, 0 otherwise
        
        Args:
            retrieved_indices: Array of retrieved result indices (ranked)
            relevant_indices: List of ground truth relevant indices
            k: Optional cutoff (if None, use all retrieved items)
        
        Returns:
            Average Precision score (0.0 to 1.0)
        
        Note:
            This is a key component of Mean Average Precision (MAP)
        """
        if len(relevant_indices) == 0:
            return 0.0
        
        relevant_set = set(relevant_indices)
        
        if k is not None:
            retrieved_indices = retrieved_indices[:k]
        
        num_relevant = 0
        sum_precisions = 0.0
        
        for i, idx in enumerate(retrieved_indices):
            if idx in relevant_set:
                num_relevant += 1
                precision_at_i = num_relevant / (i + 1)
                sum_precisions += precision_at_i
        
        if num_relevant == 0:
            return 0.0
        
        return sum_precisions / len(relevant_indices)
    
    def compute_reciprocal_rank(
        self,
        retrieved_indices: np.ndarray,
        relevant_indices: List[int]
    ) -> float:
        """
        Calculate Reciprocal Rank (RR).
        
        RR is the multiplicative inverse of the rank of the first
        relevant item.
        
        Formula:
            RR = 1 / (rank of first relevant item)
        
        Args:
            retrieved_indices: Array of retrieved result indices (ranked)
            relevant_indices: List of ground truth relevant indices
        
        Returns:
            Reciprocal Rank (0.0 to 1.0)
        
        Example:
            If first relevant item is at position 3: RR = 1/3 = 0.333
        
        Note:
            Mean Reciprocal Rank (MRR) averages this across multiple queries
        """
        relevant_set = set(relevant_indices)
        
        for i, idx in enumerate(retrieved_indices):
            if idx in relevant_set:
                return 1.0 / (i + 1)
        
        return 0.0
    
    def compute_ndcg(
        self,
        retrieved_indices: np.ndarray,
        relevant_indices: List[int],
        k: int,
        relevance_scores: Optional[Dict[int, float]] = None
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG@K).
        
        NDCG measures the quality of ranking by considering both relevance
        and position. Items higher in the ranking contribute more to the score.
        
        Formula:
            DCG@K = Σ (rel_i / log2(i + 1)) for i in [1, K]
            NDCG@K = DCG@K / IDCG@K
        
        Args:
            retrieved_indices: Array of retrieved result indices (ranked)
            relevant_indices: List of ground truth relevant indices
            k: Number of top results to consider
            relevance_scores: Optional dict mapping index to relevance score
                             If None, binary relevance (0 or 1) is used
        
        Returns:
            NDCG score (0.0 to 1.0)
        
        Example:
            Higher NDCG means better ranking of relevant items
        """
        if len(relevant_indices) == 0 or k == 0:
            return 0.0
        
        relevant_set = set(relevant_indices)
        
        # Use binary relevance if scores not provided
        if relevance_scores is None:
            relevance_scores = {idx: 1.0 for idx in relevant_indices}
        
        # Calculate DCG
        dcg = 0.0
        for i, idx in enumerate(retrieved_indices[:k]):
            if idx in relevant_set:
                rel = relevance_scores.get(idx, 1.0)
                dcg += rel / np.log2(i + 2)  # i+2 because index starts at 0
        
        # Calculate IDCG (ideal DCG - if all relevant items were at top)
        ideal_scores = sorted(
            [relevance_scores.get(idx, 1.0) for idx in relevant_indices],
            reverse=True
        )[:k]
        
        idcg = sum(
            score / np.log2(i + 2) 
            for i, score in enumerate(ideal_scores)
        )
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def compute_average_similarity_score(
        self,
        similarity_scores: np.ndarray,
        k: int
    ) -> float:
        """
        Calculate average similarity score for top-K results.
        
        This metric measures the overall quality of retrieved results
        based on their similarity scores.
        
        Args:
            similarity_scores: Array of similarity scores (higher = more similar)
            k: Number of top results to consider
        
        Returns:
            Average similarity score
        
        Example:
            If top-3 scores are [0.9, 0.85, 0.8]: average = 0.85
        """
        if len(similarity_scores) == 0 or k == 0:
            return 0.0
        
        top_k_scores = similarity_scores[:k]
        return float(np.mean(top_k_scores))
    
    def evaluate_query(
        self,
        retrieved_indices: np.ndarray,
        similarity_scores: np.ndarray,
        relevant_indices: Optional[List[int]] = None,
        k: int = 5,
        relevance_scores: Optional[Dict[int, float]] = None
    ) -> Dict[str, float]:
        """
        Compute all metrics for a single query.
        
        This is the main evaluation function that computes all available
        metrics for a retrieval result.
        
        Args:
            retrieved_indices: Array of retrieved result indices (ranked by score)
            similarity_scores: Array of similarity scores for retrieved results
            relevant_indices: Optional list of ground truth relevant indices.
                            If None, metrics requiring ground truth will be skipped.
            k: Number of top results to evaluate (default: 5)
            relevance_scores: Optional dict of custom relevance scores
        
        Returns:
            Dictionary containing all computed metrics
        
        Example:
            >>> metrics = RetrievalMetrics()
            >>> results = metrics.evaluate_query(
            ...     retrieved_indices=np.array([5, 12, 8, 20, 3]),
            ...     similarity_scores=np.array([0.95, 0.89, 0.87, 0.82, 0.80]),
            ...     relevant_indices=[5, 8, 15, 30],
            ...     k=5
            ... )
            >>> print(f"Top-K Retrieval: {results['top_k_retrieval_percentage']:.2f}%")
        """
        results = {}
        
        # Always compute average similarity score (doesn't need ground truth)
        results["avg_similarity_score"] = self.compute_average_similarity_score(
            similarity_scores, k
        )
        results["k_value"] = k
        results["total_retrieved"] = len(retrieved_indices)
        
        # If ground truth is provided, compute all metrics
        if relevant_indices is not None and len(relevant_indices) > 0:
            results["top_k_retrieval_percentage"] = self.compute_top_k_retrieval_percentage(
                retrieved_indices, relevant_indices, k
            )
            results["precision_at_k"] = self.compute_precision_at_k(
                retrieved_indices, relevant_indices, k
            )
            results["recall_at_k"] = self.compute_recall_at_k(
                retrieved_indices, relevant_indices, k
            )
            results["average_precision"] = self.compute_average_precision(
                retrieved_indices, relevant_indices, k
            )
            results["reciprocal_rank"] = self.compute_reciprocal_rank(
                retrieved_indices, relevant_indices
            )
            results["ndcg_at_k"] = self.compute_ndcg(
                retrieved_indices, relevant_indices, k, relevance_scores
            )
            results["num_relevant_total"] = len(relevant_indices)
            results["num_relevant_in_top_k"] = int(
                len(set(retrieved_indices[:k]).intersection(set(relevant_indices)))
            )
        else:
            logger.info("No ground truth provided - skipping ground-truth-dependent metrics")
        
        # Track for aggregate statistics
        self.queries_evaluated += 1
        for key, value in results.items():
            if key in self.accumulated_metrics and isinstance(value, (int, float)):
                self.accumulated_metrics[key].append(value)
        
        return results
    
    def get_aggregate_metrics(self) -> Dict[str, float]:
        """
        Get aggregated metrics across all evaluated queries.
        
        Returns:
            Dictionary with mean metrics across all queries
        
        Example:
            >>> metrics = RetrievalMetrics()
            >>> # ... evaluate multiple queries ...
            >>> aggregate = metrics.get_aggregate_metrics()
            >>> print(f"Mean Top-K Retrieval: {aggregate['mean_top_k_retrieval_percentage']:.2f}%")
        """
        aggregate = {
            "total_queries_evaluated": self.queries_evaluated
        }
        
        for metric_name, values in self.accumulated_metrics.items():
            if len(values) > 0:
                aggregate[f"mean_{metric_name}"] = float(np.mean(values))
                aggregate[f"std_{metric_name}"] = float(np.std(values))
                aggregate[f"min_{metric_name}"] = float(np.min(values))
                aggregate[f"max_{metric_name}"] = float(np.max(values))
        
        return aggregate
    
    def reset(self):
        """Reset all accumulated metrics."""
        self.queries_evaluated = 0
        for key in self.accumulated_metrics:
            self.accumulated_metrics[key] = []
    
    def export_to_json(self, filepath: str):
        """
        Export aggregate metrics to JSON file.
        
        Args:
            filepath: Path to save JSON file
        
        Example:
            >>> metrics = RetrievalMetrics()
            >>> # ... evaluate queries ...
            >>> metrics.export_to_json("evaluation_results.json")
        """
        aggregate = self.get_aggregate_metrics()
        aggregate["timestamp"] = datetime.now().isoformat()
        aggregate["metric_descriptions"] = {
            "top_k_retrieval_percentage": "Percentage of relevant chunks retrieved in top-K",
            "precision_at_k": "Fraction of top-K results that are relevant",
            "recall_at_k": "Fraction of all relevant items found in top-K",
            "average_precision": "Average of precision values at relevant positions",
            "reciprocal_rank": "Inverse rank of first relevant item",
            "ndcg_at_k": "Normalized Discounted Cumulative Gain at K",
            "avg_similarity_score": "Average similarity score of top-K results"
        }
        
        with open(filepath, 'w') as f:
            json.dump(aggregate, f, indent=2)
        
        logger.info(f"Metrics exported to {filepath}")
    
    def export_to_csv(self, filepath: str):
        """
        Export aggregate metrics to CSV file.
        
        Args:
            filepath: Path to save CSV file
        """
        import csv
        
        aggregate = self.get_aggregate_metrics()
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            for key, value in sorted(aggregate.items()):
                writer.writerow([key, value])
        
        logger.info(f"Metrics exported to {filepath}")
    
    def print_summary(self):
        """
        Print a formatted summary of metrics to console.
        
        Useful for quick inspection during development/research.
        """
        aggregate = self.get_aggregate_metrics()
        
        print("\n" + "="*70)
        print("VIDEO RAG RETRIEVAL METRICS SUMMARY")
        print("="*70)
        print(f"\nTotal Queries Evaluated: {self.queries_evaluated}")
        print("\n" + "-"*70)
        
        if "mean_top_k_retrieval_percentage" in aggregate:
            print("\n📊 PRIMARY METRIC: Top-K Chunks Retrieval")
            print(f"  Mean: {aggregate['mean_top_k_retrieval_percentage']:.2f}%")
            print(f"  Std:  {aggregate.get('std_top_k_retrieval_percentage', 0):.2f}%")
            print(f"  Min:  {aggregate.get('min_top_k_retrieval_percentage', 0):.2f}%")
            print(f"  Max:  {aggregate.get('max_top_k_retrieval_percentage', 0):.2f}%")
        
        if "mean_precision_at_k" in aggregate:
            print("\n🎯 PRECISION & RECALL")
            print(f"  Precision@K: {aggregate['mean_precision_at_k']:.4f}")
            print(f"  Recall@K:    {aggregate['mean_recall_at_k']:.4f}")
        
        if "mean_average_precision" in aggregate:
            print("\n📈 RANKING QUALITY")
            print(f"  MAP:  {aggregate['mean_average_precision']:.4f}")
            print(f"  MRR:  {aggregate.get('mean_reciprocal_rank', 0):.4f}")
            print(f"  NDCG: {aggregate.get('mean_ndcg_at_k', 0):.4f}")
        
        if "mean_avg_similarity_score" in aggregate:
            print("\n⭐ SIMILARITY SCORES")
            print(f"  Mean: {aggregate['mean_avg_similarity_score']:.4f}")
            print(f"  Std:  {aggregate.get('std_avg_similarity_score', 0):.4f}")
        
        print("\n" + "="*70 + "\n")


def format_metrics_for_display(metrics: Dict[str, float]) -> str:
    """
    Format metrics dictionary into a readable string for UI display.
    
    Args:
        metrics: Dictionary of metric names and values
    
    Returns:
        Formatted string ready for display
    """
    lines = []
    lines.append("### 📊 Retrieval Metrics\n")
    
    if "top_k_retrieval_percentage" in metrics:
        lines.append(f"**🎯 Top-K Retrieval:** {metrics['top_k_retrieval_percentage']:.2f}%")
        lines.append(f"  - {metrics.get('num_relevant_in_top_k', 0)}/{metrics.get('num_relevant_total', 0)} relevant chunks found in top-{metrics['k_value']}")
    
    if "precision_at_k" in metrics:
        lines.append(f"\n**📈 Precision@{metrics['k_value']}:** {metrics['precision_at_k']:.4f}")
        lines.append(f"**📊 Recall@{metrics['k_value']}:** {metrics['recall_at_k']:.4f}")
    
    if "average_precision" in metrics:
        lines.append(f"\n**⭐ Average Precision:** {metrics['average_precision']:.4f}")
    
    if "reciprocal_rank" in metrics:
        lines.append(f"**🔝 Reciprocal Rank:** {metrics['reciprocal_rank']:.4f}")
    
    if "ndcg_at_k" in metrics:
        lines.append(f"**🏆 NDCG@{metrics['k_value']}:** {metrics['ndcg_at_k']:.4f}")
    
    if "avg_similarity_score" in metrics:
        lines.append(f"\n**💯 Avg Similarity Score:** {metrics['avg_similarity_score']:.4f}")
    
    return "\n".join(lines)
