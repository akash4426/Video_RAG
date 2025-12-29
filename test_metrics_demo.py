"""
Demo script to test the metrics module functionality.

This script demonstrates how to use the RetrievalMetrics class
to evaluate retrieval performance.
"""

import numpy as np
from metrics import RetrievalMetrics, format_metrics_for_display


def demo_single_query():
    """Demonstrate metrics calculation for a single query."""
    print("\n" + "="*80)
    print("DEMO 1: Single Query Evaluation")
    print("="*80)
    
    # Initialize metrics calculator
    metrics = RetrievalMetrics()
    
    # Simulate search results
    print("\nScenario: Searching for 'person walking' in a video")
    print("-"*80)
    
    retrieved_indices = np.array([5, 12, 8, 20, 3, 15, 30, 45, 60, 75])
    similarity_scores = np.array([0.95, 0.89, 0.87, 0.82, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55])
    relevant_indices = [5, 8, 15, 30]  # Ground truth: frames that actually show "person walking"
    k = 5
    
    print(f"Retrieved Frame IDs (top-{k}): {retrieved_indices[:k].tolist()}")
    print(f"Ground Truth (relevant frames): {relevant_indices}")
    print(f"Similarity Scores: {similarity_scores[:k].tolist()}")
    
    # Calculate metrics
    results = metrics.evaluate_query(
        retrieved_indices=retrieved_indices,
        similarity_scores=similarity_scores,
        relevant_indices=relevant_indices,
        k=k
    )
    
    print("\n" + "-"*80)
    print("COMPUTED METRICS:")
    print("-"*80)
    
    # Display key metrics
    print(f"\n🎯 Top-K Retrieval Percentage: {results['top_k_retrieval_percentage']:.2f}%")
    print(f"   → Found {results['num_relevant_in_top_k']}/{results['num_relevant_total']} relevant frames in top-{k}")
    
    print(f"\n📈 Precision@{k}: {results['precision_at_k']:.4f}")
    print(f"   → {results['num_relevant_in_top_k']}/{k} retrieved results are relevant")
    
    print(f"\n📊 Recall@{k}: {results['recall_at_k']:.4f}")
    print(f"   → Found {results['num_relevant_in_top_k']}/{results['num_relevant_total']} of all relevant frames")
    
    print(f"\n⭐ Average Precision (AP): {results['average_precision']:.4f}")
    print(f"🔝 Reciprocal Rank (RR): {results['reciprocal_rank']:.4f}")
    print(f"🏆 NDCG@{k}: {results['ndcg_at_k']:.4f}")
    print(f"💯 Avg Similarity Score: {results['avg_similarity_score']:.4f}")
    
    print("\n" + "="*80 + "\n")
    
    return metrics


def demo_multiple_queries():
    """Demonstrate metrics aggregation across multiple queries."""
    print("\n" + "="*80)
    print("DEMO 2: Multiple Query Evaluation & Aggregation")
    print("="*80)
    
    metrics = RetrievalMetrics()
    
    # Simulate 3 different queries
    queries_data = [
        {
            "query": "person walking",
            "retrieved": np.array([5, 12, 8, 20, 3]),
            "scores": np.array([0.95, 0.89, 0.87, 0.82, 0.80]),
            "relevant": [5, 8, 15, 30]
        },
        {
            "query": "red car",
            "retrieved": np.array([120, 150, 125, 200, 180]),
            "scores": np.array([0.92, 0.88, 0.85, 0.79, 0.75]),
            "relevant": [120, 125, 150]
        },
        {
            "query": "sunset",
            "retrieved": np.array([450, 460, 455, 470, 480]),
            "scores": np.array([0.98, 0.94, 0.91, 0.87, 0.83]),
            "relevant": [450, 455, 460, 465]
        }
    ]
    
    k = 5
    
    # Evaluate each query
    for i, data in enumerate(queries_data, 1):
        print(f"\nQuery {i}: '{data['query']}'")
        print("-"*80)
        
        results = metrics.evaluate_query(
            retrieved_indices=data["retrieved"],
            similarity_scores=data["scores"],
            relevant_indices=data["relevant"],
            k=k
        )
        
        print(f"  Top-K Retrieval: {results['top_k_retrieval_percentage']:.2f}%")
        print(f"  Precision@{k}: {results['precision_at_k']:.4f}")
        print(f"  Recall@{k}: {results['recall_at_k']:.4f}")
        print(f"  NDCG@{k}: {results['ndcg_at_k']:.4f}")
    
    # Get aggregated metrics
    print("\n" + "="*80)
    print("AGGREGATED METRICS ACROSS ALL QUERIES:")
    print("="*80)
    
    aggregate = metrics.get_aggregate_metrics()
    
    print(f"\nTotal Queries Evaluated: {aggregate['total_queries_evaluated']}")
    print("\n" + "-"*80)
    
    print(f"\n📊 Mean Top-K Retrieval: {aggregate['mean_top_k_retrieval_percentage']:.2f}% "
          f"(±{aggregate['std_top_k_retrieval_percentage']:.2f}%)")
    print(f"📈 Mean Precision@{k}: {aggregate['mean_precision_at_k']:.4f} "
          f"(±{aggregate['std_precision_at_k']:.4f})")
    print(f"📊 Mean Recall@{k}: {aggregate['mean_recall_at_k']:.4f} "
          f"(±{aggregate['std_recall_at_k']:.4f})")
    print(f"⭐ Mean AP: {aggregate['mean_average_precision']:.4f} "
          f"(±{aggregate['std_average_precision']:.4f})")
    print(f"🔝 Mean RR (MRR): {aggregate['mean_reciprocal_rank']:.4f} "
          f"(±{aggregate['std_reciprocal_rank']:.4f})")
    print(f"🏆 Mean NDCG@{k}: {aggregate['mean_ndcg_at_k']:.4f} "
          f"(±{aggregate['std_ndcg_at_k']:.4f})")
    
    print("\n" + "="*80 + "\n")
    
    # Export to files
    print("Exporting metrics...")
    metrics.export_to_json("/tmp/metrics_results.json")
    metrics.export_to_csv("/tmp/metrics_results.csv")
    print("✓ Exported to /tmp/metrics_results.json")
    print("✓ Exported to /tmp/metrics_results.csv")
    
    return metrics


def demo_without_ground_truth():
    """Demonstrate metrics calculation without ground truth data."""
    print("\n" + "="*80)
    print("DEMO 3: Evaluation Without Ground Truth")
    print("="*80)
    
    print("\nScenario: User doesn't provide ground truth labels")
    print("-"*80)
    
    metrics = RetrievalMetrics()
    
    retrieved_indices = np.array([10, 25, 30, 45, 60])
    similarity_scores = np.array([0.94, 0.88, 0.82, 0.76, 0.70])
    k = 5
    
    print(f"Retrieved Frame IDs: {retrieved_indices.tolist()}")
    print(f"Similarity Scores: {similarity_scores.tolist()}")
    print("Ground Truth: Not provided")
    
    # Calculate metrics (without ground truth)
    results = metrics.evaluate_query(
        retrieved_indices=retrieved_indices,
        similarity_scores=similarity_scores,
        relevant_indices=None,  # No ground truth
        k=k
    )
    
    print("\n" + "-"*80)
    print("AVAILABLE METRICS (without ground truth):")
    print("-"*80)
    
    print(f"\n💯 Average Similarity Score: {results['avg_similarity_score']:.4f}")
    print(f"📊 Number of Results: {results['total_retrieved']}")
    print(f"🔢 K Value: {results['k_value']}")
    
    print("\n⚠️  Metrics requiring ground truth are not available:")
    print("   - Top-K Retrieval Percentage")
    print("   - Precision@K / Recall@K")
    print("   - MAP, MRR, NDCG")
    
    print("\n💡 To enable full metrics, provide ground truth frame IDs.")
    
    print("\n" + "="*80 + "\n")


def demo_metrics_display_format():
    """Demonstrate formatted metrics display for UI."""
    print("\n" + "="*80)
    print("DEMO 4: Formatted Metrics Display (for Streamlit UI)")
    print("="*80)
    
    metrics = RetrievalMetrics()
    
    retrieved_indices = np.array([5, 12, 8, 20, 3])
    similarity_scores = np.array([0.95, 0.89, 0.87, 0.82, 0.80])
    relevant_indices = [5, 8, 15, 30]
    k = 5
    
    results = metrics.evaluate_query(
        retrieved_indices=retrieved_indices,
        similarity_scores=similarity_scores,
        relevant_indices=relevant_indices,
        k=k
    )
    
    print("\nFormatted for display:")
    print("-"*80)
    formatted = format_metrics_for_display(results)
    print(formatted)
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("VIDEO RAG METRICS MODULE - DEMONSTRATION")
    print("="*80)
    print("\nThis demo shows how to use the metrics module for research evaluation.")
    print("The metrics module provides comprehensive retrieval evaluation suitable")
    print("for academic papers and system benchmarking.")
    print("="*80)
    
    # Run all demos
    demo_single_query()
    demo_multiple_queries()
    demo_without_ground_truth()
    demo_metrics_display_format()
    
    print("\n" + "="*80)
    print("✅ ALL DEMOS COMPLETED SUCCESSFULLY")
    print("="*80)
    print("\nKey Takeaways:")
    print("  1. Top-K Retrieval % measures what percentage of relevant items are found")
    print("  2. Metrics work with or without ground truth (limited without)")
    print("  3. Multiple queries can be aggregated for research papers")
    print("  4. Results can be exported to JSON/CSV for analysis")
    print("  5. Formatted display is available for UI integration")
    print("\n" + "="*80 + "\n")
