# Research Metrics Guide

## Overview

The Video RAG system now includes comprehensive retrieval metrics suitable for academic research and system evaluation. This guide explains how to use these metrics effectively.

## Quick Start

### 1. Enable Metrics in the UI

1. Open the sidebar in the Streamlit app
2. Expand the "📊 Research Metrics" section
3. Check "Enable Metrics Calculation"
4. (Optional) Enter ground truth frame IDs

### 2. Provide Ground Truth (Optional but Recommended)

For complete metrics, provide comma-separated frame IDs that are relevant to your query:

```
Ground Truth Frame IDs: 120,450,780,1200
```

**Note**: Without ground truth, only similarity-based metrics are available.

### 3. Run Your Search

Perform your search as usual. Metrics will be computed automatically.

### 4. View Results

After search completes, expand the "📊 Retrieval Metrics" section to see:
- Top-K Retrieval Percentage
- Precision@K and Recall@K
- MAP, MRR, NDCG
- Average Similarity Score

### 5. Export for Research Papers

Click "📥 Export Metrics (JSON)" or "📥 Export Metrics (CSV)" to download results for your research paper.

---

## Available Metrics

### Primary Metric: Top-K Retrieval Percentage

**Definition**: Percentage of relevant chunks found in the top-K results.

**Formula**: 
```
Top-K Retrieval % = (# relevant chunks in top-K) / (total # relevant chunks) × 100
```

**Example**:
- Query: "person walking"
- Ground truth: 4 relevant frames [120, 450, 780, 1200]
- Top-5 results: [120, 300, 450, 600, 780]
- Top-K Retrieval = 3/4 × 100 = **75%**

**Use in Papers**: "The system achieved a Top-5 Retrieval rate of 75%, indicating that 3 out of 4 relevant frames were successfully retrieved within the top-5 results."

---

### Precision@K

**Definition**: Fraction of top-K results that are actually relevant.

**Formula**: 
```
Precision@K = (# relevant items in top-K) / K
```

**Example**:
- Top-5 results contain 3 relevant items
- Precision@5 = 3/5 = **0.60**

**Use in Papers**: "The retrieval system demonstrated a Precision@5 of 0.60, meaning 60% of the top-5 results were relevant."

---

### Recall@K

**Definition**: Fraction of all relevant items found in top-K results.

**Formula**: 
```
Recall@K = (# relevant items in top-K) / (total # relevant items)
```

**Example**:
- 4 relevant items total, 3 found in top-5
- Recall@5 = 3/4 = **0.75**

**Use in Papers**: "The system achieved a Recall@5 of 0.75, successfully retrieving 75% of all relevant frames."

---

### Mean Average Precision (MAP)

**Definition**: Average of precision values at each position where a relevant item appears.

**Formula**: 
```
AP = Σ(Precision@k × rel(k)) / (# relevant items)
MAP = mean(AP) across all queries
```

**Use in Papers**: "The mean average precision (MAP) across 100 queries was 0.72, indicating strong overall retrieval performance."

---

### Mean Reciprocal Rank (MRR)

**Definition**: Average of the reciprocal ranks of the first relevant result.

**Formula**: 
```
RR = 1 / (rank of first relevant item)
MRR = mean(RR) across all queries
```

**Example**:
- First relevant item at position 2
- RR = 1/2 = **0.50**

**Use in Papers**: "The system achieved an MRR of 0.85, demonstrating that relevant results typically appear within the top 2 positions."

---

### Normalized Discounted Cumulative Gain (NDCG@K)

**Definition**: Position-aware ranking quality metric that rewards relevant items appearing earlier.

**Formula**: 
```
DCG@K = Σ(rel_i / log2(i + 1)) for i in [1, K]
NDCG@K = DCG@K / IDCG@K
```

**Use in Papers**: "The NDCG@5 score of 0.82 indicates high-quality ranking with relevant items positioned early in the results."

---

## Programmatic Usage

### Basic Example

```python
from metrics import RetrievalMetrics
import numpy as np

# Initialize
metrics = RetrievalMetrics()

# Evaluate a query
results = metrics.evaluate_query(
    retrieved_indices=np.array([5, 12, 8, 20, 3]),
    similarity_scores=np.array([0.95, 0.89, 0.87, 0.82, 0.80]),
    relevant_indices=[5, 8, 15, 30],
    k=5
)

print(f"Top-K Retrieval: {results['top_k_retrieval_percentage']:.2f}%")
print(f"Precision@5: {results['precision_at_k']:.4f}")
print(f"Recall@5: {results['recall_at_k']:.4f}")
```

### Multiple Queries

```python
metrics = RetrievalMetrics()

# Evaluate multiple queries
for query_data in dataset:
    metrics.evaluate_query(
        retrieved_indices=query_data['results'],
        similarity_scores=query_data['scores'],
        relevant_indices=query_data['ground_truth'],
        k=5
    )

# Get aggregate statistics
aggregate = metrics.get_aggregate_metrics()
print(f"Mean Top-K Retrieval: {aggregate['mean_top_k_retrieval_percentage']:.2f}%")
print(f"Mean Precision@5: {aggregate['mean_precision_at_k']:.4f}")

# Export for paper
metrics.export_to_json("results/evaluation_metrics.json")
```

---

## Research Paper Template

### Results Section

```markdown
## Evaluation Results

We evaluated our Video RAG system on [dataset name] containing [N] videos 
with [M] queries and ground truth annotations.

### Retrieval Performance

| Metric | Value |
|--------|-------|
| Top-5 Retrieval % | 78.5% ± 12.3% |
| Precision@5 | 0.72 ± 0.08 |
| Recall@5 | 0.78 ± 0.12 |
| MAP | 0.75 ± 0.11 |
| MRR | 0.89 ± 0.06 |
| NDCG@5 | 0.81 ± 0.09 |

Our system achieved a Top-5 Retrieval rate of 78.5%, indicating that 
nearly 4 out of 5 relevant frames were successfully retrieved within 
the top-5 results. The high MRR of 0.89 demonstrates that the first 
relevant result typically appears within the top 2 positions.

### Comparison with Baselines

| System | Top-5 Retrieval % | MAP | NDCG@5 |
|--------|-------------------|-----|---------|
| Baseline 1 | 65.2% | 0.63 | 0.70 |
| Baseline 2 | 71.8% | 0.68 | 0.75 |
| **Our System** | **78.5%** | **0.75** | **0.81** |

Our approach outperforms both baselines across all metrics, with 
particularly strong improvements in Top-K Retrieval (+13.3% vs Baseline 1).
```

---

## Best Practices

### 1. Ground Truth Annotation

- **Quality over Quantity**: Carefully annotate a representative subset rather than hastily annotating everything
- **Inter-Annotator Agreement**: Have multiple annotators for reliability
- **Binary vs Graded**: Use binary relevance (relevant/not relevant) for simplicity, or graded (0-3) for NDCG

### 2. Choosing K

- **K=5**: Good for user-facing applications (fits on screen)
- **K=10**: Standard in information retrieval research
- **K=20**: Suitable for comprehensive evaluation

### 3. Multiple Queries

- **Minimum 50 queries**: For statistical significance
- **Diverse queries**: Cover different object types, actions, scenes
- **Query difficulty**: Include both easy and hard queries

### 4. Statistical Significance

- Report mean and standard deviation
- Use paired t-tests to compare systems
- Report confidence intervals (e.g., 95% CI)

---

## Troubleshooting

### No Ground Truth Available

If you don't have ground truth labels, you can still:
1. Use Average Similarity Score as a proxy for quality
2. Manually inspect top-K results
3. Conduct user studies (ask users to rate relevance)

### Ground Truth Not Matching Results

If ground truth frame IDs don't match your indexed frames:
- Check that you're using the same video
- Verify frame sampling rate matches annotation
- Use timestamps instead of frame IDs if possible

### Export Not Working

- Ensure you have write permissions
- Check disk space
- Try downloading immediately after generation

---

## References

For more information on these metrics:

1. **Precision and Recall**: Manning, C. D., et al. (2008). Introduction to Information Retrieval.
2. **MAP**: Zhu, M. (2004). Recall, Precision and Average Precision.
3. **NDCG**: Järvelin, K., & Kekäläinen, J. (2002). Cumulative gain-based evaluation of IR techniques.
4. **MRR**: Voorhees, E. M. (1999). The TREC-8 Question Answering Track Report.

---

## Support

For questions or issues with metrics:
- Open an issue on GitHub
- Check the demo script: `test_metrics_demo.py`
- Review the metrics module documentation in `metrics.py`
