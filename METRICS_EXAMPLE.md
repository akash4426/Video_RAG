# Top-K Retrieval Percentage - Visual Example

## Scenario: Searching for "person walking" in a video

### Video Information
- Total frames indexed: 1000
- Sampling rate: 1 FPS
- Video duration: ~16 minutes

### Ground Truth Annotation
An expert annotator identified **4 frames** that show "person walking":
- Frame 120 (at 2:00)
- Frame 450 (at 7:30)
- Frame 780 (at 13:00)
- Frame 1200 (would be at 20:00, but video ends at 16:40)

**Total relevant frames: 4**

---

## Search Results (Top-5)

The system retrieved the following frames with their similarity scores:

| Rank | Frame ID | Timestamp | Similarity Score | Relevant? |
|------|----------|-----------|------------------|-----------|
| 1    | 120      | 02:00     | 0.95             | ✅ YES    |
| 2    | 300      | 05:00     | 0.89             | ❌ NO     |
| 3    | 450      | 07:30     | 0.87             | ✅ YES    |
| 4    | 600      | 10:00     | 0.82             | ❌ NO     |
| 5    | 780      | 13:00     | 0.80             | ✅ YES    |

---

## Metrics Calculation

### 1. Top-K Retrieval Percentage (Primary Metric)

```
Relevant frames in top-5: {120, 450, 780}
Total relevant frames: {120, 450, 780, 1200}

Top-5 Retrieval % = (3 / 4) × 100 = 75%
```

**Interpretation**: The system successfully retrieved 75% of all relevant frames within the top-5 results. Frame 1200 was missed because it's outside the video duration.

---

### 2. Precision@5

```
Relevant results in top-5: 3
Total results in top-5: 5

Precision@5 = 3 / 5 = 0.60
```

**Interpretation**: 60% of the retrieved results are relevant. This means 3 out of 5 shown frames actually depict "person walking".

---

### 3. Recall@5

```
Relevant results found: 3
Total relevant results: 4

Recall@5 = 3 / 4 = 0.75
```

**Interpretation**: The system found 75% of all relevant frames. This is the same as Top-K Retrieval % in this case.

---

### 4. Average Precision (AP)

Precision is calculated at each position where a relevant item appears:

| Position | Frame | Relevant? | # Relevant So Far | Precision at Position |
|----------|-------|-----------|-------------------|-----------------------|
| 1        | 120   | ✅ YES    | 1                 | 1/1 = 1.000          |
| 2        | 300   | ❌ NO     | 1                 | -                    |
| 3        | 450   | ✅ YES    | 2                 | 2/3 = 0.667          |
| 4        | 600   | ❌ NO     | 2                 | -                    |
| 5        | 780   | ✅ YES    | 3                 | 3/5 = 0.600          |

```
AP = (1.000 + 0.667 + 0.600) / 4 = 2.267 / 4 = 0.567
```

**Interpretation**: The average precision of 0.567 indicates moderately good ranking quality.

---

### 5. Reciprocal Rank (RR)

```
First relevant result appears at position: 1

RR = 1 / 1 = 1.000
```

**Interpretation**: Perfect score! The first result shown to the user is relevant.

---

### 6. NDCG@5

NDCG calculation (simplified, assuming binary relevance):

```
DCG@5 = 1/log2(2) + 0 + 1/log2(4) + 0 + 1/log2(6)
      = 1.000 + 0 + 0.500 + 0 + 0.387
      = 1.887

Ideal DCG (if all relevant items were at top):
IDCG@5 = 1/log2(2) + 1/log2(3) + 1/log2(4) + 1/log2(5)
       = 1.000 + 0.631 + 0.500 + 0.431
       = 2.562

NDCG@5 = 1.887 / 2.562 = 0.736
```

**Interpretation**: The ranking quality is 73.6% of ideal. Higher values mean relevant items appear earlier.

---

### 7. Average Similarity Score

```
Average of top-5 scores = (0.95 + 0.89 + 0.87 + 0.82 + 0.80) / 5
                        = 4.33 / 5
                        = 0.866
```

**Interpretation**: High average similarity (0.866) suggests the model is confident about these results.

---

## Summary Table

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Top-5 Retrieval %** | **75.0%** | Found 3 out of 4 relevant frames |
| Precision@5 | 0.600 | 60% of results are relevant |
| Recall@5 | 0.750 | Found 75% of all relevant frames |
| Average Precision | 0.567 | Good ranking quality |
| Reciprocal Rank | 1.000 | First result is relevant |
| NDCG@5 | 0.736 | Decent position-aware ranking |
| Avg Similarity | 0.866 | High confidence scores |

---

## Comparison: Different K Values

### Top-3 Results

| Rank | Frame ID | Relevant? |
|------|----------|-----------|
| 1    | 120      | ✅ YES    |
| 2    | 300      | ❌ NO     |
| 3    | 450      | ✅ YES    |

- **Top-3 Retrieval %**: 2/4 = **50%**
- **Precision@3**: 2/3 = **0.667**
- **Recall@3**: 2/4 = **0.500**

### Top-10 Results (hypothetical)

| Rank | Frame ID | Relevant? |
|------|----------|-----------|
| 1    | 120      | ✅ YES    |
| 2    | 300      | ❌ NO     |
| 3    | 450      | ✅ YES    |
| 4    | 600      | ❌ NO     |
| 5    | 780      | ✅ YES    |
| 6    | 850      | ❌ NO     |
| 7    | 900      | ❌ NO     |
| 8    | 950      | ❌ NO     |
| 9    | 975      | ❌ NO     |
| 10   | 1000     | ❌ NO     |

- **Top-10 Retrieval %**: 3/4 = **75%** (same as Top-5)
- **Precision@10**: 3/10 = **0.300** (worse)
- **Recall@10**: 3/4 = **0.750** (same)

**Key Insight**: Increasing K doesn't help Top-K Retrieval % if the 4th relevant item (Frame 1200) isn't in the dataset. Precision decreases because we're including more irrelevant results.

---

## When to Use Each Metric

| Metric | Best Used For |
|--------|---------------|
| **Top-K Retrieval %** | **Primary metric for coverage - answers "Did we find the relevant items?"** |
| Precision@K | Quality of shown results - important for user satisfaction |
| Recall@K | Completeness - did we miss any relevant items? |
| MAP | Overall system quality across multiple queries |
| MRR | User experience - how quickly do they find what they need? |
| NDCG@K | Ranking quality - are best items shown first? |
| Avg Similarity | Confidence/quality indicator without ground truth |

---

## Research Paper Usage Example

> "We evaluated our Video RAG system on the ActivityNet dataset with 250 queries. 
> The system achieved a **Top-5 Retrieval rate of 78.5%** (σ=12.3%), indicating 
> that approximately 4 out of 5 relevant frames were successfully retrieved within 
> the top-5 results. The high Precision@5 of 0.72 demonstrates that most shown 
> results are relevant to user queries. With an MRR of 0.89, the first relevant 
> result typically appears within the top 2 positions, ensuring quick access to 
> relevant content."

---

## Tips for Your Research

1. **Report multiple metrics**: Don't rely on a single metric. Top-K Retrieval % + Precision + Recall + NDCG gives a complete picture.

2. **Choose appropriate K**: 
   - K=5 for user-facing applications
   - K=10 for academic comparisons
   - K=20 for comprehensive evaluation

3. **Include variance**: Report mean ± standard deviation across queries.

4. **Visualize**: Create bar charts comparing your method against baselines.

5. **Statistical tests**: Use paired t-tests to show your improvements are significant (p < 0.05).
