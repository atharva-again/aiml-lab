# AI Lab Assignment Part B - Theory Answers
**Student:** Atharva Verma 
**Enrollment Number:** 0901AI231019
**Batch:** A

---

## Q1: Decision Tree

### Information Gain Calculations

**Parent Entropy:**  
50 pass, 50 fail → $H(S) = -0.5\log_2(0.5) - 0.5\log_2(0.5) = 1$

**Feature A Split:** (40,10) and (10,40)
- Left node: $40/50=0.8$ pass, $10/50=0.2$ fail  
  $H_L = -0.8\log_2(0.8) - 0.2\log_2(0.2) = 0.722$
- Right node: $10/50=0.2$ pass, $40/50=0.8$ fail  
  $H_R = -0.2\log_2(0.2) - 0.8\log_2(0.8) = 0.722$
- Weighted: $0.5 \times 0.722 + 0.5 \times 0.722 = 0.722$
- **IG(A) = 1 - 0.722 = 0.278**

**Feature B Split:** (50,0) and (0,50)
- Left node: $50/50=1$ pass, $0/50=0$ fail → $H_L = 0$
- Right node: $0/50=0$ pass, $50/50=1$ fail → $H_R = 0$
- Weighted: $0.5 \times 0 + 0.5 \times 0 = 0$
- **IG(B) = 1 - 0 = 1**

### Why Tree Prefers Feature B
Feature B achieves **perfect separation** (IG = 1), meaning it immediately classifies all samples correctly. Feature A still has misclassifications in both children. The tree greedily picks the split that maximizes Information Gain.

### Overfitting, Pruning, and Minimum Samples

**Overfitting:** A 20-node deep tree on 100 samples means $\approx 5$ samples per node on average. The tree memorizes noise in training data rather than learning generalizable patterns. This creates spurious branches that don't reflect true underlying relationships.

**Pruning:** Remove subtrees that don't significantly improve accuracy on validation data. Reduces complexity by collapsing nodes back into their parents.

**Minimum Samples per Leaf:** Requires at least $N$ samples in each leaf node (e.g., min_samples_leaf=10). This prevents creating branches that isolate individual outliers.

---

## Q2: K-NN and High-Dimensionality

### Why Distance Becomes Equal

In high dimensions, the **volume of a hypersphere** grows exponentially relative to its inscribing hypercube. The distance from any point to the corners of the hypercube dominates, making all pairwise distances converge to approximately the same value.

Mathematically: For $d$ dimensions, the expected distance between two random points is $\sqrt{d}$, but the variance shrinks. The ratio of distance to the farthest corner vs. nearest neighbor approaches 1 as $d \to \infty$.

### Why Euclidean Distance Fails

K-NN relies on distance metrics to identify "similar" neighbors. When all distances are nearly equal, the notion of "nearest" becomes meaningless. Every point has essentially the same set of neighbors, destroying the algorithm's ability to discriminate.

### Feature Selection Requirement

Feature selection (or dimensionality reduction via PCA/t-SNE) is mandatory to:
1. Remove irrelevant/noisy features that add noise
2. Reduce dimensionality so meaningful distances exist
3. Focus K-NN on truly discriminative attributes

---

## Q3: K-Means vs DBSCAN

### Why K-Means Fails on Concentric Circles

K-Means uses **Voronoi cells** and centroids. It partitions space into regions closest to each centroid, creating linear boundaries (straight lines). For two concentric circles:

- The smaller circle's centroid lies inside the larger ring
- K-Means will split the ring into pie-shaped sectors, not separate the inner circle from outer ring
- Both true clusters get merged across the linear boundaries

K-Means assumes spherical, equal-sized clusters — concentric circles violate this fundamentally.

### Why DBSCAN Works Better

DBSCAN uses **density-based connectivity**:
- Inner circle: dense region
- Outer ring: another dense region (but separate from inner)
- They remain disconnected even though spatially nested

DBSCAN doesn't use centroids or Voronoi cells, so it naturally separates density-based clusters.

### Epsilon Parameter

**Epsilon** defines the neighborhood radius for density calculations:
- Points within $\epsilon$ distance are considered neighbors
- A point is a **core point** if it has $\ge$ minPts neighbors
- Points not reachable from any core point → **noise** (labeled -1)

In concentric circles, points in the sparse gap between inner and outer circle become noise if they can't reach core points on either side.
