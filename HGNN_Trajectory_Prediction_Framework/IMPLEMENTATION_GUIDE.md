# HGNN Trajectory Prediction - Implementation Guide

## Project Overview

This is a **production-ready, research-grade implementation** of Sequence-to-Sequence HyperGraph Neural Networks for multi-agent trajectory prediction. The project is structured following Python best practices and deep learning project standards.

## Architecture Highlights

### 1. HyperGraph vs Standard Graph

**Standard Graph Neural Network:**
- Edges connect exactly 2 nodes (pairwise)
- Cannot model groups of 3+ agents walking together
- Limited to binary relationships

**HyperGraph Neural Network (This Implementation):**
- Hyperedges connect ANY number of nodes
- Can model families, groups, crowd dynamics
- Higher-order social interactions

![Graph vs Hypergraph](https://kimi-web-img.moonshot.cn/img/ar5iv.labs.arxiv.org/c937dbfde843e2149ba523be8c40f4c621156d3b.png)

### 2. DBSCAN Social Group Detection

![DBSCAN Clustering](https://kimi-web-img.moonshot.cn/img/miro.medium.com/7243983e0f8271363c576b277a3d01320dd16276.png)

DBSCAN parameters in `src/utils/hypergraph_builder.py`:
- `eps=2.0`: Agents within 2 meters are neighbors
- `min_samples=2`: At least 2 agents form a group
- Clusters become hyperedges in the graph

### 3. Message Passing Flow

```
Input: Past 8 timesteps (obs_len=8)
    ↓
┌─────────────────────────────────────────┐
│  GRU Encoder                            │
│  - Compress history to hidden state     │
│  - Output: (batch, agents, hidden_dim)  │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  Hypergraph Constructor (DBSCAN)        │
│  - Detect spatial groups                │
│  - Build hyperedge_index                │
│  - Compute hyperedge weights            │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  Multi-Layer HGNN                       │
│  - Nodes → Hyperedges (aggregate)       │
│  - Hyperedges → Nodes (distribute)      │
│  - Social feature extraction            │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  Fusion Layer                           │
│  - Concat [individual, social]          │
│  - Learnable fusion                     │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  GRU Decoder                            │
│  - Autoregressive prediction            │
│  - Output: Future 12 timesteps          │
└─────────────────────────────────────────┘
```

## Key Components Explained

### HypergraphConv (`src/models/hypergraph_conv.py`)

The core message passing layer:

```python
# Step 1: Aggregate nodes to hyperedges
hyperedge_features = scatter_add(node_features[members], hyperedge_idx)
hyperedge_features = hyperedge_features / hyperedge_size  # Average

# Step 2: Transform hyperedge features
hyperedge_features = self.hyperedge_transform(hyperedge_features)

# Step 3: Distribute back to nodes
node_features = scatter_mean(hyperedge_features[hyperedges], node_idx)
```

### TrajectoryAugmenter (`src/data/augmentation.py`)

Four augmentation strategies:

1. **Reversal**: `traj[::-1]` - bidirectional symmetry
2. **Missing Data**: Inject NaNs → linear interpolation
3. **Noise**: `traj + N(0, σ²)` - regularization
4. **Rotation/Scaling**: Spatial transformations

### DBSCAN Clustering (`src/utils/hypergraph_builder.py`)

```python
from sklearn.cluster import DBSCAN

# Cluster agents by spatial proximity
clusters = DBSCAN(eps=2.0, min_samples=2).fit(positions)

# Each cluster becomes a hyperedge
hyperedges = []
for cluster_id in unique_clusters:
    members = np.where(clusters == cluster_id)[0]
    hyperedges.append(members.tolist())
```
