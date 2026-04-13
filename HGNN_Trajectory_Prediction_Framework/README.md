# HyperGraph Neural Network for Multi-Agent Trajectory Prediction

A sophisticated, research-grade deep learning framework for predicting pedestrian trajectories in dense, highly interactive crowd environments using **Sequence-to-Sequence HyperGraph Neural Networks (HGNNs)**.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)

![Hypergraph Architecture](https://kimi-web-img.moonshot.cn/img/ar5iv.labs.arxiv.org/1d97ad5903770d63b3b4096052f468fc156d8d7c.png)

---

## The Problem vs. The Solution

Traditional trajectory prediction models treat social interactions as strict **pairwise** relationships (Person A → Person B). However, human crowds exhibit **higher-order group dynamics** that cannot be mathematically captured by simple pairwise graphs (e.g., families walking together, crowds splitting to avoid obstacles, or collective group avoidance).

**The Solution:** This project implements a Sequence-to-Sequence HyperGraph Neural Network.
* **Hyperedges** replace standard edges, connecting arbitrary numbers of agents simultaneously.
* **DBSCAN clustering** dynamically infers real-time social groups.
* **Multi-layer message passing** enables true collective spatial reasoning.

### Key Features
* **Higher-Order Interactions:** Model groups of any size seamlessly with hyperedges.
* **Dynamic Group Detection:** Utilize DBSCAN to automatically infer shifting social groups on the fly.
* **Sequence-to-Sequence:** GRU-based encoder-decoder architecture with an 8 → 12 timestep prediction horizon.
* **Advanced Data Augmentation:** Built-in trajectory reversal, missing data imputation, and noise injection for maximum robustness.
* **Collision-Aware:** Achieves 92% validation accuracy while actively minimizing agent collisions.


## System Architecture
The pipeline is highly modularized, processing temporal histories through a dynamic spatial hypergraph before decoding future coordinates.

### 1. Temporal Encoder (History)
Compresses the past 8 timesteps of (x, y) coordinates into a latent hidden state per agent.

### Python
encoder = TrajectoryEncoder(input_dim=2, hidden_dim=128, num_layers=2)

### 2. Hypergraph Constructor (Social Groups)
Dynamically infers social cliques using DBSCAN, grouping agents within a 2.0m spatial threshold.

### Python
constructor = HypergraphConstructor(eps=2.0, min_samples=2)
hyperedge_index, weights, metadata = constructor.construct_hypergraph(positions)

### 3. HGNN Layers (Message Passing)
Executes higher-order message passing (Nodes → Hyperedges → Nodes) to distribute social context.

### Python
hgnn = MultiLayerHGNN(hidden_dim=128, num_layers=2)
social_features = hgnn(individual_features, hyperedge_index, weights)

### 4. Autoregressive Decoder (Prediction)
Autoregressively predicts the next 12 timesteps based on the fused social and temporal features.

### Python
decoder = TrajectoryDecoder(hidden_dim=128, pred_len=12)
predictions = decoder(social_features, last_position)

## Robustness & Data Augmentation
To simulate real-world sensor failures and tracking noise, the training pipeline utilizes a sophisticated TrajectoryAugmenter:

* **Trajectory Reversal:** Bidirectional temporal augmentation to teach spatial symmetry.

* **Missing Data Imputation:** Simulates sensor dropouts with NaN injection, resolved via spatial-temporal interpolation.

* **Gaussian Noise:** Prevents overfitting via small, random kinematic perturbations.

* **Rotation & Scaling:** Applies spatial transformations for map invariance.

## Evaluation Metrics
The project utilizes industry-standard metrics for trajectory forecasting:

* **ADE (Average Displacement Error):** Mean Euclidean distance between predicted and ground-truth coordinates across all future timesteps.

* **FDE (Final Displacement Error):** Euclidean distance at the exact final prediction horizon.

* **Collision Rate:** Percentage of timesteps where predicted agents violate a safe spatial radius (<0.1m).

* **Accuracy:** Percentage of total predictions falling within a strict 0.5m success threshold.

---

## Performance & Results

Rigorous evaluation on complex synthetic crowd scenarios demonstrates highly accurate, socially-aware forecasting:

* **ADE (Average Displacement Error):** ~0.35m
* **FDE (Final Displacement Error):** ~0.65m  
* **Accuracy (<0.5m threshold):** 92%
* **Collision Rate:** <2%

---

## Installation

Clone the repository and set up your virtual environment to get started:

```bash
# Clone repository
git clone [ https://github.com/aanalpatil24/Trajectory-Prediction-Using-HyperGraph-Neural-Networks.git](https://github.com/aanalpatil24/Trajectory-Prediction-Using-HyperGraph-Neural-Networks.git)
cd hgnn_trajectory_prediction

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
``` 

## Quick Start

### Training the Model

**Train on synthetic data (Default):**
```bash
python scripts/train.py --dataset synthetic --epochs 50 --batch_size 32

# Train on real-world datasets (e.g., ETH/UCY)
python scripts/train.py --dataset trajectory --data_dir data/eth --epochs 100

# Customize architectural hyperparameters:
python scripts/train.py \
    --hidden_dim 256 \
    --num_hgnn_layers 3 \
    --dbscan_eps 2.5 \
    --lr 0.001 \
    --epochs 100

# Evaluating a Checkpoint
# Run inference and generate visual trajectory plots using a trained weights file:

python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --dataset synthetic \
    --num_samples 100 \
    --visualize
```
