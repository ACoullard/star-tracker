# SIFTER Implementation Context
## Star Identification From Transformer-Encoded Representations
### Source: Zhang, E. — 38th Annual Small Satellite Conference (SSC24-VI-05)

---

## 1. Project Overview

SIFTER is a neural network-based star identification algorithm for small satellites. It takes a list of star centroids (pixel coordinates) extracted from a star tracker image and identifies which catalog stars they correspond to. The core neural network component is called **SIFTERN**.

The algorithm is designed to be:
- Lightweight (< 15 MB at 16-bit quantization)
- Robust to noise (>97% availability, <2% error rate under realistic noise)
- Fast (12 ms on Intel Xeon CPU, ~300–400 ms estimated on 100 MHz rad-hardened processor)
- Generalizable to real satellite/observatory images after training on synthetic data

---

## 2. Full Algorithm Pipeline

The pipeline runs as follows at inference time:

```
Input: raw star image
  ↓
[External] Centroiding algorithm (Center of Gravity, Gaussian Grid, etc.)
  → Output: list of centroids C = {s_i = (x_i, y_i)}
  ↓
Step 1: Determine guide star s_G
  - s_G = centroid with shortest Euclidean distance to image center (x_c, y_c)
  - Remove s_G from C: C = C - {s_G}
  - Re-project all centroids so s_G is at the origin/center
  ↓
Step 2: SIFTERN inference
  - Input: re-projected centroid list C (including s_G at center)
  - Output: (ŷ_sG, p) where ŷ_sG is predicted catalog ID, p is confidence probability
  ↓
Step 3: Confidence threshold check
  - If p < th (threshold hyperparameter): REFUSE to identify, return "unidentified"
  - If p >= th: continue
  ↓
Step 4: False star check
  - If s_G is classified as a false star:
      Option A (default): remove s_G from C, restart from Step 1 with next guide star
      Option B: quit and re-image
  ↓
Step 5: GuidedMatch (identify remaining centroids)
  - Uses known identity of s_G to match all other centroids to catalog stars
  - If insufficient matches: suspect s_G is a false star, refuse identification
  - Output: full scene identification OR refusal
  ↓
Output: Successful identification OR Refused identification
```

---

## 3. SIFTERN Neural Network Architecture

### 3.1 High-Level Structure

SIFTERN has two sequential components:
1. **Transformer Encoder** — takes variable-length centroid list, outputs contextual representation
2. **Classifier Head** — maps representation to catalog star probabilities

### 3.2 Input

- Raw input: centroid list `C = {s_i = (x_i, y_i)}` — pixel coordinates of each detected star
- Each centroid is a 2D vector: `s_i ∈ ℝ²`
- Input is variable-length (number of stars per scene varies)
- The guide star `s_G` is re-projected to the center before being passed in

### 3.3 Input Embedding Layer

- Projects each centroid from `ℝ²` → `ℝ^d`
- This is a learned linear projection (embedding layer)
- Embedding dimension `d = 256` (from evaluated model)
- Purpose: expand representation so encoder can learn more complex inter-centroid relationships

```python
# Pseudocode
embedding = nn.Linear(2, d_model)  # maps (x, y) → ℝ^256
```

### 3.4 Transformer Encoder

Based on Vaswani et al. "Attention is All You Need" (2017).

**Encoder hyperparameters (evaluated model):**
| Parameter | Value |
|---|---|
| Input dimensions | 2 |
| Embedding dimensions (d) | 256 |
| Number of attention heads | 8 |
| Number of encoder blocks | 5 |
| Activation function | ReLU |
| Encoder output scheme | Global averaging |

**Each Encoder Block contains:**

1. **Multi-Head Self-Attention**
   - Input: `x ∈ ℝ^(n×d)` where n = number of centroids
   - For each head j, compute learned projections:
     - `Q = x · W_q^j`, `K = x · W_k^j`, `V = x · W_v^j`
     - where `W_q^j, W_k^j, W_v^j ∈ ℝ^(d×d_head)`, `d_head = d / num_heads = 32`
   - Attention scores: `A = softmax(QK^T / √d_head)`
   - Head output: `H_j = A · V`
   - Concatenate all heads, project back to `ℝ^d`
   - Alignment score formula: `e_ji = q_i · k_j` (dot product = relevance of x_j to x_i)
   - Scale by `1/√d` to prevent vanishing gradients from large dot products
   - Full attention output: `H = softmax(QK^T / √d) · V`

2. **Residual connection** (skip connection around attention)
   - `x = x + Attention(x)`
   - Mitigates vanishing/exploding gradients in deep networks

3. **LayerNorm** (applied after residual)
   - Standardizes features and activations
   - Improves Transformer expressivity (Brody et al., 2023)

4. **MLP block** (position-wise feed-forward)
   - Two linear layers with ReLU activation
   - Introduces nonlinear transformations
   - Increases encoder capability for complex relational patterns

5. **Residual connection** (skip connection around MLP)
   - `x = x + MLP(x)`

6. **LayerNorm** (applied after MLP residual)

Standard Transformer encoder block (Pre-LN or Post-LN — paper does not specify, Post-LN shown above, Pre-LN also common):
```
x → MultiHeadSelfAttention → + residual → LayerNorm → MLP → + residual → LayerNorm → output
```

**Self-attention mathematical summary:**
```
Q = xW_q,  K = xW_k,  V = xW_v
E = QK^T
A = softmax(E / √d)
H = AV
```

### 3.5 Aggregation (Encoder Output → Context Vector)

After the final encoder block, output is `H ∈ ℝ^(n×d)` — one vector per centroid.

**Method used: Global Average Pooling**
- `c = mean(H, dim=0)` → `c ∈ ℝ^d`
- Averages across all centroid representations to get a single fixed-size scene vector

**Alternative explored but left for future work: [CLS] token** (BERT-style)
- Prepend a special learnable [CLS] token to every centroid list
- After encoding, use the output vector `H_0^T` corresponding to [CLS] as context vector `c`
- The paper notes nonnegligible performance differences between the two schemes

### 3.6 Classifier Head

- Single linear layer: `ℝ^d → ℝ^N_C`
- `N_C = 4954` (number of identifiable catalog stars after filtering)
- Softmax applied to logits to get class probabilities
- Prediction: `ŷ = argmax_k P(k)`
- Confidence: `p = max_k P(k)`

```python
# Pseudocode
classifier = nn.Linear(d_model, num_classes)  # 256 → 4954
logits = classifier(c)
probs = F.softmax(logits, dim=-1)
pred = probs.argmax(dim=-1)
confidence = probs.max(dim=-1).values
```

### 3.7 Total Model Size

- **8,090,631 total parameters**
- Storage: 30 MB (32-bit float), 15 MB (16-bit quantization), 7.5 MB (8-bit quantization)
- This is 4x+ smaller than FCNet (16M params), 17x smaller than VGG (138M params)

---

## 4. GuidedMatch Algorithm

GuidedMatch identifies all non-guide-star centroids once `s_G`'s identity is known. It is a hybrid of the Polestar and Tetra algorithms.

**Steps:**
1. For each centroid in C, compute its angular distance to `s_G`
2. Assign each centroid to all plausible catalog stars within angular distance tolerance `ε` (hyperparameter)
3. Construct 4-star patterns from these initial matchings
4. Uniquely identify centroids as catalog stars via pattern matching
5. Rejection rules:
   - Centroids with conflicting identifications → treated as false stars
   - If insufficient number of centroids matched → suspect `s_G` is false star → refuse scene identification entirely

---

## 5. Dataset

### 5.1 Star Catalog

- Source: **Yale Bright Star Catalog (YBSC)** — 9,110 brightest stars
- Filtering: retain only stars with magnitude ≤ 6.0 Mv
- Remove double stars: pairs with angular distance < 0.05° (avoid classifier confusion)
- **Final catalog: 4,954 identifiable stars** → this is `N_C`, the number of output classes

### 5.2 Synthetic Dataset Generation

1. **Star scene generation:**
   - Camera FOV: 12° × 12° (narrow)
   - Sky scan: 1° increments in right ascension and declination, 5° increments in roll
   - For each attitude: project all stars in FOV to sensor coordinates
   - Label: YBSC ID of guide star `s_G` (centroid closest to image center)

2. **Data augmentation (applied per sample):**

   | Noise Type | Augmentation Method |
   |---|---|
   | Centroid error | Shift each centroid (x, y) by Gaussian noise: μ=0, σ=5 pixels. Drop centroids that exit sensor view. |
   | False stars | Add 0–5 random centroids with random pixel coordinates and random magnitude ∈ [0, 6] |
   | Magnitude error | Add Gaussian noise to each centroid's magnitude: μ=0, σ=0.5 Mv. Drop star if adjusted magnitude > 6 Mv (simulates missing stars). Allow catalog stars with adjusted magnitude < 6 Mv to enter the scene. |
   | Rotation invariance | Implicitly handled: all roll increments (0°, 5°, 10°, ..., 355°) included in dataset generation, making the model rotationally invariant without a handcrafted feature |

3. **Final dataset size: 13,000,000+ synthetic star scene samples**
4. **Dataset is open-sourced on GitHub**

### 5.3 Data Format

Each sample:
- Input `x`: list of centroid coordinates `[(x_0, y_0), (x_1, y_1), ..., (x_{n-1}, y_{n-1})]`
  - Variable length (n varies per scene)
  - Re-projected so guide star is at center
- Label `y`: YBSC catalog ID of guide star `s_G` (integer, 0-indexed into 4954 classes)

---

## 6. Training Procedure

### 6.1 Loss Function

Cross-entropy loss:
```
L = -Σ_i t_i · log(p_i)
```
where `p_i` is predicted probability of class i, `t_i = 1` if i is ground truth else 0.

### 6.2 Optimizer

**AdamW** (Adam with decoupled weight decay)

### 6.3 Hyperparameters

| Hyperparameter | Value |
|---|---|
| Max learning rate (lr_max) | 5e-5 |
| Warmup steps (T_max) | 7000 |
| Weight decay | 1e-2 |
| Batch size | 256 |

### 6.4 Learning Rate Schedule

Two-stage scheduling:
1. **Warm-up phase:** Linear ramp from lr=0 to lr_max over T_max=7000 steps
   - Prevents divergence from large weight updates on randomly initialized parameters
2. **Decay phase:** Learning rate scheduler attached to optimizer after warm-up
   - Options explored: cosine decay, inverse square root decay, PyTorch ReduceLROnPlateau
   - Gradually decreases LR as training progresses

### 6.5 Transfer Learning / Partitioned Training Scheme

Training all 4,954 classes at once is extremely challenging. The solution:

**Step 1 — Partition the sky:**
- Divide celestial sphere into 10 bands by declination
- Each band: 18° declination window
- Partitioning by declination empirically outperforms right ascension (lesser class imbalance)
- Example splits: [-90..-72], [-72..-54], ..., [72..90]

**Step 2 — Train base encoder:**
- Train full SIFTERN (encoder + classifier head) on one arbitrary partition (e.g. [0..18])
- This yields pretrained base model `M_P` with a trained encoder `M_E`

**Step 3 — Fine-tune classifier heads for remaining partitions:**
- For each of the other 9 partitions `S`:
  - **Freeze all encoder layers** (keep `M_E` weights fixed)
  - Initialize a new classifier head `ℝ^d → ℝ^(N_C_S)` where `N_C_S` = number of stars in partition S
  - Fine-tune only the classifier head on partition S's dataset

**Result:**
- 1 shared trained encoder `M_E`
- 10 fine-tuned classifier heads (one per declination partition)

**Runtime aggregation:**
- Given a star scene: pass through `M_E` to get encoded representation `f`
- Matrix multiply `f` by each of the 10 classifier heads to get 10 probability distributions
- Select the identification with the highest probability across all heads
- **Optimization:** if approximate attitude is known (e.g. from IMU or previous identification), only query the relevant partition's classifier head

**Benefits of this scheme:**
1. Significantly reduced time to convergence per training run (smaller dataset per run)
2. Can apply more aggressive augmentations per partition
3. Encoder learns transferable centroid-relationship features generalizable across the sky

---

## 7. Evaluation Results

Tested on real images from: Portland State Aerospace Society's OreSat project, Astrometry.net database, LOST: Open-source Star Tracker project, and vendor samples.

Ground truth obtained via Astrometry.net plate solver. Each experiment: 1000+ images.

**Metrics:**
- Availability = % images correctly identified
- Error rate = % images incorrectly identified
- Unidentified rate = % images SIFTER refused to identify
- Constraint: Availability + Error Rate + Unidentified Rate = 100%

### 7.1 Centroid Error Results

| Centroid Error (σ pixels) | SIFTER Availability | SIFTER Error Rate | Tetra Availability |
|---|---|---|---|
| 0 | 99.83% | 0.01% | ~100% |
| 1 | ~99.5% | ~0.1% | ~60% |
| 2 | ~98.5% | ~0.3% | ~40% |
| 3 | ~98% | ~0.5% | <40% |
| 5 | >97% | 0.93% | <40% |

SIFTER maintains >97% availability even at 5 pixels of centroid error. Tetra drops below 40% after 1.6 pixels.

### 7.2 False Stars Results

| False Stars | SIFTER Availability | SIFTER Error Rate |
|---|---|---|
| 0 | ~99.8% | ~0% |
| 5 | 96.89% | 0.97% |
| 10 | ~70% | 2.31% |

Tetra maintains ~100% availability even at 10 false stars (does not rely on guide star classification). SIFTER is better than Pyramid (80% at 5 false stars) and better than RPNet (80% at 3 false stars).

### 7.3 Magnitude Error Results

| Magnitude Error (σ Mv) | SIFTER Availability | SIFTER Error Rate |
|---|---|---|
| 0 | ~99.8% | ~0% |
| 0.5 | >98% | 0.69% |

Comparable or better than all surveyed NN algorithms. Magnitude not used as a classification feature, so magnitude noise reduces to dropped/extra stars.

### 7.4 Runtime and Storage

| Device | Inference Time |
|---|---|
| NVIDIA T4 GPU | 4 ms |
| Intel Xeon CPU (2.30 GHz) | 12 ms |
| Estimated rad-hardened 100 MHz CPU | 300–400 ms |

Time complexity: O(n²) due to self-attention, where n = number of centroids. In practice n is small (typically 5–15 stars in a 12°×12° FOV), so this is fast.

| Algorithm | Parameters | Storage |
|---|---|---|
| SIFTER (8-bit QNT) | 8M | 7.5 MB |
| SIFTER (16-bit QNT) | 8M | 15 MB |
| SIFTER (32-bit) | 8M | 30 MB |
| Tetra | N/A | 40 MB |
| 1D-CNN | 186K | 0.74 MB |
| RPNet | 819K | 3.27 MB |
| FCNet | 16M | 60 MB |
| VGG | 138M | 552 MB |

---

## 8. Implementation Notes and Design Rationale

### 8.1 Why Transformer over CNN/FCN

- CNNs (VGG, 1D-CNN) suffer from large model sizes and poor generalization to real data
- FCNs (FCNet) underfit: training accuracy stalls below 80% — model too simple for ~5000-class problem
- Transformer self-attention directly models pairwise relationships between centroids without needing to discretize or handcraft features
- The Transformer naturally handles **variable-length inputs** (different numbers of stars per scene)

### 8.2 Why Centroids, Not Raw Images (The "Centroid Abstraction")

- Training on synthetic images requires highly realistic image simulation (sensor noise, PSF, optics)
- The gap between synthetic and real images causes severe overfitting (VGG: <70% availability on real images despite ~100% training accuracy)
- Operating on centroid outputs means we only need to model noise at the centroid level:
  - False star = extra centroid with random coordinates
  - Centroid error = Gaussian offset to (x, y)
  - Missing star = dropped centroid
- This dramatically reduces simulation complexity and the sim-to-real gap
- Note: skipping centroiding entirely (image → attitude directly) is NOT feasible — many missions require very small attitude error, and centroid algorithms are already well-optimized

### 8.3 Why No Handcrafted Feature Extraction

- Type A algorithms (FCNet, RPNet, 1D-CNN) use log-polar transform to build fixed-size bit vectors
- This requires choosing a discretization factor N_bin — a critical hyperparameter
- Discretization inherently loses centroid coordinate information
- This information loss is hypothesized to be why Type A algorithms degrade under centroid error
- SIFTER feeds raw (x, y) coordinates directly into the encoder, preserving all spatial information

### 8.4 Rotational Invariance

- Handcrafted log-polar features are naturally rotationally invariant (encode angular distance, not absolute position)
- SIFTER has no such feature, so rotational invariance must be learned
- It is achieved implicitly: the training dataset includes all roll angles (0°, 5°, 10°, ...) for each (RA, Dec) pair
- The model learns to be invariant to rotation through data diversity rather than architectural inductive bias

### 8.5 Confidence Threshold and Error Rate Control

- SIFTER prioritizes low error rate over high availability
- The threshold `th` is a tunable hyperparameter that trades unidentified rate for error rate
- A satellite can re-image; an incorrect attitude reading is far more damaging
- GuidedMatch provides an additional rejection layer: insufficient matches → refuse identification

---

## 9. Key References

1. Vaswani et al. "Attention is All You Need." NeurIPS (2017) — Transformer architecture
2. Devlin et al. "BERT." NAACL (2019) — [CLS] token scheme
3. Brody et al. "On the Expressivity Role of LayerNorm in Transformers' Attention." ACL (2023) — LayerNorm justification
4. Polyakov, Zhang, Haining. "LOST: An Open-Source Suite of Star Tracking Software." SmallSat (2023) — classical algorithm benchmarks
5. Rijlaarsdam et al. "Efficient Star Identification Using a Neural Network." Sensors (2020) — FCNet baseline
6. Wang et al. "An Efficient and Robust Star Identification Algorithm Based on Neural Networks." Sensors (2021) — 1D-CNN baseline
7. Wang et al. "An artificial intelligence enhanced star identification algorithm." Frontiers of IT & EE (2020) — VGG baseline
8. Xu et al. "RPNet: A Representation Learning-Based Star Identification Algorithm." IEEE Access (2019) — RPNet baseline

---

## 10. Suggested Implementation Stack

- **Language:** Python 3.10+
- **Framework:** PyTorch (referenced in paper for ReduceLROnPlateau scheduler)
- **Key PyTorch components:**
  - `nn.Linear` — embedding layer and classifier head
  - `nn.TransformerEncoderLayer` — or custom implementation per above spec
  - `nn.TransformerEncoder` — stacks N encoder blocks
  - `torch.nn.functional.cross_entropy` — loss
  - `torch.optim.AdamW` — optimizer
  - `torch.optim.lr_scheduler` — LR scheduling (cosine or ReduceLROnPlateau)
- **Data:** Yale Bright Star Catalog (publicly available); synthetic dataset open-sourced by Zhang on GitHub
- **Quantization:** PyTorch `torch.quantization` for 16-bit or 8-bit post-training quantization

---

## 11. Hyperparameter Summary (Quick Reference)

| Hyperparameter | Value | Notes |
|---|---|---|
| d_model (embedding dim) | 256 | Core model width |
| n_heads | 8 | Attention heads per block |
| n_encoder_blocks | 5 | Depth of encoder |
| d_head | 32 | d_model / n_heads |
| activation | ReLU | In MLP sub-block |
| encoder_output | global average | Mean pool over centroid dim |
| N_C | 4954 | Output classes (catalog stars) |
| total_params | 8,090,631 | Measured in paper |
| batch_size | 256 | Training |
| lr_max | 5e-5 | Peak learning rate |
| warmup_steps | 7000 | Linear warmup |
| weight_decay | 1e-2 | AdamW |
| centroid_noise_sigma | 5 px | Training augmentation |
| false_stars_range | 0–5 | Training augmentation |
| magnitude_noise_sigma | 0.5 Mv | Training augmentation |
| sky_partitions | 10 | Declination bands, 18° each |
| catalog_mag_limit | 6.0 Mv | Star catalog filter |
| catalog_min_separation | 0.05° | Double star filter |
| FOV | 12° × 12° | Simulated camera |
| sky_scan_ra_dec_step | 1° | Dataset generation |
| sky_scan_roll_step | 5° | Dataset generation |
| dataset_size | ~13M | Total synthetic samples |
| confidence_threshold_th | tunable | Controls availability vs error rate tradeoff |
| guidedmatch_tolerance_ε | tunable | Angular distance tolerance for centroid-catalog matching |
