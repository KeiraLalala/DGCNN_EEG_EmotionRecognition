# EEG Emotion Recognition using DGCNN

This repository provides an implementation of a Dynamic Graph Convolutional Neural Network (DGCNN) for EEG-based emotion recognition followed by the paper: "EEG Emotion Recognition Using Dynamical Graph Convolutional Neural Networks". It supports both subject-independent (Leave-One-Subject-Out, LOSO) and subject-dependent (cross-session) evaluation protocols.

The approach leverages Differential Entropy (DE) features extracted from EEG signals and models spatial relationships among EEG channels using dynamic, learnable graph structures. The implementation is tested on the SEED dataset and demonstrates strong performance in emotion classification tasks.

---

## 📂 Dataset: SEED (Preprocessed)

The data from SEED dataset are supposed have been preprocessed, which contains EEG recordings from 15 subjects over 3 sessions

- Sampling rate: 200 Hz
- Channels: 62
- Each session: 24 trials per subject
- Preprocessing includes:
  - Bandpass filtering [1–75 Hz]
  - Line noise removal
  - Average re-referencing
  - Segmentation into 4-second samples

> ⚠️ Note: It is assumed that the dataset has already been preprocessed and saved as `.csv` files before using this repo.
> Data contains one sample per row:  
`200 time points × 4 seconds × 62 channels = 49600 columns`, followed by:
- `SessionLabel` (1–3)
- `SubjectLabel` (1–15)
- `EmotionLabel` (0: neutral, 1: sad, 2: fear, 3: happy)

---

## 📦 Project Modules

### 1. `precompute_de_features.ipynb`
Extracts and saves DE features from `.csv` EEG recordings to `.pt` or `.npy` files to speed up training.

- Segments EEG into 1-second windows
- Calculates DE as:  
  \[
  \text{DE} = \frac{1}{2} \log(2\pi e \sigma^2)
  \]
- Saves features with subject/session identifiers

---

### 2. `DGCNN_Data_Model.py`
Defines the DGCNN architecture.

- `PrecomputedEEGDataset`: Loads `.pt` DE features and `.csv` labels
- `compute_normalized_laplacian`: Constructs the Laplacian from learnable adjacency matrix
- `chebyshev_polynomials`: Generates $K$-order Chebyshev polynomials
- `ChebConv`: Graph convolution using Chebyshev approximation
- `DGCNNBlock`: Full model including  
  - Learnable $\mathbf{W}^*$
  - ChebConv + 1×1 Conv + ReLU
  - Global average pooling
  - Fully connected classification

---

### 3. `DGCNN_LOSO.py`
Performs **subject-independent evaluation** using Leave-One-Subject-Out (LOSO) cross-validation.

- Loads precomputed `.pt` features and `.csv` labels
- For each subject:
  - Train on 14 others
  - Test on the held-out subject
- Logs per-subject and average accuracy

---

### 4. `DGCNN_CrossSession.py`
Performs **subject-dependent, cross-session evaluation**.

- Loads all sessions for a subject
- Trains on Session 1 & 2, tests on Session 3 (and vice versa)
- Logs session-wise accuracy and mean performance

---

## ⚠️ Common Issues & Fixes

| Issue | Solution |
|-------|----------|
| **NaNs in Laplacian** | Set `eigval_max = 2.0` to avoid instability in normalization |
| **Broadcast errors in ChebConv** | Use `torch.einsum('nm,bmf->bnf', T_k, x)` instead of `T_k @ x` |
| **Unstable `W*` init** | Use `nn.init.xavier_normal_` |
| **Gradient explosion** | Add `W_pos += 1e-5 * I` and symmetrize with `W_pos = 0.5 * (W + W.T)` |
| **Debugging NaNs** | Use loop to check:  
| **Loss diverges** | Reduce `learning_rate` to `1e-5` and restart |

---

## 🛠️ Requirements

- Python ≥ 3.8  
- PyTorch ≥ 1.11  
- NumPy, Pandas, Matplotlib  
- tqdm (for training logs)

---

## 🚀 Usage

```bash
# Step 1: Precompute DE features
Run: precompute_de_features.ipynb

# Step 2: Subject-Independent Training
python DGCNN_LOSO.py

# Step 3: Subject-Dependent Cross-Session Evaluation
python DGCNN_CrossSession.py

## 📄 Citation

This implementation is based on the following paper:

> **Tengfei Song, Wenming Zheng, Peng Song, and Zhen Cui**  
> *EEG Emotion Recognition Using Dynamical Graph Convolutional Neural Networks*  
> IEEE Transactions on Affective Computing, vol. 11, no. 3, pp. 532–541, 2020.  
> [DOI: 10.1109/TAFFC.2018.2819665](https://doi.org/10.1109/TAFFC.2018.2819665)

Please cite this work if you use the algorithm or code in your research:

```bibtex
@ARTICLE{Song2020,
  author={Song, Tengfei and Zheng, Wenming and Song, Peng and Cui, Zhen},
  journal={IEEE Transactions on Affective Computing}, 
  title={{EEG} Emotion Recognition Using Dynamical Graph Convolutional Neural Networks}, 
  year={2020},
  volume={11},
  number={3},
  pages={532--541}
}
