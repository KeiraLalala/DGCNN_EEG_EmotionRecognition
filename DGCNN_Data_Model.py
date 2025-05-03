#!/usr/bin/env python
# coding: utf-8

# # RGCNN Implementation Roadmap 

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader


# ### 1. Loading pretrained data and extracted DE Features
# ### Custom Dataset Class
# ##### The reshape assumes that the EEG data in the .csv file is organized sequentially:
# ##### First 200 columns → EEG data for 1st second.
# ##### Next 200 columns → EEG data for 2nd second.
# ##### Repeats for 4 seconds × 62 channels.

class PrecomputedEEGDataset(Dataset):
    def __init__(self, features_list, labels_list):
        assert len(features_list) == len(labels_list), "Mismatch between features and labels files"

        self.features = []
        self.labels = []

        for f_path, l_path in zip(features_list, labels_list):
            feat = torch.load(f_path, weights_only=True)  # shape: [N, 62, 5]
            label_df = pd.read_csv(l_path)

            assert len(feat) == len(label_df), f"Mismatch in samples: {f_path}"

            self.features.append(feat)
            self.labels.append(label_df)

        # Concatenate all data across files
        self.features = torch.cat(self.features, dim=0)  # shape: [Total_N, 62, 5]
        self.labels_df = pd.concat(self.labels, ignore_index=True)  # pandas DataFrame

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]  # [62, 5]
        label_row = self.labels_df.iloc[idx]

        return feature, {
            "subject": int(label_row['subject']),
            "session": int(label_row['session']),
            "emotion": int(label_row['emotion'])
        }

# ### 2. Laplacian + Normalization
def compute_normalized_laplacian(W):
    # W: [N, N]
    D = torch.diag(torch.sum(W, dim=1) + 1e-6)
    L = D - W
    D_inv_sqrt = torch.diag(torch.pow(torch.sum(W, dim=1), -0.5))
    D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0  # stability
    # Rescale L to [-1, 1] for Chebyshev polynomial approximation
    L_norm = D_inv_sqrt @ L @ D_inv_sqrt


    eigval_max = 2.0 # It's necessary to use this line otherwise, it may introduce NaNs due to instabiility in L_norm
    #eigval_max = torch.linalg.eigvalsh(L_norm).max() #
    L_tilde = (2.0 * L_norm) / eigval_max - torch.eye(W.size(0), device=W.device)

    return L_tilde

# ### 3. Generate Chebyshev Polynomials
def chebyshev_polynomials(L_tilde, K):
    """
    L_tilde: [N, N]
    Returns: list of [N, N] matrices: [T_0, T_1, ..., T_{K-1}]
    """
    N = L_tilde.size(0)
    T_k = []
    T_k.append(torch.eye(N, device=L_tilde.device))         # T_0
    if K > 1:
        T_k.append(L_tilde)                                 # T_1
    for k in range(2, K):
        T_k.append(2 * torch.matmul(L_tilde, T_k[-1]) - T_k[-2])        # T_k = 2L T_{k-1} - T_{k-2}
    return T_k

# ### 4. Define the ChebConv Layer
class ChebConv(nn.Module):
    def __init__(self, in_features, out_features, K):
        super(ChebConv, self).__init__()
        self.K = K
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Parameter(torch.Tensor(K, in_features, out_features))
        nn.init.xavier_uniform_(self.linear)

    def forward(self, x, T_k_list):
        # x: [B, N, Fin], T_k_list: list of [N, N], length K
        out = 0
        for k in range(self.K):
            T_k = T_k_list[k]  # [N, N]
            Tx = torch.einsum('nm,bmf->bnf', T_k, x)  # Using this to replace T_k @ x, it's safer and clearer to avoide broadcast error
            out += torch.matmul(Tx, self.linear[k])  # [B, N, Fout]
        return out # [B, N, Fout]
        
# ### 5. DGCNNBlock:
# #### Chebyshev Graph Convolution (ChebConv)
# #### ReLU Activation
# #### Optional 1×1 Conv or projection
# #### Fully Connected (FC) Layer
# #### Softmax for classification

class DGCNNBlock(nn.Module):
    def __init__(self, in_features, hidden_features, num_classes, K, num_nodes):
        super(DGCNNBlock, self).__init__()
        self.K = K
        self.num_nodes = num_nodes
        self.W_star = nn.Parameter(torch.empty(num_nodes, num_nodes))
        nn.init.xavier_normal_(self.W_star)
        self.cheb_conv = ChebConv(in_features, hidden_features, K)
        self.conv1x1 = nn.Linear(hidden_features, hidden_features)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(hidden_features)
        self.dropout = nn.Dropout(0.5) 
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_features, num_classes)
        
    def forward(self, x):
        """
        x: [B, 62, 5]
        T_k_list: list of [N, N] Chebyshev polynomials (length K)
        returns: [B, num_classes]
        """
        W_pos = F.relu(self.W_star) # Laplacian
        # W_pos += 1e-3 * torch.eye(self.num_nodes, device=x.device)
        # W_pos = 0.5 * (W_pos + W_pos.T)  # symmetric
        # if torch.isnan(W_pos).any():
        #     print("❌ W_pos contains NaN!")
        #     print(W_pos)
        #     raise ValueError("NaN in adjacency matrix.")
        L_tilde = compute_normalized_laplacian(W_pos) # [62, 62]
        T_k_list = chebyshev_polynomials(L_tilde, self.K) # list of [62, 62]
        
        out = self.cheb_conv(x, T_k_list)  # [B, 62, Fout]
        out = self.conv1x1(out)          
        out = self.relu(out) # non-linearity
        #out = self.dropout(out) adding if you need

        out = out.permute(0, 2, 1)          # [B, Fout, 62]
        # out = self.bn(out)
        out = self.global_pool(out)        # [B, Fout, 1]
        out = out.squeeze(-1)              # [B, Fout]

        logits = self.fc(out)              # [B, num_classes]
        return logits

        # eps = 1e-5
        # W_pos += eps * torch.eye(W_pos.size(0), device=W_pos.device)
        # W_pos = 0.5 * (W_pos + W_pos.T)
        # L = torch.eye(W_pos.size(0), device=W_pos.device) - normalize_adjacency(W_pos)
        # L_tilde = 2 * L - torch.eye(W_pos.size(0), device=W_pos.device)