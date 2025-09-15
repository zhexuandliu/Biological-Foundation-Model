import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import coo_matrix

def build_G_from_edges(edges_df, gene_names, col1="gene1", col2="gene2", make_dense=False):
    """
    edges_df: DataFrame with two gene columns (undirected edges)
    gene_names: list/array of genes defining the node order in G (n x n)
    """
    # normalize inputs
    genes = pd.Index(gene_names)
    gene2idx = pd.Series(range(len(genes)), index=genes)

    # keep only edges with both ends in the list and drop self-loops
    df = edges_df[[col1, col2]].dropna().copy()
    df = df[df[col1].isin(genes) & df[col2].isin(genes)]
    df = df[df[col1] != df[col2]]

    # deduplicate undirected edges (sort endpoints within each row)
    a = df[col1].to_numpy()
    b = df[col2].to_numpy()
    left = np.minimum(a, b)
    right = np.maximum(a, b)
    df_undirected = pd.DataFrame({col1: left, col2: right}).drop_duplicates()

    # indices
    i = gene2idx.loc[df_undirected[col1]].to_numpy()
    j = gene2idx.loc[df_undirected[col2]].to_numpy()
    n = len(genes)

    # build upper triangle, then symmetrize
    data = np.ones_like(i, dtype=np.int8)
    G_upper = coo_matrix((data, (i, j)), shape=(n, n))
    G = (G_upper + G_upper.T).tocsr()  # symmetric 0/1
    G.data[:] = 1                      # ensure binary if duplicates slipped in

    if make_dense:
        return G.toarray()
    return G  # scipy.sparse CSR (preferred)

# --- example ---
# STRING_gene_interaction_high_conf: DataFrame with columns "gene1","gene2"
# gene_names: list of genes in your desired order
# G = build_G_from_edges(STRING_gene_interaction_high_conf, gene_names)


def train_A(X, G, X_val, G_val, k=16, lr=1e-2, steps=200, batch_size=1000, lam=1e-4, device="cpu"):
    """
    X: (n,d) tensor, gene embeddings
    G: (n,n) tensor, adjacency (0/1)
    k: embedding dim after A
    """
    
    X = torch.as_tensor(X, dtype=torch.float32, device=device)   # (n,d) torch
    np.fill_diagonal(G, 0)
    G = torch.from_numpy((G > 0).astype(np.bool_)).to(device)    # (n,n) torch.bool
    
    X_val = torch.as_tensor(X_val, dtype=torch.float32, device=device)   # (n,d) torch
    np.fill_diagonal(G_val, 0)
    G_val = torch.from_numpy((G_val > 0).astype(np.bool_)).to(device)    # (n,n) torch.bool
    
    n, d = X.shape
    A = torch.nn.Parameter(0.01 * torch.randn(d, k, device=device))
    w = torch.nn.Parameter(torch.zeros((), device=device))
    opt = torch.optim.Adam([A, w], lr=lr)
    bce = torch.nn.BCEWithLogitsLoss()
    
    # build pools of pos/neg pairs once (torch indexing throughout)
    I, J = torch.triu_indices(n, n, offset=1, device=device)
    ymask = G[I, J]                     # torch.bool
    pos = torch.stack([I[ymask], J[ymask]], dim=1)
    neg = torch.stack([I[~ymask], J[~ymask]], dim=1)
    
    nv = X_val.shape[0]
    Iv, Jv = torch.triu_indices(nv, nv, offset=1, device=device)
    yv = G_val[Iv, Jv].float() 
    
    train_loss_trace, val_loss_trace = [], []
    
    for step in range(steps):
        m = batch_size // 2
        ip = torch.randint(len(pos), (m,), device=device)
        ineg = torch.randint(len(neg), (m,), device=device)
        pairs = torch.cat([pos[ip], neg[ineg]], dim=0)           # (B,2)
        i, j = pairs[:, 0].long(), pairs[:, 1].long()
        y = G[i, j].float()                                      # (B,)
    
        delta = X[i] - X[j]                                      # (B,d) torch
        z = delta @ A                                            # (B,k) torch
        sqdist = (z * z).sum(dim=1)                              # (B,)
        logits = w - sqdist
        loss = bce(logits, y) + lam * (A * A).sum()
    
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step%20==0:
          with torch.no_grad():
            vdelta = X_val[Iv] - X_val[Jv]                           # (N_val,d)
            vz = vdelta @ A                                          # (N_val,k)
            vsqdist = (vz * vz).sum(dim=1)                           # (N_val,)
            vlogits = w - vsqdist
    
            pos_mask = (yv == 1)
            neg_mask = ~pos_mask
            pos_loss = bce(vlogits[pos_mask], yv[pos_mask])
            neg_loss = bce(vlogits[neg_mask], yv[neg_mask])
            val_loss = (pos_loss + neg_loss)/2
            # val_loss = bce(vlogits, yv)
            # val_loss = bce(logits, y)
            val_loss_trace.append(float(val_loss.detach().cpu()))

        # record losses
        train_loss_trace.append(float(loss.detach().cpu()))

    return A.detach(), float(w.detach()), train_loss_trace, val_loss_trace