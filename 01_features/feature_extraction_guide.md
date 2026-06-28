# Feature Extraction Guide

The 1066-dim features used for training are precomputed and stored in `precomputed/`. These can be loaded directly for model training or prediction.

## Feature Composition

Each peptide is a 1066-dim vector:

| Feature | Dim | Description |
|:-------:|:---:|-------------|
| ESM-2 150M  | 640 | 30-layer Transformer, mean pooling over sequence |
| AA Composition | 20 | Frequency of 20 standard amino acids |
| Dipeptide Composition | 400 | Frequency of 400 dipeptide pairs |
| Net Charge | 1 | Charged residues (D/E=-1, K/R=1, H=0.5) |
| Hydrophobicity (KD) | 1 | Kyte-Doolittle index mean |
| Molecular Weight | 1 | Total MW / 1000 (kDa) |
| Isoelectric Point | 1 | Estimated from charged amino acids |
| Aromaticity | 1 | (F+W+Y) / sequence length |
| Flexibility | 1 | (G+P+S+N) / sequence length |

## Pipeline

```
Raw sequences (CSV: seq + label columns)
  |
  +-- Shallow features (426 dim) -- computed directly per sequence
  |     +-- AA frequencies: 20 dim
  |     +-- Dipeptide frequencies: 400 dim
  |     +-- Physicochemical properties: 6 dim
  |
  +-- ESM-2 150M embedding (640 dim) -- requires model inference
        +-- batch_size = 32
        +-- Layer 30 (last layer) token representations
        +-- Mean pooling per sequence
```

## Using Precomputed Features

```python
import numpy as np

X_train = np.load('precomputed/X_train.npy')  # (6910, 1066)
y_train = np.load('precomputed/y_train.npy')  # (6910,)
X_val   = np.load('precomputed/X_val.npy')    # (864, 1066)
y_val   = np.load('precomputed/y_val.npy')    # (864,)
X_test  = np.load('precomputed/X_test.npy')   # (864, 1066)
y_test  = np.load('precomputed/y_test.npy')   # (864,)

# Merge train + val for cross-validation
X_all = np.vstack([X_train, X_val])
y_all = np.hstack([y_train, y_val])
```

## Custom Feature Extraction (Reference Code)

### Shallow Features (426 dim)

```python
import numpy as np

AA = 'ACDEFGHIKLMNPQRSTVWY'
KD = {'A':1.8,'C':2.5,'D':-3.5,'E':-3.5,'F':2.8,'G':-0.4,'H':-3.2,'I':4.5,'K':-3.9,'L':3.8,'M':1.9,'N':-3.5,'P':-1.6,'Q':-3.5,'R':-4.5,'S':-0.8,'T':-0.7,'V':4.2,'W':-0.9,'Y':-1.3}
CHG = {'D':-1,'E':-1,'K':1,'R':1,'H':0.5}
AAM = {'A':89.1,'C':121.2,'D':133.1,'E':147.1,'F':165.2,'G':75.1,'H':155.2,'I':131.2,'K':146.2,'L':131.2,'M':149.2,'N':132.1,'P':115.1,'Q':146.1,'R':174.2,'S':105.1,'T':119.1,'V':117.1,'W':204.2,'Y':181.2}

def shallow_features(seq):
    L = len(seq)
    aac = np.zeros(20)
    for c in seq:
        i = AA.find(c)
        if i >= 0: aac[i] += 1
    f = (aac / L).tolist()
    
    dp = {a+b: 0 for a in AA for b in AA}
    for i in range(L - 1):
        di = seq[i:i+2]
        if di in dp: dp[di] += 1
    f.extend([dp[a+b] / max(L-1, 1) for a in AA for b in AA])
    
    chg = sum(CHG.get(c, 0) for c in seq)
    f.extend([
        chg,                                                  # net charge
        np.mean([KD.get(c, 0) for c in seq]),                 # hydrophobicity
        (sum(AAM.get(c, 0) for c in seq) + 18.015) / 1000,   # MW (kDa)
        min(max(9.0+(seq.count('K')+seq.count('R')+seq.count('H')*0.5)/max(L,1),3),12),  # pI
        (seq.count('F')+seq.count('W')+seq.count('Y'))/max(L,1),  # aromaticity
        (seq.count('G')+seq.count('P')+seq.count('S')+seq.count('N'))/max(L,1),  # flexibility
    ])
    return np.array(f, dtype=np.float32)
```

### ESM-2 150M Embedding (640 dim)

```python
import torch, esm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_esm, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
model_esm = model_esm.to(device).eval()
batch_converter = alphabet.get_batch_converter()

def esm_embedding(seqs, batch_size=32):
    """Extract ESM-2 150M embeddings. Returns (n, 640) array."""
    embs = []
    for start in range(0, len(seqs), batch_size):
        batch = seqs[start:start+batch_size]
        batch_data = [(str(i), s) for i, s in enumerate(batch)]
        _, _, tokens = batch_converter(batch_data)
        tokens = tokens.to(device)
        with torch.no_grad():
            result = model_esm(tokens, repr_layers=[30], return_contacts=False)
        reprs = result['representations'][30]
        for i, s in enumerate(batch):
            sl = len(s)
            embs.append(reprs[i, 1:sl+1, :].mean(dim=0).cpu().numpy())
    return np.array(embs)

# Usage
seqs = ['LLYQQPV', 'YRGDVFPK']
X_shallow = np.array([shallow_features(s) for s in seqs])  # (n, 426)
X_esm = esm_embedding(seqs)                                   # (n, 640)
X_full = np.hstack([X_shallow, X_esm])                        # (n, 1066)
```

## Notes

1. ESM-2 150M is downloaded on first use (~2.4GB)
2. Set `device='cuda'` for GPU inference
3. Shallow features are position-independent (frequency-based); ESM-2 embeddings are position-aware
4. ESM model version: `esm2_t30_150M_UR50D`
