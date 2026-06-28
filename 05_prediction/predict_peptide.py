#!/usr/bin/env python3
"""Predict peptide activity using a trained LightGBM model."""
import os, pickle, numpy as np, pandas as pd, torch, esm, warnings, sys
warnings.filterwarnings('ignore')

# ========== 1. Shallow features (same as training) ==========
AA = 'ACDEFGHIKLMNPQRSTVWY'
KD = {'A':1.8,'C':2.5,'D':-3.5,'E':-3.5,'F':2.8,'G':-0.4,'H':-3.2,'I':4.5,'K':-3.9,'L':3.8,'M':1.9,'N':-3.5,'P':-1.6,'Q':-3.5,'R':-4.5,'S':-0.8,'T':-0.7,'V':4.2,'W':-0.9,'Y':-1.3}
CHG = {'D':-1,'E':-1,'K':1,'R':1,'H':0.5}
AAM = {'A':89.1,'C':121.2,'D':133.1,'E':147.1,'F':165.2,'G':75.1,'H':155.2,'I':131.2,'K':146.2,'L':131.2,'M':149.2,'N':132.1,'P':115.1,'Q':146.1,'R':174.2,'S':105.1,'T':119.1,'V':117.1,'W':204.2,'Y':181.2}

def shallow(seq):
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
    f.append(chg); f.append(np.mean([KD.get(c, 0) for c in seq]))
    f.append((sum(AAM.get(c, 0) for c in seq) + 18.015) / 1000)
    f.append(min(max(9.0 + (seq.count('K')+seq.count('R')+seq.count('H')*0.5)/max(L,1),3),12))
    f.append((seq.count('F')+seq.count('W')+seq.count('Y'))/max(L,1))
    f.append((seq.count('G')+seq.count('P')+seq.count('S')+seq.count('N'))/max(L,1))
    return np.array(f, dtype=np.float32)

# ========== 2. Load model ==========
_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Default: Isotonic calibrated model
# Options:
#   original: ../02_optimal_model/model/LightGBM_Optuna_best.pkl
#   isotonic: ../03_calibration_isotonic/model/LightGBM_Isotonic_calibrated.pkl
#   platt:    ../04_calibration_platt/model/LightGBM_Platt_calibrated.pkl
MODEL = os.path.join(_BASE, '03_calibration_isotonic', 'model', 'LightGBM_Isotonic_calibrated.pkl')

if not os.path.exists(MODEL):
    MODEL = os.path.join(_BASE, '02_optimal_model', 'model', 'LightGBM_Optuna_best.pkl')

with open(MODEL, 'rb') as f:
    model = pickle.load(f)
print(f'Model loaded! Feature dim: {model.n_features_}')
print(f'LightGBM, trees: {model.n_estimators_}\n')

# ========== 3. Load ESM-2 150M ==========
print('Loading ESM-2 150M (first run downloads)...')
m, a = esm.pretrained.esm2_t30_150M_UR50D()
m.eval()
bc = a.get_batch_converter()
print('ESM-2 ready!\n')

# ========== 4. Single prediction ==========
def predict_peptide(seq, show_detail=True):
    """Predict bioactivity for one peptide sequence."""
    seq = seq.strip().upper()
    if not seq:
        return None
    # Shallow features
    f_shallow = shallow(seq)
    # ESM-2 embedding
    batch = [('0', seq)]
    _, _, bt = bc(batch)
    with torch.no_grad():
        r = m(bt, repr_layers=[30], return_contacts=False)
    tr = r['representations'][30]
    emb = tr[0, 1:len(seq)+1, :].mean(dim=0).numpy()
    # Combine
    X = np.hstack([f_shallow, emb]).reshape(1, -1)
    # Predict
    prob = model.predict_proba(X)[0, 1]
    pred = model.predict(X)[0]
    label = 'Active' if pred == 1 else 'Inactive'
    if show_detail:
        print(f'  Sequence: {seq}')
        print(f'  Length: {len(seq)}aa')
        print(f'  Prediction: {label}')
        print(f'  Activity probability: {prob:.6f}')
    return {'seq': seq, 'label': label, 'probability': prob}

# ========== 5. Batch prediction from CSV ==========
def predict_from_csv(csv_path, seq_col='seq', output_path=None):
    """Batch predict from a CSV file."""
    df = pd.read_csv(csv_path)
    if seq_col not in df.columns:
        print(f'Error: Column "{seq_col}" not found in CSV')
        return
    results = []
    for seq in df[seq_col]:
        r = predict_peptide(seq, show_detail=False)
        if r:
            results.append(r)
    res_df = pd.DataFrame(results)
    if output_path:
        res_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f'Results saved: {output_path}')
    return res_df

# ========== 6. Predict a list of sequences ==========
def predict_list(seqs):
    """Predict for a list of sequences (returns DataFrame)."""
    results = []
    for seq in seqs:
        r = predict_peptide(seq, show_detail=False)
        if r:
            results.append(r)
    return pd.DataFrame(results)


if __name__ == '__main__':
    print('=' * 55)
    print('  LightGBM Peptide Activity Predictor')
    print('=' * 55)
    print()
    print('Usage:')
    print(f'  python {__file__} <SEQ>')
    print(f'  python {__file__} "LLYQQPV"')
    print(f'  python {__file__} --csv input.csv [--seq_col seq] [--output result.csv]')
    print()

    if len(sys.argv) >= 3 and sys.argv[1] == '--csv':
        csv_path = sys.argv[2]
        seq_col = sys.argv[4] if len(sys.argv) >= 5 and sys.argv[3] == '--seq_col' else 'seq'
        output = sys.argv[6] if len(sys.argv) >= 7 and sys.argv[5] == '--output' else None
        results = predict_from_csv(csv_path, seq_col, output)
        if results is not None:
            print(f'\nPredicted {len(results)} sequences')
    elif len(sys.argv) >= 2:
        seq = sys.argv[1]
        print('Single prediction:')
        predict_peptide(seq)
    else:
        print('Demo:')
        predict_peptide('LLYQQPV')
        print()
        predict_peptide('MEQNPNPNNL')
