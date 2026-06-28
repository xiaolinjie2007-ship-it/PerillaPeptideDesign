#!/usr/bin/env python3
"""Isotonic Regression calibration + cognitive/storage protein prediction."""
import os, pickle, time, numpy as np, pandas as pd, torch, esm, warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

# ========== Shallow features ==========
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
    f.append(min(max(9.0+(seq.count('K')+seq.count('R')+seq.count('H')*0.5)/max(L,1),3),12))
    f.append((seq.count('F')+seq.count('W')+seq.count('Y'))/max(L,1))
    f.append((seq.count('G')+seq.count('P')+seq.count('S')+seq.count('N'))/max(L,1))
    return np.array(f, dtype=np.float32)

AA_valid = set('ACDEFGHIKLMNPQRSTVWYU')
def valid_pep(s):
    return not (pd.isna(s) or str(s).strip() == '') and all(c in AA_valid for c in str(s).upper().strip())

# ========== Paths (relative to repo root) ==========
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FEAT_DIR   = os.path.join(_REPO_ROOT, '01_features', 'precomputed')
MODEL_PATH = os.path.join(_REPO_ROOT, '02_optimal_model', 'model', 'LightGBM_Optuna_best.pkl')
DST_DIR    = os.path.join(_REPO_ROOT, '03_calibration_isotonic')

# External prediction data (set your own paths)
COG_FEAT = None  # e.g. '/path/to/cognitive_features.npy'
COG_META = None  # e.g. '/path/to/cognitive_metadata.csv'
STO_FILE = None  # e.g. '/path/to/storage_protein.xlsx'

COG_DIR = os.path.join(DST_DIR, 'cognitive_peptide')
STO_DIR = os.path.join(DST_DIR, 'storage_protein')
MODEL_SUB = os.path.join(DST_DIR, 'model')
for d in [COG_DIR, STO_DIR, MODEL_SUB]:
    os.makedirs(d, exist_ok=True)

# ========== 1. Load original model ==========
print('='*55)
print('  1. Loading original model')
print('='*55)
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)
print(f'  LightGBM, trees={model.n_estimators_}, dim={model.n_features_}')

# ========== 2. Isotonic calibration ==========
print('\n' + '='*55)
print('  2. Isotonic Regression calibration')
print('='*55)
X_all = np.load(f'{FEAT_DIR}/X_train.npy'); y_all = np.load(f'{FEAT_DIR}/y_train.npy')
X_val = np.load(f'{FEAT_DIR}/X_val.npy'); y_val = np.load(f'{FEAT_DIR}/y_val.npy')
X_all = np.vstack([X_all, X_val]); y_all = np.hstack([y_all, y_val])
X_test_orig = np.load(f'{FEAT_DIR}/X_test.npy'); y_test_orig = np.load(f'{FEAT_DIR}/y_test.npy')

from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier
X_train_cal, X_cal, y_train_cal, y_cal = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)
print(f'  Total: {len(X_all)} samples (pos {sum(y_all==1)}/neg {sum(y_all==0)})')
print(f'  Training(80%): {len(X_train_cal)} samples')
print(f'  Calibration(20%): {len(X_cal)} samples')
print(f'  Test set: {len(X_test_orig)} samples (untouched)')

t0 = time.time()
best_params = {
    'max_depth': 8, 'num_leaves': 62, 'learning_rate': 0.0635,
    'subsample': 0.8529, 'colsample_bytree': 0.6746, 'bagging_freq': 10,
    'reg_lambda': 0.8928, 'lambda_l1': 0.0010, 'min_child_samples': 16,
    'min_gain_to_split': 0.0355, 'max_bin': 110, 'max_delta_step': 3.7896,
    'min_sum_hessian_in_leaf': 0.1193, 'boost_from_average': True,
    'extra_trees': False, 'random_state': 42, 'verbose': -1,
}
cal_model = LGBMClassifier(n_estimators=1000, **best_params)
cal_model.fit(X_train_cal, y_train_cal)
calibrated = CalibratedClassifierCV(cal_model, cv='prefit', method='isotonic')
calibrated.fit(X_cal, y_cal)
t_cal = time.time() - t0
print(f'  Calibration complete ({t_cal:.0f}s)')
print(f'  Model saved as: 1 LightGBM + 1 isotonic calibrator = 1 file')

prob_before = model.predict_proba(X_test_orig)[:, 1]
prob_after = calibrated.predict_proba(X_test_orig)[:, 1]
print(f'\n  Test set probability comparison:')
print(f'    Before: min={prob_before.min():.4f} max={prob_before.max():.4f} mean={prob_before.mean():.4f}')
print(f'    After:  min={prob_after.min():.4f} max={prob_after.max():.4f} mean={prob_after.mean():.4f}')

# Save model & notes
with open(f'{MODEL_SUB}/LightGBM_Isotonic_calibrated.pkl', 'wb') as f:
    pickle.dump(calibrated, f)
with open(f'{DST_DIR}/calibration_notes.txt', 'w') as f:
    f.write('Calibration: Isotonic Regression\n')
    f.write('='*50)
    f.write('\n\n[Method] Isotonic Regression (non-parametric, monotonic)')
    f.write('\n1 LightGBM + 1 isotonic calibrator = 1 file')
    f.write('\n\n[LightGBM params]')
    f.write('\nSource: Optuna 300-trial search')
    for k, v in best_params.items():
        if k not in ['random_state', 'verbose']:
            f.write(f'\n  {k}: {v}')
    f.write(f'\n\n[Data split]')
    f.write(f'\nTotal: {len(X_all)} samples (train+val)')
    f.write(f'\n  Training(80%): {len(X_train_cal)}')
    f.write(f'\n  Calibration(20%): {len(X_cal)}')
    f.write(f'\n  Test({len(X_test_orig)}): untouched')
    f.write(f'\n\n[Calibration effect on test set]')
    f.write(f'\nBefore: min={prob_before.min():.4f} max={prob_before.max():.4f} mean={prob_before.mean():.4f}')
    f.write(f'\nAfter:  min={prob_after.min():.4f} max={prob_after.max():.4f} mean={prob_after.mean():.4f}')

# ========== 3. Cognitive peptide prediction ==========
if COG_FEAT and COG_META and os.path.exists(COG_FEAT):
    print('\nPredicting cognitive peptides...')
    X_cog = np.load(COG_FEAT)
    cog_meta = pd.read_csv(COG_META)
    prob_cog = calibrated.predict_proba(X_cog)[:, 1]
    pred_cog = calibrated.predict(X_cog)
    cog_meta['source'] = 'cognitive_peptide'
    cog_meta['pred_label'] = ['active' if p == 1 else 'inactive' for p in pred_cog]
    cog_meta['pred_prob'] = [round(p, 6) for p in prob_cog]
    cog_meta.to_csv(f'{COG_DIR}/full_predictions.csv', index=False, encoding='utf-8-sig')
    cog_ranked = cog_meta.sort_values('pred_prob', ascending=False)
    cog_ranked.insert(0, 'rank', range(1, len(cog_ranked)+1))
    cog_ranked.to_csv(f'{COG_DIR}/probability_ranking.csv', index=False, encoding='utf-8-sig')
    print(f'  {len(cog_meta)} peptides, prob range {prob_cog.min():.4f}~{prob_cog.max():.4f}')
    prob_cog = prob_cog
else:
    print('Skipping cognitive peptide prediction (set COG_FEAT/COG_META)')
    prob_cog = None

# ========== 4. Storage protein prediction ==========
if STO_FILE and os.path.exists(STO_FILE):
    print('\nPredicting storage proteins...')
    sto_df = pd.read_excel(STO_FILE, sheet_name='Peptide_Details')
    sto_df = sto_df.rename(columns={'peptide': 'Peptide'}, errors='ignore')
    sto_df['Peptide'] = sto_df['Peptide'].astype(str).str.upper()
    sto_df = sto_df[sto_df['Peptide'].apply(valid_pep)]
    seq_info = sto_df.groupby('Peptide', sort=False).agg({'AO_Max_Score': 'mean'}).reset_index()
    seq_info = seq_info[seq_info['Peptide'].apply(valid_pep)]
    sto_seqs = seq_info['Peptide'].tolist()
    print(f'  {len(sto_df)} rows, {len(sto_seqs)} unique sequences')

    Xs_list = [shallow(s) for s in tqdm(sto_seqs, desc='Shallow', unit='seq')]
    Xs = np.array(Xs_list)
    print('  Loading ESM-2 150M...')
    m, a = esm.pretrained.esm2_t30_150M_UR50D(); m.eval(); bc = a.get_batch_converter()
    embs = []
    for start in tqdm(range(0, len(sto_seqs), 32), desc='ESM-2', unit='batch'):
        end = min(start + 32, len(sto_seqs))
        bd = [(str(i), sto_seqs[start+i]) for i in range(end-start)]
        _, _, bt = bc(bd)
        with torch.no_grad():
            r = m(bt, repr_layers=[30], return_contacts=False)
        tr = r['representations'][30]
        for i in range(end-start):
            embs.append(tr[i, 1:len(sto_seqs[start+i])+1, :].mean(dim=0).numpy())
    X_full = np.hstack([Xs, np.array(embs)])
    prob_sto = calibrated.predict_proba(X_full)[:, 1]
    pred_sto = calibrated.predict(X_full)
    print(f'  Prob range: {prob_sto.min():.4f}~{prob_sto.max():.4f}')
    print(f'  Active: {sum(pred_sto)}/{len(pred_sto)}')

    seq_info['pred_label'] = ['active' if p == 1 else 'inactive' for p in pred_sto]
    seq_info['pred_prob'] = [round(p, 6) for p in prob_sto]
    seq_info.insert(0, 'id', range(1, len(seq_info)+1))
    sto_df = sto_df.merge(seq_info[['Peptide', 'pred_label', 'pred_prob']], on='Peptide', how='left')
    sto_df.to_csv(f'{STO_DIR}/full_predictions.csv', index=False, encoding='utf-8-sig')
    ranked = seq_info.sort_values('pred_prob', ascending=False)
    ranked.insert(0, 'rank', range(1, len(ranked)+1))
    ranked.to_csv(f'{STO_DIR}/unique_ranking.csv', index=False, encoding='utf-8-sig')
else:
    print('Skipping storage protein prediction (set STO_FILE)')
    sto_seqs, prob_sto, pred_sto = [], None, None

print(f'\n{"="*55}')
print(f'  Done!')
print(f'{"="*55}')
