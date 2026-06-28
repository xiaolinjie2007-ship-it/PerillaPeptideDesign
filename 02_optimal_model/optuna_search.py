#!/usr/bin/env python3
"""LightGBM CPU + Optuna Bayesian search (300 trials x 5-fold CV)"""
import os, pickle, time, numpy as np, pandas as pd, warnings, optuna
from lightgbm import LGBMClassifier, early_stopping
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, matthews_corrcoef, confusion_matrix
warnings.filterwarnings('ignore')

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(_REPO_ROOT, '01_features', 'precomputed')
OUT_DIR = os.path.join(_REPO_ROOT, '02_optimal_model')
LOG_DIR = os.path.join(OUT_DIR, 'search_logs')
EVAL_DIR = os.path.join(OUT_DIR, 'evaluation')
MODEL_DIR = os.path.join(OUT_DIR, 'model')
for d in [LOG_DIR, EVAL_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

X_all = np.load(f'{DATA_DIR}/X_train.npy'); y_all = np.load(f'{DATA_DIR}/y_train.npy')
X_v = np.load(f'{DATA_DIR}/X_val.npy'); y_v = np.load(f'{DATA_DIR}/y_val.npy')
X_test = np.load(f'{DATA_DIR}/X_test.npy'); y_test = np.load(f'{DATA_DIR}/y_test.npy')
X_all = np.vstack([X_all, X_v]); y_all = np.hstack([y_all, y_v])
N_TRIALS = 300
print(f'Train: {X_all.shape}, Test: {X_test.shape}')

def objective(trial):
    params = {
        'n_estimators': 1000,
        'max_depth': trial.suggest_int('max_depth', 4, 8),
        'num_leaves': trial.suggest_int('num_leaves', 15, 63),
        'learning_rate': trial.suggest_float('lr', 0.02, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample', 0.6, 0.9),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'reg_lambda': trial.suggest_float('L2', 0.5, 20),
        'lambda_l1': trial.suggest_float('L1', 0, 5),
        'min_child_samples': trial.suggest_int('min_child', 5, 50),
        'min_gain_to_split': trial.suggest_float('min_gain', 0, 2),
        'max_bin': trial.suggest_int('max_bin', 63, 255),
        'max_delta_step': trial.suggest_float('max_delta', 0, 5),
        'min_sum_hessian_in_leaf': trial.suggest_float('min_hessian', 0.001, 10, log=True),
        'boost_from_average': trial.suggest_categorical('boost_avg', [True, False]),
        'extra_trees': trial.suggest_categorical('extra_trees', [False, True]),
        'verbose': -1, 'random_state': 42,
    }
    model = LGBMClassifier(**params)
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    scores = []
    t0 = time.time()
    for ti, vi in skf.split(X_all, y_all):
        model.fit(X_all[ti], y_all[ti], eval_set=[(X_all[vi], y_all[vi])],
                  eval_metric='auc', callbacks=[early_stopping(20, first_metric_only=True)])
        prob = model.predict_proba(X_all[vi], num_iteration=model.best_iteration_)[:, 1]
        scores.append(roc_auc_score(y_all[vi], prob))
    trial.set_user_attr('time_s', round(time.time()-t0, 1))
    return np.mean(scores)

print(f'\n=== LightGBM CPU Optuna ({N_TRIALS} trials x 5-fold) ===\n')
t_start = time.time()
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
t_total = time.time() - t_start
best = study.best_params

# --- Trial log ---
log_data = []
for t in study.trials:
    if t.state != optuna.trial.TrialState.COMPLETE: continue
    row = {'trial': t.number, 'CV_AUC': round(t.value, 4) if t.value else None}
    for k, v in t.params.items(): row[k] = round(v, 4) if isinstance(v, float) else v
    row['time_s'] = t.user_attrs.get('time_s')
    log_data.append(row)
log_df = pd.DataFrame(log_data)
log_df.to_csv(f'{LOG_DIR}/trial_log_300rounds.csv', index=False)
print(f'\nSearch complete: {len(log_df)} trials, {t_total/60:.1f} min')

# Convergence trend
log_df['batch'] = log_df['trial'] // 10
trend = log_df.groupby('batch').agg({'CV_AUC': ['mean', 'max', 'std'], 'trial': 'count'}).round(4)
trend.columns = ['AUC_mean', 'AUC_max', 'AUC_std', 'trials']
trend.to_csv(f'{LOG_DIR}/convergence_trend.csv')

# Best value evolution
cur = 0; best_sofar = []
for _, r in log_df.iterrows():
    if r['CV_AUC'] > cur: cur = r['CV_AUC']
    best_sofar.append(cur)
pd.DataFrame({'trial': log_df['trial'], 'best_AUC': best_sofar}).to_csv(f'{LOG_DIR}/best_value_evolution.csv', index=False)

# --- 10-fold CV ---
print(f'\n=== 10-fold CV ===')
model = LGBMClassifier(n_estimators=1000, random_state=42, verbose=-1, **best)
skf = StratifiedKFold(10, shuffle=True, random_state=42)
cv10 = {'AUC': [], 'ACC': [], 'F1': [], 'MCC': []}
for fold, (ti, vi) in enumerate(skf.split(X_all, y_all), 1):
    model.fit(X_all[ti], y_all[ti], eval_set=[(X_all[vi], y_all[vi])],
              eval_metric='auc', callbacks=[early_stopping(20, first_metric_only=True)])
    prob = model.predict_proba(X_all[vi], num_iteration=model.best_iteration_)[:, 1]
    pred = model.predict(X_all[vi], num_iteration=model.best_iteration_)
    cv10['AUC'].append(roc_auc_score(y_all[vi], prob))
    cv10['ACC'].append(accuracy_score(y_all[vi], pred))
    cv10['F1'].append(f1_score(y_all[vi], pred))
    cv10['MCC'].append(matthews_corrcoef(y_all[vi], pred))
    print(f'  Fold {fold:2d}: AUC={cv10["AUC"][-1]:.4f} ACC={cv10["ACC"][-1]:.4f} MCC={cv10["MCC"][-1]:.4f}')
pd.DataFrame(cv10).to_csv(f'{EVAL_DIR}/cv_10fold_results.csv', index=False)

# --- Full retrain + Test ---
print(f'\n=== Full retrain ===')
final = LGBMClassifier(n_estimators=1000, random_state=42, verbose=-1, **best)
final.fit(X_all, y_all)

prob_test = final.predict_proba(X_test)[:, 1]; pred_test = final.predict(X_test)
ta = roc_auc_score(y_test, prob_test); tc = accuracy_score(y_test, pred_test)
tf = f1_score(y_test, pred_test); tm = matthews_corrcoef(y_test, pred_test)
cm = confusion_matrix(y_test, pred_test)

print(f'Test: AUC={ta:.4f} ACC={tc:.4f} F1={tf:.4f} MCC={tm:.4f}')
print(f'TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}')

# Confusion matrix
pd.DataFrame(cm, index=['True Neg', 'True Pos'], columns=['Pred Neg', 'Pred Pos']).to_csv(f'{EVAL_DIR}/confusion_matrix.csv', encoding='utf-8-sig')
# Test set predictions
pd.DataFrame({'true_label': y_test, 'pred_label': pred_test, 'pred_prob': [round(x, 6) for x in prob_test]}).to_csv(f'{EVAL_DIR}/test_set_predictions.csv', index=False)

# Hyperparameter importance
param_imp = optuna.importance.get_param_importances(study)
pd.DataFrame([{'param': k, 'importance': round(v, 4)} for k, v in sorted(param_imp.items(), key=lambda x: -x[1])]).to_csv(f'{EVAL_DIR}/hyperparameter_importance.csv', index=False)

# Feature importance
fi = final.feature_importances_
pd.DataFrame({'feature_idx': range(len(fi)), 'importance': fi}).sort_values('importance', ascending=False).to_csv(f'{EVAL_DIR}/feature_importance_906dim.csv', index=False)

# Complete parameter record
all_params = {
    'train_samples': len(X_all), 'test_samples': len(X_test),
    'pos_samples': int(sum(y_all == 1)), 'neg_samples': int(sum(y_all == 0)),
    'feature_dim': X_all.shape[1], 'feature_source': 'ESM-2 150M(640) + shallow(426)',
    'model': 'LightGBM CPU', 'n_trials': N_TRIALS,
    'cv_strategy': '5-fold search + 10-fold validation', 'search_time_min': round(t_total/60, 1),
    'best_5fold_CV_AUC': round(study.best_value, 4),
    '10fold_CV_AUC_mean': round(np.mean(cv10['AUC']), 4),
    '10fold_CV_ACC_mean': round(np.mean(cv10['ACC']), 4),
    '10fold_CV_F1_mean': round(np.mean(cv10['F1']), 4),
    '10fold_CV_MCC_mean': round(np.mean(cv10['MCC']), 4),
}
all_params.update({k: v for k, v in best.items()})
all_params.update({'Test_AUC': round(ta, 4), 'Test_ACC': round(tc, 4), 'Test_F1': round(tf, 4), 'Test_MCC': round(tm, 4)})
pd.DataFrame([all_params]).T.to_csv(f'{EVAL_DIR}/complete_parameters.csv', header=False)

# Pipeline notes
with open(f'{EVAL_DIR}/pipeline_notes.txt', 'w') as f:
    f.write('LightGBM CPU Bayesian hyperparameter search\n')
    f.write('=' * 50)
    f.write(f'\nTrain: {len(X_all)} ({sum(y_all==1)} pos / {sum(y_all==0)} neg)')
    f.write(f'\nTest: {len(X_test)} ({sum(y_test==1)} pos / {sum(y_test==0)} neg)')
    f.write(f'\nFeatures: ESM-2 150M 640 + shallow 426 = {X_all.shape[1]} dim')
    f.write(f'\nSearch: {N_TRIALS} trials x 5-fold CV')
    f.write(f'\nTime: {t_total/60:.1f} min')
    f.write(f'\nBest 5-fold CV AUC: {study.best_value:.4f}')
    f.write(f'\n10-fold CV AUC: {np.mean(cv10["AUC"]):.4f}')
    f.write(f'\nTest AUC: {ta:.4f}')

# Search report
rows = [{'category': 'best_param', 'param': k, 'value': v} for k, v in best.items()] + [
    {'category': 'test_result', 'param': 'AUC', 'value': round(ta, 4)},
    {'category': 'test_result', 'param': 'ACC', 'value': round(tc, 4)},
    {'category': 'test_result', 'param': 'F1', 'value': round(tf, 4)},
    {'category': 'test_result', 'param': 'MCC', 'value': round(tm, 4)},
]
pd.DataFrame(rows).to_csv(f'{EVAL_DIR}/search_report.csv', index=False)

# Model
with open(f'{MODEL_DIR}/LightGBM_Optuna_best.pkl', 'wb') as f: pickle.dump(final, f)

print(f'\n{"="*55}')
print(f'  LightGBM CPU Optuna search complete!')
print(f'  Test AUC: {ta:.4f}')
print(f'{"="*55}')
