import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kNN_MTD_model import kNNMTD
import optuna
from utils import *
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

def objective(trial, data, class_col):
    k = trial.suggest_int('k', 1, 10)
    # n_obs = trial.suggest_int('n_obs', 20, 200, step=20)
    knn_mtd = kNNMTD(n_obs=360, k=k, random_state=42)
    synthetic_data = knn_mtd.fit(data, class_col)

    # Calculate the Spearman correlation coefficient matrix of the real data and the synthetic data
    original_corr = data.corr(method='spearman')
    synthetic_corr = synthetic_data.corr(method='spearman')
    diff = np.sum(np.abs(original_corr.values - synthetic_corr.values))
    return diff

data = pd.read_csv('data/NaturalFrequency_real_40.csv')
class_col = 'frequency'

# Create an Optuna study object
study = optuna.create_study(direction='minimize')
study.optimize(lambda trial: objective(trial, data, class_col), n_trials=10)

# 获取最佳参数
best_params = study.best_params
print("Optimal parameters：", best_params)

# Generate synthetic data using the optimal parameters
knn_mtd_best = kNNMTD(n_obs=360, k=int(best_params['k']), random_state=42)
synthetic_data_best = knn_mtd_best.fit(data, class_col)

# # Merge the real dataset with the synthetic dataset
# df_enhanced = pd.concat([data, synthetic_data_best])

# Save the synthetic dataset
synthetic_data_best.to_csv('data/data_MTD/MY_MTD_40-360.csv', index=False)

# Calculate and print PCD
pcd = PCD(data, synthetic_data_best)
print(pcd)
