import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.svm import SVR
import optuna

np.random.seed(42)

# Load data
real_data = pd.read_csv('data/NaturalFrequency_all_400_optuna.csv')
# real_data = pd.read_csv('data/kNNMTD_40-360/kNNMTD_40-360.csv')
# real_data = pd.read_csv('data/CTGAN_40-360/CTGAN_40-360.csv')
target_column = 'frequency'
feature_columns = [col for col in real_data.columns if col not in [target_column]]
X = real_data[feature_columns]
y = real_data[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define Optuna objective function
def objective(trial):
    # Define hyperparameter search space
    C = trial.suggest_float('C', 1e-2, 1e2, log=True)
    gamma = trial.suggest_float('gamma', 1e-2, 1e2, log=True)
    kernel = trial.suggest_categorical('kernel', ['rbf'])

    # Create SVR model
    svr = SVR(kernel=kernel, C=C, gamma=gamma)
    scores = cross_val_score(svr, X_train_scaled, y_train, cv=5, scoring='r2')
    mean_r2 = scores.mean()
    return mean_r2

# Create an Optuna study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Get the best hyperparameters
best_params = study.best_params
print("Best params:", best_params)

# Train the model using optimal hyperparameters
best_svr = SVR(kernel=best_params['kernel'], C=best_params['C'], gamma=best_params['gamma'], degree=best_params.get('degree', 3))
best_svr.fit(X_train_scaled, y_train)

# Prediction and evaluation
y_train_pred = best_svr.predict(X_train_scaled)
y_test_pred = best_svr.predict(X_test_scaled)

train_r2 = r2_score(y_train, y_train_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_mape = np.mean(np.abs((y_train-y_train_pred)/y_train)) * 100
test_r2 = r2_score(y_test, y_test_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_mape = np.mean(np.abs((y_test-y_test_pred)/y_test)) * 100

print(f"Training R²: {train_r2}")
print(f"Training RMSE: {train_rmse}")
print(f"Training MAPE: {train_mape}")
print("\n")
print(f"Testing R²: {test_r2}")
print(f"Testing RMSE: {test_rmse}")
print(f"Testing MAPE: {test_mape}")

# Plot the scatter plot
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_train, y_train_pred, color='blue', label='Training Data')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=2, label='Perfect Fit')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Training Data')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_test_pred, color='green', label='Testing Data')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Perfect Fit')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Testing Data')
plt.legend()
plt.tight_layout()
plt.show()

# Export data
train_data = pd.DataFrame({
    'Actual': y_train,
    'Predicted': y_train_pred
})
train_data.to_csv('data/CamFreqSyn/train_data_SVR_R2.csv', index=False)

test_data = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_test_pred
})
test_data.to_csv('data/CamFreqSyn/test_data_SVR_R2.csv', index=False)