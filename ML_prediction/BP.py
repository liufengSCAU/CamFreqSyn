import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neural_network import MLPRegressor

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

# Create BP neural network model
mlp = MLPRegressor(
    # hidden_layer_sizes=(32, 256, 32),
    # hidden_layer_sizes=(128, 256, 128),
    hidden_layer_sizes=(128, 128, 128, 128),
    activation='relu',
    # solver='adam',
    solver='sgd',
    alpha=0.01,  # L2 regularization term
    learning_rate_init=0.01,
    max_iter=1000,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.2,
    batch_size=64,
)

# Training and prediction
mlp.fit(X_train_scaled, y_train)
y_train_pred = mlp.predict(X_train_scaled)
y_test_pred = mlp.predict(X_test_scaled)

# Evaluate
train_r2 = r2_score(y_train, y_train_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_mape = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
test_r2 = r2_score(y_test, y_test_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100

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

# Plot the loss curve
plt.figure(figsize=(10, 5))
plt.plot(mlp.loss_curve_, label='Training Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Export data
train_data = pd.DataFrame({
    'Actual': y_train,
    'Predicted': y_train_pred
})
train_data.to_csv('data/CamFreqSyn/train_data_BP_R2.csv', index=False)

test_data = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_test_pred
})
test_data.to_csv('data/CamFreqSyn/test_data_BP_R2.csv', index=False)

# Print hyperparameters
print(f"Number of iterations: {mlp.n_iter_}")
