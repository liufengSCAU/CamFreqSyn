import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ctgan import CTGAN
import pickle

np.random.seed(42)
real_data = pd.read_csv('data/NaturalFrequency_real_40.csv')
# real_data = pd.read_csv('data_CamFreqSyn/MTD_40_and_real_40.csv')

discrete_columns = [
    'main_number',
    'secondary_number'
]

## 40 generates 360
ctgan = CTGAN(
                epochs=70000,
                embedding_dim=128,
                generator_dim=(512, 512),
                discriminator_dim=(512, 512),
                generator_lr=2e-5,
                generator_decay=1e-7,
                discriminator_lr=2e-4,
                discriminator_decay=1e-5,
                batch_size=70,
                verbose=True,
                cuda=True
)

ctgan.fit(real_data, discrete_columns)

# Save the model
with open('models/optuna_ctgan_model_40_360_trial20.pkl', 'wb') as file:
    pickle.dump(ctgan, file)

# Conditional Constraint Module
synthetic_data = ctgan.sample(360)
synthetic_data = synthetic_data[(synthetic_data['A'] > 0) & (synthetic_data['B'] > 0)]
deleted_count = 360 - len(synthetic_data)
while deleted_count > 0:
    additional_data = ctgan.sample(deleted_count)
    additional_data = additional_data[(additional_data['A'] > 0) & (additional_data['B'] > 0)]
    synthetic_data = pd.concat([synthetic_data, additional_data.sample(min(deleted_count, len(additional_data)))], axis=0)
    deleted_count = 360 - len(synthetic_data)
if len(synthetic_data) > 360:
    synthetic_data = synthetic_data.sample(360)

synthetic_data.to_csv('data_CTGAN/kde_trial20/NaturalFrequency_synthetic_360.csv', index=False)

# ## 80 generates 320
# ctgan = CTGAN(
#                 epochs=60000,
#                 embedding_dim=64,
#                 generator_dim=(1024, 1024),
#                 discriminator_dim=(1024, 1024),
#                 generator_lr=2e-4,
#                 generator_decay=1e-6,
#                 discriminator_lr=2e-4,
#                 discriminator_decay=1e-7,
#                 batch_size=170,
#                 verbose=True,
#                 cuda=True
# )

# # Conditional Constraint Module
# synthetic_data = ctgan.sample(320)
# synthetic_data = synthetic_data[(synthetic_data['A'] > 0) & (synthetic_data['B'] > 0)]
# deleted_count = 320 - len(synthetic_data)
# while deleted_count > 0:
#     additional_data = ctgan.sample(deleted_count)
#     additional_data = additional_data[(additional_data['A'] > 0) & (additional_data['B'] > 0)]
#     synthetic_data = pd.concat([synthetic_data, additional_data.sample(min(deleted_count, len(additional_data)))], axis=0)
#     deleted_count = 320 - len(synthetic_data)
# if len(synthetic_data) > 320:
#     synthetic_data = synthetic_data.sample(320)
#
# synthetic_data.to_csv('data_CamFreqSyn/NaturalFrequency_synthetic_320_trial98.csv', index=False)

# Drawing
fig, axs = plt.subplots(nrows=len(real_data.columns), figsize=(5, 10))
kde_data = {}

for i, column in enumerate(real_data.columns):
    ax = sns.kdeplot(real_data[column], ax=axs[i], label='Real Data')
    sns.kdeplot(synthetic_data[column], ax=axs[i], label='Synthetic Data')

    real_kde_line = ax.lines[0]
    synthetic_kde_line = ax.lines[1]
    x_real = real_kde_line.get_xdata()
    y_real = real_kde_line.get_ydata()
    x_synthetic = synthetic_kde_line.get_xdata()
    y_synthetic = synthetic_kde_line.get_ydata()

    kde_data[column] = {
        'x_real': x_real,
        'y_real': y_real,
        'x_synthetic': x_synthetic,
        'y_synthetic': y_synthetic
    }

    axs[i].set_xlabel(column)
plt.show()

# Save kde data as csv file, containing only x and y data, without index
for column_name in kde_data.keys():
    real_xy = pd.DataFrame(
        {
            "x_real": kde_data[column_name]["x_real"],
            "y_real": kde_data[column_name]["y_real"],
        }
    )
    synthetic_xy = pd.DataFrame(
        {
            "x_synthetic": kde_data[column_name]["x_synthetic"],
            "y_synthetic": kde_data[column_name]["y_synthetic"],
        }
    )
    real_xy.to_csv(f"data_CTGAN/kde_trial20/{column_name}_real_kde_data_optuna.csv", index=False)
    synthetic_xy.to_csv(f"data_CTGAN/kde_trial20/{column_name}_synthetic_kde_data_optuna.csv", index=False)

real_data_labeled = real_data.copy()
real_data_labeled['Data_Type'] = 'Real Data'

synthetic_data_labeled = synthetic_data.copy()
synthetic_data_labeled['Data_Type'] = 'Synthetic Data'

combined_data = pd.concat([real_data_labeled, synthetic_data_labeled], ignore_index=True)
combined_data.to_csv('data_CTGAN/kde_trial20/NaturalFrequency_all_400_optuna.csv', index=False)

fig, axs = plt.subplots(ncols=2, figsize=(20, 10))
sns.heatmap(real_data.corr(method='spearman'), cmap='coolwarm', annot=True, ax=axs[0])
axs[0].set_title('Real Data Correlation Matrix')
sns.heatmap(synthetic_data.corr(method='spearman'), cmap='coolwarm', annot=True, ax=axs[1])
axs[1].set_title('Synthetic Data Correlation Matrix')
plt.show()
