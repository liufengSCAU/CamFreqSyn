import pandas as pd
import numpy as np
from ctgan import CTGAN
import optuna

real_data = pd.read_csv('data/NaturalFrequency_real_40.csv')
# real_data = pd.read_csv('data_CamFreqSyn/MTD_40_and_real_40.csv')

discrete_columns = [
    'main_number',
    'secondary_number'
]

def objective(trial):
    epochs = trial.suggest_categorical('epochs', [1000, 5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000])
    embedding_dim = trial.suggest_categorical('embedding_dim', [64, 128])
    generator_dim = trial.suggest_categorical('generator_dim', [256, 512, 1024])
    discriminator_dim = trial.suggest_categorical('discriminator_dim', [256, 512, 1024])
    generator_lr = trial.suggest_categorical('generator_lr', [2e-5, 2e-4, 2e-3])
    generator_decay = trial.suggest_categorical('generator_decay', [1e-7, 1e-6, 1e-5])
    discriminator_lr = trial.suggest_categorical('discriminator_lr', [2e-5, 2e-4, 2e-3])
    discriminator_decay = trial.suggest_categorical('discriminator_decay', [1e-7, 1e-6, 1e-5])
    batch_size = trial.suggest_categorical('batch_size', [10, 40, 70, 100])
    # batch_size = trial.suggest_categorical('batch_size', [20, 50, 80, 110, 140, 170, 200])

    # Initialize CTGAN
    ctgan = CTGAN(
        epochs=epochs,
        embedding_dim=embedding_dim,
        generator_dim=(generator_dim, generator_dim),
        discriminator_dim=(discriminator_dim, discriminator_dim),
        generator_lr=generator_lr,
        generator_decay=generator_decay,
        discriminator_lr=discriminator_lr,
        discriminator_decay=discriminator_decay,
        batch_size=batch_size,
        verbose=True,
        cuda=True
    )

    # Train CTGAN and generate samples
    ctgan.fit(real_data, discrete_columns)

    ## Conditional Constraint Module
    synthetic_data = ctgan.sample(360)
    synthetic_data = synthetic_data[(synthetic_data['A'] > 0) & (synthetic_data['B'] > 0)]
    deleted_count = 360 - len(synthetic_data)
    while deleted_count > 0:
        additional_data = ctgan.sample(deleted_count)
        additional_data = additional_data[(additional_data['A'] > 0) & (additional_data['B'] > 0)]
        synthetic_data = pd.concat([synthetic_data, additional_data.sample(min(deleted_count, len(additional_data)))],
                                   axis=0)
        deleted_count = 360 - len(synthetic_data)
    if len(synthetic_data) > 360:
        synthetic_data = synthetic_data.sample(360)

    # Compute the difference between two correlation coefficient matrices
    real_corr = real_data.corr(method='spearman').values
    synthetic_corr = synthetic_data.corr(method='spearman').values
    diff_corr = np.sum(np.abs(real_corr - synthetic_corr))
    print(diff_corr)
    print(trial.params)

    # Returns the difference as the value of the objective function
    return diff_corr

# Create an Optuna study object, Minimize the difference in correlation coefficient matrix
study = optuna.create_study(direction='minimize')

# Run the optimization and print the best parameters
study.optimize(objective, n_trials=100)
print(study.best_params)
