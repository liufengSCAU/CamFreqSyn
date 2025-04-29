import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils import shuffle

class kNNMTD():
    def __init__(self, n_obs=100, k=3, random_state=1):
        self.n_obs = n_obs
        self._gen_obs = self.n_obs * 10
        self.k = k
        np.random.seed(random_state)

    def diffusion(self, sample):
        if np.issubdtype(sample.dtype, np.number):
            new_sample = []
            n = len(sample)
            min_val = np.min(sample)
            max_val = np.max(sample)
            u_set = (min_val + max_val) / 2
            Nl = len([i for i in sample if i < u_set])
            Nu = len([i for i in sample if i > u_set])
            if Nl + Nu == 0:
                return np.random.uniform(min_val, max_val, size=self._gen_obs)
            skew_l = Nl / (Nl + Nu)
            skew_u = Nu / (Nl + Nu)
            var = np.var(sample, ddof=1)
            if var == 0:
                a = min_val / 5
                b = max_val * 5
                h = 0
                new_sample = np.random.uniform(a, b, size=self._gen_obs)
            else:
                h = var / n
                a = u_set - (skew_l * np.sqrt(-2 * (var / Nl) * np.log(10**(-20))))
                b = u_set + (skew_u * np.sqrt(-2 * (var / Nu) * np.log(10**(-20))))
                L = a if a <= min_val else min_val
                U = b if b >= max_val else max_val
                while len(new_sample) < self._gen_obs:
                    x = np.random.uniform(L, U)
                    MF = (x - L) / (U - L) if x <= u_set else (U - x) / (U - u_set)
                    rs = np.random.uniform(0, 1)
                    if MF > rs:
                        new_sample.append(x)
            return np.array(new_sample)
        else:
            raise ValueError("The 'diffusion' method is only designed for numerical data.")

    def getNeighbors(self, val_to_test, xtrain, ytrain):
        X_train = xtrain.reshape(-1, 1)
        y_train = ytrain
        knn = KNeighborsRegressor(n_neighbors=self.k)
        knn.fit(X_train, y_train)
        nn_indices = knn.kneighbors(X=np.array(val_to_test).reshape(1, -1), return_distance=False)[0]
        neighbor_df = xtrain[nn_indices]
        y_neighbor_df = ytrain[nn_indices]
        return nn_indices, neighbor_df, y_neighbor_df

    def fit(self, train, class_col=None):
        train = train.copy()
        X_train = train.drop(class_col, axis=1)
        y_train = train[class_col]
        columns = X_train.columns
        temp_surr_data = pd.DataFrame(columns=list(train.columns))
        surrogate_data = pd.DataFrame(columns=list(train.columns))
        temp = pd.DataFrame(columns=list(train.columns))

        for ix, val in train.iterrows():
            for col in columns:
                X_train = train[col].values.reshape(-1, 1)
                y_train = train[class_col].values
                knn = KNeighborsRegressor(n_neighbors=self.k)
                knn.fit(X_train, y_train)
                nn_indices = knn.kneighbors(X=np.array(val[col]).reshape(1, -1), return_distance=False)[0]
                neighbor_df = train[col].iloc[nn_indices]
                y_neighbor_df = train[class_col].iloc[nn_indices]
                if neighbor_df.dtype == np.int64 or neighbor_df.dtype == np.int32:
                    bin_val = np.unique(train[col])
                    centers = (bin_val[:-1] + bin_val[1:]) / 2
                    x = self.diffusion(neighbor_df)
                    ind = np.digitize(x, bins=centers, right=True)
                    x = np.array([bin_val[i] for i in ind])
                    y = self.diffusion(y_neighbor_df)
                    nn_indices, neighbor_val_array, _ = self.getNeighbors(val[col], x, y)
                    temp[col] = pd.Series(neighbor_val_array)
                else:
                    x = self.diffusion(neighbor_df)
                    y = self.diffusion(y_neighbor_df)
                    nn_indices, neighbor_val_array, y_neighbor_val_array = self.getNeighbors(val[col], x, y)
                    temp[col] = pd.Series(neighbor_val_array)
                    temp[class_col] = pd.Series(y_neighbor_val_array)
                temp_surr_data = pd.concat([temp_surr_data, temp]).dropna(axis=1, how='all')
                surrogate_data = pd.concat([surrogate_data, temp_surr_data]).dropna(axis=1, how='all')

        surrogate_data = pd.concat([surrogate_data, temp_surr_data])
        synth_data = self.sample(surrogate_data, train, class_col)
        return synth_data

    def sample(self, data, real, class_col):
        surrogate_data = data.copy()
        train = real.copy()
        synth_data = pd.DataFrame(columns=list(train.columns))
        for x in train.columns:
            surrogate_data[x] = surrogate_data[x].fillna(
                train[x].mean() if train[x].dtype == 'float64' else train[x].mode()[0])
            surrogate_data[x] = surrogate_data[x].astype(train[x].dtypes.name)
        surrogate_data.reset_index(inplace=True, drop=True)
        try:
            temp_data = surrogate_data.sample(np.abs(self.n_obs))
        except ValueError:
            temp_data = surrogate_data.sample(np.abs(self.n_obs), replace=True)
        synth_data = pd.concat([synth_data, temp_data], axis=0)
        for x in train.columns:
            synth_data[x] = synth_data[x].fillna(train[x].mean() if train[x].dtype == 'float64' else train[x].mode()[0])
            synth_data[x] = synth_data[x].astype(train[x].dtypes.name)
        synth_data.reset_index(inplace=True, drop=True)
        return synth_data