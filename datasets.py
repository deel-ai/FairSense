from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import sklearn.datasets as ds
import torch


class Datasets:
    
    """
    Boston Housing dataset features :
    -------------------------
    CRIM - per capita crime rate by town
    ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
    INDUS - proportion of non-retail business acres per town.
    CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
    NOX - nitric oxides concentration (parts per 10 million)
    RM - average number of rooms per dwelling
    AGE - proportion of owner-occupied units built prior to 1940
    DIS - weighted distances to five Boston employment centres
    RAD - index of accessibility to radial highways
    TAX - full-value property-tax rate per $10,000
    PTRATIO - pupil-teacher ratio by town
    B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
    LSTAT - % lower status of the population
    MEDV - Median value of owner-occupied homes in $1000's
    """
    
    @staticmethod
    def load_Boston_Housing():
        # Load the Boston dataset from the scikit-learn module
        data = ds.load_boston()
        # Split data (X) from target (Y)
        x, y = data.data, data.target
        # Split the dataset into training/test sets (for Machine Learning use)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
        print('Boston Housing dataset loaded, shapes :', x_train.shape, y_train.shape, x_test.shape, y_test.shape)
        # Return the extracted data/targets as Pandas dataframes
        return (x_train, y_train), (x_test, y_test)
    
    @staticmethod
    def list_Boston_Housing_features():
        # Load the Boston dataset from the scikit-learn module
        data = ds.load_boston()
        # Return the list of the dataset features
        return data.feature_names



    @staticmethod
    def load_Breaking_Distance_Estimation():
        # Define both BDE file paths
        INPUT_PATH = "./data_BDE/"
        input_file = "%sDATA_GENERIC.csv" % INPUT_PATH
        input_file_test = "%sDATA_GENERIC_VALIDATION.csv" % INPUT_PATH
        # Load both files from the BDE dataset into Pandas dataframes
        df = pd.read_csv(input_file, delimiter=";")
        df_test = pd.read_csv(input_file_test, delimiter=";")
        # Concatenate both dataframes into a single one
        df = pd.concat([df, df_test])
        # One-hot encode the "state" input column
        df_dummies = pd.get_dummies(df["Input_08"])
        df_global = pd.concat([df.drop(['OUTPUT', 'Input_08'], axis = 1), df_dummies], axis=1, join_axes=[df.index])
        # Use standard scaler to scale real-valued input columns
        fit_colnames = ['Input_01', 'Input_02', 'Input_03', 'Input_04', 'Input_05', 'Input_06']
        scaler = StandardScaler()
        scaler.fit(df_global[fit_colnames])
        # Split the dataset into training and validation sets
        df_train, df_test, y_train, y_test = train_test_split(df_global, df['OUTPUT'], test_size=0.4, random_state=42)
        df_test, df_val, y_test, y_val = train_test_split(df_test, y_test, test_size=0.5, random_state=42)
        # Create a PyTorch dataloader from the training dataframe
        bs = 10000
        class DataSetRops(Dataset):
            def __init__(self, X, y):
                if not torch.is_tensor(X):
                    self.X = torch.tensor(X.values, dtype=torch.float64)
                if not torch.is_tensor(y):
                    self.y = torch.tensor(y.values, dtype=torch.float64)
            def __len__(self):
                return len(self.X)
            def __getitem__(self, idx):
                return self.X[idx], self.y[idx]
        data_torch = DataSetRops(df_train, y_train)
        dataloader = DataLoader(data_torch, batch_size=bs,shuffle=True)
        return dataloader, scaler, fit_colnames, df_train, y_train, df_val, y_val, df_test, y_test









