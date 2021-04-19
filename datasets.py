import sklearn.datasets as ds
from sklearn.model_selection import train_test_split


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
