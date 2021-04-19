from sklearn.ensemble import RandomForestRegressor


def fit_random_forest(x_train, y_train, x_test, y_test):
    # Create a Random Forest model of 50 decision trees
    random_forest_model = RandomForestRegressor(50, max_depth=5, min_samples_leaf=5)
    # Fit the model onto the training dataset
    random_forest_model.fit(x_train, y_train)
    # Evaluate the model on both training/test datasets
    train_score = 100.0 * random_forest_model.score(x_train, y_train)
    test_score = 100.0 * random_forest_model.score(x_test, y_test)
    print('Random Forest performance (train/test) : {:.02f}% / {:.02f}%'.format(train_score, test_score))
    # Return the model for further usage
    return random_forest_model
