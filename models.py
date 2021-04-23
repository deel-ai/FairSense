from sklearn.ensemble import RandomForestRegressor
import torch
import model_torch


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


def create_BDE_model(scaler, fit_colnames):
    # Manually set the random seed for reproducibility
    torch.manual_seed(533)
    # Set the parameters of the network to be created
    nb_in, nb_out = 13, 1
    nb_layer = 4
    nb_hidden = [64, 64, 64, 64]
    # Create the model
    model = model_torch.Net(nb_layer, nb_in, nb_out, nb_hidden)
    model.set_scaler(scaler, fit_colnames)
    return model

def evaluate_BDE_model(model, Xs, Ys):
    # Extract different dataframes and targets
    df_train, df_val, df_test = Xs
    y_train, y_val, y_test = Ys
    # Format data into tensors
    tensor_X_train = torch.tensor(df_train.values, dtype=torch.float64, requires_grad=False)
    tensor_y_train = torch.tensor(y_train.values, dtype=torch.float64, requires_grad=False).unsqueeze(-1)
    tensor_X_val = torch.tensor(df_val.values, dtype=torch.float64, requires_grad=False)
    tensor_y_val = torch.tensor(y_val.values, dtype=torch.float64, requires_grad=False).unsqueeze(-1)
    tensor_X_test = torch.tensor(df_test.values, dtype=torch.float64, requires_grad=False)
    tensor_y_test = torch.tensor(y_test.values, dtype=torch.float64, requires_grad=False).unsqueeze(-1)
    # Get the corresponding loss function
    criterion = torch.nn.MSELoss()
    with torch.no_grad():
        # Predictions on train dataset
        train_predictions = model(tensor_X_train)[0]
        train_loss = criterion(train_predictions, tensor_y_train)
        print("[TRAIN] loss = {:.03f}".format(train_loss))
        # Predictions on validation dataset
        val_predictions = model(tensor_X_val)[0]
        val_loss = criterion(val_predictions, tensor_y_val)
        print("[VALID] loss = {:.03f}".format(val_loss))
        # Predictions on test dataset
        test_predictions = model(tensor_X_test)[0]
        test_loss = criterion(test_predictions, tensor_y_test)
        print("[TEST ] loss = {:.03f}".format(test_loss))

def fit_BDE_model(model, dataloader, df_train, y_train, df_val, y_val, epochs=5):
    print('*** Model started training...')
    # Create the loss object and the optimizer for the model training
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    def rmse_loss(ypred, y):
        return torch.mean(torch.sqrt((ypred - y)**2) / y)

    for epoch in range(epochs):
        # Switch the model to training mode (weights update for model optimization)
        model.train()
        for local_batch, local_labels in enumerate(dataloader):
            # Iterate over data contained in the data loader
            X, y = local_labels[0], local_labels[1].unsqueeze(-1)
            # Predict with the model and evaluate losses (MSE & RMSE)
            y_pred = model(X, 'Relu')[0]
            loss = criterion(y_pred, y)
            rmse_val_loss = rmse_loss(y_pred, y)
            optimizer.zero_grad()
            #loss.backward()
            rmse_val_loss.backward()
            optimizer.step()
        # Switch the model to evaluation mode (no weights update)
        model.eval()
        with torch.no_grad():
            # Build training/validation Torch tensors for data/labels
            Xtrain = torch.tensor(df_train.values, dtype=torch.float64,requires_grad=False)
            Ytrain = torch.tensor(y_train.values, dtype=torch.float64,requires_grad=False).unsqueeze(-1)
            Xvalidation = torch.tensor(df_val.values, dtype=torch.float64,requires_grad=False)
            Yvalidation = torch.tensor(y_val.values, dtype=torch.float64,requires_grad=False).unsqueeze(-1)
            # Compute training losses (MSE & RMSE)
            ytrain_pred = model(Xtrain, 'Relu')[0]
            loss_train = criterion(ytrain_pred, Ytrain)
            rmse_train = rmse_loss(ytrain_pred, Ytrain)
            # Compute validation losses (MSE & RMSE)
            yvalidation_pred = model(Xvalidation,'Relu')[0]
            loss_val = criterion(yvalidation_pred, Yvalidation)
            rmse_val = rmse_loss(yvalidation_pred, Yvalidation)
            # Evaluation of the model at each epoch
            print('Epoch {:2} -> MSE (train/val) : {:.03f}/{:.03f} ; RMSE (train/val) : {:.03f}/{:.03f}'.format(
                epoch, loss_train, loss_val, rmse_train, rmse_val))
    print('*** Model finished training !')

def save_model(model, path):
    # Save PyTorch model on disk
    model.save_model_state_dict(path)

def load_model(model, path):
    # Load PyTorch model on disk
    model.load_model_state_dict(path)
