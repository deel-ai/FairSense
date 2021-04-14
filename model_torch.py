# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F



class Net(nn.Module):

    def __init__(self, nb_layer, nb_in, nb_out, nb_hidden):
        super(Net, self).__init__()
        # Check that nb_layer and nb_hidden are coherent
        if len(nb_hidden) != nb_layer:
            print("ERROR : Number of Layer not compatible")
            return
        self.scaler = None
        self.layer = {}
        self.layer["Linear1"] = nn.Linear(nb_in, nb_hidden[0], bias=True)
        self.layer["Linear1"].to(torch.double)
        self.add_module("Linear1", self.layer["Linear1"])

        for i in range(1,nb_layer):
            lname = "Linear%d" % (i+1)
            self.layer[lname] = nn.Linear(nb_hidden[i-1],nb_hidden[i], bias=True)
            self.layer[lname].to(torch.double)
            self.add_module("Linear"+str(i+1), self.layer[lname])

        self.out = nn.Linear(nb_hidden[nb_layer-1], nb_out, bias=True)
        self.out.to(torch.double)

    def init_weight(self,val):
        for l in self.layer:
            s = self.layer[l].weight.data.size()
            w = (torch.zeros(s)+val).to(torch.double)
            self.layer[l].weight.data = w
        s = self.out.weight.data.size()
        w = (torch.zeros(s)+val).to(torch.double)
        self.out.weight.data = w
            #self.layer[l].weight.data.abs_()

    def get_nb_negative(self):
        nb = 0
        for l in self.layer:
            nb += (self.layer[l].weight.data<0).sum()
        return nb

    def abs_weight(self):
        for l in self.layer:
            self.layer[l].weight.data.abs_()
        self.out.weight.data.abs_()

    def clamp_weight(self):
        for l in self.layer:
            self.layer[l].weight.data.clamp_(min=0.)
        self.out.weight.data.clamp_(min=0.)

    def set_scaler(self,sc, fc):
        self.scaler = sc
        self.fit_colnames = fc

    def heavy_side(self,x):
        s = x.size()
        zeros = torch.zeros(s, dtype=torch.double)
        ones = torch.ones(s, dtype=torch.double)
        res = (x > zeros).to(torch.double) * ones
        # print(res)
        return res




    def heavy_alpha(self, x, alpha):
        y = 0.5+0.5*torch.sign(x)*(1.0 - (1.0+torch.abs(x))**(-alpha))
        return y

    def forward(self, x, activation='Relu', alpha=1.0):
        if self.scaler is not None:
            nb_scale =  self.scaler.scale_.shape[0]
            #print(x[:,0:nb_scale])
            x[:,0:nb_scale] = torch.tensor(self.scaler.transform(x[:,0:nb_scale]),dtype=torch.float64)
            #print(x)
        ne=[]
        if activation == 'Relu':
            relu_h = F.relu(self.layer["Linear1"](x))
            # relu_h = self.heavy_side(self.layer["Linear1"](x))
            # print(relu_h.size())
            #print(relu_h)
            nb_layer = len(self.layer)
            for i in range(1, nb_layer):
                lname = "Linear%d" % (i+1)
                relu_h = F.relu(self.layer[lname](relu_h))
                # relu_h = self.heavy_side(self.layer[lname](relu_h))
            y_pred = self.out(relu_h)
        if activation == 'Tanh':
            relu_h = torch.tanh(self.layer["Linear1"](x))
            nb_layer = len(self.layer)
            for i in range(1, nb_layer):
                lname = "Linear%d" % (i+1)
                relu_h = torch.tanh(self.layer[lname](relu_h))
            y_pred = self.out(relu_h)

        if activation == 'Sigmoid':
            relu_h = torch.sigmoid(self.layer["Linear1"](x))
            nb_layer = len(self.layer)
            for i in range(1, nb_layer):
                lname = "Linear%d" % (i+1)
                relu_h = torch.sigmoid(self.layer[lname](relu_h))
            y_pred = self.out(relu_h)

        if activation == 'Heavy':
            ne =[]
            ne.append(self.layer["Linear1"](x))
            relu_h = self.heavy_alpha(self.layer["Linear1"](x), alpha)
            nb_layer = len(self.layer)
            for i in range(1, nb_layer):
                lname = "Linear%d" % (i+1)
                ne.append(self.layer[lname](relu_h))
                relu_h = self.heavy_alpha(self.layer[lname](relu_h), alpha)
#                ne.append(relu_h)
            y_pred = self.out(relu_h)
#        y_pred = F.relu(y_pred)
#        print(y_pred)
        return y_pred, ne

    def print_state_model(self, optimizer):
        print("Model's state_dict")
        for param_tensor in self.state_dict():
            print(param_tensor, "\t", self.state_dict()[param_tensor].size())

        print("\nOptimizer's state_dict:")
        for var_name in optimizer.state_dict():
            print(var_name, "\t", optimizer.state_dict()[var_name])

    def save_model_state_dict(self, path):
        torch.save(self.state_dict(), path)

    def load_model_state_dict(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()

    def print_parameters(self):
        print("##### Parameters of the model")
        nb_layer = len(self.layer)
        for i in range(0, nb_layer):
            lname = "Linear%d" % (i+1)
            print("\n############# Layer %s ###############"% lname)
            print("Weights")
            print(self.layer[lname].weight.data.numpy())
            print("Bias")
            print(self.layer[lname].bias.data.numpy())

            print("##############################\n")

        print("\n############# Out Layer ###############")
        print("Weights")
        print(self.out.weight.data.numpy())
        print("Bias")
        print(self.out.bias.data.numpy())
        print("##############################\n")






def main():
    # TESTING THE NETWORK

    # Set the seed
    torch.manual_seed(533)


    # Parameters of the NN architecture
    nb_in, nb_out= 8, 1
    nb_layer = 4
    nb_hidden = [20, 30, 40, 10]
    bs = 64


    # Create random tensor for inputs and outputs
    x = torch.randn(bs, nb_in)
    y = torch.randn(bs, nb_out)

    # Create the model
    model = Net(nb_layer, nb_in, nb_out, nb_hidden)
    print("Model ARCHITECTURE :")
    print(model)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-4)
    # Learning Loop
    print("\nSTART TRAINING")
    for t in range(5000):
        # Forward Pass
        y_pred = model(x)

        # Compute and print loss
        loss = criterion(y_pred, y)
        if t % 100 == 99:
            print("STEP", t, "\n\tloss:", loss.item())
            # print("\tGradient Layer1 Gradient")
            # print(model.linear1.weight.grad)

        # Backward with gradient descent
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == '__main__':
    main()
