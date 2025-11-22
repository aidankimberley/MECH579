import numpy as np
import torch
from objective_function import obj_func

# Neural network (NN) for objective function. 
# The NN has two inputs, three hidden layers and one output.
# torch.nn.Linear stores weights and biases, which are trained to minimize error with respect to the objective function.

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        n_inputs = 2;
        n_layer1 = 50;
        n_layer2 = 100;
        n_layer3 = 100;
        n_layer4 = 100;
        n_layer5 = 100;
        n_layer6 = 50;
        n_outputs = 1;
        # Specify layers of the neural network.
        self.hidden1 = torch.nn.Linear(n_inputs, n_layer1) # 2 inputs, n_layer1 neurons
        self.hidden2 = torch.nn.Linear(n_layer1, n_layer2) # n_layer1 inputs, n_layer2 neurons
        self.hidden3 = torch.nn.Linear(n_layer2, n_layer3) # n_layer2 inputs, n_layer3 neurons
        self.hidden4 = torch.nn.Linear(n_layer3, n_layer4) # n_layer3 inputs, n_layer4 neurons
        self.hidden5 = torch.nn.Linear(n_layer4, n_layer5) # n_layer4 inputs, n_layer5 neurons
        self.hidden6 = torch.nn.Linear(n_layer5, n_layer6) # n_layer5 inputs, n_layer6 neurons
        self.output_layer = torch.nn.Linear(n_layer6, n_outputs) # n_layer3 inputs, 1 output
        
        # Initialize weights and biases.
        # torch.nn.init.eye_(self.hidden1.weight);
        # torch.nn.init.ones_(self.hidden1.bias);
        # torch.nn.init.eye_(self.hidden2.weight);
        # torch.nn.init.ones_(self.hidden2.bias);
        # torch.nn.init.eye_(self.hidden3.weight);
        # torch.nn.init.ones_(self.hidden3.bias);
        # torch.nn.init.eye_(self.output_layer.weight);
        # torch.nn.init.ones_(self.output_layer.bias);


    def forward(self, x): # this is called implicitly when you call the object
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = torch.relu(self.hidden3(x))
        x = torch.relu(self.hidden4(x))
        x = torch.relu(self.hidden5(x))
        x = torch.relu(self.hidden6(x))
        x = self.output_layer(x) #keep inputting x through the network, keep updating until at output layer
        return x

def get_trained_objfunc(epochs=20000, n_vel_samples_1d=75, n_altitude_samples_1d=75): 
    loss_history = []
    epoch_history = []
    # Data to train the neural network.
    velocity_1d =  torch.linspace(50,300, n_vel_samples_1d); #x
    altitude_1d =  torch.linspace(1000,20000, n_altitude_samples_1d); #y
    x_samples,y_samples = np.meshgrid(velocity_1d.detach().numpy(), altitude_1d.detach().numpy());
    xy_tensor = torch.from_numpy(np.concatenate((x_samples.reshape(n_vel_samples_1d*n_altitude_samples_1d,1), y_samples.reshape(n_vel_samples_1d*n_altitude_samples_1d,1)),axis=1));
    print("Size of training data: ", xy_tensor.shape)
    f_exact = obj_func(xy_tensor).unsqueeze(1);

    f_neural_net = Net()
    loss_function = torch.nn.MSELoss() # Using squared L2 norm of the error.
    optimizer = torch.optim.Adam(f_neural_net.parameters(), lr=0.005)  # Using Adam optimizer.

    print("Training neural network of objective function");
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = f_neural_net(xy_tensor)
        loss = loss_function(outputs, f_exact)
        loss.backward() # Backpropagation and computing gradients w.r.t. weights and biases.
        optimizer.step() # Update weights and biases.
        loss_history.append(loss.detach().numpy())
        epoch_history.append(epoch)
        if epoch % 500 == 0:
            print("Epoch {}: Loss = {}".format(epoch, loss.detach().numpy()))

    print("Finished training neural network of objective function");

    return f_neural_net, loss_history, epoch_history;
