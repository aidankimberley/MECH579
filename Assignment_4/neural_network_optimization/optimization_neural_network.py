import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import time
from train_objfunc import get_trained_objfunc, Net
from constraints import constraint_1, constraint_2
from objective_function import obj_func

train_objfunc = True

# Create directories if they don't exist
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)


# Modified implementation of: Chen J, Liu Y. 2023 Neural optimization machine: a neural network approach for optimization and its application in additive manufacturing with physics-guided learning. Phil. Trans. R. Soc. A.

# Define the neural network for optimization problem
class Net_optimizationproblem(torch.nn.Module):
    def __init__(self,f_neural_net, constraint):
        super(Net_optimizationproblem, self).__init__()
        self.f_neural_net = f_neural_net; # NN of Objective function
        for param in self.f_neural_net.parameters(): # Keep the NN objective function weights and biases constant
            param.requires_grad = False; # Does not compute gradients wrt weights and biases of the objective function.
        self.constraint = constraint; 
        # x_opt = Wx + b where we are optimizing W and b
        self.linear = torch.nn.Linear(2,2); # Define NN of the Optimization Problem
        torch.nn.init.eye_(self.linear.weight);
        torch.nn.init.zeros_(self.linear.bias);


    def forward(self, x):
        x = self.linear(x);
        f_val = self.f_neural_net(x).squeeze();
        #f_val = obj_func(x); # Can also use the analytical objective function.
        constraints_val = self.constraint(x); 
        #Using a penalty method to enforce constraints (suboptimal algorithm)
        #TRYING NO CONSTRAINT PENALTY FOR NOW
        output = f_val #+ 10*(torch.relu(constraints_val) + torch.relu(-constraints_val));
        return output;

#Custom loss function for optimization.
class CustomLoss(torch.nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, predictions):
        return predictions;


def run_optimizer(objfunc_neural_net, constraint):
    optimization_problem = Net_optimizationproblem(objfunc_neural_net, constraint); 
    loss_function = CustomLoss() 
    optimizer = torch.optim.NAdam(optimization_problem.parameters(), lr=0.005); #NAdam is a variant of Adam that is designed to handle non-convex objectives.

    x_initial = torch.Tensor(1,2); #just a regular array essentially -> intialize size -> [x1, x2]
    x_initial[:,0] = 150; #velocity
    x_initial[:,1] = 14000; #altitude
    
    # Track objective function values and timing
    objective_history = []
    epoch_history = []
    time_per_iteration = []
    total_start_time = time.time()
    
    # Use fewer epochs for faster testing - can be increased for better convergence
    n_epochs = 10000  # Reduced from 40000 for faster execution
    for epoch in range(n_epochs):
        iter_start_time = time.time()
        optimizer.zero_grad() # Gradient of wrt weights and biases are set to zero.
        outputs = optimization_problem(x_initial)
        loss = loss_function(outputs)
        loss.backward() # Backward propagation with automatic differentiation. Compute d (Obj_fun) / d (weights and biases)
        optimizer.step() # Updates weights and biases with specified learning rate
        
        # Get current optimal point and evaluate objective function
        x_current = optimization_problem.linear(x_initial)
        # Evaluate objective function (obj_func expects tensor input)
        obj_result = obj_func(x_current.detach())
        # Handle both tensor and numpy return values
        if isinstance(obj_result, torch.Tensor):
            obj_val = -obj_result.item()  # Negative because we minimize -range
        else:
            obj_val = -float(obj_result)  # Negative because we minimize -range
        objective_history.append(obj_val)
        epoch_history.append(epoch)
        
        iter_time = time.time() - iter_start_time
        time_per_iteration.append(iter_time)
        
        if epoch % 1000 == 0:
            print("Epoch {}: Loss = {}, Objective = {:.2f} km, Time = {:.4f} s".format(
                epoch, loss.detach().numpy(), obj_val, iter_time))

    total_time = time.time() - total_start_time
    x_optimal = optimization_problem.linear(x_initial);
    print("Optimal point = ", x_optimal);
    print("Total optimization time: {:.2f} s".format(total_time))
    print("Average time per iteration: {:.4f} s".format(np.mean(time_per_iteration)))
    
    return x_optimal, objective_history, epoch_history, time_per_iteration;



print("===================================================================");
print("                 Training NN objective function                       ");
print("===================================================================");

if train_objfunc:
    objfunc_neuralnet, loss_history, epoch_history = get_trained_objfunc(epochs=25000, n_vel_samples_1d=70, n_altitude_samples_1d=70);
    #save objfunc neural network
    torch.save(objfunc_neuralnet.state_dict(), "models/objfunc_neuralnet.pth");
    plt.plot(epoch_history, loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss History")
    plt.savefig("plots/loss_history_objfunc.png")
else:
    objfunc_neuralnet = Net()
    objfunc_neuralnet.load_state_dict(torch.load("models/objfunc_neuralnet_1.pth"));
    objfunc_neuralnet.eval();

# #test values going through the NN and compare to analytical objective function
print("Testing values going through the NN and comparing to analytical objective function")
print(objfunc_neuralnet(torch.Tensor([[100, 14000]])))
print(obj_func(torch.Tensor([[100, 14000]])),"\n")
print("Difference: ", abs(objfunc_neuralnet(torch.Tensor([[150, 14000]])) - obj_func(torch.Tensor([[150, 14000]]))))
print(objfunc_neuralnet(torch.Tensor([[160, 14500]])))
print(obj_func(torch.Tensor([[160, 14500]])),"\n")
print("Difference: ", abs(objfunc_neuralnet(torch.Tensor([[160, 14500]])) - obj_func(torch.Tensor([[160, 14500]]))))
print(objfunc_neuralnet(torch.Tensor([[170, 15000]])))
print(obj_func(torch.Tensor([[170, 15000]])),"\n")
print("Difference: ", abs(objfunc_neuralnet(torch.Tensor([[170, 15000]])) - obj_func(torch.Tensor([[170, 15000]]))))
print(objfunc_neuralnet(torch.Tensor([[180, 15500]])))
print(obj_func(torch.Tensor([[180, 15500]])),"\n")
print("Difference: ", abs(objfunc_neuralnet(torch.Tensor([[180, 15500]])) - obj_func(torch.Tensor([[180, 15500]]))))
print(objfunc_neuralnet(torch.Tensor([[190, 16000]])))
print(obj_func(torch.Tensor([[190, 16000]])),"\n")
print(objfunc_neuralnet(torch.Tensor([[200, 16500]])))
print(obj_func(torch.Tensor([[200, 16500]])),"\n")
print(objfunc_neuralnet(torch.Tensor([[210, 17000]])))
print(obj_func(torch.Tensor([[210, 17000]])),"\n")
print(objfunc_neuralnet(torch.Tensor([[220, 17500]])))
print(obj_func(torch.Tensor([[220, 17500]])),"\n")
print("===================================================================");
print('\n\n');
print("===================================================================");
print("                 Optimization using Neural Network                 ");
print("===================================================================");
print('\n');
print("===================================================================");
print("Running optimization for brequet range equation, altitude constraint");
print("===================================================================");
nn_optimal, nn_objective_history, nn_epoch_history, nn_time_per_iter = run_optimizer(objfunc_neuralnet, constraint_1);

# Plot neural network optimization convergence
plt.figure(figsize=(10, 6))
plt.plot(nn_epoch_history, nn_objective_history, 'b-', linewidth=2, label='Neural Network Optimization')
plt.xlabel('Epoch', fontsize=12, fontweight='bold')
plt.ylabel('Objective Function (Range in km)', fontsize=12, fontweight='bold')
plt.title('Neural Network Optimization Convergence', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('../plots/nn_optimization_convergence.png', dpi=300, bbox_inches='tight')
print("Saved neural network convergence plot to plots/nn_optimization_convergence.png")
plt.close()

# Save timing data for comparison
np.savez('../plots/nn_timing_data.npz', 
         epochs=nn_epoch_history, 
         objective=nn_objective_history, 
         time_per_iter=nn_time_per_iter)
print("===================================================================");
print('\n\n');
print("===================================================================");
print("Running optimization for brequet range equation, velocity constraint");
print("===================================================================");
#run_optimizer(objfunc_neuralnet, constraint_2);
print("===================================================================");

