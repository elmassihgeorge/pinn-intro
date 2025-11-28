"""
Physics-Informed Neural Network (PINN)
for solving the specific ODE dy/dt + y = 0
with initial condition y(0) = 1.
The analytical solution is y(t) = e^(-t)
"""

import torch
import torch.nn as nn

class PINN(nn.Module):
    """
    Physics-Informed Neural Network for solving dy/dt + y = 0.
    
    The network learns the solution by minimizing a loss function that combines:
    1. Physics loss: residual of the differential equation dy/dt + y = 0
    2. Boundary loss: error in satisfying initial condition y(0) = 1
    """
    
    def __init__(self):
        """Initialize the neural network with 2 hidden layers of 20 neurons each."""
        super(PINN, self).__init__()
        self.layer_1 = nn.Linear(1, 20)
        self.layer_2 = nn.Linear(20, 20)
        self.output_layer = nn.Linear(20, 1)

    def forward(self, t):
        """
        Forward pass through the network.
        
        Args:
            t: Input tensor representing time values
            
        Returns:
            Predicted value of y(t)
        """
        y = torch.tanh(self.layer_1(t))
        y = torch.tanh(self.layer_2(y))
        y = self.output_layer(y)
        return y

    def physics_loss(self, t_physics):
        """
        Compute the physics-informed loss for the differential equation dy/dt + y = 0
        
        Args:
            t_physics: Collocation points where the PDE is enforced
            
        Returns:
            Mean squared error of the differential equation residual
        """
        y = self(t_physics)
        dy_dt = torch.autograd.grad(
            y, t_physics,
            grad_outputs=torch.ones_like(y),
            create_graph=True,
        )[0]
        L_E = dy_dt + y
        MSE = torch.mean(L_E**2)
        return MSE

    def boundary_loss(self, t_0=torch.tensor([0.0]).float().unsqueeze(1)):
        """
        Compute the loss for the initial condition y(0) = 1.
        
        Args:
            t_0: Time point for initial condition
            
        Returns:
            Mean squared error of the initial condition residual
        """
        y_0 = self(t_0)
        L_IC = y_0 - 1
        MSE = torch.mean(L_IC**2)
        return MSE

    def loss(self, t_physics, t_0=torch.tensor([0.0]).float().unsqueeze(1)):
        """
        Total loss combining physics loss and boundary loss.
        
        Args:
            t_physics: Collocation points for physics loss
            t_0: Time point for initial condition
            
        Returns:
            Combined loss value
        """
        return self.physics_loss(t_physics) + self.boundary_loss(t_0)