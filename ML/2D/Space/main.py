import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

from domain import sample_points
from energy import energy_loss, boundary_loss
from plot_machine import plotter

"""
This script tries to learn a solution to the 2D time independnet schrödinger equation.
"""

# Simple Logger
logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("train.log"),  
    ]
)

# Not used here --> In the future for time dependent equation 
class ComplexNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.Tanh(self.fc1(x))
        x = torch.Tanh(self.fc2(x))
        x = torch.Tanh(self.fc2(x))
        x = torch.Tanh(self.fc2(x))
        re, im = self.out(x).unbind(dim=-1)
        return torch.complex(re, im)

# Simple network, with smooth activation function
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim = 2, hidden_dim = 50, output_dim = 1):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)
    
def main():
    """
    Simple training loop with Adam optimizer.
    """
    model = NeuralNetwork()
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)
    n_epochs = 12000
    lambda_boundary = 20
    loss = []
    best_loss = float('inf')
    best_model_path = "best_model.pth"

    # Visualize the sampled points in the domain
    interior, boundary = sample_points()
    plt.scatter(interior[:,0], interior[:,1],  s=1, label="Interior")
    plt.scatter(boundary[:, 0], boundary[:, 1], s=1, label="Boundary")
    plt.xlabel("x"); plt.ylabel("y"); plt.legend()
    plt.title("Sampled Points")
    plt.show()

    # Training loop
    for epoch in tqdm(range(n_epochs)):
        optimizer.zero_grad()

        # Sample points
        interior, boundary = sample_points()

        # Compute loss in interior and boundary
        loss_interior = energy_loss(model, interior)
        loss_boundary = boundary_loss(model, boundary)
        total_loss = loss_interior + lambda_boundary * loss_boundary

        # Backpropagation and weight update
        total_loss.backward()
        optimizer.step()

        loss.append(total_loss.item())

        #Save the model with the best loss
        if total_loss < best_loss:
            print(f"Epoch {epoch}, Total Loss: {total_loss.item():.6f}, "
                         f"Interior Loss: {loss_interior.item():.6f}, "
                         f"Boundary Loss: {loss_boundary.item():.6f}, "
                         f"Best Loss: {best_loss:.7f}")
            
            logging.info(f"Epoch {epoch}: Best Loss: {best_loss:.4f}, saving model")
            best_loss = total_loss.item()
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, best_model_path)
        
        # Log info
        logging.info(f"Epoch {epoch}, Total Loss: {total_loss.item():.6f}, "
                         f"Interior Loss: {loss_interior.item():.6f}, "
                         f"Boundary Loss: {loss_boundary.item():.6f}"
                         f"Best Loss: {best_loss:.7f}")


    # Save loss for plotting
    np.save("loss.npy", np.array(loss))
    plotter()
    logging.info("DONE")    
if __name__ == '__main__':
    main()
