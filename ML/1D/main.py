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
This script aims to learn an approximative solution to the 1D - time independent schrödinger Equation:
H psi(x) = E psi(x)

E: Energy of the system
H: Is the Hamiltonian operator
    H = h²/2m d²/dx² + V(x) 
V: Some potential (we use the harmonic oscillator)


"""

#Logger to for debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("train.log"),
    ]
)
#Not used here --> For Time dependent schrödinger equation
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

# Simple neural network with smooth activation function
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim = 1, hidden_dim = 50, output_dim = 1):
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
    Classic training loop:
    
    """
    model = NeuralNetwork()
    optimizer = optim.Adam(model.parameters(), lr = 1e-4)
    n_epochs = 1000
    lambda_boundary = 300
    loss = []
    best_loss = float('inf')
    best_model_path = "best_model.pth"

    for epoch in tqdm(range(n_epochs)):
        optimizer.zero_grad()
        # Sample points and compute loss
        interior, boundary = sample_points()
        loss_interior = energy_loss(model, interior)
        loss_boundary = boundary_loss(model, boundary)
        total_loss = loss_interior + lambda_boundary * loss_boundary
        total_loss.backward()
        optimizer.step()
        
        # To plot loss later
        loss.append(total_loss.item())

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
        
        logging.info(f"Epoch {epoch}, Total Loss: {total_loss.item():.6f}, "
                         f"Interior Loss: {loss_interior.item():.6f}, "
                         f"Boundary Loss: {loss_boundary.item():.6f}, "
                         f"Best Loss: {best_loss:.7f}")


    #Save loss and plot it
    np.save("loss.npy", np.array(loss))
    plotter()
    logging.info("DONE")    
if __name__ == '__main__':
    main()
