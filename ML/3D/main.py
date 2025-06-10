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
from l2_error import compute_l2_error
"""
Same routine as in 1D and 2D case.

"""

# Logger
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("train.log"),
    ]
)
def true_psi0_3d(xx, yy, zz):
    """
    Returns the 3d groundstate soltuion
    """
    return (1 / np.pi**(3/4)) * np.exp(-0.5 * (xx**2 + yy**2 + zz**2))

# Complex network for future; not used here
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

# Simple network
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim = 3, hidden_dim = 125, output_dim = 1):
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
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)
    
def main():
    # Initialize model and set hyperparameters
    model = NeuralNetwork()
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
    n_epochs = 10 # tried 50k steps. Not much change
    lambda_boundary = 1000
    loss = []
    l2 = []
    best_loss = float('inf')
    best_model_path = "best_model.pth"
    interior, boundary = sample_points()
    
    # Plot the sampled points (can be commented out)
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(interior[:,0], interior[:,1], interior[:,2], label="Interior")
    ax.scatter(boundary[:, 0], boundary[:, 1], boundary[:,2], label="Boundary")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title("Sampled Points")
    plt.show()
    L = 2.5  # Domain size [-L, L] x [-L, L]
    n_points = 50
    x = torch.linspace(-L, L, n_points)
    y = torch.linspace(-L, L, n_points)
    z = torch.linspace(-L, L, n_points)
    xx, yy , zz= torch.meshgrid(x, y, z)
    grid_points = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1) 
    dx = 2 * L / (n_points - 1)
    volume_element = dx **3
    psi_true = true_psi0_3d(xx, yy, zz).cpu().numpy()
    # Training loop. Same as before
    for epoch in tqdm(range(n_epochs)):
        optimizer.zero_grad()
        interior, boundary = sample_points()
        loss_interior = energy_loss(model, interior)
        loss_boundary = boundary_loss(model, boundary)
        total_loss = loss_interior + lambda_boundary * loss_boundary
        total_loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        loss.append(total_loss.item())
        l2_error = compute_l2_error(model, grid_points, n_points, psi_true, volume_element)
        l2.append(l2_error)
        # Log losses and save best model
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
                         f"Boundary Loss: {loss_boundary.item():.6f}"
                         f"Best Loss: {best_loss:.7f}")


    # Save loss and plot it
    np.save("loss.npy", np.array(loss))
    np.save("l2.npy", np.array(l2))
    plotter()
    logging.info("DONE")    
if __name__ == '__main__':
    main()
