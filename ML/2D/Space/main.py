import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from torch.optim.lr_scheduler import OneCycleLR

from domain import sample_points
from energy import energy_loss, boundary_loss
from error import convservation_loss, compute_error
from plot_machine import plotter
import random

"""
This script tries to learn a solution to the 2D time independnet schrödinger equation.
"""

#Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Set seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Ensures deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Simple Logger
logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("train.log"),  
    ]
)
#Initialize weights
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias) 
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
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.apply(init_weights)
    def forward(self, x):
        return self.layers(x)
    
def main():
    """
    Simple training loop with Adam optimizer.
    """
    model = NeuralNetwork()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr = 1e-3, weight_decay=1e-3)
    n_epochs = 15000
    loss = []
    l2 = []
    best_loss = float('inf')
    best_l2 = float('inf')
    best_model_path = "best_model.pth"
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
    scheduler = OneCycleLR(optimizer, max_lr=1e-3, total_steps=n_epochs, pct_start=0.3, anneal_strategy='cos')

    # Visualize the sampled points in the domain
    """ interior, boundary = sample_points()
    plt.scatter(interior[:,0], interior[:,1],  s=1, label="Interior")
    plt.scatter(boundary[:, 0], boundary[:, 1], s=1, label="Boundary")
    plt.xlabel("x"); plt.ylabel("y"); plt.legend()
    plt.title("Sampled Points")
    plt.show() """

    # Training loop
    for epoch in tqdm(range(n_epochs)):
        set_seed(epoch)
        optimizer.zero_grad()

        # Sample points
        interior, boundary = sample_points(device)

        # Compute loss in interior and boundary
        loss_interior = energy_loss(model, interior)
        loss_boundary = boundary_loss(model, boundary)
        loss_conv = convservation_loss(model,device)
        total_loss = loss_interior + loss_boundary  + loss_conv
        # Backpropagation and weight update
        total_loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        l2_error = compute_error(model,device)

        loss.append(total_loss.item())
        l2.append(l2_error.item())

        #Save the model with the best loss
        if l2_error.item() < best_l2:
            best_loss = total_loss.item()
            best_l2 = l2_error.item()
            print(f"Epoch {epoch}, Total Loss: {total_loss.item():.6f}, "
                         f"Interior Loss: {loss_interior.item():.6f}, "
                         f"Boundary Loss: {loss_boundary.item():.6f}, "
                         f"Consv Loss: {loss_conv.item():.4f}, "
                         f"L2: {l2_error:.5f}, "
                         f"Best Loss: {best_loss:.7f}")
            
            logging.info(f"Epoch {epoch}: Best Loss: {best_loss:.4f}, saving model")
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'l2': l2_error.item()
            }, best_model_path)
        
        # Log info
        logging.info(f"Epoch {epoch}, Total Loss: {total_loss.item():.6f}, "
                         f"Interior Loss: {loss_interior.item():.6f}, "
                         f"Boundary Loss: {loss_boundary.item():.6f}"
                         f"Consv Loss: {loss_conv.item():.4f}, "
                         f"L2: {l2_error:.5f}, "
                         f"Best Loss: {best_loss:.7f}")


    # Save loss for plotting
    np.save("loss.npy", np.array(loss))
    np.save("l2.npy", np.array(l2))
    plotter()
    logging.info("DONE")    
if __name__ == '__main__':
    main()
