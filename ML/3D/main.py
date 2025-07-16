import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import random
from torch.optim.lr_scheduler import OneCycleLR


from domain import sample_points
from energy import energy_loss, boundary_loss
from plot_machine import plotter
from error import compute_l2_error, conservation_loss
"""
Same routine as in 1D and 2D case.

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

# Logger
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


# Simple network
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim = 3, hidden_dim = 60, output_dim = 1):
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
        self.apply(init_weights)
    def forward(self, x):
        return self.layers(x)
    
def main():
    # Initialize model and set hyperparameters
    model = NeuralNetwork()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr = 1e-2)#, weight_decay=1e-3)

    n_epochs = 7000 # tried 50k steps. Not much change
    loss = []
    l2 = []
    best_loss = float('inf')
    best_l2 = float('inf')
    best_model_path = "best_model.pth"
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.5)
    scheduler = OneCycleLR(optimizer, max_lr=1e-2, total_steps=n_epochs, pct_start=0.3, anneal_strategy='cos')
    """ 
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
     """
    # Training loop. Same as before
    for epoch in tqdm(range(n_epochs)):
        set_seed(epoch)
        optimizer.zero_grad()

        interior, boundary = sample_points(device)
        loss_interior = energy_loss(model, interior)
        loss_boundary = boundary_loss(model, boundary)
        loss_conv = conservation_loss(model, device)
        total_loss = loss_interior +  500*loss_boundary + loss_conv
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        l2_error = compute_l2_error(model, device)

        loss.append(total_loss.item())
        l2.append(l2_error.item())
        # Log losses and save best model
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
                'l2': best_l2
            }, best_model_path)
        
        logging.info(f"Epoch {epoch}, Total Loss: {total_loss.item():.6f}, "
                         f"Interior Loss: {loss_interior.item():.6f}, "
                         f"Boundary Loss: {loss_boundary.item():.6f}"
                         f"Consv Loss: {loss_conv.item():.4f}, "
                         f"L2: {l2_error:.5f}, "
                         f"Best Loss: {best_loss:.7f}")


    # Save loss and plot it
    np.save("loss.npy", np.array(loss))
    np.save("l2.npy", np.array(l2))
    plotter()
    logging.info("DONE")    
if __name__ == '__main__':
    main()
