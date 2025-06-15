import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

from domain import sample_space_time_points
from energy import initial_loss, tdse_residual_loss,boundary_loss, weak_loss
from plot_machine import plotter
"""
This script aims to learn an approximative solution to the 1D - time dependent schrödinger Equation:

"""

#Logger to for debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("train.log"),
    ]
)


class ComplexNetwork(nn.Module):
    def __init__(self, input_dim = 2, hidden_dim =100, output_dim =2):
        super(ComplexNetwork, self).__init__()
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

    def forward(self, x, t):
        #x, t = interior
        x_t = torch.cat([x, t], dim=1)
        output = self.layers(x_t)
        re, im = output.unbind(dim=-1)
        return torch.complex(re, im)

def initial_condition(x):
    psi = torch.exp(-x**2 / 2)
    return psi.unsqueeze(1), torch.zeros_like(psi).unsqueeze(1)

def conservation_loss(model, x_range=(-5, 5), num_points=100):
    x = torch.linspace(*x_range, num_points).view(-1, 1).to(device)
    dx = (x_range[1] - x_range[0]) / num_points
    loss = 0
    #t_vals = range(0, 20,)
    t_vals = np.linspace(0,2,21)

    # New
    #extension = np.linspace(5,7,10)
    #t_vals = np.concatenate((t_vals,extension))
    for t_val in t_vals:
        t = torch.full_like(x, t_val)
        u = model(x, t)
        prob = u.real**2 + u.imag**2
        integral = torch.sum(prob) * dx
        loss += (integral - 1.0)**2
    return loss / len(t_vals)

device = torch.device("cuda") #if torch.cuda.is_available() else "cpu")
def main():
    """
    Classic training loop:
    
    """
    model = ComplexNetwork()
    
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)

    epoch = 0
    """ checkpoint = torch.load("best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    best_loss = checkpoint['loss'] """
   
    n_epochs = 5000
    lambda_boundary = 200
    loss = []
    best_loss = float('inf')
    best_model_path = "best_model.pth"

    x_interior,t_interior, x_initial, t_initial, x_boundary, t_boundary = sample_space_time_points()
    x_interior = x_interior.to(device)
    t_interior = t_interior.to(device)
    x_initial = x_initial.to(device)
    t_initial = t_initial.to(device)
    x_boundary = x_boundary.to(device)
    t_boundary = t_boundary.to(device)
    u0, v0 = initial_condition(x_interior)
    u0 = u0.to(device)
    v0 = v0.to(device)
    psi_initial = torch.complex(u0,v0)
    for epoch in tqdm(range(epoch +1, n_epochs)):
        optimizer.zero_grad()
        # Sample points and compute loss
        #interior, initial, boundary = sample_space_time_points()
        x_interior,t_interior, x_initial, t_initial, x_boundary, t_boundary = sample_space_time_points()
        x_interior = x_interior.to(device)
        t_interior = t_interior.to(device)
        x_initial = x_initial.to(device)
        t_initial = t_initial.to(device)
        x_boundary = x_boundary.to(device)
        t_boundary = t_boundary.to(device)
        #interior = interior.to(device)
        #initial = initial.to(device)
        #boundary = boundary.to(device)

        loss_interior = tdse_residual_loss(model, x_interior,t_interior)
        loss_initial = initial_loss(model,  x_initial, t_initial, psi_initial)
        loss_boundary = boundary_loss(model, x_boundary, t_boundary)
        loss_conservation = conservation_loss(model)
        total_loss = loss_interior  + 40 *loss_initial + 40*loss_boundary + 200* loss_conservation
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        
        # To plot loss later
        loss.append(total_loss.item())

        if total_loss < best_loss:
            print(f"Epoch {epoch}, Total Loss: {total_loss.item():.6f}, "
                         f"Interior Loss: {loss_interior.item():.6f}, "
                         f"Boundary Loss: {loss_boundary.item():.6f}, "
                         f"Initial Loss: {loss_initial.item():.6f}, "
                         f"Conservation Loss: {loss_conservation.item():.6f}",
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
                         f"Initial Loss: {loss_initial.item():.6f}, "
                         f"Conservation Loss: {loss_conservation.item():.6f}",
                         f"Best Loss: {best_loss:.7f}")


    #Save loss and plot it
    np.save("loss.npy", np.array(loss))
    plotter()
    logging.info("DONE")    
if __name__ == '__main__':
    main()
