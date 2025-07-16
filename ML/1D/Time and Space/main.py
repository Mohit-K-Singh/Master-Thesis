import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import random
from domain import sample_space_time_points
from energy import initial_loss, tdse_residual_loss,boundary_loss, weak_loss , variational_loss,  phase_evolution_loss
from plot_machine import plotter
from error import compute_error,conservation_loss
from torch.optim.lr_scheduler import OneCycleLR

from scipy.integrate import trapezoid

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Ensures deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def init_weights2(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias) 

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias) 

class ComplexNetwork(nn.Module):
    def __init__(self, input_dim = 2, hidden_dim =60, output_dim =2):
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
            nn.Linear(hidden_dim, 30),
            nn.Tanh(),
            nn.Linear(30, output_dim)

        )
        self.apply(init_weights)  


    
    
    def forward(self, x, t):
        #x, t = interior
        x_t = torch.cat([x, t], dim=1)
        #print("X_T:", x_t.shape)
        output = self.layers(x_t)
        #print("OUTPUT: ",output.shape)
        re, im = output.unbind(dim=-1)
        #print("Re: ", re.shape, " Im: ", im.shape)
        return torch.complex(re, im)

def psi_true(x, t, m=1.0, omega=1.0, hbar=1.0):
    """
    Ground state wavefunction of 1D quantum harmonic oscillator (complex-valued)
    """
    prefactor = (m * omega / (np.pi * hbar))**0.25
    exp_space = torch.exp(-m * omega * x**2 / (2 * hbar))
    exp_time = torch.exp(-1j * omega * t / 2)
    return prefactor * exp_space * exp_time
def psi_true_second_state(x, t, m=1.0, omega=1.0, hbar=1.0):
    """
    Second excited state (n=2) of 1D quantum harmonic oscillator.
    """
    xi = torch.sqrt(m * omega / hbar) * x
    H2 = 4 * xi**2 - 2  # Hermite polynomial H_2(xi)
    
    prefactor = (m * omega / (np.pi * hbar))**0.25 * (1 / torch.sqrt(torch.tensor(2.0)))
    spatial_part = H2 * torch.exp(-m * omega * x**2 / (2 * hbar))
    time_part = torch.exp(-1j * 2.5 * omega * t)  # E_2 = 5/2 ħω
    
    return prefactor * spatial_part * time_part



def main():
    """
    Classic training loop:
    
    """
    
    model = ComplexNetwork()
    
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)#,  weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

    epoch = 0
    best_loss = float('inf')
    best_l2 = float('inf')

    """ 
    checkpoint = torch.load("best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    best_loss = checkpoint['loss']
    best_l2 =checkpoint['l2'] 
    """
   
    n_epochs = 20000#5500 
    loss = []
    l2 = []
    scheduler = OneCycleLR(optimizer, max_lr=1e-3, total_steps=n_epochs -epoch, pct_start=0.3, anneal_strategy='cos')
    best_model_path = "best_model.pth"

    x_interior,t_interior, x_initial, t_initial, x_boundary, t_boundary = sample_space_time_points(device)
    t_zero = torch.tensor(0.0).to(device)

    loss_history = {
        'phase_evo': [],
        'interior': [],
        'initial': [],
        'boundary': [],
        'conservation': [],
        'neg': [],
        'l2': [],
        'total': []     
    }
    """
    #Sanity check
    psi_initial = psi_true(x_initial.squeeze(), t_zero) 
    plt.figure(figsize=(10, 6))
    plt.plot(x_initial.detach().cpu().squeeze().numpy(), psi_initial.detach().cpu().numpy())
    plt.xlabel("x")
    plt.ylabel("psi_initial")
    plt.title("Initial sanity check")
    plt.legend()
    plt.grid(True)
    plt.show() """
    for epoch in tqdm(range(epoch +1, n_epochs)):

        set_seed(epoch)
        optimizer.zero_grad()
        #if epoch % 10 == 0:
        x_interior,t_interior, x_initial, t_initial, x_boundary, t_boundary = sample_space_time_points(device)

        psi_initial = psi_true(x_initial.squeeze(),t_zero)


        x_phase = torch.linspace(-5, 5, 100).to(device)  # shape [5, 1]
        t_phase = torch.linspace(0, 1, 100).to(device)
        loss_phase_evo = phase_evolution_loss(model, x_phase, t_phase)
        loss_interior = tdse_residual_loss(model, x_interior,t_interior)
        loss_initial =initial_loss(model,  x_initial, t_initial, psi_initial)
        loss_boundary = boundary_loss(model, x_boundary, t_boundary)
        loss_conservation = conservation_loss(model,device)
        loss_neg =  1.05*torch.relu(-loss_interior)

        
        #220 init 100 boundary 300 loss was good with 5000 steps
        total_loss = loss_initial +  loss_boundary +  loss_interior +  loss_conservation  + loss_phase_evo #+ loss_neg
        
        loss_history['phase_evo'].append(loss_phase_evo.item())
        loss_history['interior'].append(loss_interior.item())
        loss_history['initial'].append(loss_initial.item())
        loss_history['boundary'].append(loss_boundary.item())
        loss_history['conservation'].append(loss_conservation.item())
        loss_history['neg'].append(loss_neg.item())
        loss_history['total'].append(total_loss.item())

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        l2_error = compute_error(model,device)
        loss_history['l2'].append(l2_error.item())

        # To plot loss later
        loss.append(total_loss.item())
        l2.append(l2_error.cpu())
        if l2_error.item() < best_l2:
            best_loss = total_loss.item()
            best_l2 = l2_error.item()
            print(f"{epoch}, Total : {total_loss.item():.6f}, "
                            f"Interior : {loss_interior.item():.6f}, "
                            f"Boundary : {loss_boundary.item():.6f}, "
                            f"Initial : {loss_initial.item():.6f}, "
                            f"Conservation : {loss_conservation.item():.6f}",
                            f"Evo: {loss_phase_evo.item():.4f}",
                            f"L2: {l2_error:.4f}",
                            f"Best: {best_loss:.7f}")
            
            logging.info(f"Epoch {epoch}: Best Loss: {best_loss:.4f}, saving model")
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'l2': best_l2,
            }, best_model_path)
           
        
        logging.info(f"Epoch {epoch}, Total : {total_loss.item():.6f}, "
                            f"Interior : {loss_interior.item():.6f}, "
                            f"Boundary : {loss_boundary.item():.6f}, "
                            f"Initial : {loss_initial.item():.6f}, "
                            f"Conservation : {loss_conservation.item():.6f},"
                            f"Best : {best_loss:.7f}")

        
    #Save loss and plot it
    np.save("loss.npy", np.array(loss))
    np.save("l2.npy", np.array(l2))
    plotter()
    # Plot individual losses
    for loss_name, values in loss_history.items():
        if loss_name != 'total':  # We'll plot total separately
            plt.plot(values, label=loss_name)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.yscale('log')  # Often helpful for PINN losses
    plt.legend()
    plt.grid(True)
    plt.savefig("mix.png")
    #plt.show()
    logging.info("DONE")    
if __name__ == '__main__':
    main()
