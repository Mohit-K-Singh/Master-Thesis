import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

from domain import sample_space_time_points
from energy import initial_loss, tdse_residual_loss,boundary_loss, weak_loss , variational_loss,  phase_evolution_loss
from plot_machine import plotter
from error import compute_error

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


class ComplexNetwork(nn.Module):
    def __init__(self, input_dim = 2, hidden_dim =60, output_dim =2):
        super(ComplexNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 30),
            nn.SiLU(),
            nn.Linear(30, output_dim)

        )
    """    self._initialize_weights()

    def _initialize_weights(self):
        for m in self.layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))
                nn.init.zeros_(m.bias) """
    
    
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
def conservation_loss(model,x_range=(-7, 7), num_points=100):
    x = torch.linspace(*x_range, num_points).to(device)
    t = torch.linspace(0, 1, 100).to(device)
    X, T = torch.meshgrid(x, t, indexing='ij')
    #print(X.shape, T.shape)
    x_flat, t_flat = X.flatten().unsqueeze(-1), T.flatten().unsqueeze(-1)
    #print(x.shape)
    psi = model(x_flat, t_flat)
    psi = psi.reshape(X.shape)
    prob_density = torch.abs(psi)**2
    norm = torch.trapezoid(prob_density, x=x, dim = 0) 
    final_norm = torch.trapezoid(norm, x=t)
    norm_loss = (final_norm - 1.0)**2
    return norm_loss



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def main():
    """
    Classic training loop:
    
    """
    
    model = ComplexNetwork()
    
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)#,  weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

    epoch = 0
    
    checkpoint = torch.load("Best/true_1.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    best_loss = checkpoint['loss']
   
    n_epochs = 25000#5500 
    lambda_boundary = 200
    loss = []
    l2 = []
    best_loss = float('inf')
    best_l2 = float('inf')
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
        optimizer.zero_grad()
        #if epoch % 10 == 0:
        x_interior,t_interior, x_initial, t_initial, x_boundary, t_boundary = sample_space_time_points(device)

        psi_initial = psi_true(x_initial.squeeze(),t_zero)

        #x_phase = torch.tensor([[0.0]], device=device) #[[0]] with shape[1,1]
        #x_phase = torch.tensor([[-1.5], [0.0], [1.5]], device = device)
        #t_phase = torch.linspace(0, 1, 20).view(-1, 1).to(device) #[30,1]
        t_phase,_ = torch.sort(torch.rand(30,1).view(-1,1).to(device), dim=0)
        
        x_phase = torch.linspace(-3, 3, 20).view(-1, 1).to(device)  # shape [5, 1]
        t_phase = torch.linspace(0, 1, 20).view(-1, 1).to(device)
        #loss_phase_evo = 20 * phase_evolution_loss(model, x_phase.expand(t_phase.size(0), -1), t_phase) #x_phase brought into t_phase shape [30,1].. 30 zeros
        loss_phase_evo = 10 * phase_evolution_loss(model, x_phase, t_phase)
        loss_interior = variational_loss(model, x_interior,t_interior)
        loss_initial =70 *initial_loss(model,  x_initial, t_initial, psi_initial)
        loss_boundary = 50 *boundary_loss(model, x_boundary, t_boundary)
        loss_conservation = 1000 *conservation_loss(model)
        loss_neg =  1.05*torch.relu(-loss_interior)

        
        #220 init 100 boundary 300 loss was good with 5000 steps
        total_loss = loss_initial +  loss_boundary +  loss_interior +  loss_conservation  + loss_phase_evo + loss_neg
        l2_error = compute_error(model,device)
        loss_history['phase_evo'].append(loss_phase_evo.item())
        loss_history['interior'].append(loss_interior.item())
        loss_history['initial'].append(loss_initial.item())
        loss_history['boundary'].append(loss_boundary.item())
        loss_history['conservation'].append(loss_conservation.item())
        loss_history['neg'].append(loss_neg.item())
        loss_history['total'].append(total_loss.item())
        loss_history['l2'].append(l2_error.cpu())

        total_loss.backward()
        optimizer.step()
        scheduler.step()
        
        # To plot loss later
        loss.append(total_loss.item())
        l2.append(l2_error.cpu())
        if l2_error < best_l2:
            best_loss = total_loss.item()
            best_l2 = l2_error
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
    plt.show()
    logging.info("DONE")    
if __name__ == '__main__':
    main()
