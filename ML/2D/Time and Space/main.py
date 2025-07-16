import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim.lr_scheduler import OneCycleLR
import random


from domain import sample_space_time_points
from energy import variational_loss, initial_loss, boundary_loss, phase_evolution_loss, tdse_residual_loss
from error import compute_error, conservation_loss
from plot_machine import plotter

#torch.manual_seed(0)

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
    if isinstance(m, nn.Linear) :
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias) 

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias) 

class ComplexNetwork(nn.Module):
    def __init__(self, input_dim = 3, hidden_dim =70, output_dim =2):
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
        self.apply(init_weights2)  
    def forward(self, x, t):
        x_t = torch.cat([x, t], dim=1)
        output = self.layers(x_t)
        re, im = output.unbind(dim=-1)
        return torch.complex(re, im)


def psi_true1(x,y, t, m=1.0, omega=1.0, hbar=1.0):
    """
    Ground state wavefunction of 2D quantum harmonic oscillator (complex-valued)
    """
    #x,_ = torch.sort(xy[:,0], dim=0)
    #y,_ = torch.sort(xy[:,1], dim=0)
    #X, Y= torch.meshgrid(x,y, indexing='ij')
    #x, y = X.flatten(), Y.flatten()
    prefactor = (m * omega / (np.pi * hbar))**0.5
    exp_space = torch.exp(-m * omega * (x**2 +y**2)/ (2 * hbar))
    exp_time = torch.exp(-1j * omega * t)
    return prefactor * exp_space * exp_time    
def psi_true(xy, t, m=1.0, omega=1.0, hbar=1.0):
    """
    Ground state wavefunction of 2D quantum harmonic oscillator (complex-valued)
    """
    x= xy[:,0]
    y= xy[:,1]

    prefactor = (m * omega / (np.pi * hbar))**0.5
    exp_space = torch.exp(-m * omega * (x**2 +y**2)/ (2 * hbar))
    exp_time = torch.exp(-1j * omega * t)
    return prefactor * exp_space * exp_time  
def main():
    model = ComplexNetwork()
    
    model.to(device)


    optimizer = optim.Adam(model.parameters(), lr = 1e-3 )#,  weight_decay=1e-3)
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999),weight_decay=1e-4)

    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
    a,b = -3.5 , 3.5
    t0, t1 = 0, 1
    t_zero = torch.tensor(0.0).to(device)
    #x_phase = torch.linspace(-3.5,3.5,200).to(device)
    #t_phase = torch.linspace(0, 1,200).to(device)
    
    epoch = 0
    best_loss = float('inf')
    best_l2 = float('inf')
    best_model_path = "best_model.pth"
    """  
    checkpoint = torch.load("best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    best_loss = checkpoint['loss']
    best_l2 =checkpoint['l2']    
    """
   
    n_epochs =25000

    
    scheduler = OneCycleLR(optimizer, max_lr=1e-3, total_steps=n_epochs -epoch, pct_start=0.3, anneal_strategy='cos')

   
    loss = []
    l2 = []
    loss_history = {
        'phase_evo': [],
        'interior': [],
        'initial': [],
        'boundary': [],
        'conservation': [],
        #'neg': [],
        'L2 ERROR': [],
        'total': []     
    }

    #set_seed(42 + epoch)
    #x_interior,t_interior, x_initial, t_initial, x_boundary, t_boundary = sample_space_time_points(device,0)
    #x_initial = torch.linspace(-2,2, 30).to(device)
    #t_initial = torch.zeros_like(x_initial).to(device)
    #x,_ = torch.sort(x_initial[:,0], dim=0)
    #y,_ = torch.sort(x_initial[:,1], dim=0)
    #t,_ = torch.sort(t_initial, dim=0)
    #X, Y= torch.meshgrid(x_initial,x_initial, indexing='ij')
    #psi_initial = psi_true(x_initial,t_zero).reshape(X.shape)
    #psi_initial = psi_true(X.flatten(), Y.flatten(),t_zero).reshape(X.shape)
    #prob_density = torch.abs(psi_initial)**2 
    """     
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf =ax.plot_surface(X.detach().cpu().numpy(), Y.detach().cpu().numpy(), prob_density.detach().cpu().numpy(),cmap='viridis', rstride=1, cstride=1, alpha=0.9)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_zlabel('$psi(x, y, t)|^2$', fontsize=12)
    ax.set_title('2D Ground State Probability Density', fontsize=14)

    # Add colorbar
    fig.colorbar(surf, shrink=0.5, aspect=10, label='Probability Density')
    plt.tight_layout()
    #plt.show()
    
    """ 
    

    for epoch in tqdm(range(epoch+1, n_epochs)):
        set_seed(epoch)
        #x_phase = torch.rand(300, 1 ,device=device).squeeze() * (b-a) + a
        #t_phase = torch.rand(300,1,device=device).squeeze() * (t1 - t0) + t0
        optimizer.zero_grad()
        x_interior,t_interior, x_boundary, t_boundary ,x_initial, t_initial= sample_space_time_points(device)
        psi_initial = psi_true(x_initial,t_zero)
        loss_interior = tdse_residual_loss(model, x_interior, t_interior)
        loss_initial = initial_loss(model, x_initial, t_initial, psi_initial)
        loss_boundary =  boundary_loss(model, x_boundary, t_boundary)
        loss_conservation =conservation_loss(model,device)
        loss_phase_evo =phase_evolution_loss(model, x_interior, t_interior)
        #loss_neg = 1.05*torch.relu(-loss_interior)



        total_loss =   loss_phase_evo  + loss_interior + loss_initial + loss_conservation +loss_boundary  #+ loss_neg
        loss_history['phase_evo'].append(loss_phase_evo.item())
        loss_history['interior'].append(loss_interior.item())
        loss_history['initial'].append(loss_initial.item())
        loss_history['boundary'].append(loss_boundary.item())
        loss_history['conservation'].append(loss_conservation.item())
        #loss_history['neg'].append(loss_neg.item())
        loss_history['total'].append(total_loss.item())
        

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        
        l2_error = compute_error(model,device)
        loss_history['L2 ERROR'].append(l2_error.item())
        loss.append(total_loss.item())
        l2.append(l2_error.item())
        
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
           
        
        logging.info(
            f"Epoch {epoch}, "
            f"Total: {total_loss.item():.6f}, "
            f"Interior: {loss_interior.item():.6f}, "
            f"Boundary: {loss_boundary.item():.6f}, "
            f"Initial: {loss_initial.item():.6f}, "
            f"Conservation: {loss_conservation.item():.6f}, "
            f"Evo: {loss_phase_evo.item():.4f}, "
            f"L2: {l2_error:.4f}, "
            f"Best: {best_loss:.7f}"
        )
            
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
    plt.yscale('log')  
    plt.legend()
    plt.grid(True)
    plt.savefig("mix.png")
    plt.show()
    logging.info("DONE") 

if __name__ == '__main__':
    main()
