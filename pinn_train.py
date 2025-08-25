"""
Physics-Informed Neural Networks (PINNs) Hyperparameter Study

Comprehensive evaluation of different architectures and activation functions
for solving the 1D diffusion equation:
∂y/∂t = ∂²y/∂x² - e^(-t)(sin(πx) - π²sin(πx))

Domain: x ∈ [-1,1], t ∈ [0,1]
Initial Condition: y(x,0) = sin(πx)
Boundary Conditions: y(-1,t) = 0, y(1,t) = 0
Exact solution: y(x,t) = e^(-t)sin(πx)
"""

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import time
from pyDOE import lhs
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set style for professional plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Set default dtype and seeds
torch.set_default_dtype(torch.float)
torch.manual_seed(1234)
np.random.seed(1234)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Fixed parameters
STEPS = 20000
TOTAL_POINTS_X = 200
TOTAL_POINTS_T = 100
X_MIN, X_MAX = -1, 1
T_MIN, T_MAX = 0, 1
NU = 100    # Number of training points
NF = 10000  # Number of collocation points

# Hyperparameter configurations to test (ReLU removed due to poor performance)
CONFIGS = {
    'Small_Tanh': {
        'layers': [2, 32, 32, 1],
        'activation': 'tanh',
        'lr': 1e-3
    },
    'Medium_Tanh': {
        'layers': [2, 64, 64, 1],
        'activation': 'tanh',
        'lr': 1e-3
    },
    'Large_Tanh': {
        'layers': [2, 128, 128, 1],
        'activation': 'tanh',
        'lr': 1e-3
    },
    'Deep_Tanh': {
        'layers': [2, 64, 64, 64, 1],
        'activation': 'tanh',
        'lr': 1e-3
    },
    'Medium_GELU': {
        'layers': [2, 64, 64, 1],
        'activation': 'gelu',
        'lr': 1e-3
    },
    'Medium_Swish': {
        'layers': [2, 64, 64, 1],
        'activation': 'swish',
        'lr': 1e-3
    },
    'Medium_Tanh_HighLR': {
        'layers': [2, 64, 64, 1],
        'activation': 'tanh',
        'lr': 5e-3
    },
    'Medium_Tanh_LowLR': {
        'layers': [2, 64, 64, 1],
        'activation': 'tanh',
        'lr': 1e-4
    },
}

def f_real(x, t):
    """Exact solution of the diffusion equation"""
    return torch.exp(-t) * torch.sin(np.pi * x)

class FCN(nn.Module):
    """Fully Connected Neural Network for PINNs"""
    
    def __init__(self, layers, activation_type='tanh'):
        super().__init__()
        
        # Activation function selection
        if activation_type.lower() == 'tanh':
            self.activation = nn.Tanh()
        elif activation_type.lower() == 'relu':
            self.activation = nn.ReLU()
        elif activation_type.lower() == 'gelu':
            self.activation = nn.GELU()
        elif activation_type.lower() == 'swish':
            self.activation = nn.SiLU()  # SiLU is Swish
        else:
            raise ValueError(f"Unknown activation: {activation_type}")
            
        self.loss_function = nn.MSELoss(reduction='mean')
        self.linears = nn.ModuleList([
            nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)
        ])
        
        # Xavier initialization
        for i in range(len(layers)-1):
            if activation_type.lower() == 'relu':
                nn.init.kaiming_normal_(self.linears[i].weight.data, nonlinearity='relu')
            else:
                nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linears[i].bias.data)
        
        self.layers = layers
    
    def forward(self, x):
        """Forward pass through the network"""
        if not torch.is_tensor(x):         
            x = torch.from_numpy(x)                
        
        a = x.float()
        for i in range(len(self.layers)-2):  
            z = self.linears[i](a)              
            a = self.activation(z)    
        a = self.linears[-1](a)
        return a
    
    def lossBC(self, x_BC, y_BC):
        """Loss function for boundary conditions"""
        return self.loss_function(self.forward(x_BC), y_BC)
    
    def lossPDE(self, x_PDE):
        """Loss function for PDE residual"""
        g = x_PDE.clone()
        g.requires_grad = True
        
        f = self.forward(g)
        
        # Compute derivatives
        f_x_t = autograd.grad(
            f, g, torch.ones([g.shape[0], 1]).to(device), 
            retain_graph=True, create_graph=True
        )[0]
        
        f_xx_tt = autograd.grad(
            f_x_t, g, torch.ones(g.shape).to(device), 
            create_graph=True
        )[0]
        
        f_t = f_x_t[:, [1]]   # ∂f/∂t
        f_xx = f_xx_tt[:, [0]]  # ∂²f/∂x²
        
        # PDE residual
        f_residual = (f_t - f_xx + 
                     torch.exp(-g[:, 1:]) * 
                     (torch.sin(np.pi * g[:, 0:1]) - 
                      np.pi ** 2 * torch.sin(np.pi * g[:, 0:1])))
        
        f_hat = torch.zeros_like(f_residual)
        return self.loss_function(f_residual, f_hat)

    def loss(self, x_BC, y_BC, x_PDE):
        """Total loss function"""
        loss_bc = self.lossBC(x_BC, y_BC)
        loss_pde = self.lossPDE(x_PDE)
        return loss_bc + loss_pde

def prepare_data():
    """Prepare all training and test data"""
    # Create grids
    x = torch.linspace(X_MIN, X_MAX, TOTAL_POINTS_X).view(-1, 1)
    t = torch.linspace(T_MIN, T_MAX, TOTAL_POINTS_T).view(-1, 1)
    X, T = torch.meshgrid(x.squeeze(1), t.squeeze(1), indexing='ij')
    y_real = f_real(X, T)
    
    # Test data
    x_test = torch.hstack((
        X.transpose(1, 0).flatten()[:, None],
        T.transpose(1, 0).flatten()[:, None]
    ))
    y_test = y_real.transpose(1, 0).flatten()[:, None]
    
    # Training data (boundary/initial conditions)
    left_X = torch.hstack((X[:, 0][:, None], T[:, 0][:, None]))
    left_Y = torch.sin(np.pi * left_X[:, 0]).unsqueeze(1)
    
    bottom_X = torch.hstack((X[0, :][:, None], T[0, :][:, None]))
    bottom_Y = torch.zeros(bottom_X.shape[0], 1)
    
    top_X = torch.hstack((X[-1, :][:, None], T[-1, :][:, None]))
    top_Y = torch.zeros(top_X.shape[0], 1)
    
    X_train = torch.vstack([left_X, bottom_X, top_X])
    Y_train = torch.vstack([left_Y, bottom_Y, top_Y])
    
    # Sample training points
    idx = np.random.choice(X_train.shape[0], NU, replace=False)
    X_train_Nu = X_train[idx, :]
    Y_train_Nu = Y_train[idx, :]
    
    # Collocation points
    lb = x_test[0]
    ub = x_test[-1]
    X_train_Nf = lb + (ub - lb) * lhs(2, NF)
    X_train_Nf = torch.vstack((X_train_Nf, X_train_Nu))
    
    return X_train_Nu, Y_train_Nu, X_train_Nf, x_test, y_test

def train_config(config_name, config, X_train_Nu, Y_train_Nu, X_train_Nf, x_test, y_test):
    """Train a single configuration and return metrics"""
    print(f"\nTraining {config_name}...")
    
    # Reset seed for fair comparison
    torch.manual_seed(123)
    
    # Move data to device
    X_train_Nu_gpu = X_train_Nu.float().to(device)
    Y_train_Nu_gpu = Y_train_Nu.float().to(device)
    X_train_Nf_gpu = X_train_Nf.float().to(device)
    X_test_gpu = x_test.float().to(device)
    Y_test_gpu = y_test.float().to(device)
    
    # Create model
    model = FCN(config['layers'], config['activation'])
    model.to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    
    # Training metrics
    train_losses = []
    test_losses = []
    times = []
    
    start_time = time.time()
    
    for i in range(STEPS):
        # Training step
        loss = model.loss(X_train_Nu_gpu, Y_train_Nu_gpu, X_train_Nf_gpu)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Record metrics every 1000 steps
        if i % 1000 == 0:
            with torch.no_grad():
                test_loss = model.lossBC(X_test_gpu, Y_test_gpu)
                train_losses.append(loss.detach().cpu().numpy())
                test_losses.append(test_loss.detach().cpu().numpy())
                times.append(time.time() - start_time)
    
    # Final evaluation
    with torch.no_grad():
        final_test_loss = model.lossBC(X_test_gpu, Y_test_gpu).cpu().numpy()
        y_pred = model(X_test_gpu)
        error = torch.abs(y_pred - Y_test_gpu).detach().cpu()
        mae = torch.mean(error).numpy()
        max_error = torch.max(error).numpy()
        rel_l2_error = (torch.norm(error) / torch.norm(Y_test_gpu.detach().cpu())).numpy()
    
    total_time = time.time() - start_time
    params = sum(p.numel() for p in model.parameters())
    
    results = {
        'config_name': config_name,
        'final_test_loss': final_test_loss,
        'mae': mae,
        'max_error': max_error,
        'rel_l2_error': rel_l2_error,
        'training_time': total_time,
        'parameters': params,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'times': times,
        'config': config
    }
    
    print(f"  Final Test Loss: {final_test_loss:.6f}")
    print(f"  Relative L2 Error: {rel_l2_error:.4f}")
    print(f"  Training Time: {total_time:.1f}s")
    print(f"  Parameters: {params:,}")
    
    return results

def create_performance_plots(all_results):
    """Create comprehensive performance comparison plots"""
    
    # Convert to DataFrame for easier plotting
    df_data = []
    for result in all_results:
        row = {
            'Config': result['config_name'],
            'Architecture': 'x'.join(map(str, result['config']['layers'])),
            'Activation': result['config']['activation'].upper(),
            'Learning Rate': result['config']['lr'],
            'Test Loss': result['final_test_loss'],
            'MAE': result['mae'],
            'Rel L2 Error': result['rel_l2_error'],
            'Training Time (s)': result['training_time'],
            'Parameters': result['parameters']
        }
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    
    # Create subplot layout
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Test Loss Comparison
    plt.subplot(2, 3, 1)
    bars = plt.bar(range(len(df)), df['Test Loss'], color=plt.cm.Set3(np.arange(len(df))))
    plt.xlabel('Configuration')
    plt.ylabel('Final Test Loss')
    plt.title('Test Loss Comparison', fontsize=14, fontweight='bold')
    plt.xticks(range(len(df)), df['Config'], rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, df['Test Loss'])):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{val:.4f}', ha='center', va='bottom', fontsize=8)
    
    # 2. Relative L2 Error Comparison
    plt.subplot(2, 3, 2)
    bars = plt.bar(range(len(df)), df['Rel L2 Error'], color=plt.cm.Set3(np.arange(len(df))))
    plt.xlabel('Configuration')
    plt.ylabel('Relative L2 Error')
    plt.title('Relative L2 Error Comparison', fontsize=14, fontweight='bold')
    plt.xticks(range(len(df)), df['Config'], rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    for i, (bar, val) in enumerate(zip(bars, df['Rel L2 Error'])):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 3. Training Time vs Performance
    plt.subplot(2, 3, 3)
    scatter = plt.scatter(df['Training Time (s)'], df['Rel L2 Error'], 
                         s=df['Parameters']/1000, alpha=0.6, c=range(len(df)), cmap='viridis')
    plt.xlabel('Training Time (seconds)')
    plt.ylabel('Relative L2 Error')
    plt.title('Efficiency vs Accuracy\n(Bubble size = Parameters/1000)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add labels to points
    for i, config in enumerate(df['Config']):
        plt.annotate(config, (df['Training Time (s)'].iloc[i], df['Rel L2 Error'].iloc[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # 4. Training Curves
    plt.subplot(2, 3, 4)
    for result in all_results:
        steps = np.arange(0, STEPS, 1000)
        plt.plot(steps, result['test_losses'], label=result['config_name'], linewidth=2)
    plt.xlabel('Training Steps')
    plt.ylabel('Test Loss')
    plt.title('Training Curves Comparison', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # 5. Architecture Performance Heatmap
    plt.subplot(2, 3, 5)
    
    # Group by activation function
    activation_performance = {}
    architecture_performance = {}
    
    for _, row in df.iterrows():
        activation = row['Activation']
        arch = row['Architecture']
        error = row['Rel L2 Error']
        
        if activation not in activation_performance:
            activation_performance[activation] = []
        activation_performance[activation].append(error)
        
        if arch not in architecture_performance:
            architecture_performance[arch] = []
        architecture_performance[arch].append(error)
    
    # Create heatmap data
    activations = list(activation_performance.keys())
    avg_errors = [np.mean(activation_performance[act]) for act in activations]
    
    bars = plt.bar(activations, avg_errors, color=plt.cm.plasma(np.linspace(0, 1, len(activations))))
    plt.xlabel('Activation Function')
    plt.ylabel('Average Rel L2 Error')
    plt.title('Performance by Activation Function', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    for bar, val in zip(bars, avg_errors):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 6. Parameter Efficiency
    plt.subplot(2, 3, 6)
    plt.scatter(df['Parameters'], df['Rel L2 Error'], s=100, alpha=0.7, c=range(len(df)), cmap='coolwarm')
    plt.xlabel('Number of Parameters')
    plt.ylabel('Relative L2 Error')
    plt.title('Parameter Efficiency', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    for i, config in enumerate(df['Config']):
        plt.annotate(config, (df['Parameters'].iloc[i], df['Rel L2 Error'].iloc[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('pinn_hyperparameter_study.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Summary table
    print("\n" + "="*100)
    print("HYPERPARAMETER STUDY SUMMARY")
    print("="*100)
    
    # Sort by performance
    df_sorted = df.sort_values('Rel L2 Error')
    print(f"{'Rank':<4} {'Configuration':<20} {'Architecture':<12} {'Activation':<10} {'LR':<8} {'Rel L2 Error':<12} {'Time(s)':<8} {'Params':<8}")
    print("-"*100)
    
    for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
        print(f"{i:<4} {row['Config']:<20} {row['Architecture']:<12} {row['Activation']:<10} "
              f"{row['Learning Rate']:<8.0e} {row['Rel L2 Error']:<12.4f} {row['Training Time (s)']:<8.1f} {row['Parameters']:<8,}")
    
    # Best performers analysis
    best_config = df_sorted.iloc[0]
    print(f"\n🏆 BEST PERFORMER: {best_config['Config']}")
    print(f"   Architecture: {best_config['Architecture']}")
    print(f"   Activation: {best_config['Activation']}")
    print(f"   Relative L2 Error: {best_config['Rel L2 Error']:.4f}")
    print(f"   Training Time: {best_config['Training Time (s)']:.1f}s")
    print(f"   Parameters: {best_config['Parameters']:,}")

def main():
    """Main execution function for hyperparameter study"""
    print("🚀 Physics-Informed Neural Networks (PINNs) Hyperparameter Study")
    print("="*80)
    print(f"Device: {device}")
    print(f"Training Steps: {STEPS:,}")
    print(f"Configurations to test: {len(CONFIGS)}")
    print("="*80)
    
    # Prepare data once
    print("\nPreparing data...")
    X_train_Nu, Y_train_Nu, X_train_Nf, x_test, y_test = prepare_data()
    print(f"Training points: {X_train_Nu.shape[0]}")
    print(f"Collocation points: {X_train_Nf.shape[0]}")
    print(f"Test points: {x_test.shape[0]}")
    
    # Train all configurations
    all_results = []
    
    for i, (config_name, config) in enumerate(CONFIGS.items(), 1):
        print(f"\n[{i}/{len(CONFIGS)}] Configuration: {config_name}")
        print(f"  Architecture: {config['layers']}")
        print(f"  Activation: {config['activation']}")
        print(f"  Learning Rate: {config['lr']}")
        
        result = train_config(config_name, config, X_train_Nu, Y_train_Nu, 
                            X_train_Nf, x_test, y_test)
        all_results.append(result)
    
    # Create comprehensive analysis
    print("\n" + "="*80)
    print("CREATING PERFORMANCE ANALYSIS...")
    print("="*80)
    
    create_performance_plots(all_results)
    
    print("\n✅ Hyperparameter study completed!")
    print("📊 Results saved as 'pinn_hyperparameter_study.png'")

if __name__ == "__main__":
    main()