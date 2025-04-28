import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import our models
import sys
import os

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.resnet import ResNet
from models.neural_ode import NeuralODE
from models.adaptive_resnet import AdaptiveResNet

def load_data(system_name, train_split=0.8, batch_size=32):
    """Load and prepare data for training."""
    data_path = os.path.join('data', f'{system_name}.npz')
    data = np.load(data_path)
    t, y = data['t'], data['y']
    
    # Convert to torch tensors
    t = torch.FloatTensor(t)
    y = torch.FloatTensor(y)
    
    # Create input-output pairs (y_t -> y_{t+1})
    X = y[:-1]  # All points except last
    Y = y[1:]   # All points except first
    
    # Split into train and validation
    n_train = int(len(X) * train_split)
    X_train, Y_train = X[:n_train], Y[:n_train]
    X_val, Y_val = X[n_train:], Y[n_train:]
    
    # Create dataloaders
    train_ds = TensorDataset(X_train, Y_train)
    val_ds = TensorDataset(X_val, Y_val)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    
    return train_loader, val_loader, t, y

def train_epoch(model, train_loader, optimizer, criterion):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for X, Y in train_loader:
        optimizer.zero_grad()
        Y_pred = model(X)
        loss = criterion(Y_pred, Y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion):
    """Validate the model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for X, Y in val_loader:
            Y_pred = model(X)
            loss = criterion(Y_pred, Y)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

def train_model(model_name, system_name, epochs=100, lr=1e-3):
    """Train a specific model on a specific system."""
    # Load data
    train_loader, val_loader, t, y = load_data(system_name)
    state_dim = y.shape[1]
    
    # Initialize model
    if model_name == 'resnet':
        model = ResNet(state_dim=state_dim)
    elif model_name == 'neural_ode':
        model = NeuralODE(state_dim=state_dim)
    else:  # adaptive_resnet
        model = AdaptiveResNet(state_dim=state_dim)
    
    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in tqdm(range(epochs), desc=f'Training {model_name} on {system_name}'):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss = validate(model, val_loader, criterion)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 
                      f'results/models/{model_name}_{system_name}_best.pt')
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}: train_loss = {train_loss:.6f}, '
                  f'val_loss = {val_loss:.6f}')
    
    return model, train_losses, val_losses

def plot_training_curves(train_losses, val_losses, model_name, system_name):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title(f'{model_name} on {system_name}')
    plt.legend()
    plt.savefig(f'results/plots/{model_name}_{system_name}_loss.png')
    plt.close()

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('results/models', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)
    
    # Systems to train on
    systems = ['van_der_pol', 'lotka_volterra']
    models = ['resnet', 'neural_ode', 'adaptive_resnet']
    
    # Train all models on all systems
    for system in systems:
        for model_name in models:
            print(f'\nTraining {model_name} on {system}')
            model, train_losses, val_losses = train_model(model_name, system)
            plot_training_curves(train_losses, val_losses, model_name, system)