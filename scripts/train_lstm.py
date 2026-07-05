import sys
from pathlib import Path

# Add project root directory to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from src.config import Config
from src.models.lstm_autoencoder import LSTMAutoencoder, calculate_reconstruction_error

def generate_normal_data(steps=50000):
    """Generates synthetic normal time-series data for Gas, Temperature, and Pressure.
    Reads occur every 5 seconds.
    - Gas: mean ~18.0 LEL%, random walk with reversion
    - Temp: mean ~28.0 C, random walk with reversion
    - Pressure: mean ~1.2 bar, random walk with reversion
    """
    np.random.seed(42)
    
    # Initialize baselines
    gas = 18.0
    temp = 28.0
    pressure = 1.2
    
    data = []
    for _ in range(steps):
        # Random walks with mean reversion to simulate normal processes
        gas += np.random.normal(0, 0.05) + (18.0 - gas) * 0.01
        temp += np.random.normal(0, 0.02) + (28.0 - temp) * 0.01
        pressure += np.random.normal(0, 0.005) + (1.2 - pressure) * 0.01
        
        data.append([gas, temp, pressure])
        
    return np.array(data)

def create_dataset(data, window_size=12):
    """Creates sliding windows of length window_size."""
    X = []
    for i in range(len(data) - window_size + 1):
        X.append(data[i:i+window_size])
    return np.array(X)

def main():
    print("--- Starting LSTM Autoencoder Training Pipeline ---")
    
    # 1. Generate normal training data (72h equivalent ~ 51,840 steps)
    # For speed during build, we will generate 40,000 steps which is plenty for training
    print("Generating synthetic normal data...")
    raw_data = generate_normal_data(steps=40000)
    print(f"Generated data shape: {raw_data.shape}")
    
    # 2. Normalize raw data using standard scaling
    means = np.mean(raw_data, axis=0)
    stds = np.std(raw_data, axis=0)
    # Avoid zero division
    stds[stds == 0] = 1.0
    
    normalized_data = (raw_data - means) / stds
    print(f"Normalized data means: {np.mean(normalized_data, axis=0)}")
    print(f"Normalized data stds: {np.std(normalized_data, axis=0)}")
    
    # Save training history data to disk for visualization/referencing later
    np.save(str(Config.DATA_DIR / "normal_sensor_data.npy"), raw_data)
    
    # 3. Slice into sliding windows
    print(f"Creating sliding windows of size {Config.LSTM_WINDOW_SIZE}...")
    windows = create_dataset(normalized_data, window_size=Config.LSTM_WINDOW_SIZE)
    print(f"Dataset windowed shape: {windows.shape}")
    
    # 4. Train-Test Split (80% train, 20% validation)
    split_idx = int(len(windows) * 0.8)
    train_windows = windows[:split_idx]
    val_windows = windows[split_idx:]
    
    # Convert to PyTorch tensors
    train_tensor = torch.tensor(train_windows, dtype=torch.float32)
    val_tensor = torch.tensor(val_windows, dtype=torch.float32)
    
    train_loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=Config.LSTM_BATCH_SIZE,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(val_tensor),
        batch_size=Config.LSTM_BATCH_SIZE,
        shuffle=False
    )
    
    # 5. Initialize LSTM Autoencoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using training device: {device}")
    
    model = LSTMAutoencoder(
        seq_len=Config.LSTM_WINDOW_SIZE,
        no_features=Config.LSTM_FEATURES,
        embedding_dim=Config.LSTM_HIDDEN_DIM
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LSTM_LR)
    
    # 6. Training Loop
    epochs = 15 # Lower epochs for fast compilation in hackathon environment (can be adjusted)
    print(f"Training model for {epochs} epochs...")
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x = batch[0].to(device)
            
            optimizer.zero_grad()
            reconstructed = model(x)
            loss = criterion(reconstructed, x)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * x.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                reconstructed = model(x)
                loss = criterion(reconstructed, x)
                val_loss += loss.item() * x.size(0)
        val_loss /= len(val_loader.dataset)
        
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")
            
    # 7. Threshold Calibration (mean + 3 * std on validation set)
    print("Calibrating anomaly threshold...")
    val_errors = calculate_reconstruction_error(model, val_loader, criterion, device=device)
    mean_val_err = np.mean(val_errors)
    std_val_err = np.std(val_errors)
    
    # Formula: Mean + 3 * Std Deviation
    threshold = mean_val_err + 3 * std_val_err
    print(f"Calibration results: Mean error = {mean_val_err:.6f}, Std error = {std_val_err:.6f}")
    print(f"Calibrated Anomaly Threshold (Mean + 3*Std) = {threshold:.6f}")
    
    # 8. Save Model Weights and Metadata
    print(f"Saving model artifacts to {Config.LSTM_MODEL_PATH}...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'threshold': float(threshold),
        'means': means.tolist(),
        'stds': stds.tolist(),
        'window_size': Config.LSTM_WINDOW_SIZE,
        'embedding_dim': Config.LSTM_HIDDEN_DIM,
        'no_features': Config.LSTM_FEATURES
    }, Config.LSTM_MODEL_PATH)
    print("Training complete and weights successfully saved!")

if __name__ == "__main__":
    main()
