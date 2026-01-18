import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import time
from scipy.linalg import sqrtm

from src.data import QuantumDataset
from src.model import QuantumStateReconstructor


def quantum_fidelity(rho_pred, rho_true):
    """
    Computes Quantum Fidelity F(rho_pred, rho_true) = (Tr(sqrt(sqrt(rho_pred) * rho_true * sqrt(rho_pred))))^2
    """
    fidelities = []
    
   
    rho_pred = rho_pred.detach().cpu().numpy()
    rho_true = rho_true.detach().cpu().numpy()
    
    for i in range(rho_pred.shape[0]):
        rp = rho_pred[i]
        rt = rho_true[i]
        
        try:
            
            srp = sqrtm(rp)
            
            temp = srp @ rt @ srp
            
            stemp = sqrtm(temp)
            
            tr = np.trace(stemp)
            
            fid = np.real(tr) ** 2
            fidelities.append(fid)
        except Exception:
            
            fidelities.append(0.0)
        
    return np.mean(fidelities)

def trace_distance(rho_pred, rho_true):
    """
    Computes Trace Distance D = 1/2 * Tr(|rho_pred - rho_true|)
    """
    rho_pred = rho_pred.detach().cpu().numpy()
    rho_true = rho_true.detach().cpu().numpy()
    
    distances = []
    for i in range(rho_pred.shape[0]):
        diff = rho_pred[i] - rho_true[i]
        
        
        try:
            evals = np.linalg.eigvalsh(diff)
            dist = 0.5 * np.sum(np.abs(evals))
            distances.append(dist)
        except Exception:
            distances.append(1.0) 
            
    return np.mean(distances)


def train_and_test(num_qubits=2, num_shadows=50, batch_size=32, epochs=20, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    
    print("\n--- Generating Datasets ---")
    train_dataset = QuantumDataset(num_qubits, num_samples=1000, num_shadows=num_shadows)
    val_dataset = QuantumDataset(num_qubits, num_samples=200, num_shadows=num_shadows)
    test_dataset = QuantumDataset(num_qubits, num_samples=500, num_shadows=num_shadows) 
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
   
    model = QuantumStateReconstructor(num_qubits, num_shadows).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    def frobenius_loss(pred, target):
        return torch.norm(pred - target) ** 2 / pred.size(0)

    best_val_fidelity = 0.0
    os.makedirs('outputs', exist_ok=True)
    save_path = f"outputs/model_q{num_qubits}.pth"
    
   
    print(f"\n--- Starting Training ({epochs} epochs) ---")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            shadows = batch['shadows'].to(device)
            rho_true = batch['rho'].to(device)
            
            optimizer.zero_grad()
            rho_pred = model(shadows)
            
            loss = frobenius_loss(rho_pred, rho_true)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        
        
        model.eval()
        val_fidelity = 0.0
        with torch.no_grad():
            for batch in val_loader:
                s = batch['shadows'].to(device)
                r = batch['rho'].to(device)
                p = model(s)
                val_fidelity += quantum_fidelity(p, r)
        
        val_fidelity /= len(val_loader)
        
        print(f"Epoch {epoch+1}: Loss {avg_loss:.4f} | Val Fidelity {val_fidelity:.4f}")
        
        if val_fidelity > best_val_fidelity:
            best_val_fidelity = val_fidelity
            torch.save(model.state_dict(), save_path)
            
    print(f"Training Complete. Best Val Fidelity: {best_val_fidelity:.4f}")
    
   
    print("\n--- Running Final Evaluation on Test Set ---")
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()
    
    fidelities = []
    trace_dists = []
    latencies = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            shadows = batch['shadows'].to(device)
            rho_true = batch['rho'].to(device)
            
            
            start = time.time()
            rho_pred = model(shadows)
            end = time.time()
            latencies.append((end - start) / shadows.size(0))
            
            
            fidelities.append(quantum_fidelity(rho_pred, rho_true))
            trace_dists.append(trace_distance(rho_pred, rho_true))
            
    mean_fid = np.mean(fidelities)
    mean_td = np.mean(trace_dists)
    mean_lat = np.mean(latencies) * 1000 
    
    print("\n" + "="*40)
    print("FINAL TEST RESULTS")
    print("="*40)
    print(f"Mean Fidelity (Test Set):    {mean_fid:.4f}")
    print(f"Mean Trace Distance:         {mean_td:.4f}")
    print(f"Inference Latency:           {mean_lat:.2f} ms")
    print("="*40 + "\n")

if __name__ == "__main__":
    train_and_test()
