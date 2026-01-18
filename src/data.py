import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.stats import unitary_group

# Pauli Matrices
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI_BASIS = [X, Y, Z]

def generate_random_density_matrix(num_qubits):
    """Generates a random valid density matrix (Hermitian, PSD, Unit Trace)."""
    dim = 2**num_qubits
    U = unitary_group.rvs(dim)
    eigenvalues = np.random.rand(dim)
    eigenvalues /= np.sum(eigenvalues)
    rho = U @ np.diag(eigenvalues) @ U.conj().T
    return rho

def get_pauli_string(num_qubits, index):
    """Convert integer index to Pauli string indices."""
    indices = []
    for _ in range(num_qubits):
        indices.append(index % 3)
        index //= 3
    return indices[::-1]

def measure_projection(rho, basis_indices):
    """Simulates a measurement of rho in the specified Pauli basis."""
    M = np.array([1], dtype=complex)
    for idx in basis_indices:
        M = np.kron(M, PAULI_BASIS[idx])
    
    # Basis rotation logic
    dim = rho.shape[0]
    num_qubits = int(np.log2(dim))
    
    U_rot = np.array([1], dtype=complex)
    for idx in basis_indices:
        if idx == 0:
            u = 1/np.sqrt(2) * np.array([[1, 1], [1, -1]], dtype=complex)
        elif idx == 1:
            u = 1/np.sqrt(2) * np.array([[1, -1j], [1, 1j]], dtype=complex)
        else:
            u = np.eye(2, dtype=complex)
        U_rot = np.kron(U_rot, u)
        
    rho_rot = U_rot @ rho @ U_rot.conj().T
    
    probs = np.real(np.diag(rho_rot))
    probs = np.clip(probs, 0, 1)
    probs /= np.sum(probs)
    
    outcome_idx = np.random.choice(dim, p=probs)
    outcome_bits = [(outcome_idx >> i) & 1 for i in range(num_qubits)][::-1]
    
    return outcome_bits

class QuantumDataset(Dataset):
    def __init__(self, num_qubits=2, num_samples=1000, num_shadows=100):
        self.num_qubits = num_qubits
        self.data = []
        
        print(f"Generating {num_samples} density matrices...")
        for _ in range(num_samples):
            rho = generate_random_density_matrix(num_qubits)
            
            shadows = []
            for _ in range(num_shadows):
                basis_indices = np.random.randint(0, 3, size=num_qubits)
                outcomes = measure_projection(rho, basis_indices)
                
                feature = []
                for b, o in zip(basis_indices, outcomes):
                    feature.extend([b, o])
                shadows.append(feature)
            
            shadows = np.array(shadows) 
            
            self.data.append({
                'shadows': torch.tensor(shadows, dtype=torch.float32),
                'rho': torch.tensor(rho, dtype=torch.cfloat)
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

if __name__ == "__main__":
    ds = QuantumDataset(num_qubits=2, num_samples=10, num_shadows=50)
    print("Dataset created. Sample 0 shadows shape:", ds[0]['shadows'].shape)
