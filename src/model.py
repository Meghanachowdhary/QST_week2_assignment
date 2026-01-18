import torch
import torch.nn as nn
import numpy as np

class QuantumStateReconstructor(nn.Module):
    def __init__(self, num_qubits, num_shadows, embed_dim=32, num_heads=2, num_layers=2):
        super().__init__()
        self.num_qubits = num_qubits
        self.dim = 2**num_qubits
        
       
        self.embedding = nn.Embedding(6, embed_dim)
        
    
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        

        self.output_dim = self.dim ** 2
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_dim)
        )

    def forward(self, x):
    
        x = x.flatten(start_dim=1)

        
        device = x.device
        x_bases = x[:, ::2]   
        x_outcomes = x[:, 1::2] 
        tokens = x_bases * 2 + x_outcomes 
        tokens = tokens.long()
        
      
        emb = self.embedding(tokens) 
        
        
        out = self.transformer(emb)
        
        
        pooled = out.mean(dim=1)
        
        
        params = self.head(pooled)
        
        
        L = self.params_to_lower_triangular(params)
        
       
        rho = self.compute_rho(L)
        
        return rho

    def params_to_lower_triangular(self, params):
        batch_size = params.shape[0]
        D = self.dim
        
        
        diag_params = params[:, :D]
        
        off_diag_params = params[:, D:]
        
      
        L = torch.zeros(batch_size, D, D, dtype=torch.cfloat, device=params.device)
        
  
        
        rows, cols = torch.tril_indices(D, D, offset=0)
        
        diag_indices = (rows == cols)
        off_diag_indices = (rows > cols)
        
        
        L[:, range(D), range(D)] = diag_params.to(torch.cfloat)
        
        
        num_off_diag = D * (D - 1) // 2
        real_part = off_diag_params[:, :num_off_diag]
        imag_part = off_diag_params[:, num_off_diag:]
        
        L[:, rows[off_diag_indices], cols[off_diag_indices]] = \
            real_part.to(torch.cfloat) + 1j * imag_part.to(torch.cfloat)
            
        return L

    def compute_rho(self, L):
        
        L_dagger = torch.conj(L).transpose(1, 2)
        rho_raw = torch.bmm(L, L_dagger)
        
        
        trace = torch.diagonal(rho_raw, dim1=-2, dim2=-1).sum(dim=-1)
        trace = trace.view(-1, 1, 1) + 1e-8 
        
        rho = rho_raw / trace
        return rho
