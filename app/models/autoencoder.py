import torch
from torch import nn
import torch.nn.functional as F
import lightning as L

class PavimentoAutoencoder(L.LightningModule):
    def __init__(self, in_dim):
        super().__init__()
        self.save_hyperparameters()
        self.train_losses = []
        self.val_losses = []
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.BatchNorm1d(32),
            nn.SELU(),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.SELU(),
            nn.Linear(16, 8),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.BatchNorm1d(16),
            nn.SELU(),
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.SELU(),
            nn.Linear(32, in_dim),
        )
    
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    
    def training_step(self, batch, batch_idx):
        x = batch
        x_hat = self(x)
        loss = F.mse_loss(x_hat, x)
        self.train_losses.append(loss.item())
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch
        x_hat = self(x)
        val_loss = F.mse_loss(x_hat, x)
        self.val_losses.append(val_loss.item())
        self.log('val_loss', val_loss)
        return val_loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3) 