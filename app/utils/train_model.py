import torch
import lightning as L
from app.models.autoencoder import PavimentoAutoencoder
from app.models.dataset import PavimentoDataset
from app.utils.model_manager import ModelManager
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
import os

def train_autoencoder(data_path):
    """
    Treina o autoencoder com dados normais
    """
    try:
        # Carregar e preparar dados
        manager = ModelManager()
        df = pd.read_excel(data_path)
        df_limpo = manager.limpar_dados_imtraff(df)
        X_full_scaled, dados_normais = manager.processar_dados_pavimento(df_limpo)
        
        # Criar datasets
        train_size = int(0.8 * len(X_full_scaled))
        train_data = X_full_scaled[:train_size]
        val_data = X_full_scaled[train_size:]
        
        train_ds = PavimentoDataset(train_data)
        val_ds = PavimentoDataset(val_data)
        
        train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=32)
        
        # Configurar modelo
        model = PavimentoAutoencoder(in_dim=10)  # 5 janelas x 2 features
        
        # Callbacks
        checkpoint_callback = ModelCheckpoint(
            dirpath='checkpoints',
            filename='autoencoder',
            save_top_k=1,
            monitor='val_loss'
        )
        
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=10,
            mode='min'
        )
        
        # Treinar
        trainer = L.Trainer(
            max_epochs=100,
            callbacks=[checkpoint_callback, early_stop_callback],
            accelerator='auto'
        )
        
        trainer.fit(model, train_dl, val_dl)
        print(f"Modelo salvo em: {checkpoint_callback.best_model_path}")
        
    except Exception as e:
        print(f"Erro durante treinamento: {str(e)}")

if __name__ == "__main__":
    # Criar diretório para checkpoints se não existir
    os.makedirs("checkpoints", exist_ok=True)
    
    # Treinar modelo
    data_path = "500-IMT-BR080GO - INVENTARIO km 114.00 ao km 181.00.xlsx"
    train_autoencoder(data_path) 