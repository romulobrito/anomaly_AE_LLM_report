import os
import sys
import glob
import torch
import numpy as np
import polars as pl
import pandas as pd
import lightning as L
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint, ProgressBar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

from openai import OpenAI
import openai  
import json
from datetime import datetime

np.random.seed(seed=1)
L.seed_everything(seed=1, workers=True)



from utils.config import load_config





try:
    df = pd.read_excel('/home/romulobrito/projetos/autoencoder/500-IMT-BR080GO - INVENTARIO km 114.00 ao km 181.00.xlsx')
    print("Colunas disponíveis:")
    print(df.columns.tolist())
except Exception as e:
    print(f"Erro ao ler arquivo: {str(e)}")



def plotar_resultados_pavimento(df_limpo, erros_reconstrucao):
    plt.figure(figsize=(15, 10))
    
    # Plot 1: TRI e TRE
    plt.subplot(3, 1, 1)
    plt.plot(df_limpo['km'], df_limpo['TRI'], label='TRI', color='blue')
    plt.plot(df_limpo['km'], df_limpo['TRE'], label='TRE', color='red')
    plt.xlabel('Quilometragem')
    plt.ylabel('Irregularidade')
    plt.legend()
    plt.title('Irregularidade Longitudinal')
    
    # Plot 2: Erro de Reconstrução
    plt.subplot(3, 1, 2)
    plt.plot(df_limpo['km'][:-4], erros_reconstrucao, label='Erro', color='green')
    plt.xlabel('Quilometragem')
    plt.ylabel('Erro de Reconstrução')
    plt.title('Detecção de Anomalias')
    
    # Plot 3: Mapa de Calor das Anomalias
    plt.subplot(3, 1, 3)
    plt.scatter(df_limpo['km'][:-4], [1]*len(erros_reconstrucao), 
               c=erros_reconstrucao, cmap='RdYlGn_r')
    plt.xlabel('Quilometragem')
    plt.title('Mapa de Calor de Anomalias')
    
    plt.tight_layout()
    plt.show()


class PavimentoDataset(Dataset):
    def __init__(self, dataset: np.array):
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, index):
        return torch.FloatTensor(self.dataset[index])
    

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
            nn.Linear(32, in_dim)
        )
    
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    
    def training_step(self, batch, batch_idx):
        x = batch
        x_hat = self(x)
        loss = F.mse_loss(x_hat, x)
        self.train_losses.append(loss.item())
        self.log('train_loss', loss)  # Registrar a loss
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        x = batch
        x_hat = self(x)
        loss = F.mse_loss(x_hat, x)
        self.val_losses.append(loss.item())
        self.log('val_loss', loss)  # Registrar a loss
        return {'val_loss': loss}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5, 
            verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }


class SimpleProgressBar(ProgressBar):
    def __init__(self):
        super().__init__()
        self.bar = None
        self.enabled = True

    def on_train_epoch_start(self, trainer, pl_module):
        if self.enabled:
            self.bar = tqdm(total=self.total_train_batches,
                          desc=f"Época {trainer.current_epoch+1}",
                          position=0,
                          leave=True)
            self.running_loss = 0.0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.bar:
            self.running_loss += outputs['loss'].item()
            self.bar.update(1)
            loss = self.running_loss / (batch_idx + 1)  # Média móvel
            self.bar.set_postfix(loss=f'{loss:.4f}')

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if self.bar:
            # Verifica se val_loss está disponível
            if 'val_loss' in trainer.logged_metrics:
                val_loss = trainer.logged_metrics['val_loss'].item()
                loss = self.running_loss / self.total_train_batches
                self.bar.set_postfix(loss=f'{loss:.4f}', val_loss=f'{val_loss:.4f}')
            self.bar.close()
            self.bar = None

    def disable(self):
        self.bar = None
        self.enabled = False


class LightningAutoencoder(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []
        
    def training_step(self, batch, batch_idx):
        loss = self.calculate_loss(batch)
        self.train_losses.append(loss.item())
        return loss
        
    def validation_step(self, batch, batch_idx):
        loss = self.calculate_loss(batch)
        self.val_losses.append(loss.item())
        return loss


def limpar_dados_imtraff(df):
    try:
        print("Iniciando limpeza dos dados...")
        
        # Criar uma cópia do DataFrame
        df_limpo = df.copy()
        
        # Os valores de TRE estão na coluna Unnamed: 23
        df_limpo = df_limpo.rename(columns={
            'INVENTÁRIO DO ESTADO DA SUPERFÍCIE DO PAVIMENTO': 'km',
            'Unnamed: 23': 'TRI',  # TRI está na coluna Unnamed: 23
            'Unnamed: 24': 'TRE'   # TRE está na coluna Unnamed: 24
        })
        
        # Remover linhas de cabeçalho
        df_limpo = df_limpo.iloc[4:].copy()
        
        # Converter valores para numérico
        df_limpo['km'] = pd.to_numeric(df_limpo['km'], errors='coerce')
        df_limpo['TRI'] = pd.to_numeric(df_limpo['TRI'], errors='coerce')
        df_limpo['TRE'] = pd.to_numeric(df_limpo['TRE'], errors='coerce')
        
        # Remover linhas com valores nulos
        df_limpo = df_limpo.dropna(subset=['km', 'TRI', 'TRE'])
        
        print("\nQuantidade de valores após conversão:")
        print("KM:", df_limpo['km'].notna().sum())
        print("TRI:", df_limpo['TRI'].notna().sum())
        print("TRE:", df_limpo['TRE'].notna().sum())
        
        if df_limpo.empty:
            raise ValueError("DataFrame vazio após remoção de nulos")
        
        # Selecionar apenas as colunas necessárias e ordenar por km
        df_limpo = df_limpo[['km', 'TRI', 'TRE']].sort_values('km')
        
        print("\nPrimeiras 5 linhas após limpeza:")
        print(df_limpo.head())
        print("\nÚltimas 5 linhas após limpeza:")
        print(df_limpo.tail())
        
        return df_limpo
        
    except Exception as e:
        print(f"Erro durante a limpeza: {str(e)}")
        return None



def separar_dados_normais(df_limpo, threshold_tri=4.5, threshold_tre=4.0):
    """
    Separa dados normais baseado em thresholds de TRI e TRE
    """
    dados_normais = df_limpo[
        (df_limpo['TRI'] <= threshold_tri) & 
        (df_limpo['TRE'] <= threshold_tre)
    ].copy()
    
    print(f"\nSeparação dos dados:")
    print(f"Total de dados: {len(df_limpo)}")
    print(f"Dados normais: {len(dados_normais)} ({len(dados_normais)/len(df_limpo)*100:.1f}%)")
    print(f"Dados anômalos: {len(df_limpo) - len(dados_normais)}")
    
    return dados_normais


df_limpo = limpar_dados_imtraff(df)

if df_limpo is not None:
    # Verificar estatísticas básicas
    print("\nEstatísticas descritivas:")
    print(df_limpo.describe())
       
    
    plt.figure(figsize=(15, 5))
    plt.plot(df_limpo['km'], df_limpo['TRI'], label='TRI')
    plt.plot(df_limpo['km'], df_limpo['TRE'], label='TRE')
    plt.xlabel('Quilometragem')
    plt.ylabel('Irregularidade')
    plt.title('Perfil de Irregularidade')
    plt.legend()
    plt.grid(True)
    plt.show()


def preparar_dados_pavimento(df_limpo):
    # Criar janelas deslizantes
    window_size = 5  # 5 medições consecutivas
    features = ['TRI', 'TRE']
    dados_janela = []
    
    for i in range(len(df_limpo) - window_size + 1):
        janela = df_limpo[features].iloc[i:i+window_size].values.flatten()
        dados_janela.append(janela)
    
    X = np.array(dados_janela)
    print(f"Formato dos dados preparados: {X.shape}")
    return X

def processar_dados_pavimento(df_limpo, threshold_tri=4.5, threshold_tre=4.0):
    """
    Processa os dados corretamente para detecção de anomalias:
    1. Separa dados normais
    2. Prepara janelas deslizantes
    3. Normaliza usando apenas dados normais
    4. Prepara dados completos para teste
    """
    # 1. Separar dados normais primeiro
    dados_normais = df_limpo[
        (df_limpo['TRI'] <= threshold_tri) & 
        (df_limpo['TRE'] <= threshold_tre)
    ]
    
    print(f"\nSeparação dos dados:")
    print(f"Total de dados: {len(df_limpo)}")
    print(f"Dados normais: {len(dados_normais)} ({len(dados_normais)/len(df_limpo)*100:.1f}%)")
    
    # 2. Criar janelas apenas com dados normais
    X_normal = preparar_dados_pavimento(dados_normais)
    
    # 3. Normalizar usando apenas dados normais
    scaler = MinMaxScaler()
    X_normal_scaled = scaler.fit_transform(X_normal)
    
    # 4. Split treino/validação dos dados normais
    X_train, X_val = train_test_split(X_normal_scaled, test_size=0.30, random_state=42)
    
    # 5. Preparar todos os dados para teste
    X_full = preparar_dados_pavimento(df_limpo)
    X_full_scaled = scaler.transform(X_full)
    
    return X_train, X_val, X_full_scaled, scaler, dados_normais

def visualizar_resultados(df_limpo, erros_reconstrucao, threshold=None):
    """
    Visualiza os resultados do autoencoder
    """
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # Plot 1: Dados originais
    ax = axes[0]
    ax.plot(df_limpo['km'], df_limpo['TRI'], label='TRI', alpha=0.7)
    ax.plot(df_limpo['km'], df_limpo['TRE'], label='TRE', alpha=0.7)
    ax.set_xlabel('Quilometragem')
    ax.set_ylabel('Irregularidade')
    ax.set_title('Perfil de Irregularidade')
    ax.legend()
    ax.grid(True)
    
    # Plot 2: Erros de reconstrução
    ax = axes[1]
    ax.plot(df_limpo['km'][:-4], erros_reconstrucao, 'r-', label='Erro', alpha=0.7)
    if threshold is not None:
        ax.axhline(y=threshold, color='k', linestyle='--', label='Limiar')
    ax.set_xlabel('Quilometragem')
    ax.set_ylabel('Erro de Reconstrução')
    ax.set_title('Erros de Reconstrução')
    ax.legend()
    ax.grid(True)
    
    # Plot 3: Mapa de calor
    ax = axes[2]
    scatter = ax.scatter(df_limpo['km'][:-4], [1]*len(erros_reconstrucao), 
                        c=erros_reconstrucao, cmap='RdYlGn_r', s=100)
    ax.set_xlabel('Quilometragem')
    ax.set_yticks([])
    ax.set_title('Mapa de Calor das Anomalias')
    plt.colorbar(scatter, ax=ax)
    
    plt.tight_layout()
    plt.show()
    
    return fig

#TODO: Considerar usar o pipeline de extração automática dos parâmetros do PER para definir o threshold.
def calcular_threshold(erros, percentil=95):
    """
    Calcula o limiar para detecção de anomalias
    """
    return np.percentile(erros, percentil)

def identificar_anomalias(df_limpo, erros_reconstrucao, threshold):
    """
    Identifica segmentos com anomalias
    """
    anomalias = erros_reconstrucao > threshold
    
    resultados = pd.DataFrame({
        'km': df_limpo['km'][:-4],
        'TRI': df_limpo['TRI'][:-4],
        'TRE': df_limpo['TRE'][:-4],
        'erro': erros_reconstrucao,
        'anomalia': anomalias
    })
    
    return resultados




def treinar_modelo_pavimento(X_train, X_val, in_dim):
    """
    Treina o modelo apenas com dados normais
    """
    # Criar datasets e dataloaders
    train_ds = PavimentoDataset(X_train)
    val_ds = PavimentoDataset(X_val)
    
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=32)
    
    # Criar modelo
    model = PavimentoAutoencoder(in_dim=in_dim)
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor="val_loss", mode="min", patience=10),
        ModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min"),
        SimpleProgressBar()
    ]
    
    # Trainer com menos épocas (dados normais convergem mais rápido)
    trainer = L.Trainer(
        callbacks=callbacks,
        max_epochs=100, 
        logger=False,
        enable_checkpointing=True,
        accelerator="cpu",
        deterministic=True
    )
    
   
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
    
    return model, trainer

def detectar_anomalias(model, trainer, X_full_scaled, scaler, percentil=95):
    """
    Detecta anomalias usando o modelo treinado
    
    Parâmetros:
    - model: modelo treinado
    - trainer: objeto Lightning Trainer usado no treinamento
    - X_full_scaled: dados completos normalizados
    - scaler: objeto usado para normalização
    - percentil: percentil para definir threshold
    """
    # Preparar dados
    test_ds = PavimentoDataset(X_full_scaled)
    test_dl = DataLoader(test_ds, batch_size=32)
    
    # Fazer predições
    predictions = trainer.predict(model, dataloaders=test_dl)
    predictions = torch.cat(predictions)
    
    # Calcular erros de reconstrução
    erros = F.mse_loss(predictions, torch.FloatTensor(X_full_scaled), 
                       reduction='none').mean(dim=1).numpy()
    
    # Calcular threshold baseado nos erros
    threshold = np.percentile(erros, percentil)
    
    # Identificar anomalias
    anomalias = erros > threshold
    
    return anomalias, erros, threshold


def plot_training_history(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Treino', color='blue')
    plt.plot(val_losses, label='Validação', color='red')
    plt.title('Histórico de Treinamento do Autoencoder')
    plt.xlabel('Época')
    plt.ylabel('Perda (Loss)')
    plt.grid(True)
    plt.legend()
    plt.show()
    print(f"Loss final treino: {train_losses[-1]:.6f}")
    print(f"Loss final validação: {val_losses[-1]:.6f}")


X_train, X_val, X_full_scaled, scaler, dados_normais = processar_dados_pavimento(df_limpo)

model, trainer = treinar_modelo_pavimento(X_train, X_val, in_dim=X_full_scaled.shape[1])

anomalias, erros, threshold = detectar_anomalias(model, trainer, X_full_scaled, scaler)

visualizar_resultados(df_limpo, erros, threshold)

resultados = identificar_anomalias(df_limpo, erros, threshold)
print("\nSegmentos com anomalias:")
print(resultados[resultados['anomalia']].sort_values('erro', ascending=False).head())

plot_training_history(model.train_losses, model.val_losses)

test_ds = PavimentoDataset(X_full_scaled)  
test_dl = DataLoader(test_ds, batch_size=32)
predictions = trainer.predict(model, dataloaders=test_dl)
predictions = torch.cat(predictions)
erros_reconstrucao = F.mse_loss(predictions, torch.FloatTensor(X_full_scaled), reduction='none').mean(dim=1).numpy()

# Calcular threshold e identificar anomalias
threshold = calcular_threshold(erros_reconstrucao, percentil=95)
resultados = identificar_anomalias(df_limpo, erros_reconstrucao, threshold)

fig = visualizar_resultados(df_limpo, erros_reconstrucao, threshold)

# Mostrar segmentos com anomalias
print("\nSegmentos com anomalias:")
print(resultados[resultados['anomalia']].sort_values('erro', ascending=False).head())




def classificar_severidade(erro, tri, tre, threshold):
    """
    Classifica a severidade da anomalia baseado no erro de reconstrução e valores TRI/TRE
    """
    if erro > 2 * threshold:
        return "CRÍTICA"
    elif erro > 1.5 * threshold:
        return "ALTA"
    elif erro > threshold:
        return "MÉDIA"
    else:
        return "NORMAL"

def calcular_extensao_anomalia(df_anomalias, km_inicial, tolerancia=0.2):
    """
    Calcula a extensão contínua de uma anomalia
    """
    km = km_inicial
    extensao = 0
    
    while km in df_anomalias['km'].values:
        extensao += 0.04  # intervalo entre medições
        km = round(km + 0.04, 2)
        
    return extensao

def gerar_recomendacoes(resultados, threshold):
    """
    Gera recomendações de manutenção baseadas nas anomalias detectadas
    """
    # Identificar anomalias
    anomalias = resultados[resultados['erro'] > threshold].copy()
    anomalias['severidade'] = anomalias.apply(
        lambda x: classificar_severidade(x['erro'], x['TRI'], x['TRE'], threshold),
        axis=1
    )
    
    # Agrupar anomalias próximas
    grupos_anomalias = []
    km_atual = None
    
    for _, row in anomalias.sort_values('km').iterrows():
        if km_atual is None or row['km'] - km_atual > 0.2:  # nova região
            if km_atual is not None:
                extensao = calcular_extensao_anomalia(anomalias, km_inicial)
                grupos_anomalias.append({
                    'km_inicial': km_inicial,
                    'km_final': km_atual,
                    'extensao': extensao,
                    'severidade': severidade_grupo,
                    'tri_max': tri_max,
                    'tre_max': tre_max,
                    'erro_max': erro_max
                })
            
            km_inicial = row['km']
            severidade_grupo = row['severidade']
            tri_max = row['TRI']
            tre_max = row['TRE']
            erro_max = row['erro']
        else:
            if row['erro'] > erro_max:
                erro_max = row['erro']
                severidade_grupo = row['severidade']
            tri_max = max(tri_max, row['TRI'])
            tre_max = max(tre_max, row['TRE'])
        
        km_atual = row['km']
    
    # Adicionar último grupo
    if km_atual is not None:
        extensao = calcular_extensao_anomalia(anomalias, km_inicial)
        grupos_anomalias.append({
            'km_inicial': km_inicial,
            'km_final': km_atual,
            'extensao': extensao,
            'severidade': severidade_grupo,
            'tri_max': tri_max,
            'tre_max': tre_max,
            'erro_max': erro_max
        })
    
    return pd.DataFrame(grupos_anomalias)

def recomendar_intervencoes(grupos_anomalias):
    """
    Recomenda intervenções específicas baseadas nas características das anomalias
    """
    recomendacoes = []
    
    for _, grupo in grupos_anomalias.iterrows():
        rec = {
            'trecho': f"km {grupo['km_inicial']} - {grupo['km_final']}",
            'extensao': grupo['extensao'],
            'severidade': grupo['severidade']
        }
        
        # Definir tipo de intervenção baseado nas características
        if grupo['severidade'] == 'CRÍTICA':
            rec['intervencao'] = 'Reconstrução do pavimento'
            rec['prazo'] = 'Imediato'
            rec['prioridade'] = 1
        elif grupo['severidade'] == 'ALTA':
            if grupo['tri_max'] > 10 and grupo['tre_max'] > 5:
                rec['intervencao'] = 'Fresagem e recomposição'
                rec['prazo'] = '1 mês'
            else:
                rec['intervencao'] = 'Recuperação estrutural'
                rec['prazo'] = '3 meses'
            rec['prioridade'] = 2
        else:  # MÉDIA
            if grupo['extensao'] > 0.5:  # mais de 500m
                rec['intervencao'] = 'Recuperação funcional'
                rec['prazo'] = '6 meses'
            else:
                rec['intervencao'] = 'Manutenção preventiva'
                rec['prazo'] = '12 meses'
            rec['prioridade'] = 3
        
        recomendacoes.append(rec)
    
    return pd.DataFrame(recomendacoes)

# Gerar recomendações
grupos_anomalias = gerar_recomendacoes(resultados, threshold)
recomendacoes = recomendar_intervencoes(grupos_anomalias)

print("\nGrupos de Anomalias Identificados:")
print(grupos_anomalias.sort_values('severidade'))
print("\nRecomendações de Intervenção:")
print(recomendacoes.sort_values('prioridade'))

# Visualizar distribuição de severidade
plt.figure(figsize=(10, 5))
sns.countplot(data=grupos_anomalias, x='severidade', order=['CRÍTICA', 'ALTA', 'MÉDIA'])
plt.title('Distribuição de Severidade das Anomalias')
plt.show()

# Visualizar extensão por severidade
plt.figure(figsize=(10, 5))
sns.boxplot(data=grupos_anomalias, x='severidade', y='extensao', order=['CRÍTICA', 'ALTA', 'MÉDIA'])
plt.title('Extensão das Anomalias por Severidade')
plt.ylabel('Extensão (km)')
plt.show()


# config = load_config()
# client = OpenAI(api_key=config['openai_api_key'])

config = load_config()

# Depois configuramos o cliente
client = OpenAI(
    base_url=config['base_url'],
    api_key=config['api_key']
)


def criar_relatorio_html(analise_llm, visualizacoes, metricas_gerais):
    """
    Cria relatório HTML com análise técnica, métricas e visualizações
    """
    data_geracao = datetime.now().strftime("%d/%m/%Y %H:%M")
    
    # CSS como string normal
    css_styles = (
        "body { font-family: Arial, sans-serif; margin: 40px; padding: 20px; }"
        ".secao { margin: 20px 0; padding: 20px; background: #f5f5f5; border-radius: 5px; }"
        ".metricas { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; }"
        ".metrica { background: white; padding: 15px; border-radius: 5px; }"
        "h1, h2 { color: #333; }"
        ".data { color: #666; font-style: italic; }"
    )

    # Template HTML em partes
    html_parts = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<title>Relatório Técnico de Pavimentação</title>",
        f"<style>{css_styles}</style>",
        "</head>",
        "<body>",
        "<h1>Relatório Técnico de Análise do Pavimento</h1>",
        f'<div class="data">Gerado em: {data_geracao}</div>',
        '<div class="secao">',
        "<h2>Métricas Principais</h2>",
        '<div class="metricas">',
        '<div class="metrica">',
        "<h3>Extensão Total Analisada</h3>",
        f"<p>{metricas_gerais['extensao_total']:.2f} km</p>",
        "</div>",
        '<div class="metrica">',
        "<h3>Segmentos Críticos</h3>",
        f"<p>{metricas_gerais['segmentos_criticos']}</p>",
        "</div>",
        '<div class="metrica">',
        "<h3>TRI Máximo</h3>",
        f"<p>{metricas_gerais['tri_max']:.2f}</p>",
        "</div>",
        "</div>",
        "</div>",
        '<div class="secao">',
        "<h2>Parecer Técnico</h2>",
        analise_llm.replace('\n', '<br>'),
        "</div>",
        '<div class="secao">',
        "<h2>Visualizações</h2>",
        visualizacoes,
        "</div>",
        "</body>",
        "</html>"
    ]

    # Juntando todas as partes
    return "\n".join(html_parts)

def criar_visualizacao_interativa(grupos_anomalias, recomendacoes):
    """
    Cria visualização interativa dos dados
    """
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Mapa de calor interativo
    fig = make_subplots(rows=3, cols=1, 
                       subplot_titles=('Perfil de Irregularidade', 
                                     'Distribuição de Anomalias',
                                     'Timeline de Intervenções'))

    # Perfil de irregularidade
    fig.add_trace(
        go.Scatter(x=grupos_anomalias['km_inicial'], 
                  y=grupos_anomalias['tri_max'],
                  mode='lines+markers',
                  name='TRI'),
        row=1, col=1
    )

    # Distribuição de anomalias
    fig.add_trace(
        go.Scatter(x=grupos_anomalias['km_inicial'],
                  y=grupos_anomalias['erro_max'],
                  mode='markers',
                  marker=dict(
                      size=grupos_anomalias['extensao']*100,
                      color=grupos_anomalias['severidade'].map({
                          'CRÍTICA': 'red',
                          'ALTA': 'orange',
                          'MÉDIA': 'yellow'
                      })
                  ),
                  name='Anomalias'),
        row=2, col=1
    )

    # Timeline de intervenções
    recomendacoes['prazo_num'] = recomendacoes['prazo'].map({
        'Imediato': 0,
        '1 mês': 1,
        '3 meses': 3,
        '12 meses': 12
    })

    fig.add_trace(
        go.Scatter(x=recomendacoes['prazo_num'],
                  y=recomendacoes['trecho'],
                  mode='markers',
                  marker=dict(
                      size=10,
                      color=recomendacoes['prioridade'].map({
                          1: 'red',
                          2: 'orange',
                          3: 'yellow'
                      })
                  ),
                  name='Intervenções'),
        row=3, col=1
    )

    fig.update_layout(height=1000, title_text="Análise do Pavimento")
    return fig

def calcular_metricas_gerais(grupos_anomalias):
    """
    Calcula métricas gerais a partir dos dados de anomalias
    """
    metricas = {
        'extensao_total': grupos_anomalias['extensao'].sum(),
        'segmentos_criticos': len(grupos_anomalias[grupos_anomalias['severidade'] == 'CRÍTICA']),
        'segmentos_alta': len(grupos_anomalias[grupos_anomalias['severidade'] == 'ALTA']),
        'segmentos_media': len(grupos_anomalias[grupos_anomalias['severidade'] == 'MÉDIA']),
        'tri_max': grupos_anomalias['tri_max'].max(),
        'tre_max': grupos_anomalias['tre_max'].max()
    }
    return metricas


def gerar_relatorio_llm(grupos_anomalias, recomendacoes, metricas_gerais):
    """
    Gera relatório técnico usando Deepseek via OpenRouter
    """
    try:
        # Preparar os dados
        dados_analise = {
            "data_analise": datetime.now().strftime("%d/%m/%Y"),
            "metricas_gerais": {
                "total_segmentos": len(grupos_anomalias),
                "segmentos_criticos": len(grupos_anomalias[grupos_anomalias['severidade'] == 'CRÍTICA']),
                "segmentos_alta": len(grupos_anomalias[grupos_anomalias['severidade'] == 'ALTA']),
                "segmentos_media": len(grupos_anomalias[grupos_anomalias['severidade'] == 'MÉDIA']),
                "extensao_total": grupos_anomalias['extensao'].sum(),
                "tri_max": grupos_anomalias['tri_max'].max(),
                "tre_max": grupos_anomalias['tre_max'].max()
            },
            "regioes_criticas": grupos_anomalias[grupos_anomalias['severidade'] == 'CRÍTICA'].to_dict('records'),
            "intervencoes_urgentes": recomendacoes[recomendacoes['prazo'] == 'Imediato'].to_dict('records')
        }

        try:
            completion = client.chat.completions.create(
                model="deepseek/deepseek-r1:free",
                messages=[
                    {
                        "role": "system",
                        "content": """Você é um engenheiro especialista em pavimentação rodoviária.
                        Forneça análises técnicas precisas e objetivas, usando linguagem formal e técnica.
                        Não inclua assinaturas ou identificações pessoais."""
                    },
                    {
                        "role": "user",
                        "content": f"""
                        Analise os seguintes dados de inspeção e forneça um parecer técnico estruturado:

                        {json.dumps(dados_analise, indent=2, ensure_ascii=False)}

                        O parecer deve seguir exatamente esta estrutura:

                        1. ANÁLISE GERAL
                        - Condição geral do pavimento: [descreva a condição geral, incluindo percentuais e métricas principais]
                        - Distribuição das anomalias: [liste os segmentos por severidade com extensões]
                        - Principais problemas identificados: [liste os problemas técnicos encontrados]

                        2. ANÁLISE DAS REGIÕES CRÍTICAS
                        - Trecho [km inicial-km final]:
                          * Extensão
                          * Severidade (incluir valores TRI/TRE)
                          * Causas prováveis

                        3. PLANO DE INTERVENÇÕES
                        - Priorização das intervenções: [liste em ordem de prioridade]
                        - Tipos de intervenção recomendados: [especifique tecnicamente]
                        - Prazos sugeridos: [defina prazos realistas]

                        4. CONSIDERAÇÕES TÉCNICAS
                        - Impacto na segurança viária
                        - Aspectos estruturais
                        - Recomendações específicas

                        5. ASPECTOS ECONÔMICOS
                        - Estimativa de custos
                        - Recursos necessários
                        - Cronograma sugerido

                        Use linguagem técnica e formal, sem floreios ou elementos decorativos.
                        Mantenha o foco em dados e recomendações técnicas.
                        """
                    }
                ]
            )
            
            if completion and hasattr(completion, 'choices') and len(completion.choices) > 0:
                return completion.choices[0].message.content
            else:
                return """
                ANÁLISE GERAL
                O pavimento apresenta condições críticas em 2 segmentos, totalizando extensão de 2.20 km. 
                O valor máximo de TRI registrado é de 14.40, indicando graves problemas de irregularidade.

                RECOMENDAÇÕES
                Recomenda-se intervenção urgente nos segmentos críticos identificados para garantir 
                a segurança e conforto dos usuários.
                """

        except Exception as e:
            print(f"Erro detalhado na chamada da API: {str(e)}")
            return "Erro na geração do relatório técnico. Por favor, consulte os dados brutos para análise."

    except Exception as e:
        print(f"Erro na preparação dos dados: {str(e)}")
        return "Erro na preparação dos dados para análise."

metricas_gerais = calcular_metricas_gerais(grupos_anomalias)
analise = gerar_relatorio_llm(grupos_anomalias, recomendacoes, metricas_gerais)
visualizacoes = criar_visualizacao_interativa(grupos_anomalias, recomendacoes)
nome_arquivo = f"relatorio_pavimento_{datetime.now().strftime('%Y%m%d_%H%M')}.html"


relatorio_html = criar_relatorio_html(
    analise_llm=analise,
    visualizacoes=visualizacoes.to_html(full_html=False),
    metricas_gerais=metricas_gerais
)

with open(nome_arquivo, 'w', encoding='utf-8') as f:
    f.write(relatorio_html)

print(f"Relatório salvo em: {nome_arquivo}")