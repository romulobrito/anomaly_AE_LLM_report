from openai import OpenAI
from app.utils.model_config import model_settings
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from typing import Dict, List
import os
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from app.models.autoencoder import PavimentoAutoencoder
from app.models.dataset import PavimentoDataset

class ModelManager:
    def __init__(self):
        self.client = OpenAI(
            base_url=model_settings.BASE_URL,
            api_key=model_settings.OPENROUTER_API_KEY
        )
        self.autoencoder = None
        self.model_loaded = False
        self.scaler = MinMaxScaler()
    
    def limpar_dados_imtraff(self, df):
        """
        Limpa e prepara os dados do Excel da IMTraff
        """
        try:
            print("Iniciando limpeza dos dados...")
            print("Colunas originais:", df.columns.tolist())
            
            # Criar uma cópia do DataFrame
            df_limpo = df.copy()
            
            # Identificar as colunas corretas
            km_col = 'INVENTÁRIO DO ESTADO DA SUPERFÍCIE DO PAVIMENTO'
            tri_col = 'Unnamed: 23'  # Coluna do TRI
            tre_col = 'Unnamed: 24'  # Coluna do TRE
            
            # Verificar se as colunas existem
            if not all(col in df_limpo.columns for col in [km_col, tri_col, tre_col]):
                raise ValueError(f"Colunas necessárias não encontradas. Colunas disponíveis: {df_limpo.columns.tolist()}")
            
            # Renomear colunas
            df_limpo = df_limpo.rename(columns={
                km_col: 'km',
                tri_col: 'TRI',
                tre_col: 'TRE'
            })
            
            # Remover linhas de cabeçalho (primeiras 4 linhas)
            df_limpo = df_limpo.iloc[4:].copy()
            
            # Converter valores para numérico
            df_limpo['km'] = pd.to_numeric(df_limpo['km'], errors='coerce')
            df_limpo['TRI'] = pd.to_numeric(df_limpo['TRI'], errors='coerce')
            df_limpo['TRE'] = pd.to_numeric(df_limpo['TRE'], errors='coerce')
            
            # Remover linhas com valores nulos
            df_limpo = df_limpo.dropna(subset=['km', 'TRI', 'TRE'])
            
            print("\nQuantidade de valores após limpeza:")
            print("KM:", df_limpo['km'].notna().sum())
            print("TRI:", df_limpo['TRI'].notna().sum())
            print("TRE:", df_limpo['TRE'].notna().sum())
            
            if df_limpo.empty:
                raise ValueError("DataFrame vazio após remoção de nulos")
            
            # Selecionar apenas as colunas necessárias e ordenar por km
            df_limpo = df_limpo[['km', 'TRI', 'TRE']].sort_values('km')
            
            print("\nPrimeiras linhas após limpeza:")
            print(df_limpo.head())
            
            return df_limpo
            
        except Exception as e:
            raise Exception(f"Erro durante a limpeza dos dados: {str(e)}")

    def preparar_dados_pavimento(self, df_limpo):
        """
        Prepara os dados para o autoencoder usando janelas deslizantes
        """
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

    def processar_dados_pavimento(self, df_limpo, threshold_tri=4.5, threshold_tre=4.0):
        """
        Processa os dados para detecção de anomalias
        """
        # Separar dados normais
        dados_normais = df_limpo[
            (df_limpo['TRI'] <= threshold_tri) & 
            (df_limpo['TRE'] <= threshold_tre)
        ]
        
        print(f"\nSeparação dos dados:")
        print(f"Total de dados: {len(df_limpo)}")
        print(f"Dados normais: {len(dados_normais)} ({len(dados_normais)/len(df_limpo)*100:.1f}%)")
        
        # Criar janelas com dados normais
        X_normal = self.preparar_dados_pavimento(dados_normais)
        
        # Normalizar usando dados normais
        X_normal_scaled = self.scaler.fit_transform(X_normal)
        
        # Preparar todos os dados para teste
        X_full = self.preparar_dados_pavimento(df_limpo)
        X_full_scaled = self.scaler.transform(X_full)
        
        return X_full_scaled, dados_normais

    def analyze_paviment(self, df: pd.DataFrame) -> Dict:
        """
        Analisa os dados do pavimento usando o autoencoder
        """
        try:
            # Limpar e preparar dados
            df_limpo = self.limpar_dados_imtraff(df)
            
            # Processar dados
            X_full_scaled, dados_normais = self.processar_dados_pavimento(df_limpo)
            
            # Carregar modelo se necessário
            self.load_model()
            
            # Detectar anomalias
            test_ds = PavimentoDataset(X_full_scaled)
            test_dl = DataLoader(test_ds, batch_size=32)
            
            # Fazer predições
            predictions = []
            with torch.no_grad():
                for batch in test_dl:
                    pred = self.autoencoder(batch)
                    predictions.append(pred)
            
            predictions = torch.cat(predictions)
            erros_reconstrucao = F.mse_loss(
                predictions, 
                torch.FloatTensor(X_full_scaled), 
                reduction='none'
            ).mean(dim=1).numpy()
            
            # Calcular threshold e identificar anomalias
            threshold = np.percentile(erros_reconstrucao, 95)
            grupos_anomalias = self.identificar_anomalias(df_limpo, erros_reconstrucao, threshold)
            
            print("\nGrupos de anomalias identificados:")
            print(grupos_anomalias.head())
            print("\nColunas dos grupos:", grupos_anomalias.columns.tolist())
            
            # Gerar métricas e recomendações
            metricas_gerais = self.calcular_metricas(grupos_anomalias)
            recomendacoes = self.gerar_recomendacoes(grupos_anomalias)
            
            return {
                "metricas_gerais": metricas_gerais,
                "grupos_anomalias": grupos_anomalias.to_dict('records'),
                "recomendacoes": recomendacoes
            }
            
        except Exception as e:
            print(f"Erro detalhado na análise: {str(e)}")
            raise Exception(f"Erro na análise: {str(e)}")

    def identificar_anomalias(self, df_limpo, erros_reconstrucao, threshold):
        """
        Identifica e classifica anomalias
        """
        anomalias = erros_reconstrucao > threshold
        
        # Criar DataFrame com resultados
        resultados = pd.DataFrame({
            'km': df_limpo['km'][:-4],
            'TRI': df_limpo['TRI'][:-4],
            'TRE': df_limpo['TRE'][:-4],
            'erro': erros_reconstrucao,
            'anomalia': anomalias,
            'severidade': [
                self.classificar_severidade(erro, tri, tre, threshold)
                for erro, tri, tre in zip(
                    erros_reconstrucao, 
                    df_limpo['TRI'][:-4], 
                    df_limpo['TRE'][:-4]
                )
            ]
        })
        
        # Identificar segmentos contínuos
        resultados['grupo'] = (resultados['anomalia'] != resultados['anomalia'].shift()).cumsum()
        
        # Agrupar anomalias contínuas
        grupos = resultados[resultados['anomalia']].groupby('grupo').agg({
            'km': ['min', 'max'],
            'TRI': ['mean', 'max'],
            'TRE': ['mean', 'max'],
            'erro': 'mean',
            'severidade': lambda x: x.mode()[0]
        })
        
        # Flatten multi-index columns
        grupos.columns = ['km_inicial', 'km_final', 'tri_medio', 'tri_max', 'tre_medio', 'tre_max', 'erro_medio', 'severidade']
        grupos = grupos.reset_index(drop=True)
        
        # Calcular extensão
        grupos['extensao'] = grupos['km_final'] - grupos['km_inicial']
        
        return grupos

    def classificar_severidade(self, erro, tri, tre, threshold):
        """
        Classifica a severidade da anomalia
        """
        if erro > 2 * threshold:
            return "CRÍTICA"
        elif erro > 1.5 * threshold:
            return "ALTA"
        elif erro > threshold:
            return "MÉDIA"
        else:
            return "NORMAL"

    def load_model(self):
        """
        Carrega o modelo apenas quando necessário
        """
        if not self.model_loaded:
            try:
                model_path = model_settings.MODEL_PATH
                if not os.path.exists(model_path):
                    # Se o modelo não existe, criar e treinar um novo
                    print("Modelo não encontrado. Criando novo modelo...")
                    self.autoencoder = PavimentoAutoencoder(in_dim=10)  # 5 janelas x 2 features
                    self.model_loaded = True
                else:
                    # Carregar modelo existente
                    print("Carregando modelo existente...")
                    self.autoencoder = PavimentoAutoencoder.load_from_checkpoint(model_path)
                    self.model_loaded = True
                
                # Colocar modelo em modo de avaliação
                self.autoencoder.eval()
                
            except Exception as e:
                raise Exception(f"Erro ao carregar modelo: {str(e)}")

    def calcular_metricas(self, anomalias: pd.DataFrame) -> Dict:
        """
        Calcula métricas gerais
        """
        try:
            # Calcular extensão (diferença entre km final e inicial)
            km_min = anomalias['km_inicial'].min() if 'km_inicial' in anomalias.columns else anomalias['km'].min()
            km_max = anomalias['km_final'].max() if 'km_final' in anomalias.columns else anomalias['km'].max()
            extensao = km_max - km_min
            
            # Calcular máximos por segmento
            tri_max = anomalias['tri_max'].max() if 'tri_max' in anomalias.columns else anomalias['TRI'].max()
            tre_max = anomalias['tre_max'].max() if 'tre_max' in anomalias.columns else anomalias['TRE'].max()
            
            # Contar segmentos por severidade
            if 'severidade' in anomalias.columns:
                segmentos_criticos = len(anomalias[anomalias['severidade'] == 'CRÍTICA'])
                segmentos_alta = len(anomalias[anomalias['severidade'] == 'ALTA'])
                segmentos_media = len(anomalias[anomalias['severidade'] == 'MÉDIA'])
            else:
                segmentos_criticos = segmentos_alta = segmentos_media = 0
            
            return {
                "total_segmentos": len(anomalias),
                "segmentos_criticos": segmentos_criticos,
                "segmentos_alta": segmentos_alta,
                "segmentos_media": segmentos_media,
                "extensao_total": float(extensao),  # Converter para float para serialização JSON
                "tri_max": float(tri_max),
                "tre_max": float(tre_max),
                "km_inicial": float(km_min),
                "km_final": float(km_max)
            }
        except Exception as e:
            print(f"Erro detalhado no cálculo de métricas: {str(e)}")
            print("Colunas disponíveis:", anomalias.columns.tolist())
            raise Exception(f"Erro no cálculo de métricas: {str(e)}")

    def gerar_recomendacoes(self, resultados: pd.DataFrame) -> List[Dict]:
        """
        Gera recomendações baseadas nas anomalias
        """
        try:
            recomendacoes = []
            
            # Agrupar anomalias por severidade
            for _, grupo in resultados.iterrows():
                recomendacao = {
                    'trecho': f"km {grupo['km_inicial']:.1f} - {grupo['km_final']:.1f}",
                    'extensao': f"{grupo['extensao']:.1f} km",
                    'severidade': grupo['severidade'],
                    'tri_medio': f"{grupo['tri_medio']:.1f}",
                    'tri_max': f"{grupo['tri_max']:.1f}",
                    'tre_medio': f"{grupo['tre_medio']:.1f}",
                    'tre_max': f"{grupo['tre_max']:.1f}",
                    'prioridade': 1 if grupo['severidade'] == 'CRÍTICA' else (2 if grupo['severidade'] == 'ALTA' else 3),
                    'prazo': 'Imediato' if grupo['severidade'] == 'CRÍTICA' else ('1 mês' if grupo['severidade'] == 'ALTA' else '3 meses'),
                    'intervencao': self._definir_intervencao(grupo['tri_max'], grupo['tre_max'])
                }
                recomendacoes.append(recomendacao)
            
            return recomendacoes
        except Exception as e:
            raise Exception(f"Erro na geração de recomendações: {str(e)}")

    def _definir_intervencao(self, tri: float, tre: float) -> str:
        """
        Define o tipo de intervenção baseado nos valores de TRI e TRE
        """
        if tri > 5.0 and tre > 7.0:
            return "Reconstrução"
        elif tri > 4.0 or tre > 5.0:
            return "Fresagem e Recomposição"
        elif tri > 3.0 or tre > 3.0:
            return "Microrevestimento"
        else:
            return "Manutenção Preventiva"

    def generate_report(self, analysis_data: Dict) -> str:
        """
        Gera relatório usando LLM
        """
        try:
            completion = self.client.chat.completions.create(
                model="deepseek/deepseek-r1:free",
                messages=[
                    {
                        "role": "system",
                        "content": "Você é um engenheiro especialista em pavimentação rodoviária."
                    },
                    {
                        "role": "user",
                        "content": f"Analise os seguintes dados: {analysis_data}"
                    }
                ]
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"Erro na geração do relatório: {str(e)}"

    def get_report_html(self, analysis_data: Dict) -> str:
        """
        Gera relatório completo em HTML
        """
        try:
            # CSS simplificado em uma única linha
            style = "<style>body{margin:20px;font-family:sans-serif}table{width:100%;border-collapse:collapse}td,th{border:1px solid #ddd;padding:8px}th{background:#f5f5f5}.section{margin:15px 0;padding:10px;border:1px solid #ddd}.metric{margin:5px 0;padding:5px;background:#f5f5f5}.critical{background:#ffe6e6}.high{background:#fff3e0}.medium{background:#f1f8e9}</style>"
            
            # Template HTML simplificado
            html = [
                "<!DOCTYPE html><html><head><title>Relatório de Análise do Pavimento</title>",
                style,
                "</head><body>",
                "<h1>Relatório de Análise do Pavimento</h1>",
                "<div class='section'><h2>Métricas Gerais</h2>",
                self._format_metricas_html(analysis_data["metricas_gerais"]),
                "</div>",
                "<div class='section'><h2>Visualizações</h2>",
                analysis_data["visualizacoes"],
                "</div>",
                "<div class='section'><h2>Anomalias Identificadas</h2>",
                self._format_anomalias_html(analysis_data["grupos_anomalias"]),
                "</div>",
                "<div class='section'><h2>Recomendações</h2>",
                self._format_recomendacoes_html(analysis_data["recomendacoes"]),
                "</div>",
                "<div class='section'><h2>Análise Técnica</h2>",
                analysis_data["relatorio_tecnico"].replace("\n", "<br>"),
                "</div>",
                "</body></html>"
            ]
            
            return "".join(html)
            
        except Exception as e:
            print(f"Erro ao gerar HTML: {str(e)}")
            print("Dados disponíveis:", analysis_data.keys())
            print("Conteúdo das métricas:", analysis_data["metricas_gerais"])
            for key, value in analysis_data.items():
                print(f"\nTipo de {key}: {type(value)}")
                if isinstance(value, dict):
                    print(f"Conteúdo de {key}: {value}")
            raise Exception(f"Erro ao gerar relatório HTML: {str(e)}")

    def _format_metricas_html(self, metricas: Dict) -> str:
        """
        Formata métricas gerais em HTML com tratamento adequado para diferentes tipos de dados
        """
        def format_value(key, value):
            if isinstance(value, float):
                # Formatar números com 1 casa decimal
                return f"{value:.1f}"
            elif isinstance(value, (int, str)):
                # Retornar diretamente números inteiros e strings
                return str(value)
            else:
                # Para outros tipos, converter para string
                return str(value)

        metrics_html = []
        for key, value in metricas.items():
            formatted_value = format_value(key, value)
            # Adicionar "km" para valores de quilometragem
            if "km" in key.lower():
                formatted_value += " km"
            metrics_html.append(f'<div class="metric">{key}: {formatted_value}</div>')
        
        return "".join(metrics_html)

    def _format_anomalias_html(self, anomalias: List[Dict]) -> str:
        """
        Formata lista de anomalias em tabela HTML com tratamento de erro
        """
        try:
            header = "<table><thead><tr><th>Trecho (km)</th><th>Extensão (km)</th><th>Severidade</th><th>TRI Máx</th><th>TRE Máx</th></tr></thead><tbody>"
            rows = []
            for a in anomalias:
                try:
                    severity_class = ("critical" if a["severidade"]=="CRÍTICA" else 
                                    "high" if a["severidade"]=="ALTA" else "medium")
                    row = f"""<tr class="{severity_class}">
                        <td>{a["km_inicial"]:.1f} - {a["km_final"]:.1f}</td>
                        <td>{a["extensao"]:.1f}</td>
                        <td>{a["severidade"]}</td>
                        <td>{a["tri_max"]:.1f}</td>
                        <td>{a["tre_max"]:.1f}</td>
                    </tr>"""
                    rows.append(row)
                except Exception as e:
                    print(f"Erro ao formatar linha da anomalia: {str(e)}")
                    print(f"Dados da anomalia: {a}")
                    continue
            
            return f"{header}{''.join(rows)}</tbody></table>"
        except Exception as e:
            print(f"Erro ao formatar tabela de anomalias: {str(e)}")
            return "<p>Erro ao gerar tabela de anomalias</p>"

    def _format_recomendacoes_html(self, recomendacoes: List[Dict]) -> str:
        """
        Formata recomendações em tabela HTML com tratamento de erro
        """
        try:
            header = "<table><thead><tr><th>Trecho</th><th>Extensão</th><th>Severidade</th><th>Intervenção</th><th>Prazo</th></tr></thead><tbody>"
            rows = []
            for r in recomendacoes:
                try:
                    severity_class = ("critical" if r["severidade"]=="CRÍTICA" else 
                                    "high" if r["severidade"]=="ALTA" else "medium")
                    row = f"""<tr class="{severity_class}">
                        <td>{r["trecho"]}</td>
                        <td>{r["extensao"]}</td>
                        <td>{r["severidade"]}</td>
                        <td>{r["intervencao"]}</td>
                        <td>{r["prazo"]}</td>
                    </tr>"""
                    rows.append(row)
                except Exception as e:
                    print(f"Erro ao formatar linha da recomendação: {str(e)}")
                    print(f"Dados da recomendação: {r}")
                    continue
            
            return f"{header}{''.join(rows)}</tbody></table>"
        except Exception as e:
            print(f"Erro ao formatar tabela de recomendações: {str(e)}")
            return "<p>Erro ao gerar tabela de recomendações</p>"

    def criar_visualizacoes(self, df_limpo, grupos_anomalias):
        """
        Cria visualizações interativas usando plotly
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Criar subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Perfil de Irregularidade (TRI)', 'Trilha de Roda (TRE)'),
            shared_xaxes=True,
            vertical_spacing=0.1
        )
        
        # Plotar TRI
        fig.add_trace(
            go.Scatter(
                x=df_limpo['km'],
                y=df_limpo['TRI'],
                name='TRI',
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )
        
        # Adicionar anomalias TRI
        for _, grupo in grupos_anomalias.iterrows():
            fig.add_trace(
                go.Scatter(
                    x=[grupo['km_inicial'], grupo['km_final']],
                    y=[grupo['tri_max'], grupo['tri_max']],
                    name=f'Anomalia {grupo["severidade"]}',
                    line=dict(
                        color='red' if grupo['severidade'] == 'CRÍTICA' 
                              else 'orange' if grupo['severidade'] == 'ALTA'
                              else 'yellow',
                        width=2
                    ),
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # Plotar TRE
        fig.add_trace(
            go.Scatter(
                x=df_limpo['km'],
                y=df_limpo['TRE'],
                name='TRE',
                line=dict(color='green', width=1)
            ),
            row=2, col=1
        )
        
        # Adicionar anomalias TRE
        for _, grupo in grupos_anomalias.iterrows():
            fig.add_trace(
                go.Scatter(
                    x=[grupo['km_inicial'], grupo['km_final']],
                    y=[grupo['tre_max'], grupo['tre_max']],
                    name=f'Anomalia {grupo["severidade"]}',
                    line=dict(
                        color='red' if grupo['severidade'] == 'CRÍTICA' 
                              else 'orange' if grupo['severidade'] == 'ALTA'
                              else 'yellow',
                        width=2
                    ),
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # Atualizar layout
        fig.update_layout(
            height=800,
            title_text="Análise do Pavimento",
            showlegend=True
        )
        
        return fig.to_html(include_plotlyjs=True, full_html=False) 