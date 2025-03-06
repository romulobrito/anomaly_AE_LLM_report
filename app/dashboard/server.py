from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import requests
import json
import pandas as pd
import time

# Configurações de autenticação
AUTH_CONFIG = {
    "username": "admin@example.com",
    "password": "admin123"
}

def init_dashboard():
    """Inicializa e configura o servidor Dash"""
    app = Dash(
        __name__, 
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True  # Evita erros de callback
    )
    
    app.layout = dbc.Container([
        html.H1("Dashboard de Análise do Pavimento",
                className="text-center my-4"),

        # Atualização automática
        dcc.Interval(
            id='interval-component',
            interval=2000,  # Reduzido para 2 segundos
            n_intervals=0,
            max_intervals=-1
        ),

        # Armazena o último timestamp dos dados
        dcc.Store(id='last-update-time', data=None),
        
        # Armazena os últimos dados
        dcc.Store(id='last-data', data=None),

        # Status
        dbc.Alert(
            "Inicializando...",
            id='status-alert',
            color="info",
            className="mb-3",
            is_open=True
        ),

        # Gráficos
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='graficos-pavimento')
            ])
        ], className="mb-4"),

        # Métricas
        dbc.Row([
            dbc.Col([
                html.Div(id='metricas-container')
            ])
        ], className="mb-4"),

        # Relatório LLM e Recomendações
        dbc.Row([
            # Coluna para o Relatório LLM
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("Relatório de Análise (LLM)", className="text-center")),
                    dbc.CardBody([
                        html.Div(id='relatorio-llm-container', style={'white-space': 'pre-line'})
                    ])
                ])
            ], width=6),
            
            # Coluna para as Recomendações
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("Recomendações de Intervenção", className="text-center")),
                    dbc.CardBody([
                        html.Div(id='recomendacoes-container')
                    ])
                ])
            ], width=6)
        ], className="mb-4"),

        # Anomalias
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("Anomalias Detectadas", className="text-center")),
                    dbc.CardBody([
                        html.Div(id='anomalias-container')
                    ])
                ])
            ])
        ])
    ], fluid=True)

    def get_auth_token():
        """Função auxiliar para obter o token de autenticação"""
        try:
            login_response = requests.post(
                "http://localhost:8000/api/v1/auth/login",
                data={
                    "username": AUTH_CONFIG["username"],
                    "password": AUTH_CONFIG["password"],
                    "grant_type": "password"
                },
                timeout=5  # Adicionado timeout
            )
            
            if login_response.status_code == 200:
                return login_response.json()["access_token"]
            else:
                print(f"Erro na autenticação: {login_response.text}")
                return None
                
        except Exception as e:
            print(f"Erro ao obter token: {e}")
            return None

    @app.callback(
        [Output('last-data', 'data'),
         Output('status-alert', 'children'),
         Output('status-alert', 'color'),
         Output('last-update-time', 'data')],
        [Input('interval-component', 'n_intervals')],
        [State('last-update-time', 'data')]
    )
    def fetch_data(n, last_update):
        """Busca dados da API"""
        try:
            token = get_auth_token()
            if not token:
                return None, "Erro de autenticação", "danger", last_update
            
            headers = {"Authorization": f"Bearer {token}"}
            response = requests.get(
                "http://localhost:8000/api/v1/analysis/latest",
                headers=headers,
                timeout=5  # Adicionado timeout
            )
            
            if response.status_code == 200:
                current_time = time.time()
                data = response.json()
                
                if not data:
                    return None, "Aguardando dados...", "info", last_update
                
                return (
                    data,
                    f"Dados atualizados em {time.strftime('%H:%M:%S')}",
                    "success",
                    current_time
                )
            
            return None, "Erro ao buscar dados", "danger", last_update

        except Exception as e:
            print(f"Erro ao buscar dados: {e}")
            return None, f"Erro: {str(e)}", "danger", last_update

    @app.callback(
        [Output('graficos-pavimento', 'figure'),
         Output('metricas-container', 'children'),
         Output('relatorio-llm-container', 'children'),
         Output('recomendacoes-container', 'children'),
         Output('anomalias-container', 'children')],
        [Input('last-data', 'data')]
    )
    def update_components(data):
        """Atualiza os componentes com os dados"""
        if not data:
            return go.Figure(), [], "", [], []

        try:
            # Gráfico
            fig = go.Figure(json.loads(data['plotly_figure']))
            
            # Métricas com estilo melhorado
            metricas = dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Total de Segmentos", className="text-center"),
                            html.H2(
                                len(data.get('recomendacoes', [])),
                                className="text-center text-primary"
                            )
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Extensão Total", className="text-center"),
                            html.H2(f"{data['metricas']['extensao_total']:.2f} km", className="text-center text-primary")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("TRI Máximo", className="text-center"),
                            html.H2(f"{data['metricas']['tri_max']:.2f}", className="text-center text-danger")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("TRE Máximo", className="text-center"),
                            html.H2(f"{data['metricas']['tre_max']:.2f}", className="text-center text-danger")
                        ])
                    ])
                ], width=3)
            ])

            # Relatório LLM formatado
            relatorio = html.Div([
                dcc.Markdown(data.get('relatorio_llm', 'Relatório não disponível'),
                           style={'white-space': 'pre-wrap'})
            ])

            # Recomendações em cards
            recomendacoes = []
            for rec in data.get('recomendacoes', []):
                recomendacoes.append(
                    dbc.Card([
                        dbc.CardBody([
                            html.H5(f"Trecho: {rec['trecho']}", className="card-title"),
                            html.Div([
                                html.P([
                                    html.Strong("Intervenção: "), rec['intervencao']
                                ], className="mb-1"),
                                html.P([
                                    html.Strong("Prioridade: "), str(rec['prioridade'])
                                ], className="mb-1"),
                                html.P([
                                    html.Strong("Prazo: "), rec['prazo']
                                ], className="mb-1"),
                                html.P([
                                    html.Strong("Extensão: "), rec['extensao']
                                ], className="mb-1")
                            ])
                        ])
                    ], className="mb-3")
                )

            # Anomalias em tabela estilizada
            if data.get('recomendacoes', []):
                df_anomalias = pd.DataFrame(data['recomendacoes'])
                colunas = {
                    'trecho': 'Trecho',
                    'extensao': 'Extensão',
                    'severidade': 'Severidade',
                    'tri_medio': 'TRI Médio',
                    'tri_max': 'TRI Máx',
                    'tre_medio': 'TRE Médio',
                    'tre_max': 'TRE Máx',
                    'intervencao': 'Intervenção',
                    'prioridade': 'Prioridade',
                    'prazo': 'Prazo'
                }
                df_anomalias = df_anomalias.rename(columns=colunas)
                colunas_ordem = ['Trecho', 'Severidade', 'TRI Médio', 'TRI Máx', 
                               'TRE Médio', 'TRE Máx', 'Extensão', 'Intervenção']
                df_anomalias = df_anomalias[colunas_ordem]
                
                anomalias = dbc.Table.from_dataframe(
                    df_anomalias,
                    striped=True,
                    bordered=True,
                    hover=True,
                    className="text-center"
                )
            else:
                anomalias = html.Div(
                    "Nenhuma anomalia detectada", 
                    className="text-center text-muted my-3"
                )

            return fig, metricas, relatorio, recomendacoes, anomalias

        except Exception as e:
            print(f"Erro ao atualizar componentes: {e}")
            return go.Figure(), [], "", [], []

    return app 