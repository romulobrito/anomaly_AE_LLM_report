from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import requests
import json
import pandas as pd

# Configurações de autenticação
AUTH_CONFIG = {
    "username": "admin@example.com",  # Ajuste para seu usuário
    "password": "admin123"  # Ajuste para sua senha
}

def create_dashboard():
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    
    app.layout = dbc.Container([
        html.H1("Dashboard de Análise do Pavimento",
                className="text-center my-4"),

        # Atualização automática
        dcc.Interval(
            id='interval-component',
            interval=2*1000,  # 2 segundos
            n_intervals=0
        ),

        # Status
        dbc.Alert(
            "Aguardando dados...",
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
                }
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
        [Output('graficos-pavimento', 'figure'),
         Output('metricas-container', 'children'),
         Output('relatorio-llm-container', 'children'),
         Output('recomendacoes-container', 'children'),
         Output('anomalias-container', 'children'),
         Output('status-alert', 'children'),
         Output('status-alert', 'color'),
         Output('status-alert', 'is_open')],
        [Input('interval-component', 'n_intervals')]
    )
    def update_metrics(n):
        try:
            token = get_auth_token()
            if not token:
                return go.Figure(), [], "", [], [], "Erro de autenticação", "danger", True
            
            headers = {"Authorization": f"Bearer {token}"}
            response = requests.get(
                "http://localhost:8000/api/v1/analysis/latest",  # Usando latest em vez de dashboard/1
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Gráfico
                fig = go.Figure(json.loads(data['plotly_figure']))
                
                # Métricas com estilo melhorado
                metricas = dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4("Total de Segmentos", className="text-center"),
                                html.H2(
                                    len(data.get('recomendacoes', [])),  # Isso dará 9 no seu caso
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
                if data.get('recomendacoes', []):  # Verificar se há recomendações
                    # Criar DataFrame a partir das recomendações
                    df_anomalias = pd.DataFrame(data['recomendacoes'])
                    # Renomear e reorganizar colunas
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
                    # Selecionar e ordenar colunas
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

                return (
                    fig, 
                    metricas, 
                    relatorio, 
                    recomendacoes, 
                    anomalias, 
                    "Dados atualizados", 
                    "success", 
                    True
                )
            
            return (
                go.Figure(), 
                [], 
                "", 
                [], 
                [], 
                "Aguardando dados...", 
                "info", 
                True
            )

        except Exception as e:
            print(f"Erro ao atualizar dashboard: {e}")
            return (
                go.Figure(), 
                [], 
                "", 
                [], 
                [], 
                f"Erro ao atualizar: {str(e)}", 
                "danger", 
                True
            )

    return app

if __name__ == '__main__':
    app = create_dashboard()
    app.run_server(debug=True, port=8050) 