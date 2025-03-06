from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
import pandas as pd
import time

def create_dashboard():
    """
    Cria o layout do dashboard para visualização dos dados em tempo real
    """
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    
    app.layout = dbc.Container([
        html.H1("Dashboard de Análise do Pavimento",
                className="text-center my-4"),

        # Intervalo para atualização automática
        dcc.Interval(
            id='interval-component',
            interval=2*1000,  # 2 segundos
            n_intervals=0
        ),

        # Status da atualização
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

        # Anomalias
        dbc.Row([
            dbc.Col([
                html.Div(id='anomalias-container')
            ])
        ])
    ], fluid=True)

    @app.callback(
        [Output('graficos-pavimento', 'figure'),
         Output('metricas-container', 'children'),
         Output('anomalias-container', 'children'),
         Output('status-alert', 'children'),
         Output('status-alert', 'color'),
         Output('status-alert', 'is_open')],
        [Input('interval-component', 'n_intervals')]
    )
    def update_metrics(n):
        try:
            # Tentar obter dados da API
            response = requests.get("http://localhost:8000/api/v1/analysis/latest")
            
            if response.status_code == 200:
                data = response.json()
                
                # Criar gráficos com os dados do Plotly
                fig = go.Figure(json.loads(data['plotly_figure']))
                
                # Criar cards de métricas
                metricas = dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4("Total de Segmentos"),
                                html.H2(data['metricas']['total_segmentos'])
                            ])
                        ])
                    ], width=3),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4("Extensão Total"),
                                html.H2(f"{data['metricas']['extensao_total']:.2f} km")
                            ])
                        ])
                    ], width=3),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4("TRI Máximo"),
                                html.H2(f"{data['metricas']['tri_max']:.2f}")
                            ])
                        ])
                    ], width=3),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4("TRE Máximo"),
                                html.H2(f"{data['metricas']['tre_max']:.2f}")
                            ])
                        ])
                    ], width=3)
                ])

                # Criar tabela de anomalias
                anomalias = dbc.Table.from_dataframe(
                    pd.DataFrame(data['anomalias']),
                    striped=True,
                    bordered=True,
                    hover=True
                )

                return fig, metricas, anomalias, "Dados atualizados", "success", True
            
            return go.Figure(), [], [], "Aguardando dados...", "info", True

        except Exception as e:
            print(f"Erro ao atualizar dashboard: {e}")
            return go.Figure(), [], [], f"Erro ao atualizar: {str(e)}", "danger", True

    return app

if __name__ == '__main__':
    app = create_dashboard()
    app.run_server(debug=True, port=8050) 