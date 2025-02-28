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

        # Anomalias
        dbc.Row([
            dbc.Col([
                html.Div(id='anomalias-container')
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
                    "grant_type": "password"  # Necessário para o OAuth2
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
         Output('anomalias-container', 'children'),
         Output('status-alert', 'children'),
         Output('status-alert', 'color'),
         Output('status-alert', 'is_open')],
        [Input('interval-component', 'n_intervals')]
    )
    def update_metrics(n):
        try:
            # Obter token de autenticação
            token = get_auth_token()
            if not token:
                return go.Figure(), [], [], "Erro de autenticação", "danger", True
            
            # Obter dados do dashboard
            headers = {"Authorization": f"Bearer {token}"}
            response = requests.get(
                "http://localhost:8000/api/v1/analysis/dashboard/1",
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Criar gráfico do Plotly
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
                if data['anomalias']:
                    anomalias = dbc.Table.from_dataframe(
                        pd.DataFrame(data['anomalias']),
                        striped=True,
                        bordered=True,
                        hover=True
                    )
                else:
                    anomalias = html.Div("Nenhuma anomalia encontrada")

                return fig, metricas, anomalias, "Dados atualizados", "success", True
            
            return go.Figure(), [], [], "Aguardando dados...", "info", True

        except Exception as e:
            print(f"Erro ao atualizar dashboard: {e}")
            return go.Figure(), [], [], f"Erro ao atualizar: {str(e)}", "danger", True

    return app

if __name__ == '__main__':
    app = create_dashboard()
    app.run_server(debug=True, port=8050) 