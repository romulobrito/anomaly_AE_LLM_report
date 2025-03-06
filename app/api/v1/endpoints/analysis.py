from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from fastapi.responses import JSONResponse
from app.utils.model_manager import ModelManager
from app.api.deps import get_current_user
from app.schemas.user import User
import pandas as pd
from io import BytesIO
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from app.dashboard.layout import create_dashboard
from typing import Optional
import time

router = APIRouter()
model_manager = ModelManager()

# Variável global para armazenar últimos dados
latest_data = None
last_update = None

@router.post("/llm")
async def get_llm_analysis(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """
    Retorna a análise do LLM com base nas anomalias detectadas pelo autoencoder
    """
    try:
        contents = await file.read()
        df = pd.read_excel(BytesIO(contents))
        
        print("Iniciando análise do pavimento...")
        # Gerar análise com autoencoder e LLM
        analysis = model_manager.analyze_paviment(df)
        
        print("Resultado da análise:", analysis)
        
        if not analysis or not isinstance(analysis, dict):
            raise HTTPException(
                status_code=400, 
                detail="Formato de retorno inválido da análise"
            )
            
        # Gerar avaliação do LLM com base nas anomalias e recomendações
        try:
            metricas = analysis['metricas_gerais']
            anomalias = analysis['grupos_anomalias']
            recomendacoes = analysis['recomendacoes']
            
            # Criar resumo da análise
            avaliacao = f"""
            Análise do Pavimento:
            
            Foram identificados {metricas['total_segmentos']} segmentos com anomalias ao longo de {metricas['extensao_total']:.2f} km de rodovia.
            
            Principais indicadores:
            - TRI máximo: {metricas['tri_max']:.1f} mm/m
            - TRE máximo: {metricas['tre_max']:.1f} mm/m
            
            Distribuição por severidade:
            - Crítica: {metricas['segmentos_criticos']} segmentos
            - Alta: {metricas['segmentos_alta']} segmentos
            - Média: {metricas['segmentos_media']} segmentos
            
            Recomendações de intervenção:
            """
            
            for rec in recomendacoes[:3]:  # Mostrar apenas as 3 principais recomendações
                avaliacao += f"\n- Trecho {rec['trecho']}: {rec['intervencao']} (Prioridade {rec['prioridade']}, Prazo: {rec['prazo']})"
            
            analysis['avaliacao_llm'] = avaliacao.strip()
            
        except Exception as e:
            print(f"Erro ao gerar avaliação LLM: {e}")
            analysis['avaliacao_llm'] = "Não foi possível gerar a avaliação detalhada."
            
        # Converter DataFrame para lista de dicionários
        try:
            anomalias = analysis['grupos_anomalias']
            print("Anomalias encontradas:", anomalias)
            
            anomalias_list = []
            if isinstance(anomalias, pd.DataFrame) and not anomalias.empty:
                for _, row in anomalias.iterrows():
                    anomalia = {}
                    for col in row.index:
                        try:
                            if pd.notnull(row[col]):
                                if col == 'severidade':
                                    anomalia[col] = str(row[col])
                                else:
                                    anomalia[col] = float(row[col])
                            else:
                                anomalia[col] = None
                        except Exception as e:
                            print(f"Erro ao converter coluna {col}: {e}")
                            anomalia[col] = None
                    anomalias_list.append(anomalia)
            
            # Preparar métricas
            metricas = analysis.get('metricas_gerais', {})
            metricas_dict = {
                "total_segmentos": len(anomalias_list),
                "extensao_total": float(metricas.get('extensao_total', 0)),
                "tri_max": float(metricas.get('tri_max', 0)),
                "tre_max": float(metricas.get('tre_max', 0))
            }
            
            response_data = {
                "metricas": metricas_dict,
                "anomalias": anomalias_list,
                "recomendacoes": analysis.get('recomendacoes', []),
                "relatorio_llm": analysis['avaliacao_llm']
            }
            
            print("Resposta formatada:", response_data)
            return JSONResponse(content=response_data)
            
        except Exception as e:
            print(f"Erro ao processar anomalias: {e}")
            raise HTTPException(
                status_code=400, 
                detail=f"Erro ao processar dados das anomalias: {str(e)}"
            )
            
    except Exception as e:
        print(f"Erro geral na análise: {e}")
        raise HTTPException(
            status_code=400, 
            detail=f"Erro na análise: {str(e)}"
        )

@router.post("/plot")
async def get_plot_data(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """
    Processa e retorna os dados para plotagem
    """
    global latest_data, last_update
    
    try:
        contents = await file.read()
        df = pd.read_excel(BytesIO(contents))
        
        # Limpar e preparar dados para plotagem
        df_limpo = model_manager.limpar_dados_imtraff(df)
        
        if df_limpo.empty:
            raise HTTPException(status_code=400, detail="Erro ao processar dados para plotagem")
        
        # Gerar análise para obter métricas e anomalias
        analysis = model_manager.analyze_paviment(df)
        
        if not analysis:
            raise HTTPException(status_code=400, detail="Erro ao gerar análise")
            
        # Criar gráficos com Plotly
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('TRI ao longo da rodovia', 'TRE ao longo da rodovia'),
            vertical_spacing=0.12
        )
        
        # Adicionar linha TRI
        fig.add_trace(
            go.Scatter(
                x=df_limpo['km'], 
                y=df_limpo['TRI'],
                name='TRI',
                mode='lines',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Adicionar linha TRE
        fig.add_trace(
            go.Scatter(
                x=df_limpo['km'],
                y=df_limpo['TRE'],
                name='TRE',
                mode='lines',
                line=dict(color='red', width=2)
            ),
            row=2, col=1
        )
        
        # Atualizar layout
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="Perfil da Rodovia",
            template="plotly_white",
            margin=dict(t=100, b=50, l=50, r=50)
        )
        
        # Atualizar eixos
        fig.update_xaxes(title_text="Quilometragem", gridcolor='lightgray')
        fig.update_yaxes(title_text="TRI (mm/m)", row=1, col=1, gridcolor='lightgray')
        fig.update_yaxes(title_text="TRE (mm/m)", row=2, col=1, gridcolor='lightgray')
            
        # Gerar avaliação do LLM
        try:
            metricas = analysis['metricas_gerais']
            anomalias = analysis['grupos_anomalias']
            recomendacoes = analysis['recomendacoes']
            
            # Preparar métricas
            metricas_dict = {
                "total_segmentos": len(anomalias) if isinstance(anomalias, pd.DataFrame) else 0,
                "extensao_total": float(metricas.get('extensao_total', 0)),
                "tri_max": float(metricas.get('tri_max', 0)),
                "tre_max": float(metricas.get('tre_max', 0))
            }
            
            # Preparar anomalias
            anomalias_list = []
            if isinstance(anomalias, pd.DataFrame) and not anomalias.empty:
                for _, row in anomalias.iterrows():
                    anomalia = {}
                    for col in row.index:
                        try:
                            if pd.notnull(row[col]):
                                if col == 'severidade':
                                    anomalia[col] = str(row[col])
                                else:
                                    anomalia[col] = float(row[col])
                            else:
                                anomalia[col] = None
                        except Exception as e:
                            print(f"Erro ao converter coluna {col}: {e}")
                            anomalia[col] = None
                    anomalias_list.append(anomalia)
            
            # Criar resumo da análise
            avaliacao = f"""
            Análise do Pavimento:
            
            Foram identificados {metricas['total_segmentos']} segmentos com anomalias ao longo de {metricas['extensao_total']:.2f} km de rodovia.
            
            Principais indicadores:
            - TRI máximo: {metricas['tri_max']:.1f} mm/m
            - TRE máximo: {metricas['tre_max']:.1f} mm/m
            
            Distribuição por severidade:
            - Crítica: {metricas['segmentos_criticos']} segmentos
            - Alta: {metricas['segmentos_alta']} segmentos
            - Média: {metricas['segmentos_media']} segmentos
            
            Recomendações de intervenção:
            """
            
            for rec in recomendacoes[:3]:
                avaliacao += f"\n- Trecho {rec['trecho']}: {rec['intervencao']} (Prioridade {rec['prioridade']}, Prazo: {rec['prazo']})"
            
        except Exception as e:
            print(f"Erro ao gerar avaliação LLM: {e}")
            avaliacao = "Não foi possível gerar a avaliação detalhada."
            recomendacoes = []

        # Preparar dados para o dashboard
        latest_data = {
            "plot_data": {
                "km": df_limpo['km'].tolist(),
                "tri": df_limpo['TRI'].tolist(),
                "tre": df_limpo['TRE'].tolist()
            },
            "plotly_figure": fig.to_json(),
            "metricas": metricas_dict,
            "anomalias": anomalias_list,
            "relatorio_llm": avaliacao,
            "recomendacoes": recomendacoes
        }
        last_update = time.time()
        
        return JSONResponse(content=latest_data)
        
    except Exception as e:
        print(f"Erro na plotagem: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Erro na plotagem: {str(e)}")

@router.get("/dashboard/{analysis_id}")
async def view_dashboard(
    analysis_id: int,
    current_user: User = Depends(get_current_user)
):
    """
    Retorna os dados para visualização no dashboard
    """
    try:
        # Usar os dados mais recentes em vez de buscar por ID
        if not latest_data:
            raise HTTPException(status_code=404, detail="Nenhum dado disponível para visualização")
            
        return JSONResponse(content={
            "plot_data": latest_data["plot_data"],
            "plotly_figure": latest_data["plotly_figure"],
            "metricas": latest_data["metricas"],
            "anomalias": latest_data["anomalias"]
        })
        
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Erro ao obter dados do dashboard: {str(e)}")

@router.get("/latest")
async def get_latest_data():
    """
    Retorna os dados mais recentes processados
    """
    global latest_data, last_update
    
    if not latest_data:
        raise HTTPException(status_code=404, detail="Nenhum dado disponível")
        
    return JSONResponse(content=latest_data) 