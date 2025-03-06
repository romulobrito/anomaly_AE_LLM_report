from sqlalchemy.orm import Session
from datetime import datetime
from app.models.analysis import Analysis
from typing import Optional, Dict

def create_analysis(db: Session, analysis_data: Dict) -> Analysis:
    """
    Cria um novo registro de análise
    """
    try:
        db_analysis = Analysis(
            data_analise=analysis_data["data_analise"],
            metricas_gerais=analysis_data["metricas_gerais"],
            grupos_anomalias=analysis_data["grupos_anomalias"],
            recomendacoes=analysis_data["recomendacoes"],
            relatorio_llm=analysis_data["relatorio_llm"],
            dados_originais=analysis_data["dados_originais"]
        )
        db.add(db_analysis)
        db.commit()
        db.refresh(db_analysis)
        return db_analysis
    except Exception as e:
        db.rollback()
        raise e

def get_analysis_from_db(db: Session, analysis_id: int) -> Optional[Dict]:
    """
    Recupera uma análise do banco de dados
    """
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not analysis:
        return None
        
    return {
        "id": analysis.id,
        "data_analise": analysis.data_analise,
        "metricas_gerais": analysis.metricas_gerais,
        "grupos_anomalias": analysis.grupos_anomalias,
        "recomendacoes": analysis.recomendacoes,
        "relatorio_llm": analysis.relatorio_llm,
        "dados_originais": analysis.dados_originais
    } 