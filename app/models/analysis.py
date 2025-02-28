from sqlalchemy import Column, Integer, DateTime, JSON, LargeBinary, String, Text
from app.db.base_class import Base
from datetime import datetime

class Analysis(Base):
    __tablename__ = "analysis"

    id = Column(Integer, primary_key=True, index=True)
    data_analise = Column(DateTime, default=datetime.now)
    metricas_gerais = Column(JSON)
    grupos_anomalias = Column(JSON)
    recomendacoes = Column(JSON)
    relatorio_llm = Column(Text)
    dados_originais = Column(LargeBinary) 