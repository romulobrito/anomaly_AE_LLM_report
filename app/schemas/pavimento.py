from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List, Dict

class PavimentoBase(BaseModel):
    km_inicial: float
    km_final: float
    tri: float
    tre: float
    severidade: str
    observacoes: Optional[str] = None

class PavimentoCreate(PavimentoBase):
    pass

class Pavimento(PavimentoBase):
    id: int
    user_id: int
    data_criacao: datetime = datetime.now()

    class Config:
        from_attributes = True

class MedicaoBase(BaseModel):
    km: float
    tri: float
    tre: float

class ExcelData(BaseModel):
    medicoes: List[MedicaoBase]
    data_analise: datetime = datetime.now()

class AnaliseResponse(BaseModel):
    data_analise: datetime
    metricas_gerais: Dict
    grupos_anomalias: List[Dict]
    recomendacoes: List[Dict]
    relatorio_tecnico: str
    visualizacoes: Optional[str] = None

    class Config:
        from_attributes = True 