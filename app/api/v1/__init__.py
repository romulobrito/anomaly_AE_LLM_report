from fastapi import APIRouter
from app.api.v1.endpoints import analysis, auth

api_router = APIRouter()

# Registrar rotas de autenticação
api_router.include_router(auth.router, prefix="/auth", tags=["Autenticação"])

# Registrar rotas de análise
api_router.include_router(analysis.router, prefix="/analysis", tags=["Análise de Pavimento"])


