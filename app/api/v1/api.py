from fastapi import APIRouter
from app.api.v1.endpoints import auth, pavimentos, analysis

api_router = APIRouter()
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(pavimentos.router, prefix="/pavimentos", tags=["pavimentos"])
api_router.include_router(analysis.router, prefix="/analysis", tags=["analysis"]) 