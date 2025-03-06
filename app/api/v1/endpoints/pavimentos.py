from fastapi import APIRouter, Depends, HTTPException
from typing import List
from app.schemas.pavimento import PavimentoCreate, Pavimento
from app.crud.pavimento import create_pavimento, get_pavimentos
from app.api.deps import get_current_user
from app.schemas.user import User

router = APIRouter()

@router.post("/", response_model=Pavimento)
def create_new_pavimento(
    pavimento: PavimentoCreate,
    current_user: User = Depends(get_current_user)
) -> Pavimento:
    """
    Criar nova análise de pavimento
    """
    return create_pavimento(pavimento=pavimento, user_id=current_user.id)

@router.get("/", response_model=List[Pavimento])
def read_pavimentos(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_user)
) -> List[Pavimento]:
    """
    Recuperar análises de pavimento
    """
    return get_pavimentos(skip=skip, limit=limit) 