from typing import List
from app.schemas.pavimento import PavimentoCreate, Pavimento

# Simulação de banco de dados
pavimentos_db = []

def create_pavimento(pavimento: PavimentoCreate, user_id: int) -> Pavimento:
    db_pavimento = Pavimento(
        id=len(pavimentos_db) + 1,
        user_id=user_id,
        **pavimento.dict()
    )
    pavimentos_db.append(db_pavimento)
    return db_pavimento

def get_pavimentos(skip: int = 0, limit: int = 100) -> List[Pavimento]:
    return pavimentos_db[skip : skip + limit] 