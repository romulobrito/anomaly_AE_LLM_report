from app.crud.user import create_user
from app.schemas.user import UserCreate

def init_db():
    # Criar usuário inicial
    initial_user = UserCreate(
        email="admin@example.com",
        password="admin123",
        full_name="Admin User"
    )
    
    try:
        create_user(initial_user)
        print("Usuário inicial criado com sucesso!")
    except Exception as e:
        print(f"Usuário inicial já existe: {e}") 