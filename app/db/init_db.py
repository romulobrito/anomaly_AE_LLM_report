from sqlalchemy.orm import Session
from app.db.base_class import Base
from app.db.session import engine, SessionLocal
from app.core.config import settings
from app.crud.user import create_user
from app.schemas.user import UserCreate

def init_db() -> None:
    """
    Cria todas as tabelas no banco de dados e o usuário admin inicial
    """
    try:
        # Dropar todas as tabelas existentes
        Base.metadata.drop_all(bind=engine)
        
        # Criar todas as tabelas novamente
        Base.metadata.create_all(bind=engine)
        
        # Criar usuário admin inicial
        db = SessionLocal()
        try:
            user = UserCreate(
                email=settings.FIRST_SUPERUSER,
                password=settings.FIRST_SUPERUSER_PASSWORD,
                is_superuser=True
            )
            create_user(db, user)
        except Exception as e:
            print(f"Erro ao criar usuário admin: {e}")
        finally:
            db.close()
            
    except Exception as e:
        print(f"Erro ao inicializar banco de dados: {e}") 