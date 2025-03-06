from sqlalchemy.orm import Session
from app.db.base_class import Base
from app.db.session import engine, SessionLocal
from app.core.config import settings
from app.crud.user import create_user, get_user_by_email
from app.schemas.user import UserCreate

def init_db() -> None:
    """
    Cria todas as tabelas no banco de dados e o usuário admin inicial
    """
    try:
        # Criar todas as tabelas
        Base.metadata.create_all(bind=engine)
        
        # Criar usuário admin inicial se não existir
        db = SessionLocal()
        try:
            # Verificar se o usuário admin já existe
            existing_user = get_user_by_email(db, email=settings.FIRST_SUPERUSER)
            
            if not existing_user:
                user = UserCreate(
                    email=settings.FIRST_SUPERUSER,
                    password=settings.FIRST_SUPERUSER_PASSWORD,
                    is_superuser=True
                )
                create_user(db, user)
                print("Usuário admin criado com sucesso!")
            else:
                print("Usuário admin já existe!")
                
        except Exception as e:
            print(f"Erro ao verificar/criar usuário admin: {e}")
        finally:
            db.close()
            
    except Exception as e:
        print(f"Erro ao inicializar banco de dados: {e}") 