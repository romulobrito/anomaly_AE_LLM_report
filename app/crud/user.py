from typing import Optional
from sqlalchemy.orm import Session
from app.core.security import get_password_hash, verify_password
from app.models.user import User
from app.schemas.user import UserCreate

def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """
    Busca usuário pelo email
    """
    return db.query(User).filter(User.email == email).first()

def create_user(db: Session, user: UserCreate) -> User:
    """
    Cria um novo usuário
    """
    # Verificar se usuário já existe
    db_user = get_user_by_email(db, email=user.email)
    if db_user:
        return db_user
        
    # Criar novo usuário
    hashed_password = get_password_hash(user.password)
    db_user = User(
        email=user.email,
        hashed_password=hashed_password,
        is_active=True,
        is_superuser=user.is_superuser
    )
    
    try:
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        return db_user
    except Exception as e:
        db.rollback()
        raise e

def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
    """
    Autentica um usuário
    """
    user = get_user_by_email(db, email=email)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user 