from typing import Generator, Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from sqlalchemy.orm import Session
from app.db.session import SessionLocal
from app.core.config import settings
from app.crud.user import get_user_by_email
from app.schemas.token import TokenPayload
from app.schemas.user import User

# Configuração do OAuth2
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_V1_STR}/auth/login")

def get_db() -> Generator:
    """
    Dependency para obter sessão do banco de dados
    """
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()

async def get_current_user(
    db: Session = Depends(get_db),
    token: str = Depends(oauth2_scheme)
) -> User:
    """
    Dependency para obter o usuário atual através do token JWT
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Não foi possível validar as credenciais",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Decodificar o token JWT
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        token_data = TokenPayload(**payload)
    except JWTError:
        raise credentials_exception
    
    # Obter usuário do banco de dados
    user = get_user_by_email(db, email=token_data.sub)
    if not user:
        raise credentials_exception
    
    return user

def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """
    Dependency para verificar se o usuário está ativo
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=400,
            detail="Usuário inativo"
        )
    return current_user

def get_current_active_superuser(
    current_user: User = Depends(get_current_user),
) -> User:
    """
    Dependency para verificar se o usuário é superusuário
    """
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=400,
            detail="O usuário não tem privilégios suficientes"
        )
    return current_user 