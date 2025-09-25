from passlib.context import CryptContext # pyright: ignore[reportMissingModuleSource]
from jose import jwt # pyright: ignore[reportMissingModuleSource]
from datetime import datetime, timedelta
import uuid
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from .database import get_db
from .models import RevokedToken, User, Role, Permission
from typing import List, Optional

SECRET_KEY = "esap-secret-key-kb-esap-secret-key-23334-KB"   
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 10080  # 7 days (7 * 24 * 60 = 10080 minutes)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({
        "exp": expire,
        "jti": str(uuid.uuid4())  # Add unique JWT ID
    })
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

def is_token_revoked(jti: str, db: Session) -> bool:
    """Check if a token has been revoked"""
    revoked_token = db.query(RevokedToken).filter(RevokedToken.jti == jti).first()
    return revoked_token is not None

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    """Get current user from token"""
    token = credentials.credentials
    payload = verify_token(token)
    
    # Check if token is revoked
    if is_token_revoked(payload.get("jti"), db):
        raise HTTPException(status_code=401, detail="Token has been revoked")
    
    username = payload.get("sub")
    if username is None:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    return username

def get_current_user_obj(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    """Get current user object from token"""
    token = credentials.credentials
    payload = verify_token(token)
    
    # Check if token is revoked
    if is_token_revoked(payload.get("jti"), db):
        raise HTTPException(status_code=401, detail="Token has been revoked")
    
    username = payload.get("sub")
    if username is None:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    user = db.query(User).filter(User.username == username, User.is_active == True).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found or inactive")
    
    return user

def get_user_permissions(user: User, db: Session) -> List[str]:
    """Get all permissions for a user through their roles"""
    permissions = set()
    for role in user.roles:
        if role.is_active:
            for permission in role.permissions:
                if permission.is_active:
                    permissions.add(f"{permission.resource}:{permission.action}")
    return list(permissions)

def check_permission(user: User, resource: str, action: str, db: Session) -> bool:
    """Check if user has specific permission"""
    user_permissions = get_user_permissions(user, db)
    required_permission = f"{resource}:{action}"
    return required_permission in user_permissions

def require_permission(resource: str, action: str):
    """Decorator to require specific permission"""
    def permission_dependency(current_user: User = Depends(get_current_user_obj), db: Session = Depends(get_db)):
        if not check_permission(current_user, resource, action, db):
            raise HTTPException(
                status_code=403, 
                detail=f"Insufficient permissions. Required: {resource}:{action}"
            )
        return current_user
    return permission_dependency

def require_role(role_name: str):
    """Decorator to require specific role"""
    def role_dependency(current_user: User = Depends(get_current_user_obj), db: Session = Depends(get_db)):
        user_roles = [role.name for role in current_user.roles if role.is_active]
        if role_name not in user_roles:
            raise HTTPException(
                status_code=403, 
                detail=f"Insufficient permissions. Required role: {role_name}"
            )
        return current_user
    return role_dependency

def create_access_token_with_roles(user: User, db: Session, expires_delta: timedelta = None):
    """Create access token with user roles and permissions"""
    user_permissions = get_user_permissions(user, db)
    user_roles = [role.name for role in user.roles if role.is_active]
    
    to_encode = {
        "sub": user.username,
        "user_id": str(user.id),
        "roles": user_roles,
        "permissions": user_permissions
    }
    
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({
        "exp": expire,
        "jti": str(uuid.uuid4())
    })
    
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
