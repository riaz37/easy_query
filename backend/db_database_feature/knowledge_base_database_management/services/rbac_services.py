from sqlalchemy.orm import Session
from ..models import User, Role, Permission
from ..schemas import RoleCreate, RoleUpdate, PermissionCreate, PermissionUpdate, UserRoleAssignment
from typing import List, Optional
from uuid import UUID
from fastapi import HTTPException

# Permission Services
def create_permission(db: Session, permission: PermissionCreate) -> Permission:
    """Create a new permission"""
    # Check if permission already exists
    existing_permission = db.query(Permission).filter(
        Permission.name == permission.name
    ).first()
    if existing_permission:
        raise HTTPException(status_code=400, detail="Permission with this name already exists")
    
    db_permission = Permission(
        name=permission.name,
        description=permission.description,
        resource=permission.resource,
        action=permission.action
    )
    db.add(db_permission)
    db.commit()
    db.refresh(db_permission)
    return db_permission

def get_permission(db: Session, permission_id: UUID) -> Optional[Permission]:
    """Get permission by ID"""
    return db.query(Permission).filter(Permission.id == permission_id).first()

def get_permissions(db: Session, skip: int = 0, limit: int = 100) -> List[Permission]:
    """Get all permissions with pagination"""
    return db.query(Permission).offset(skip).limit(limit).all()

def update_permission(db: Session, permission_id: UUID, permission: PermissionUpdate) -> Optional[Permission]:
    """Update permission"""
    db_permission = get_permission(db, permission_id)
    if not db_permission:
        return None
    
    update_data = permission.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_permission, field, value)
    
    db.commit()
    db.refresh(db_permission)
    return db_permission

def delete_permission(db: Session, permission_id: UUID) -> bool:
    """Delete permission"""
    db_permission = get_permission(db, permission_id)
    if not db_permission:
        return False
    
    db.delete(db_permission)
    db.commit()
    return True

# Role Services
def create_role(db: Session, role: RoleCreate) -> Role:
    """Create a new role with optional permissions"""
    # Check if role already exists
    existing_role = db.query(Role).filter(Role.name == role.name).first()
    if existing_role:
        raise HTTPException(status_code=400, detail="Role with this name already exists")
    
    db_role = Role(
        name=role.name,
        description=role.description
    )
    
    # Add permissions if provided
    if role.permission_ids:
        permissions = db.query(Permission).filter(Permission.id.in_(role.permission_ids)).all()
        db_role.permissions = permissions
    
    db.add(db_role)
    db.commit()
    db.refresh(db_role)
    return db_role

def get_role(db: Session, role_id: UUID) -> Optional[Role]:
    """Get role by ID"""
    return db.query(Role).filter(Role.id == role_id).first()

def get_roles(db: Session, skip: int = 0, limit: int = 100) -> List[Role]:
    """Get all roles with pagination"""
    return db.query(Role).offset(skip).limit(limit).all()

def update_role(db: Session, role_id: UUID, role: RoleUpdate) -> Optional[Role]:
    """Update role"""
    db_role = get_role(db, role_id)
    if not db_role:
        return None
    
    update_data = role.dict(exclude_unset=True)
    permission_ids = update_data.pop('permission_ids', None)
    
    for field, value in update_data.items():
        setattr(db_role, field, value)
    
    # Update permissions if provided
    if permission_ids is not None:
        permissions = db.query(Permission).filter(Permission.id.in_(permission_ids)).all()
        db_role.permissions = permissions
    
    db.commit()
    db.refresh(db_role)
    return db_role

def delete_role(db: Session, role_id: UUID) -> bool:
    """Delete role"""
    db_role = get_role(db, role_id)
    if not db_role:
        return False
    
    db.delete(db_role)
    db.commit()
    return True

# User-Role Assignment Services
def assign_roles_to_user(db: Session, user_id: UUID, role_ids: List[UUID]) -> User:
    """Assign roles to a user"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    roles = db.query(Role).filter(Role.id.in_(role_ids)).all()
    user.roles = roles
    
    db.commit()
    db.refresh(user)
    return user

def get_user_roles(db: Session, user_id: UUID) -> List[Role]:
    """Get all roles for a user"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        return []
    return user.roles

def remove_user_roles(db: Session, user_id: UUID, role_ids: List[UUID]) -> User:
    """Remove specific roles from a user"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    roles_to_remove = db.query(Role).filter(Role.id.in_(role_ids)).all()
    for role in roles_to_remove:
        if role in user.roles:
            user.roles.remove(role)
    
    db.commit()
    db.refresh(user)
    return user

def get_users_by_role(db: Session, role_id: UUID) -> List[User]:
    """Get all users with a specific role"""
    role = db.query(Role).filter(Role.id == role_id).first()
    if not role:
        return []
    return role.users

# Utility Services
def get_permissions_by_resource(db: Session, resource: str) -> List[Permission]:
    """Get all permissions for a specific resource"""
    return db.query(Permission).filter(Permission.resource == resource).all()

def get_permissions_by_action(db: Session, action: str) -> List[Permission]:
    """Get all permissions for a specific action"""
    return db.query(Permission).filter(Permission.action == action).all()

def check_user_has_permission(db: Session, user_id: UUID, resource: str, action: str) -> bool:
    """Check if a user has a specific permission"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        return False
    
    for role in user.roles:
        if role.is_active:
            for permission in role.permissions:
                if permission.is_active and permission.resource == resource and permission.action == action:
                    return True
    return False

def get_user_permissions_list(db: Session, user_id: UUID) -> List[str]:
    """Get all permissions for a user as a list of strings"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        return []
    
    permissions = set()
    for role in user.roles:
        if role.is_active:
            for permission in role.permissions:
                if permission.is_active:
                    permissions.add(f"{permission.resource}:{permission.action}")
    return list(permissions)
