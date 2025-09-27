#!/usr/bin/env python3
"""
Initialize RBAC (Role-Based Access Control) system with default roles and permissions
"""

from sqlalchemy.orm import Session
from database import SessionLocal, engine
from models import Base, User, Role, Permission
from services.rbac_services import create_permission, create_role, assign_roles_to_user
from schemas import PermissionCreate, RoleCreate, UserRoleAssignment
from auth import hash_password
import uuid

def init_database():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)

def create_default_permissions(db: Session):
    """Create default permissions for the system"""
    permissions = [
        # User management permissions
        PermissionCreate(
            name="users_create",
            description="Create new users",
            resource="users",
            action="create"
        ),
        PermissionCreate(
            name="users_read",
            description="Read user information",
            resource="users",
            action="read"
        ),
        PermissionCreate(
            name="users_update",
            description="Update user information",
            resource="users",
            action="update"
        ),
        PermissionCreate(
            name="users_delete",
            description="Delete users",
            resource="users",
            action="delete"
        ),
        
        # Role management permissions
        PermissionCreate(
            name="roles_create",
            description="Create new roles",
            resource="roles",
            action="create"
        ),
        PermissionCreate(
            name="roles_read",
            description="Read role information",
            resource="roles",
            action="read"
        ),
        PermissionCreate(
            name="roles_update",
            description="Update role information",
            resource="roles",
            action="update"
        ),
        PermissionCreate(
            name="roles_delete",
            description="Delete roles",
            resource="roles",
            action="delete"
        ),
        
        # Permission management permissions
        PermissionCreate(
            name="permissions_create",
            description="Create new permissions",
            resource="permissions",
            action="create"
        ),
        PermissionCreate(
            name="permissions_read",
            description="Read permission information",
            resource="permissions",
            action="read"
        ),
        PermissionCreate(
            name="permissions_update",
            description="Update permission information",
            resource="permissions",
            action="update"
        ),
        PermissionCreate(
            name="permissions_delete",
            description="Delete permissions",
            resource="permissions",
            action="delete"
        ),
        
        # Company management permissions
        PermissionCreate(
            name="companies_create",
            description="Create new companies",
            resource="companies",
            action="create"
        ),
        PermissionCreate(
            name="companies_read",
            description="Read company information",
            resource="companies",
            action="read"
        ),
        PermissionCreate(
            name="companies_update",
            description="Update company information",
            resource="companies",
            action="update"
        ),
        PermissionCreate(
            name="companies_delete",
            description="Delete companies",
            resource="companies",
            action="delete"
        ),
        
        # Data source management permissions
        PermissionCreate(
            name="data_sources_create",
            description="Create new data sources",
            resource="data_sources",
            action="create"
        ),
        PermissionCreate(
            name="data_sources_read",
            description="Read data source information",
            resource="data_sources",
            action="read"
        ),
        PermissionCreate(
            name="data_sources_update",
            description="Update data source information",
            resource="data_sources",
            action="update"
        ),
        PermissionCreate(
            name="data_sources_delete",
            description="Delete data sources",
            resource="data_sources",
            action="delete"
        ),
    ]
    
    created_permissions = {}
    for permission in permissions:
        try:
            db_permission = create_permission(db, permission)
            created_permissions[permission.name] = db_permission
            print(f"✅ Created permission: {permission.name}")
        except Exception as e:
            print(f"⚠️  Permission {permission.name} already exists or error: {e}")
    
    return created_permissions

def create_default_roles(db: Session, permissions: dict):
    """Create default roles with appropriate permissions"""
    
    # Super Admin Role - has all permissions
    super_admin_permissions = list(permissions.values())
    super_admin_role = RoleCreate(
        name="super_admin",
        description="Super Administrator with all permissions",
        permission_ids=[p.id for p in super_admin_permissions]
    )
    
    # Admin Role - has most permissions except user/role/permission management
    admin_permissions = [
        permissions["companies_create"], permissions["companies_read"], 
        permissions["companies_update"], permissions["companies_delete"],
        permissions["data_sources_create"], permissions["data_sources_read"],
        permissions["data_sources_update"], permissions["data_sources_delete"]
    ]
    admin_role = RoleCreate(
        name="admin",
        description="Administrator with company and data source management permissions",
        permission_ids=[p.id for p in admin_permissions]
    )
    
    # Manager Role - can read and update companies and data sources
    manager_permissions = [
        permissions["companies_read"], permissions["companies_update"],
        permissions["data_sources_read"], permissions["data_sources_update"]
    ]
    manager_role = RoleCreate(
        name="manager",
        description="Manager with read and update permissions for companies and data sources",
        permission_ids=[p.id for p in manager_permissions]
    )
    
    # User Role - can only read companies and data sources
    user_permissions = [
        permissions["companies_read"], permissions["data_sources_read"]
    ]
    user_role = RoleCreate(
        name="user",
        description="Regular user with read-only access to companies and data sources",
        permission_ids=[p.id for p in user_permissions]
    )
    
    roles_to_create = [
        ("super_admin", super_admin_role),
        ("admin", admin_role),
        ("manager", manager_role),
        ("user", user_role)
    ]
    
    created_roles = {}
    for role_name, role_data in roles_to_create:
        try:
            db_role = create_role(db, role_data)
            created_roles[role_name] = db_role
            print(f"✅ Created role: {role_name}")
        except Exception as e:
            print(f"⚠️  Role {role_name} already exists or error: {e}")
    
    return created_roles

def create_default_admin_user(db: Session, roles: dict):
    """Create a default super admin user"""
    # Check if admin user already exists
    existing_admin = db.query(User).filter(User.username == "admin").first()
    if existing_admin:
        print("⚠️  Admin user already exists")
        return existing_admin
    
    # Create admin user
    admin_user = User(
        username="admin",
        email="admin@example.com",
        password_hash=hash_password("admin123"),  # Change this in production!
        is_active=True
    )
    db.add(admin_user)
    db.commit()
    db.refresh(admin_user)
    
    # Assign super admin role
    if "super_admin" in roles:
        assign_roles_to_user(db, admin_user.id, [roles["super_admin"].id])
        print("✅ Created admin user with super_admin role")
    else:
        print("⚠️  Could not assign super_admin role - role not found")
    
    return admin_user

def main():
    """Main initialization function"""
    print("🚀 Initializing RBAC system...")
    
    # Initialize database
    init_database()
    print("✅ Database tables created")
    
    # Create database session
    db = SessionLocal()
    try:
        # Create default permissions
        print("\n📋 Creating default permissions...")
        permissions = create_default_permissions(db)
        
        # Create default roles
        print("\n👥 Creating default roles...")
        roles = create_default_roles(db, permissions)
        
        # Create default admin user
        print("\n👤 Creating default admin user...")
        admin_user = create_default_admin_user(db, roles)
        
        print("\n🎉 RBAC system initialization completed!")
        print(f"📧 Admin user: admin@example.com")
        print(f"🔑 Admin password: admin123")
        print("⚠️  IMPORTANT: Change the admin password in production!")
        
    except Exception as e:
        print(f"❌ Error during initialization: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    main()
