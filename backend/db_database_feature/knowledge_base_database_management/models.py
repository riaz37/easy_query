from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Table, Boolean
# from sqlalchemy.dialects.mssql import UNIQUEIDENTIFIER  # Commented out - MSSQL specific
from sqlalchemy.dialects.postgresql import UUID  # PostgreSQL UUID type
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .database import Base
import uuid

# Association tables for many-to-many relationships
user_roles = Table(
    'user_roles',
    Base.metadata,
    Column('user_id', UUID(as_uuid=True), ForeignKey('users.id'), primary_key=True),  # Removed 'dbo.' schema prefix
    Column('role_id', UUID(as_uuid=True), ForeignKey('roles.id'), primary_key=True),  # Removed 'dbo.' schema prefix
    # schema='dbo'  # Commented out - PostgreSQL doesn't use dbo schema by default
)

role_permissions = Table(
    'role_permissions',
    Base.metadata,
    Column('role_id', UUID(as_uuid=True), ForeignKey('roles.id'), primary_key=True),  # Removed 'dbo.' schema prefix
    Column('permission_id', UUID(as_uuid=True), ForeignKey('permissions.id'), primary_key=True),  # Removed 'dbo.' schema prefix
    # schema='dbo'  # Commented out - PostgreSQL doesn't use dbo schema by default
)

class Company(Base):
    __tablename__ = "companies"
    # __table_args__ = {"schema": "dbo"}  # Commented out - PostgreSQL doesn't use dbo schema by default

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(String)
    parent_id = Column(Integer, ForeignKey("companies.id"))  # Removed 'dbo.' schema prefix
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Define relationship with DataSource
    data_sources = relationship("DataSource", back_populates="company", cascade="all, delete-orphan")

class DataSource(Base):
    __tablename__ = "data_sources"
    # __table_args__ = {"schema": "dbo"}  # Commented out - PostgreSQL doesn't use dbo schema by default

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(String)
    business_rules_file_name = Column(String)
    file_path = Column(String)
    file_name = Column(String)
    db_path = Column(String)
    db_name = Column(String)
    company_id = Column(Integer, ForeignKey("companies.id"))  # Removed 'dbo.' schema prefix
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Define relationship with Company
    company = relationship("Company", back_populates="data_sources")
    
    # --------------Phase 3--------------------

from sqlalchemy import Column, Integer, String, DateTime, text
from sqlalchemy.sql import func
from .database import Base

class User(Base):
    __tablename__ = "users"
    # __table_args__ = {"schema": "dbo"}  # Commented out - PostgreSQL doesn't use dbo schema by default
    id = Column(
        UUID(as_uuid=True),  # Changed from UNIQUEIDENTIFIER to UUID
        primary_key=True,
        index=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()")  # Changed from NEWID() to gen_random_uuid() for PostgreSQL
    )
    username = Column(String(100), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    roles = relationship("Role", secondary=user_roles, back_populates="users")

class Role(Base):
    __tablename__ = "roles"
    # __table_args__ = {"schema": "dbo"}  # Commented out - PostgreSQL doesn't use dbo schema by default
    
    id = Column(
        UUID(as_uuid=True),  # Changed from UNIQUEIDENTIFIER to UUID
        primary_key=True,
        index=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()")  # Changed from NEWID() to gen_random_uuid() for PostgreSQL
    )
    name = Column(String(100), unique=True, nullable=False, index=True)
    description = Column(String(255))
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    users = relationship("User", secondary=user_roles, back_populates="roles")
    permissions = relationship("Permission", secondary=role_permissions, back_populates="roles")

class Permission(Base):
    __tablename__ = "permissions"
    # __table_args__ = {"schema": "dbo"}  # Commented out - PostgreSQL doesn't use dbo schema by default
    
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        index=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()")  # Changed from NEWID() to gen_random_uuid() for PostgreSQL
    )
    name = Column(String(100), unique=True, nullable=False, index=True)
    description = Column(String(255))
    resource = Column(String(100), nullable=False)  # e.g., "companies", "users", "roles"
    action = Column(String(50), nullable=False)     # e.g., "create", "read", "update", "delete"
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    roles = relationship("Role", secondary=role_permissions, back_populates="permissions")

class RevokedToken(Base):
    __tablename__ = "revoked_tokens"
    # __table_args__ = {"schema": "dbo"}  # Commented out - PostgreSQL doesn't use dbo schema by default
    id = Column(Integer, primary_key=True, index=True)
    jti = Column(String(255), unique=True, nullable=False, index=True)
    revoked_at = Column(DateTime(timezone=True), server_default=func.now())

