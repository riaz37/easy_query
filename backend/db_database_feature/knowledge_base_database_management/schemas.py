from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List
from uuid import UUID

class DataSourceBase(BaseModel):
    name: str
    description: Optional[str] = None
    business_rules_file_name: Optional[str] = None
    file_path: Optional[str] = None
    file_name: Optional[str] = None

class DataSourceCreate(BaseModel):
    name: str
    description: Optional[str] = None
    business_rules_file_name: Optional[str] = None

    class Config:
        from_attributes = True

class DataSourceUpdate(DataSourceBase):
    pass

class DataSourceOut(DataSourceBase):
    id: int
    company_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class CompanyBase(BaseModel):
    name: str
    description: Optional[str] = None
    parent_id: Optional[int] = None

class CompanyCreate(BaseModel):
    name: str
    description: str | None = None
    parent_id: int | None = None

class CompanyUpdate(CompanyBase):
    pass



class ParentCompany(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class CompanyOut(CompanyBase):
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    parent_details: Optional[ParentCompany] = None
    data_sources: List[DataSourceOut] = []

    class Config:
        from_attributes = True

    class Config:
        from_attributes = True




# --------------------Phase 3-------------------


from pydantic import BaseModel, EmailStr, Field

class UserCreate(BaseModel):
    username: str
    email: str
    password: str


class UserLogin(BaseModel):
    username: str
    password: str


class UserOut(BaseModel):
    id: UUID = Field(serialization_alias="user_id")
    username: str
    email: str
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class LogoutResponse(BaseModel):
    message: str
    success: bool = True


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


class ChangePasswordResponse(BaseModel):
    message: str
    success: bool = True

# --------------------Phase 4: Role and Permission Schemas-------------------

class PermissionBase(BaseModel):
    name: str
    description: Optional[str] = None
    resource: str
    action: str

class PermissionCreate(PermissionBase):
    pass

class PermissionUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    resource: Optional[str] = None
    action: Optional[str] = None
    is_active: Optional[bool] = None

class PermissionOut(PermissionBase):
    id: UUID
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class RoleBase(BaseModel):
    name: str
    description: Optional[str] = None

class RoleCreate(RoleBase):
    permission_ids: Optional[List[UUID]] = None

class RoleUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    is_active: Optional[bool] = None
    permission_ids: Optional[List[UUID]] = None

class RoleOut(RoleBase):
    id: UUID
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime] = None
    permissions: List[PermissionOut] = []

    class Config:
        from_attributes = True

class UserRoleAssignment(BaseModel):
    user_id: UUID
    role_ids: List[UUID]

class UserWithRoles(UserOut):
    roles: List[RoleOut] = []

    class Config:
        from_attributes = True

class RoleWithUsers(RoleOut):
    users: List[UserOut] = []

    class Config:
        from_attributes = True

class PermissionCheck(BaseModel):
    resource: str
    action: str

class PermissionCheckResponse(BaseModel):
    has_permission: bool
    message: str