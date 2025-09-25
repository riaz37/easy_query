from fastapi import FastAPI, Depends, HTTPException, File, UploadFile, Request, Query, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
import traceback
import os
import shutil
from uuid import UUID

from database import SessionLocal, engine
from models import Base, User, RevokedToken
from services import (
    create_company,
    get_company,
    update_company,
    delete_company,
    create_data_source,
    update_data_source,
    delete_data_source,
    get_all_companies,
    get_company_details
)
from schemas import (
    CompanyCreate, 
    CompanyUpdate, 
    CompanyOut, 
    DataSourceCreate, 
    DataSourceUpdate, 
    DataSourceOut,
    PermissionCreate, PermissionUpdate, PermissionOut,
    RoleCreate, RoleUpdate, RoleOut, UserRoleAssignment, UserWithRoles, RoleWithUsers,
    PermissionCheck, PermissionCheckResponse,
    UserCreate, UserLogin, UserOut, LogoutResponse, ChangePasswordRequest, ChangePasswordResponse
)

# RBAC imports
from services.rbac_services import (
    create_permission, get_permission, get_permissions, update_permission, delete_permission,
    create_role, get_role, get_roles, update_role, delete_role,
    assign_roles_to_user, get_user_roles, remove_user_roles, get_users_by_role,
    get_permissions_by_resource, get_permissions_by_action, check_user_has_permission,
    get_user_permissions_list
)
from auth import (
    require_permission, require_role, get_current_user_obj,
    hash_password, verify_password, create_access_token_with_roles,
    SECRET_KEY, ALGORITHM, verify_token, get_current_user
)

# Initialize FastAPI app and database
app = FastAPI(title="Company Management API")
Base.metadata.create_all(bind=engine)
# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Company Routes
# @route.post("/companies", response_model=CompanyOut, tags=["Companies"])
# def create_company_route(company: CompanyCreate, db: Session = Depends(get_db)):
#     """Create a new company"""
#     return create_company(db, company)

@app.post("/companies", response_model=CompanyOut, tags=["Companies"])
def create_company_route(company: CompanyCreate, db: Session = Depends(get_db)):
    """Create a new company"""
    # Validate parent_id if it's provided
    if company.parent_id:
        parent_company = get_company(db, company.parent_id)
        if not parent_company:
            raise HTTPException(
                status_code=400,
                detail=f"Parent company with id {company.parent_id} does not exist"
            )
    
    # If parent_id is 0 or None, set it to None to make it a root company
    if not company.parent_id:
        company.parent_id = None
        
    return create_company(db, company)

# @route.get("/companies/{company_id}", tags=["Companies"])
# def read_company_route(company_id: int, db: Session = Depends(get_db)):
#     """Get company details by ID"""
#     company = get_company_details(db, company_id)
#     if not company:
#         raise HTTPException(status_code=404, detail="Company not found")
#     return company


@app.get("/companies/{company_id}", response_model=CompanyOut, tags=["Companies"])
def read_company_route(company_id: int, db: Session = Depends(get_db)):
    """Get company details by ID"""
    company = get_company_details(db, company_id)
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")
    return company



@app.get("/companies", response_model=list[CompanyOut], tags=["Companies"])
def get_all_companies_route(db: Session = Depends(get_db)):
    """Get list of all companies"""
    return get_all_companies(db)

@app.put("/companies/{company_id}", response_model=CompanyOut, tags=["Companies"])
def update_company_route(company_id: int, company: CompanyUpdate, db: Session = Depends(get_db)):
    """Update company details"""
    updated_company = update_company(db, company_id, company)
    if not updated_company:
        raise HTTPException(status_code=404, detail="Company not found")
    return updated_company

@app.delete("/companies/{company_id}", tags=["Companies"])
def delete_company_route(company_id: int, db: Session = Depends(get_db)):
    """Delete a company"""
    if not delete_company(db, company_id):
        raise HTTPException(status_code=404, detail="Company not found")
    return {"detail": "Company deleted"}

# DataSource Routes
@app.post("/companies/{company_id}/data-sources", response_model=DataSourceOut, tags=["DataSources"])
async def create_data_source_route(
    company_id: int,
    name: str,
    description: str = None,
    business_rules_file_name: str = None,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Create a new data source with file upload"""
    try:
        # Create storage directory
        storage_path = f"storage/{company_id}"
        os.makedirs(storage_path, exist_ok=True)
        
        # Save the uploaded file
        file_path = os.path.join(storage_path, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Create data source object
        data_source = DataSourceCreate(
            name=name,
            description=description,
            business_rules_file_name=business_rules_file_name
        )
        
        # Save to database
        return create_data_source(
            db=db,
            company_id=company_id,
            data=data_source,
            file_path=file_path,
            file_name=file.filename
        )
    except Exception as e:
        # Clean up file if database operation fails
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/data-sources/{data_source_id}", response_model=DataSourceOut, tags=["DataSources"])
def update_data_source_route(data_source_id: int, data: DataSourceUpdate, db: Session = Depends(get_db)):
    """Update data source details"""
    updated_source = update_data_source(db, data_source_id, data)
    if not updated_source:
        raise HTTPException(status_code=404, detail="Data source not found")
    return updated_source

@app.delete("/data-sources/{data_source_id}", tags=["DataSources"])
def delete_data_source_route(data_source_id: int, db: Session = Depends(get_db)):
    """Delete a data source"""
    if not delete_data_source(db, data_source_id):
        raise HTTPException(status_code=404, detail="Data source not found")
    return {"detail": "Data source deleted"}

# Error handling middleware
@app.middleware("http")
async def db_session_middleware(request, call_next):
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"detail": str(e)}
        )

    


#----------------Phase 3------------------------

@app.post("/signup", response_model=UserOut, tags=["Auth"])
def signup(user: UserCreate, db: Session = Depends(get_db)):
    # Check if user already exists
    if db.query(User).filter((User.username == user.username) | (User.email == user.email)).first():
        raise HTTPException(status_code=400, detail="Username or email already registered")
    user_obj = User(
        username=user.username,
        email=user.email,
        password_hash=hash_password(user.password)
    )
    db.add(user_obj)
    db.commit()
    db.refresh(user_obj)
    return user_obj

@app.post("/login", tags=["Auth"])
def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if not db_user or not verify_password(user.password, db_user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username or password")
    
    if not db_user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User account is deactivated")
    
    access_token = create_access_token_with_roles(db_user, db)
    return {"access_token": access_token, "token_type": "bearer"}




#-----------Phase 3-1---------------------------

from jose import jwt, JWTError # pyright: ignore[reportMissingModuleSource]

@app.post("/logout", response_model=LogoutResponse, tags=["Auth"])
def logout(request: Request, db: Session = Depends(get_db), current_user: str = Depends(get_current_user)):
    """
    Sign out the current user by revoking their token
    """
    try:
        # Get the token from the request headers
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid authorization header"
            )
        
        token = auth_header.split(" ")[1]
        payload = verify_token(token)
        jti = payload.get("jti")
        
        if not jti:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid token format"
            )

        # Check if token is already revoked
        existing_revoked = db.query(RevokedToken).filter(RevokedToken.jti == jti).first()
        if existing_revoked:
            return LogoutResponse(message="Token already revoked")

        # Save the revoked token
        revoked = RevokedToken(jti=jti)
        db.add(revoked)
        db.commit()

        return LogoutResponse(message="Successfully signed out")
        
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during sign out: {str(e)}"
        )

@app.post("/logout-all", response_model=LogoutResponse, tags=["Auth"])
def logout_all_sessions(request: Request, db: Session = Depends(get_db), current_user: str = Depends(get_current_user)):
    """
    Sign out from all sessions for the current user
    Note: This is a simplified implementation. In a production system,
    you might want to track user sessions more granularly.
    """
    try:
        # Get the current user object
        user = db.query(User).filter(User.username == current_user).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # For this implementation, we'll just revoke the current token
        # In a more sophisticated system, you might want to track all active sessions
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            payload = verify_token(token)
            jti = payload.get("jti")
            
            if jti:
                # Check if token is already revoked
                existing_revoked = db.query(RevokedToken).filter(RevokedToken.jti == jti).first()
                if not existing_revoked:
                    revoked = RevokedToken(jti=jti)
                    db.add(revoked)
        
        db.commit()
        return LogoutResponse(message="Successfully signed out from all sessions")
        
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during sign out: {str(e)}"
        )

# Protected routes example
@app.get("/profile", response_model=UserOut, tags=["Auth"])
def get_user_profile(current_user: str = Depends(get_current_user), db: Session = Depends(get_db)):
    """
    Get current user profile (protected route example)
    """
    user = db.query(User).filter(User.username == current_user).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    return user

@app.get("/protected", tags=["Auth"])
def protected_route(current_user: str = Depends(get_current_user)):
    """
    Example of a protected route that requires authentication
    """
    return {"message": f"Hello {current_user}, this is a protected route!"}


@app.post("/change-password", response_model=ChangePasswordResponse, tags=["Auth"])
def change_password(
    payload: ChangePasswordRequest,
    request: Request,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user),
    local_kw: str = Query(default="")
):
    """
    Change the current user's password.
    - Verifies current password
    - Updates stored password hash
    - Revokes current token so user must re-login
    """
    # Fetch user
    user = db.query(User).filter(User.username == current_user).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    # Verify current password
    if not verify_password(payload.current_password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Current password is incorrect")

    # Update password hash
    user.password_hash = hash_password(payload.new_password)
    db.add(user)

    # Revoke current token
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        payload_token = verify_token(token)
        jti = payload_token.get("jti")
        if jti:
            existing_revoked = db.query(RevokedToken).filter(RevokedToken.jti == jti).first()
            if not existing_revoked:
                revoked = RevokedToken(jti=jti)
                db.add(revoked)

    db.commit()
    return ChangePasswordResponse(message="Password changed successfully. Please login again.")

#----------------Phase 4: Role and Permission Management------------------------

# Permission Management Endpoints
@app.post("/permissions", response_model=PermissionOut, tags=["Permissions"])
def create_permission_route(
    permission: PermissionCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission("permissions", "create"))
):
    """Create a new permission"""
    return create_permission(db, permission)

@app.get("/permissions", response_model=list[PermissionOut], tags=["Permissions"])
def get_permissions_route(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission("permissions", "read"))
):
    """Get all permissions with pagination"""
    return get_permissions(db, skip=skip, limit=limit)

@app.get("/permissions/{permission_id}", response_model=PermissionOut, tags=["Permissions"])
def get_permission_route(
    permission_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission("permissions", "read"))
):
    """Get permission by ID"""
    permission = get_permission(db, permission_id)
    if not permission:
        raise HTTPException(status_code=404, detail="Permission not found")
    return permission

@app.put("/permissions/{permission_id}", response_model=PermissionOut, tags=["Permissions"])
def update_permission_route(
    permission_id: UUID,
    permission: PermissionUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission("permissions", "update"))
):
    """Update permission"""
    updated_permission = update_permission(db, permission_id, permission)
    if not updated_permission:
        raise HTTPException(status_code=404, detail="Permission not found")
    return updated_permission

@app.delete("/permissions/{permission_id}", tags=["Permissions"])
def delete_permission_route(
    permission_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission("permissions", "delete"))
):
    """Delete permission"""
    if not delete_permission(db, permission_id):
        raise HTTPException(status_code=404, detail="Permission not found")
    return {"detail": "Permission deleted"}

# Role Management Endpoints
@app.post("/roles", response_model=RoleOut, tags=["Roles"])
def create_role_route(
    role: RoleCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission("roles", "create"))
):
    """Create a new role"""
    return create_role(db, role)

@app.get("/roles", response_model=list[RoleOut], tags=["Roles"])
def get_roles_route(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission("roles", "read"))
):
    """Get all roles with pagination"""
    return get_roles(db, skip=skip, limit=limit)

@app.get("/roles/{role_id}", response_model=RoleOut, tags=["Roles"])
def get_role_route(
    role_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission("roles", "read"))
):
    """Get role by ID"""
    role = get_role(db, role_id)
    if not role:
        raise HTTPException(status_code=404, detail="Role not found")
    return role

@app.put("/roles/{role_id}", response_model=RoleOut, tags=["Roles"])
def update_role_route(
    role_id: UUID,
    role: RoleUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission("roles", "update"))
):
    """Update role"""
    updated_role = update_role(db, role_id, role)
    if not updated_role:
        raise HTTPException(status_code=404, detail="Role not found")
    return updated_role

@app.delete("/roles/{role_id}", tags=["Roles"])
def delete_role_route(
    role_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission("roles", "delete"))
):
    """Delete role"""
    if not delete_role(db, role_id):
        raise HTTPException(status_code=404, detail="Role not found")
    return {"detail": "Role deleted"}

# User-Role Assignment Endpoints
@app.post("/users/{user_id}/roles", response_model=UserWithRoles, tags=["User Management"])
def assign_roles_to_user_route(
    user_id: UUID,
    role_assignment: UserRoleAssignment,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission("users", "update"))
):
    """Assign roles to a user"""
    return assign_roles_to_user(db, user_id, role_assignment.role_ids)

@app.get("/users/{user_id}/roles", response_model=list[RoleOut], tags=["User Management"])
def get_user_roles_route(
    user_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission("users", "read"))
):
    """Get all roles for a user"""
    return get_user_roles(db, user_id)

@app.delete("/users/{user_id}/roles", response_model=UserWithRoles, tags=["User Management"])
def remove_user_roles_route(
    user_id: UUID,
    role_ids: list[UUID],
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission("users", "update"))
):
    """Remove specific roles from a user"""
    return remove_user_roles(db, user_id, role_ids)

@app.get("/roles/{role_id}/users", response_model=list[UserOut], tags=["Roles"])
def get_users_by_role_route(
    role_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission("roles", "read"))
):
    """Get all users with a specific role"""
    return get_users_by_role(db, role_id)

# Permission Check Endpoints
@app.post("/check-permission", response_model=PermissionCheckResponse, tags=["Permissions"])
def check_permission_route(
    permission_check: PermissionCheck,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_obj)
):
    """Check if current user has a specific permission"""
    has_perm = check_user_has_permission(db, current_user.id, permission_check.resource, permission_check.action)
    return PermissionCheckResponse(
        has_permission=has_perm,
        message=f"User {'has' if has_perm else 'does not have'} permission: {permission_check.resource}:{permission_check.action}"
    )

@app.get("/my-permissions", tags=["Permissions"])
def get_my_permissions_route(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_obj)
):
    """Get all permissions for the current user"""
    permissions = get_user_permissions_list(db, current_user.id)
    return {
        "user_id": str(current_user.id),
        "username": current_user.username,
        "permissions": permissions
    }

# Utility Endpoints
@app.get("/permissions/resource/{resource}", response_model=list[PermissionOut], tags=["Permissions"])
def get_permissions_by_resource_route(
    resource: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission("permissions", "read"))
):
    """Get all permissions for a specific resource"""
    return get_permissions_by_resource(db, resource)

@app.get("/permissions/action/{action}", response_model=list[PermissionOut], tags=["Permissions"])
def get_permissions_by_action_route(
    action: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission("permissions", "read"))
):
    """Get all permissions for a specific action"""
    return get_permissions_by_action(db, action)



