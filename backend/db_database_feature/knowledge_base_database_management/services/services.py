from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from ..models import Company, DataSource
from ..schemas import CompanyCreate, CompanyUpdate, DataSourceCreate, DataSourceUpdate

from ..config import STORAGE_DIR

from pathlib import Path






#-------------------Company Services----------------------------


def create_company(db: Session, company: CompanyCreate):
    try:
        db_company = Company(**company.dict())
        db.add(db_company)
        db.commit()
        db.refresh(db_company)
        return db_company
    except SQLAlchemyError as e:
        db.rollback()
        raise Exception(f"Database error: {str(e)}")

def get_company(db: Session, company_id: int):
    try:
        return db.query(Company).filter(Company.id == company_id).first()
    except SQLAlchemyError as e:
        raise Exception(f"Database error: {str(e)}")

def update_company(db: Session, company_id: int, company: CompanyUpdate):
    try:
        db_company = get_company(db, company_id)
        if db_company:
            for key, value in company.dict(exclude_unset=True).items():
                setattr(db_company, key, value)
            db.commit()
            db.refresh(db_company)
        return db_company
    except SQLAlchemyError as e:
        db.rollback()
        raise Exception(f"Database error: {str(e)}")

def delete_company(db: Session, company_id: int):
    try:
        db_company = get_company(db, company_id)
        if db_company:
            db.delete(db_company)
            db.commit()
            return True
        return False
    except SQLAlchemyError as e:
        db.rollback()
        raise Exception(f"Database error: {str(e)}")
    



def get_company_details(db: Session, company_id: int):
    try:
        # Query company with related data including parent
        db_company = db.query(Company).filter(Company.id == company_id).first()
        
        if not db_company:
            return None
            
        # Get parent company details if exists
        parent_company = None
        if db_company.parent_id:
            parent = db.query(Company).filter(Company.id == db_company.parent_id).first()
            if parent:
                parent_company = {
                    "id": parent.id,
                    "name": parent.name,
                    "description": parent.description,
                    "created_at": parent.created_at,
                    "updated_at": parent.updated_at
                }
            
        # Convert to dict and structure the response
        company_data = {
            "name": db_company.name,
            "description": db_company.description,
            "parent_id": db_company.parent_id,
            "parent_details": parent_company,  # Add parent details
            "id": db_company.id,
            "created_at": db_company.created_at,
            "updated_at": db_company.updated_at,
            "data_sources": []
        }
        
        # Add data sources if they exist
        for source in db_company.data_sources:
            data_source = {
                "name": source.name,
                "description": source.description,
                "business_rules_file_name": source.business_rules_file_name,
                "file_path": source.file_path,
                "file_name": source.file_name,
                "id": source.id,
                "company_id": source.company_id,
                "created_at": source.created_at,
                "updated_at": source.updated_at
            }
            company_data["data_sources"].append(data_source)
            
        return company_data
    except SQLAlchemyError as e:
        raise Exception(f"Database error: {str(e)}")



#-------------------Data SDource Service----------------------------

def create_data_source(db: Session, company_id: int, data: DataSourceCreate, file_path: str, file_name: str):
    try:
        db_data_source = DataSource(
            name=data.name,
            description=data.description,
            business_rules_file_name=data.business_rules_file_name,
            company_id=company_id,
            file_path=file_path,
            file_name=file_name
        )
        db.add(db_data_source)
        db.commit()
        db.refresh(db_data_source)
        return db_data_source
    except SQLAlchemyError as e:
        db.rollback()
        raise Exception(f"Database error: {str(e)}")


def update_data_source(db: Session, data_source_id: int, data: DataSourceUpdate):
    try:
        db_data_source = db.query(DataSource).filter(DataSource.id == data_source_id).first()
        if db_data_source:
            for key, value in data.dict(exclude_unset=True).items():
                setattr(db_data_source, key, value)
            db.commit()
            db.refresh(db_data_source)
        return db_data_source
    except SQLAlchemyError as e:
        db.rollback()
        raise Exception(f"Database error: {str(e)}")

def delete_data_source(db: Session, data_source_id: int):
    try:
        db_data_source = db.query(DataSource).filter(DataSource.id == data_source_id).first()
        if db_data_source:
            db.delete(db_data_source)
            db.commit()
            return True
        return False
    except SQLAlchemyError as e:
        db.rollback()
        raise Exception(f"Database error: {str(e)}")
    


def get_all_companies(db: Session):
    try:
        return db.query(Company).all()
    except SQLAlchemyError as e:
        raise Exception(f"Database error: {str(e)}")
