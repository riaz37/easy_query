import os
from fastapi import FastAPI, UploadFile, File, APIRouter
from typing import List, Optional
from pydantic import BaseModel
from dotenv import load_dotenv
import pypdf
import pandas as pd
import docx
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.embeddings import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalQA
from langchain.memory import ConversationBufferMemory
import psycopg2
from datetime import datetime

router = APIRouter()
load_dotenv()

# Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PG_CONNECTION_STRING = os.getenv("PG_CONNECTION_STRING")

# Initialize models
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3, google_api_key=GOOGLE_API_KEY)

# Initialize vector store
vectorstore = PGVector(
    connection_string=PG_CONNECTION_STRING,
    embedding_function=embeddings,
    collection_name="document_store"
)

class FileInfo(BaseModel):
    file_id: str
    filename: str
    upload_date: str
    file_type: str

class QueryRequest(BaseModel):
    question: str
    file_ids: List[str]
    user_id: str = "default"

# File processors
def process_pdf(file_path: str) -> str:
    with open(file_path, 'rb') as file:
        pdf = pypdf.PdfReader(file)
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

def process_excel(file_path: str) -> str:
    df = pd.read_excel(file_path)
    return df.to_string()

def process_docx(file_path: str) -> str:
    doc = docx.Document(file_path)
    return " ".join([paragraph.text for paragraph in doc.paragraphs])

def process_text(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# File processor mapping
FILE_PROCESSORS = {
    '.pdf': process_pdf,
    '.xlsx': process_excel,
    '.docx': process_docx,
    '.txt': process_text
}

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Save file temporarily
        temp_path = f"temp_{file.filename}"
        with open(temp_path, 'wb') as f:
            content = await file.read()
            f.write(content)

        # Process file based on extension
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in FILE_PROCESSORS:
            raise ValueError(f"Unsupported file type: {ext}")

        # Extract text
        text = FILE_PROCESSORS[ext](temp_path)

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_text(text)

        # Generate unique file ID
        file_id = f"doc_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # Add metadata to chunks
        docs = [{"text": chunk, "metadata": {
            "file_id": file_id,
            "filename": file.filename,
            "upload_date": datetime.now().isoformat()
        }} for chunk in chunks]

        # Store in vector database
        vectorstore.add_texts(
            texts=[doc["text"] for doc in docs],
            metadatas=[doc["metadata"] for doc in docs]
        )

        # Cleanup
        os.remove(temp_path)

        return {"status": "success", "file_id": file_id}

    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.get("/files")
async def list_files() -> List[FileInfo]:
    # Query unique files from vector store
    query = """
    SELECT DISTINCT metadata->>'file_id' as file_id,
                    metadata->>'filename' as filename,
                    metadata->>'upload_date' as upload_date
    FROM langchain_pg_embedding;
    """
    conn = psycopg2.connect(PG_CONNECTION_STRING)
    cursor = conn.cursor()
    cursor.execute(query)
    files = cursor.fetchall()
    cursor.close()
    conn.close()

    return [
        FileInfo(
            file_id=file[0],
            filename=file[1],
            upload_date=file[2],
            file_type=os.path.splitext(file[1])[1]
        )
        for file in files
    ]

@router.post("/query")
async def query_documents(request: QueryRequest):
    try:
        # Create memory for conversation history
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # Filter documents based on selected file_ids
        filtered_vectorstore = PGVector(
            connection_string=PG_CONNECTION_STRING,
            embedding_function=embeddings,
            collection_name="document_store",
            filter={"file_id": {"$in": request.file_ids}}
        )

        # Create retrieval chain with hybrid search
        retriever = filtered_vectorstore.as_retriever(
            search_type="hybrid",  # Combines similarity and keyword search
            search_kwargs={
                "k": 5,  # Number of documents to retrieve
                "score_threshold": 0.7,  # Minimum similarity score
            }
        )

        # Create QA chain
        qa_chain = ConversationalRetrievalQA.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            verbose=True
        )

        # Get response
        response = qa_chain({"question": request.question})

        # Format response with sources
        sources = [
            {
                "file_id": doc.metadata["file_id"],
                "filename": doc.metadata["filename"],
                "content": doc.page_content[:200] + "..."  # Preview of content
            }
            for doc in response["source_documents"]
        ]

        return {
            "status_code": 200,
            "payload": {
                "answer": response["answer"],
                "sources": sources,
                "conversation_history": memory.chat_memory.messages
            }
        }

    except Exception as e:
        return {"status_code": 500, "payload": {"error": str(e)}}

@router.get("/conversation-history/{user_id}")
async def get_history(user_id: str):
    # Implement conversation history retrieval similar to MSSQL agent
    pass

@router.post("/clear-history/{user_id}")
async def clear_history(user_id: str):
    # Implement conversation history clearing similar to MSSQL agent
    pass