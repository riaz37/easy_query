import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
# Load environment variables
load_dotenv(override=True)
# Retrieve API key and model name from environment variables
gemini_apikey = os.getenv("google_api_key")
if gemini_apikey is None:
    print("Warning: 'google_api_key' not found in environment variables.")



gemini_model_name = os.getenv("google_gemini_name", "gemini-1.5-pro")
if gemini_model_name is None:
    print("Warning: 'google_gemini_name' not found in environment variables, using default 'gemini-1.5-pro'.")


google_gemini_name_light = os.getenv("google_gemini_name_light", "gemini-1.5-pro")
if google_gemini_name_light is None:
    print("Warning: 'google_gemini_name_light' not found in environment variables, using default 'gemini-1.5-pro'.")
thinking_model = os.getenv("thinking_model", "deepseek-r1-distill-llama-70b")
if thinking_model is None:
    print("Warning: 'groq_thinking_model' not found in environment variables, using default 'deepseek-r1-distill-llama-70b'.")

def initialize_llm_gemini(api_key: str = gemini_apikey, temperature: int = 0, model: str = gemini_model_name) -> ChatGoogleGenerativeAI:
    """
    Initialize and return the ChatGoogleGenerativeAI LLM instance.
    """
    return ChatGoogleGenerativeAI(model=model, temperature=temperature, google_api_key=api_key)

def initialize_llm_gemini_light(api_key: str = gemini_apikey, temperature: int = 0, model: str = google_gemini_name_light) -> ChatGoogleGenerativeAI:
    """
    Initialize and return the ChatGoogleGenerativeAI LLM instance with light settings.
    """
    return ChatGoogleGenerativeAI(model=model, temperature=temperature, google_api_key=api_key)

groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key is None:
    print("Warning: 'GROQ_API_KEY' not found in environment variables.")

groq_model_name = os.getenv("Groqmodelname", "llama3-70b-8192")
if groq_model_name is None:
    print("Warning: 'Groqmodelname' not found in environment variables, using default.")

def initialize_llm(api_key: str, temperature: int = 0, model_name: str = "llama3-70b-8192") -> ChatGroq:
    """
    Initialize and return the ChatGroq LLM instance.
    """
    return ChatGroq(groq_api_key=api_key, temperature=temperature, model_name=model_name)

print("LLM initialized with model:", gemini_model_name)
print("LLM initialized with model:", groq_model_name)