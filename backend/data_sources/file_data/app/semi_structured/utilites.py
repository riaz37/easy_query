import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
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

def initialize_llm_gemini(api_key: str = gemini_apikey, temperature: int = 0, model: str = gemini_model_name) -> ChatGoogleGenerativeAI:
    """
    Initialize and return the ChatGoogleGenerativeAI LLM instance.
    """
    return ChatGoogleGenerativeAI(model=model, temperature=temperature, google_api_key=api_key)