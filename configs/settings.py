import os
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

class Settings:
    PROVIDER = os.getenv("PROVIDER", "azure")

    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_CHAT_DEPLOYMENT = os.getenv("AZURE_CHAT_DEPLOYMENT")
    AZURE_EMBED_DEPLOYMENT = os.getenv("AZURE_EMBED_DEPLOYMENT")

    CHAT_MODEL = os.getenv("CHAT_MODEL")
    EMBED_MODEL = os.getenv("EMBED_MODEL")
    VECTOR_DB = os.getenv("VECTOR_DB")
    INDEX_PATH = os.getenv("INDEX_PATH")
    PORT = int(os.getenv("PORT", "8000"))

    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP"))
    CHUNK_STRATEGY = os.getenv("CHUNK_STRATEGY")
    INPUT_DIR = os.getenv("INPUT_DIR")
    OUTPUT_DIR = os.getenv("OUTPUT_DIR")
    MANIFEST_FILE = os.getenv("MANIFEST_FILE")

settings = Settings()
