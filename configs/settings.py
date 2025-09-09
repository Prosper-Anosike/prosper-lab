import os
from dotenv import load_dotenv

# Load environment variables from .env file, override system variables
load_dotenv(override=True)

class Settings:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

settings = Settings()
