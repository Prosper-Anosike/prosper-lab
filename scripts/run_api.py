import uvicorn
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

if __name__ == "__main__":
    uvicorn.run("apps.api.main:app", host="0.0.0.0", port=8000, reload=True)
