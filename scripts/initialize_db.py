import sys
from pathlib import Path

# Ensure the project root is available on sys.path so `packages` resolves when running as a script.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from packages.rag.store.vector_store import VectorStore

if __name__ == "__main__":
    input_dir = "data/index"
    index_dir = "data/index"
    store = VectorStore(index_dir)
    store.build_index(input_dir)
    print("Vector database initialized.")
