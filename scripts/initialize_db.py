from packages.rag.store.vector_store import VectorStore

if __name__ == "__main__":
    input_dir = "data/index"
    index_dir = "data/index"
    store = VectorStore(index_dir)
    store.build_index(input_dir)
    print("Vector database initialized.")
