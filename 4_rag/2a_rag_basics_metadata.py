import os

# Disable ChromaDB telemetry for cleaner output
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
# Use fast open-source embeddings instead of OpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_openai import OpenAIEmbeddings

# Define the directory containing the text files and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
books_dir = os.path.join(current_dir, "books")
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

print(f"Books directory: {books_dir}")
print(f"Persistent directory: {persistent_directory}")

# Check if the Chroma vector store already exists
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    # Ensure the books directory exists
    if not os.path.exists(books_dir):
        raise FileNotFoundError(
            f"The directory {books_dir} does not exist. Please check the path."
        )

    # List all text files in the directory
    book_files = [f for f in os.listdir(books_dir) if f.endswith(".txt")]

    # Read the text content from each file and store it with metadata
    documents = []
    for book_file in book_files:
        file_path = os.path.join(books_dir, book_file)
        try:
            # Use UTF-8 encoding to handle special characters
            loader = TextLoader(file_path, encoding='utf-8')
            book_docs = loader.load()
            for doc in book_docs:
                # Add metadata to each document indicating its source
                doc.metadata = {"source": book_file}
                documents.append(doc)
            print(f"✅ Loaded: {book_file}")
        except UnicodeDecodeError:
            print(f"⚠️  Skipping {book_file} due to encoding issues")
            continue

    # Split the documents into chunks (optimized for faster processing)
    text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=200)  # Larger chunks = fewer embeddings needed
    docs = text_splitter.split_documents(documents)

    # Display information about the split documents
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Books processed: {len(book_files)}")
    
    # Estimate processing time
    estimated_time = len(docs) * 1  # Rough estimate: 1 second per chunk with fast embeddings
    print(f"⏱️  Estimated processing time: ~{estimated_time//60} minutes {estimated_time%60} seconds")

    # Create embeddings using FAST open-source model (same as 1a_rag_basics.py)
    print("\n--- Creating embeddings ---")
    print("🚀 Using fast open-source embedding model...")
    
    # Use HuggingFace Sentence Transformers - extremely fast and open source
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",  # Small, fast, high-quality model
        model_kwargs={'device': 'cpu'},  # Use CPU (change to 'cuda' if you have GPU)
        encode_kwargs={'normalize_embeddings': True}  # Normalize for better similarity search
    )
    print("✅ Embedding model initialized successfully")

    # Create the vector store and persist it
    print("\n--- Creating and persisting vector store ---")
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_directory)
    print("\n--- Finished creating and persisting vector store ---")

else:
    print("Vector store already exists. No need to initialize.")
