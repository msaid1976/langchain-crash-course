import os
import time
from concurrent.futures import ThreadPoolExecutor

# Disable ChromaDB telemetry for cleaner output
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
# Use the modern Chroma import that supports pydantic v2
from langchain_chroma import Chroma
# from langchain_openai import OpenAIEmbeddings
# Use stable community embeddings instead of the problematic langchain-huggingface
from langchain_community.embeddings import HuggingFaceEmbeddings

# Define the directory containing the text file and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "odyssey.txt")
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# Check if the Chroma vector store already exists
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    # Ensure the text file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_path} does not exist. Please check the path."
        )

    # Read the text content from the file
    loader = TextLoader(file_path, encoding='utf-8')
    documents = loader.load()

    # Split the document into chunks (optimized for faster processing)
    text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=100)  # Much larger chunks = way fewer embeddings needed
    docs = text_splitter.split_documents(documents)

    # Display information about the split documents
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Sample chunk:\n{docs[0].page_content}\n")
    
    # Estimate processing time
    estimated_time = len(docs) * 2  # Rough estimate: 2 seconds per chunk
    print(f"⏱️  Estimated processing time: ~{estimated_time//60} minutes {estimated_time%60} seconds")

    # # Create embeddings
    # print("\n--- Creating embeddings ---")
    # embeddings = OpenAIEmbeddings(
    #     model="text-embedding-3-small"
    # )  # Update to a valid embedding model if needed
    # print("\n--- Finished creating embeddings ---")

    # Create embeddings using FAST open-source model (much faster than Ollama!)
    print("\n--- Creating embeddings ---")
    print("🚀 Using fast open-source embedding model...")
    embedding_start = time.time()
    
    # Use HuggingFace Sentence Transformers - extremely fast and open source
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",  # Small, fast, high-quality model
        model_kwargs={'device': 'cpu'},  # Use CPU (change to 'cuda' if you have GPU)
        encode_kwargs={'normalize_embeddings': True}  # Normalize for better similarity search
    )
    
    embedding_time = time.time() - embedding_start
    print(f"✅ Embedding model initialized in {embedding_time:.2f} seconds")

    # Create the vector store and persist it automatically
    print("\n--- Creating vector store ---")
    print(f"🔄 Processing {len(docs)} document chunks...")
    vectorstore_start = time.time()
    
    # Process in batches for better performance and progress tracking
    batch_size = 100  # Larger batches for HuggingFace (it's much faster)
    total_batches = (len(docs) + batch_size - 1) // batch_size
    
    # Create initial empty vector store
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
    
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i:i+batch_size]
        batch_num = (i // batch_size) + 1
        
        print(f"  📦 Processing batch {batch_num}/{total_batches} ({len(batch_docs)} documents)...")
        batch_start = time.time()
        
        # Add batch to vector store
        db.add_documents(batch_docs)
        
        batch_time = time.time() - batch_start
        elapsed_total = time.time() - vectorstore_start
        remaining_batches = total_batches - batch_num
        estimated_remaining = (elapsed_total / batch_num) * remaining_batches
        
        print(f"    ✅ Batch completed in {batch_time:.1f}s | ETA: {estimated_remaining/60:.1f} minutes")
    
    vectorstore_time = time.time() - vectorstore_start
    total_time = time.time() - embedding_start
    
    print(f"✅ Vector store created in {vectorstore_time:.2f} seconds")
    print(f"🎉 Total processing time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    print("\n--- Finished creating vector store ---")

else:
    print("Vector store already exists. No need to initialize.")
