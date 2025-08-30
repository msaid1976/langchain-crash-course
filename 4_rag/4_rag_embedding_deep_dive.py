import os

# Disable ChromaDB telemetry for cleaner output
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Use community embeddings for consistency
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
# from langchain_openai import OpenAIEmbeddings

# Define the directory containing the text file and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "odyssey.txt")
db_dir = os.path.join(current_dir, "db")

# Check if the text file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"The file {file_path} does not exist. Please check the path."
    )

# Read the text content from the file
loader = TextLoader(file_path, encoding='utf-8')
documents = loader.load()

# Split the document into chunks
text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=200)  # Optimized chunk size
docs = text_splitter.split_documents(documents)

# Display information about the split documents
print("\n--- Document Chunks Information ---")
print(f"Number of document chunks: {len(docs)}")
print(f"Sample chunk:\n{docs[0].page_content}\n")


# Function to create and persist vector store
def create_vector_store(docs, embeddings, store_name):
    persistent_directory = os.path.join(db_dir, store_name)
    if not os.path.exists(persistent_directory):
        print(f"\n--- Creating vector store {store_name} ---")
        Chroma.from_documents(
            docs, embeddings, persist_directory=persistent_directory)
        print(f"--- Finished creating vector store {store_name} ---")
    else:
        print(
            f"Vector store {store_name} already exists. No need to initialize.")


# 1. HuggingFace Embeddings (Fast and Free)
# Uses open-source models from HuggingFace.
# Ideal for fast, local embeddings without API costs.
# Note: Running HuggingFace models locally incurs no cost other than computational resources.
print("\n--- Using HuggingFace Embeddings (all-MiniLM-L6-v2) ---")
fast_embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",  # Fast, lightweight model
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
create_vector_store(docs, fast_embeddings, "chroma_db_fast_all-MiniLM")

# 2. HuggingFace Embeddings (Higher Quality)
# Uses a larger, more accurate model for better embeddings.
# Note: Find other models at https://huggingface.co/models?other=embeddings
print("\n--- Using HuggingFace Embeddings (all-mpnet-base-v2) ---")
quality_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",  # Higher quality model
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
create_vector_store(docs, quality_embeddings, "chroma_db_huggingface")

print("Embedding demonstrations for OpenAI and Hugging Face completed.")


# Function to query a vector store
def query_vector_store(store_name, query, embedding_function):
    persistent_directory = os.path.join(db_dir, store_name)
    if os.path.exists(persistent_directory):
        print(f"\n--- Querying the Vector Store {store_name} ---")
        db = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embedding_function,
        )
        retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 3, "score_threshold": 0.1},
        )
        relevant_docs = retriever.invoke(query)
        # Display the relevant results with metadata
        print(f"\n--- Relevant Documents for {store_name} ---")
        for i, doc in enumerate(relevant_docs, 1):
            print(f"Document {i}:\n{doc.page_content}\n")
            if doc.metadata:
                print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
    else:
        print(f"Vector store {store_name} does not exist.")


# Define the user's question
query = "Who is Odysseus wife"  # Simplified query for better matching

# Query each vector store
query_vector_store("chroma_db_fast", query, fast_embeddings)
query_vector_store("chroma_db_quality", query, quality_embeddings)

print("Querying demonstrations completed.")
