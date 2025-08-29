import os

# Disable ChromaDB telemetry for cleaner output
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from langchain_community.vectorstores import Chroma
# Fast open-source embeddings using community package
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_openai import OpenAIEmbeddings  # Commented out due to pydantic compatibility issues

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# Define the embedding model (using fast open-source model)
print("🚀 Loading fast open-source embedding model...")
# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Use HuggingFace Sentence Transformers - extremely fast and open source
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",  # Small, fast, high-quality model
    model_kwargs={'device': 'cpu'},  # Use CPU (change to 'cuda' if you have GPU)
    encode_kwargs={'normalize_embeddings': True}  # Normalize for better similarity search
)
print("✅ Embedding model loaded successfully")

# Load the existing vector store with the embedding function
db = Chroma(persist_directory=persistent_directory,
            embedding_function=embeddings)

# Define the user's question
query = "Who is Odysseus' wife?"
print(f"🔍 Searching for: '{query}'")

# Retrieve relevant documents based on the query (optimized settings)
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 3,           # Get top 3 results
        "score_threshold": 0.6  # Lowered from 0.9 for better results with normalized embeddings
    },
)
relevant_docs = retriever.invoke(query)
print(f"📄 Found {len(relevant_docs)} relevant documents")

# Display the relevant results with metadata
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
