import os
import time

# Disable ChromaDB telemetry for cleaner output
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Use the modern Chroma import that supports pydantic v2
from langchain_chroma import Chroma
# Use stable community embeddings instead of the problematic langchain-huggingface
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
query = "Who is Odysseus wife"  # Simplified query (removed apostrophe for better matching)
print(f"� Searching for: '{query}'")

# Retrieve relevant documents based on the query (optimized settings)
print("🔍 First, let's check similarity scores without threshold...")

# Step 1: Get similarity scores to diagnose the issue
docs_with_scores = db.similarity_search_with_score(query, k=5)
print(f"📊 Top 5 similarity scores:")
for i, (doc, score) in enumerate(docs_with_scores, 1):
    print(f"  {i}. Score: {score:.3f} - Preview: {doc.page_content[:100]}...")

print(f"\n🔍 Now retrieving with reasonable threshold...")

# Step 2: Use a much lower, realistic threshold
retriever = db.as_retriever(
    search_type="similarity_score_threshold", 
    search_kwargs={
        "k": 10,           # Get top 10 results
        "score_threshold": 0.5  # Much more realistic threshold (50% similarity)
    },
)
relevant_docs = retriever.invoke(query)
print(f"📄 Found {len(relevant_docs)} relevant documents with threshold 0.3")

# Step 3: If still no results, use basic similarity search (no threshold)
if len(relevant_docs) == 0:
    print("⚠️  No documents found with threshold. Trying basic similarity search...")
    relevant_docs = db.similarity_search(query, k=3)
    print(f"📄 Found {len(relevant_docs)} documents with basic similarity search")

# Display the relevant results with metadata
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
