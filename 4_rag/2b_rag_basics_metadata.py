import os

# Disable ChromaDB telemetry for cleaner output
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from langchain_community.vectorstores import Chroma
# Use fast open-source embeddings instead of OpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_openai import OpenAIEmbeddings

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

# Define the embedding model (using fast open-source model)
print("🚀 Initializing fast HuggingFace embedding model...")

# Use HuggingFace Sentence Transformers - extremely fast and open source
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",  # Small, fast, high-quality model
    model_kwargs={'device': 'cpu'},  # Use CPU (change to 'cuda' if you have GPU)
    encode_kwargs={'normalize_embeddings': True}  # Normalize for better similarity search
)
print("✅ Embedding model loaded successfully")

# Load the existing vector store with the embedding function
print("🔍 Loading vector store...")
db = Chroma(persist_directory=persistent_directory,
            embedding_function=embeddings)
print("✅ Vector store loaded successfully")

# Define the user's question
query = "How did Juliet die?"
print(f"🔎 Searching for: '{query}'")

# First, let's check similarity scores without threshold for diagnostics
print("🔍 Checking similarity scores...")
docs_with_scores = db.similarity_search_with_score(query, k=5)
print(f"📊 Top 5 similarity scores:")
for i, (doc, score) in enumerate(docs_with_scores, 1):
    source = doc.metadata.get('source', 'Unknown')
    print(f"  {i}. Score: {score:.3f} - Source: {source} - Preview: {doc.page_content[:100]}...")

print(f"\n🔍 Now retrieving with reasonable threshold...")

# Retrieve relevant documents based on the query (with realistic threshold)
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 3, 
        "score_threshold": 0.3  # Realistic threshold instead of 0.1
    },
)
relevant_docs = retriever.invoke(query)
print(f"📄 Found {len(relevant_docs)} relevant documents with threshold 0.3")

# If no results with threshold, try basic similarity search
if len(relevant_docs) == 0:
    print("⚠️  No documents found with threshold. Trying basic similarity search...")
    relevant_docs = db.similarity_search(query, k=3)
    print(f"📄 Found {len(relevant_docs)} documents with basic similarity search")

# Display the relevant results with metadata
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    source = doc.metadata.get('source', 'Unknown')
    print(f"📖 Document {i} (from {source}):")
    print(f"{doc.page_content}\n")
    print("-" * 50)

# Summary of sources
if relevant_docs:
    sources = [doc.metadata.get('source', 'Unknown') for doc in relevant_docs]
    unique_sources = list(set(sources))
    print(f"\n📚 Sources found: {', '.join(unique_sources)}")
else:
    print("\n❌ No relevant documents found. Try a different query or check if the vector store exists.")
