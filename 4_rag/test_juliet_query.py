import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Disable ChromaDB telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"

print("🔍 Testing Juliet death query...")

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Test queries
queries = [
    "How did Juliet die?",
    "Juliet death poison dagger", 
    "Juliet tomb death",
    "What happened to Juliet?"
]

# Test with existing vector store
db_path = os.path.join("db", "chroma_db_char")
if os.path.exists(db_path):
    print(f"Loading vector store from: {db_path}")
    db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    
    for query in queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print('='*50)
        
        # Get top 2 results with scores
        results = db.similarity_search_with_score(query, k=2)
        
        if results:
            for i, (doc, score) in enumerate(results, 1):
                print(f"\nResult {i} (score: {score:.3f}):")
                # Show more content to see if Juliet's death is described
                content = doc.page_content
                print(f"Content ({len(content)} chars): {content}")
                print("-" * 30)
        else:
            print("No results found")
else:
    print(f"Vector store not found at: {db_path}")
    print("Please run the main script first to create the vector stores.")
