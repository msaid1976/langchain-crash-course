import os

# Disable ChromaDB telemetry for cleaner output
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
# Use fast open-source embeddings and Ollama instead of OpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(
    current_dir, "db", "chroma_db_with_metadata")

# Define the embedding model (using fast open-source model)
print("🚀 Initializing fast HuggingFace embedding model...")
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
query = "How can I learn more about LangChain?"

# Retrieve relevant documents based on the query
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)
relevant_docs = retriever.invoke(query)

# Display the relevant results with metadata
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")

# Combine the query and the relevant document contents
combined_input = (
    "Here are some documents that might help answer the question: "
    + query
    + "\n\nRelevant Documents:\n"
    + "\n\n".join([doc.page_content for doc in relevant_docs])
    + "\n\nPlease provide an answer based only on the provided documents. If the answer is not found in the documents, respond with 'I'm not sure'."
)

# Create an Ollama model
model = Ollama(model="llama3.2")

# Define the messages for the model
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]

# Invoke the model with the combined input
result = model.invoke(messages)

# Display the full result and content only
print("\n--- Generated Response ---")
# print("Full result:")
# print(result)
print("Content only:")
# Check if result is a string or has content attribute
if isinstance(result, str):
    print(result)
else:
    print(result.content)
