# Chat Model Documents: https://python.langchain.com/v0.2/docs/integrations/chat/
# OpenAI Chat Model Documents: https://python.langchain.com/v0.2/docs/integrations/chat/openai/

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables from .env
load_dotenv()

# # Create a ChatOpenAI model
# model = ChatOpenAI(model="gpt-4o")

# Alternative: Use Ollama for open source models
from langchain_ollama import ChatOllama
model = ChatOllama(model="llama3.2") 

# Alternative: Use Hugging Face models
# from langchain_huggingface import ChatHuggingFace
# from langchain_huggingface import HuggingFaceEndpoint
# llm = HuggingFaceEndpoint(repo_id="microsoft/DialoGPT-medium")
# model = ChatHuggingFace(llm=llm)

# Invoke the model with a message
result = model.invoke("What is 81 divided by 9?")
print("Full result:")
print(result)
print("Content only:")
print(result.content)
