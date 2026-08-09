from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2:1b", temperature=0)

response = llm.invoke("Say hello")

print(response)
print(response.content)