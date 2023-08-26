from langchain.document_loaders import WebBaseLoader
from langchain.indexes import VectorstoreIndexCreator
from dotenv import load_dotenv
load_dotenv()

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
index = VectorstoreIndexCreator().from_loaders([loader])

print(index.query("What is Task Decomposition?"))

# >> Task decomposition is a technique used to break down complex tasks into smaller and simpler steps. It can be done using LLM with simple prompting, task-specific instructions, or with human input.