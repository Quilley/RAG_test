# Import the Pinecone library
from pinecone import Pinecone
from langchain.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Initialize a Pinecone client with your API key
pc = Pinecone(api_key="pcsk_4hpJ1p_GBB4PSUEzr8gfG4FybzmoYph1WNu1y9nBN668vj61wd4SaRLGfx2hdygbmQmQS4")

# Create a dense index with integrated embedding
index_name = "dense-index"
if not pc.has_index(index_name):
    pc.create_index_for_model(
        name=index_name,
        cloud="aws",
        region="us-east-1",
        embed={
            "model":"llama-text-embed-v2",
            "field_map":{"text": "chunk_text"}
        }
    )

# Install and import the required library



def websiteloaders(url):
    loader = WebBaseLoader(url)
    return loader.load()

urls =["https://medium.com/@Shamimw/building-a-rag-based-chatbot-using-pinecone-langchain-and-streamlit-e7a8cea277c5"]
website_docs = WebBaseLoader(urls)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,chunk_overlap=100, length_function=len,
    is_separator_regex=False,
)


# Load the documents from the WebBaseLoader instance
loaded_documents = website_docs.load()

# Split the loaded documents
splited_documents = text_splitter.split_documents(loaded_documents)

# Index the split documents into Pinecone
for doc in splited_documents:
    pc.index(index_name).upsert({
        "id": doc.metadata.get("id", str(hash(doc.page_content))),
        "values": doc.page_content
    })