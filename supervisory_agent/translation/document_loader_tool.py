import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from crewai_tools import tool


@tool("File loader")
def file_loader_tool(file_path: str) -> str:
    """"Useful for loading documents. Supported types: pdf, docx, txt"""
    _, file_extension = os.path.splitext(file_path)

    if file_extension.lower() == '.pdf':
        loader = PyPDFLoader(file_path)
    elif file_extension.lower() == '.docx':
        loader = Docx2txtLoader(file_path)
    elif file_extension.lower() == '.txt':
        loader = TextLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

    documents = loader.load()

    # Split the document into chunks
    # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    # texts = text_splitter.split_documents(documents)

    return documents[0].page_content
