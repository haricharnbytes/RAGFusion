import os
import base64
from typing import List, Dict, Any
import uuid
import hashlib
import warnings
warnings.filterwarnings("ignore")

# LangChain imports
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.vectorstores import InMemoryVectorStore


# Document parsing (using PyPDF and PyMuPDF for image extraction)
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("Warning: PyMuPDF not available. PDF image extraction disabled. Install with: pip install PyMuPDF")


import google.generativeai as genai

class MultimodalRAGSystem:

    def __init__(self, google_api_key: str, persist_directory: str = "./chroma_db_v2", 
                 main_model: str = "gemini-1.5-pro", vision_model: str = "gemini-1.5-flash", 
                 embedding_model: str = "models/text-embedding-004", temperature: float = 0.1, 
                 max_tokens: int = 8192):
        """
        Initialize the Multimodal RAG system
        
        Args:
            google_api_key: Google API key for Gemini
            persist_directory: Directory to persist vector database
            main_model: Main LLM model for text generation
            vision_model: Vision model for image processing
            embedding_model: Embedding model for vector embeddings
            temperature: Temperature for model responses
            max_tokens: Maximum tokens for model responses
        """
        self.google_api_key = google_api_key
        self.persist_directory = persist_directory

        # Configure Google AI
        genai.configure(api_key=google_api_key)
        os.environ["GOOGLE_API_KEY"] = google_api_key

        # Initialize Gemini models with configurable parameters
        self.llm = ChatGoogleGenerativeAI(
            model=main_model,
            google_api_key=google_api_key,
            temperature=temperature,
            max_tokens=max_tokens
        )
        # Vision model for image processing
        self.vision_model = ChatGoogleGenerativeAI(
            model=vision_model,
            google_api_key=google_api_key,
            temperature=temperature
        )

        # Embedding model for vector embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=embedding_model,
            google_api_key=google_api_key
        )
        
        # text splitter with better chunking strategy
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,  # Increased for better context
            chunk_overlap=300,  # Better overlap for continuity
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        # Multi-vector retriever components
        self.vectorstore = None
        self.retriever = None
        self.doc_store = InMemoryByteStore()
        self.image_metadata = {}

        def setup_multi_vector_retriever(self):
            """Setup the multi-vector retriever with enhanced capabilities."""
        if not self.vectorstore:
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name="multimodal_rag_v2"
            )
        
        # Initialize multi-vector retriever with latest pattern
        self.retriever = MultiVectorRetriever(
            vectorstore=self.vectorstore,
            byte_store=self.doc_store,
            id_key="doc_id",
            search_kwargs={"k": 6}  # Retrieve more candidates
        )