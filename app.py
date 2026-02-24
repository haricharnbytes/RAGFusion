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
from langchain_core.stores import InMemoryByteStore


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


        def parse_documents(self, file_paths: List[str]) -> Dict[str, List]:
            """
        Parse documents using basic document loaders.
        
        Args:
            file_paths: List of file paths to parse
            
        Returns:
            Dictionary containing separated elements (text, tables, images)
        """
        all_elements = {"texts": [], "tables": [], "images": []}
        
        for file_path in file_paths:
            try:
                if file_path.endswith('.pdf'):
                    # Extract text using PyPDF
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    
                    for doc in docs:
                        all_elements["texts"].append({
                            "content": doc.page_content,
                            "metadata": {"source": file_path, "type": "text", "page": doc.metadata.get("page", 0)}
                        })
                    
                    # Extract images from PDF using PyMuPDF (if available)
                    if PYMUPDF_AVAILABLE:
                        pdf_images = self._extract_images_from_pdf(file_path)
                        all_elements["images"].extend(pdf_images)
                    else:
                        print(f"Skipping image extraction from {file_path} - PyMuPDF not installed")
                    
                elif file_path.endswith('.txt'):
                    loader = TextLoader(file_path)
                    docs = loader.load()
                    
                    for doc in docs:
                        all_elements["texts"].append({
                            "content": doc.page_content,
                            "metadata": {"source": file_path, "type": "text", "page": 0}
                        })
                else:
                    continue
                    
            except Exception as e:
                print(f"Error parsing {file_path}: {str(e)}")
        
        return all_elements
    
    def _extract_images_from_pdf(self, pdf_path: str) -> List[Dict]:
        """
        Extract images from PDF using PyMuPDF.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of extracted image data
        """
        extracted_images = []
        
        if not PYMUPDF_AVAILABLE:
            print("PyMuPDF not available - cannot extract images from PDF")
            return extracted_images
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    # Create unique identifier
                    image_hash = hashlib.md5(image_bytes).hexdigest()
                    
                    extracted_images.append({
                        "content": image_bytes,
                        "metadata": {
                            "source": pdf_path,
                            "type": "image",
                            "page": page_num + 1,
                            "image_index": img_index,
                            "extension": image_ext,
                            "hash": image_hash
                        }
                    })
                    
                    print(f"Extracted image {img_index + 1} from page {page_num + 1} of {pdf_path}")
            
            doc.close()
            
        except Exception as e:
            print(f"Error extracting images from PDF {pdf_path}: {str(e)}")
        
        return extracted_images
    
    def process_images_advanced(self, image_paths: List[str]) -> List[Dict]:
        """
        Image processing Gemini Vision Model.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of processed image data
        """
        processed_images = []
        
        for image_path in image_paths:
            try:
                # Load and process image
                with open(image_path, 'rb') as img_file:
                    image_data = img_file.read()
                
                # Create image hash for unique identification
                image_hash = hashlib.md5(image_data).hexdigest()
                
                # Image description with Gemini LLM
                description = self._generate_enhanced_image_description(image_data)
                
                # Extract any text from the image
                text_content = self._extract_text_from_image(image_data)
                
                processed_image = {
                    "path": image_path,
                    "hash": image_hash,
                    "description": description,
                    "text_content": text_content,
                    "type": "image",
                    "metadata": {
                        "source": image_path,
                        "type": "image",
                        "hash": image_hash
                    }
                }
                
                processed_images.append(processed_image)
                print(f"Processed image: {image_path}")
                
            except Exception as e:
                print(f"Error processing image {image_path}: {str(e)}")
        
        return processed_images
    
    def _generate_enhanced_image_description(self, image_data: bytes) -> str:
        """Generate detailed description using latest Gemini vision model capabilities."""
        try:
            # Convert to base64 for API
            image_b64 = base64.b64encode(image_data).decode()
            
            prompt = """
            Analyze this image comprehensively and provide a detailed description that includes:
            
            1. **Main Content**: Objects, people, scenes, activities
            2. **Visual Elements**: Colors, composition, style, lighting
            3. **Text Content**: Any visible text, signs, labels, captions
            4. **Context & Setting**: Environment, location, time period if apparent
            5. **Technical Details**: Charts, graphs, diagrams, technical elements
            6. **Relationships**: How elements in the image relate to each other
            7. **Actionable Information**: Key insights or data points visible
            
            Format your response to be detailed yet concise, optimized for retrieval and understanding.
            Focus on information that would be valuable for question-answering scenarios.
            """
            
            # Use latest vision model with correct message format
            
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]
            )
            
            response = self.vision_model.invoke([message])
            
            return response.content
            
        except Exception as e:
            print(f"Error generating enhanced image description: {str(e)}")
            return "Error generating image description"
    
    def _extract_text_from_image(self, image_data: bytes) -> str:
        """Extract text content from images using Gemini OCR capabilities."""
        try:
            image_b64 = base64.b64encode(image_data).decode()
            
            prompt = """
            Extract all visible text from this image. Include:
            - Any readable text, numbers, labels
            - Text in charts, graphs, diagrams
            - Signs, captions, headers
            - Technical annotations
            
            Provide only the extracted text content, maintaining structure where possible.
            If no text is visible, respond with "No text detected".
            """
            
            
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]
            )
            
            response = self.vision_model.invoke([message])
            
            return response.content
            
        except Exception as e:
            print(f"Error extracting text from image: {str(e)}")
            return "No text detected"
        

    def create_summaries_for_retrieval(self, elements: List[Dict]) -> List[str]:
        """
        Create optimized summaries for better retrieval using Gemini LLM.
        
        Args:
            elements: List of elements (text, table, image data)
            
        Returns:
            List of summaries optimized for retrieval
        """
        summaries = []
        
        for element in elements:
            try:
                content = element.get("content", "")
                element_type = element.get("metadata", {}).get("type", "text")
                
                if element_type == "table":
                    prompt = f"""
                    Summarize this table for optimal retrieval. Focus on:
                    - Key data points and metrics
                    - Column headers and categories
                    - Notable trends or patterns
                    - Actionable insights
                    
                    Table content:
                    {content[:2000]}  # Limit content length
                    
                    Provide a concise summary optimized for semantic search.
                    """
                elif element_type == "image":
                    # For images, use the description we already generated
                    summaries.append(content)
                    continue
                else:
                    prompt = f"""
                    Create a retrieval-optimized summary of this text. Include:
                    - Main topics and themes
                    - Key facts and figures
                    - Important concepts
                    - Actionable information
                    
                    Text content:
                    {content[:2000]}
                    
                    Provide a comprehensive yet concise summary.
                    """
                
                response = self.llm.invoke(prompt)
                summaries.append(response.content)
                
            except Exception as e:
                print(f"Error creating summary: {str(e)}")
                summaries.append(str(element.get("content", ""))[:500])
        
        return summaries