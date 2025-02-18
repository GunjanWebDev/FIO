from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.embeddings import AzureOpenAIEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
from pathlib import Path
from typing import List, Dict
import json
import time
import numpy as np
from tenacity import retry, wait_exponential, stop_after_attempt

class LangChainSearchHandler:
    def __init__(self):
        load_dotenv()
        
        # Initialize paths and directories
        self.base_dir = Path(__file__).parent.parent
        self.index_dir = self.base_dir / 'data' / 'vector_store'
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure embeddings based on environment setting
        self.embedding_type = os.getenv("EMBEDDING_TYPE", "azure").lower()
        if self.embedding_type == "azure":
            self.embeddings = self._init_azure_embeddings()
        else:
            self.embeddings = self._init_local_embeddings()
        
        # Initialize vector store
        self.vector_store = None
        self.load_or_create_store()
        
        # Batch processing settings
        self.batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "20"))
        self.delay_seconds = float(os.getenv("EMBEDDING_DELAY_SECONDS", "1.0"))

    def _init_azure_embeddings(self):
        """Initialize Azure OpenAI embeddings with retry logic"""
        return AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview"),
            openai_api_key=os.getenv("AZURE_OPENAI_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            chunk_size=self.batch_size  # Enable batch processing
        )

    def _init_local_embeddings(self):
        """Initialize local HuggingFace embeddings as fallback"""
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

    @retry(wait=wait_exponential(multiplier=1, min=4, max=60),
           stop=stop_after_attempt(5))
    def get_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Get embeddings for a batch of texts with retry logic - now synchronous"""
        try:
            embeddings = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                if self.embedding_type == "azure":
                    # Add delay between batches for Azure API
                    if i > 0:
                        time.sleep(self.delay_seconds)
                batch_embeddings = self.embeddings.embed_documents(batch)
                embeddings.extend(batch_embeddings)
                print(f"Processed batch {i//self.batch_size + 1}/{(len(texts)-1)//self.batch_size + 1}")
            return embeddings
        except Exception as e:
            print(f"Error in batch embedding: {e}")
            if "429" in str(e):
                print("Rate limit exceeded, retrying with exponential backoff...")
                raise  # Trigger retry
            return []

    def create_index_from_text(self, text: str, sections: Dict[str, List[str]] = None):
        """Create new index from text with optimized batch processing - now synchronous"""
        try:
            self.clear_store()
            documents = []
            
            if sections:
                for section, content in sections.items():
                    chunks = self._chunk_by_headers(
                        "\n".join(content),
                        headers=self.section_headers.get(section, [])
                    )
                    documents.extend([
                        Document(
                            page_content=chunk,
                            metadata={"section": section}
                        ) for chunk in chunks
                    ])
            else:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50,
                    separators=["\n\n", "\n", ". ", " ", ""]
                )
                chunks = text_splitter.split_text(text)
                documents = [Document(page_content=chunk) for chunk in chunks]

            print(f"Creating vector store with {len(documents)} documents...")
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            
            # Process embeddings in batches - now synchronous
            embeddings = self.get_embeddings_batch(texts)  # Removed await
            
            # Create FAISS index
            self.vector_store = FAISS.from_embeddings(
                text_embeddings=list(zip(texts, embeddings)),
                embedding=self.embeddings,
                metadatas=metadatas
            )
            
            # Save to disk
            self.vector_store.save_local(str(self.index_dir))
            print("Vector store created and saved successfully")

        except Exception as e:
            print(f"Error creating vector store: {e}")
            raise

    def load_or_create_store(self):
        """Load existing vector store or create new one"""
        try:
            if (self.index_dir / "index.faiss").exists():
                print("Loading existing vector store...")
                self.vector_store = FAISS.load_local(
                    folder_path=str(self.index_dir),
                    embeddings=self.embeddings
                )
            else:
                print("No existing vector store found")
                self.vector_store = None
        except Exception as e:
            print(f"Error loading vector store: {e}")
            self.vector_store = None

    def _chunk_by_headers(self, text: str, headers: List[str]) -> List[str]:
        """Custom chunking based on section headers"""
        chunks = []
        current_chunk = ""
        
        for line in text.split('\n'):
            # Check if line contains a header
            is_header = any(header.lower() in line.lower() for header in headers)
            
            if is_header and current_chunk:
                # Save current chunk and start new one
                chunks.append(current_chunk.strip())
                current_chunk = line
            else:
                current_chunk += "\n" + line
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search vector store and return results with metadata"""
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
            
        try:
            results = self.vector_store.similarity_search_with_score(
                query,
                k=top_k
            )
            
            return [{
                "text": doc.page_content,
                "score": float(score),
                "section": doc.metadata.get("section", "general")
            } for doc, score in results]
            
        except Exception as e:
            print(f"Error during search: {e}")
            return []

    def clear_store(self):
        """Clear existing vector store"""
        try:
            if self.index_dir.exists():
                for file in self.index_dir.glob("*"):
                    file.unlink()
                self.vector_store = None
                print("Vector store cleared successfully")
        except Exception as e:
            print(f"Error clearing vector store: {e}")
