import PyPDF2
from typing import List, Dict
import io
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings

import os
from dotenv import load_dotenv

class PDFHandler:
    def __init__(self):
        load_dotenv()
        self.extracted_text = ""
        
        # Initialize Azure OpenAI embeddings
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
            openai_api_version="2023-12-01-preview",
            openai_api_key=os.getenv("AZURE_OPENAI_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            chunk_size=2048
        )
        
        # Initialize vector store
        self.vector_store = None
        self.sections = {
            'description': ['description', 'scope', 'work item'],
            'amount': ['amount', 'cost', 'price', 'value'],
            'quantity': ['quantity', 'qty', 'count'],
            'status': ['status', 'completion', 'progress']
        }

    def extract_text(self, pdf_file) -> str:
        """Extract text and create vector store"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            section_texts = {section: [] for section in self.sections}
            current_section = None
            
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                text += page_text
                
                for section, keywords in self.sections.items():
                    for keyword in keywords:
                        if keyword in page_text.lower():
                            current_section = section
                            break
                    if current_section:
                        section_texts[current_section].append(page_text)
            
            self.extracted_text = text
            
            # Create vector store
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            
            documents = []
            for section, content in section_texts.items():
                if content:
                    chunks = text_splitter.split_text("\n".join(content))
                    documents.extend([{
                        "text": chunk,
                        "metadata": {"section": section}
                    } for chunk in chunks])
            
            # Create FAISS index
            texts = [doc["text"] for doc in documents]
            metadatas = [doc["metadata"] for doc in documents]
            self.vector_store = FAISS.from_texts(
                texts,
                self.embeddings,
                metadatas=metadatas
            )
            
            return text
            
        except Exception as e:
            raise Exception(f"Error extracting PDF text: {e}")

    def get_context(self, query: str, section: str = None) -> str:
        """Get relevant context using vector store"""
        try:
            if not self.vector_store:
                return ""
                
            results = self.vector_store.similarity_search_with_score(query, k=2)
            
            if section:
                results = [r for r, _ in results if r.metadata.get("section") == section]
            
            if results:
                return results[0][0].page_content
            return ""
            
        except Exception as e:
            print(f"Error getting context: {e}")
            return ""

    def validate_context(self, query: str, section: str = None) -> Dict:
        """Validate query against context"""
        try:
            results = self.vector_store.similarity_search_with_score(query, k=1)
            if not results:
                return {
                    "isValid": False,
                    "confidence": 0.0,
                    "context": ""
                }
                
            result = results[0]
            confidence = result[1]
            
            return {
                "isValid": confidence > 0.7,
                "confidence": confidence,
                "context": result[0].page_content[:200]
            }
            
        except Exception as e:
            print(f"Error validating context: {e}")
            return {
                "isValid": False,
                "confidence": 0.0,
                "context": ""
            }
