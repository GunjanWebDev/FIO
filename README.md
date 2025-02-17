from llama_index.core import Settings
from llama_index.core.schema import Document
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.indices.vector_store.base import VectorStoreIndex
import os
from dotenv import load_dotenv
from pathlib import Path
from typing import List, Dict
import json
#from llama_index import Settings

class LlamaSearchHandler:
    def __init__(self):
        load_dotenv()
        
        # Initialize paths and directories
        self.base_dir = Path(__file__).parent.parent
        self.index_dir = self.base_dir / 'data' / 'llama_index'
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        self.index_file = self.index_dir / "faiss.index"
        self.docstore_file = self.index_dir / "docstore.json"
        
        # Access the existing Settings singleton
        self.settings = Settings
        
        # Initialize or load index
        self.load_or_create_index()

# class LlamaSearchHandler:
#     def __init__(self):
#         load_dotenv()
        
#         # Initialize paths and directories
#         self.base_dir = Path(__file__).parent.parent
#         self.index_dir = self.base_dir / 'data' / 'llama_index'
#         self.index_dir.mkdir(parents=True, exist_ok=True)
        
#         self.index_file = self.index_dir / "faiss.index"
#         self.docstore_file = self.index_dir / "docstore.json"
        
#         # Initialize settings
#         self.settings = Settings(
#             chunk_size=500,
#             chunk_overlap=50
#         )
        
#         # Initialize or load index
#         self.load_or_create_index()
        
    def load_or_create_index(self):
        """Load existing index or create new empty one"""
        try:
            if self.index_file.exists() and self.docstore_file.exists():
                print("Loading existing LlamaIndex...")
                vector_store = FaissVectorStore.from_persist_dir(
                    persist_dir=str(self.index_dir)
                )
                docstore = SimpleDocumentStore.from_persist_dir(
                    persist_dir=str(self.index_dir)
                )
                self.index = VectorStoreIndex.from_vector_store(
                    vector_store=vector_store,
                    docstore=docstore,
                    settings=self.settings
                )
            else:
                print("Creating new LlamaIndex...")
                self.index = None
                
        except Exception as e:
            print(f"Error loading index: {e}")
            self.index = None
            
    def create_index_from_text(self, text: str, sections: Dict[str, List[str]] = None):
        """Create new index from text with optional sections"""
        try:
            documents = []
            
            if sections:
                # Create documents with section metadata
                for section, content in sections.items():
                    doc = Document(
                        text="\n".join(content),
                        metadata={"section": section}
                    )
                    documents.append(doc)
            else:
                # Create single document
                doc = Document(text=text)
                documents.append(doc)
                
            # Create new index
            print("Creating new index with documents...")
            self.index = VectorStoreIndex.from_documents(
                documents,
                settings=self.settings
            )
            
            # Persist index
            self.index.storage_context.persist(persist_dir=str(self.index_dir))
            print("Index created and persisted successfully")
            
        except Exception as e:
            print(f"Error creating index: {e}")
            raise
            
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search index and return results with metadata"""
        if not self.index:
            raise ValueError("Index not initialized")
            
        try:
            retriever = self.index.as_retriever(similarity_top_k=top_k)
            nodes = retriever.retrieve(query)
            
            results = []
            for node in nodes:
                results.append({
                    "text": node.get_text(),
                    "score": float(node.score) if hasattr(node, 'score') else 0.0,
                    "section": node.metadata.get("section", "general")
                })
                
            return results
            
        except Exception as e:
            print(f"Error during search: {e}")
            return []

    def clear_index(self):
        """Clear existing index"""
        try:
            if self.index_dir.exists():
                for file in self.index_dir.glob("*"):
                    file.unlink()
                self.index = None
                print("Index cleared successfully")
        except Exception as e:
            print(f"Error clearing index: {e}")


========================
error: Reading PDF from C:\Users\sarkarg\Downloads\referencepdfforColumnMapping.pdf...
Creating index with sectioned documents...
Creating new index with documents...
Error creating index: 
******
Could not load OpenAI embedding model. If you intended to use OpenAI, please check your OPENAI_API_KEY.
Original error:
No API key found for OpenAI.
Please set either the OPENAI_API_KEY environment variable or openai.api_key prior to initialization.
API keys can be found or created at https://platform.openai.com/account/api-keys

Consider using embed_model='local'.
Visit our documentation for more embedding options: https://docs.llamaindex.ai/en/stable/module_guides/models/embeddings.html#modules  
******
Initialization error: Error extracting PDF text:
******
Could not load OpenAI embedding model. If you intended to use OpenAI, please check your OPENAI_API_KEY.
Original error:
No API key found for OpenAI.
Please set either the OPENAI_API_KEY environment variable or openai.api_key prior to initialization.
API keys can be found or created at https://platform.openai.com/account/api-keys

Consider using embed_model='local'.
Visit our documentation for more embedding options: https://docs.llamaindex.ai/en/stable/module_guides/models/embeddings.html#modules  
******
Traceback (most recent call last):
  File "C:\Users\sarkarg\OneDrive - Moody's\Desktop\ColumnMappingV2\venv\Lib\site-packages\llama_index\core\embeddings\utils.py", line 59, in resolve_embed_model
    validate_openai_api_key(embed_model.api_key)  # type: ignore
    ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\sarkarg\OneDrive - Moody's\Desktop\ColumnMappingV2\venv\Lib\site-packages\llama_index\embeddings\openai\utils.py", line 103, in validate_openai_api_key
    raise ValueError(MISSING_API_KEY_ERROR_MESSAGE)
ValueError: No API key found for OpenAI.
Please set either the OPENAI_API_KEY environment variable or openai.api_key prior to initialization.
API keys can be found or created at https://platform.openai.com/account/api-keys


During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\Users\sarkarg\OneDrive - Moody's\Desktop\ColumnMappingV2\utils\pdf_handler.py", line 56, in extract_text
    self.search_handler.create_index_from_text(
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        text=text,
        ^^^^^^^^^^
        sections=section_texts
        ^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "C:\Users\sarkarg\OneDrive - Moody's\Desktop\ColumnMappingV2\utils\llama_search_handler.py", line 97, in create_index_from_text
    self.index = VectorStoreIndex.from_documents(
                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        documents,
        ^^^^^^^^^^
        settings=self.settings
        ^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "C:\Users\sarkarg\OneDrive - Moody's\Desktop\ColumnMappingV2\venv\Lib\site-packages\llama_index\core\indices\base.py", line 119, in from_documents
    return cls(
        nodes=nodes,
    ...<4 lines>...
        **kwargs,
    )
  File "C:\Users\sarkarg\OneDrive - Moody's\Desktop\ColumnMappingV2\venv\Lib\site-packages\llama_index\core\indices\vector_store\base.py", line 72, in __init__
    else Settings.embed_model
         ^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\sarkarg\OneDrive - Moody's\Desktop\ColumnMappingV2\venv\Lib\site-packages\llama_index\core\settings.py", line 64, in embed_model
    self._embed_model = resolve_embed_model("default")
                        ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^
  File "C:\Users\sarkarg\OneDrive - Moody's\Desktop\ColumnMappingV2\venv\Lib\site-packages\llama_index\core\embeddings\utils.py", line 66, in resolve_embed_model
    raise ValueError(
    ...<10 lines>...
    )
ValueError:
******
Could not load OpenAI embedding model. If you intended to use OpenAI, please check your OPENAI_API_KEY.
Original error:
No API key found for OpenAI.
Please set either the OPENAI_API_KEY environment variable or openai.api_key prior to initialization.
API keys can be found or created at https://platform.openai.com/account/api-keys

Consider using embed_model='local'.
Visit our documentation for more embedding options: https://docs.llamaindex.ai/en/stable/module_guides/models/embeddings.html#modules  
******

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\Users\sarkarg\OneDrive - Moody's\Desktop\ColumnMappingV2\server.py", line 107, in initialize_processing
    pdf_text = pdf_handler.extract_text(pdf_file)
  File "C:\Users\sarkarg\OneDrive - Moody's\Desktop\ColumnMappingV2\utils\pdf_handler.py", line 64, in extract_text
    raise Exception(f"Error extracting PDF text: {e}")
Exception: Error extracting PDF text:
******
Could not load OpenAI embedding model. If you intended to use OpenAI, please check your OPENAI_API_KEY.
Original error:
No API key found for OpenAI.
Please set either the OPENAI_API_KEY environment variable or openai.api_key prior to initialization.
API keys can be found or created at https://platform.openai.com/account/api-keys

Consider using embed_model='local'.
Visit our documentation for more embedding options: https://docs.llamaindex.ai/en/stable/module_guides/models/embeddings.html#modules  
******
>>> Response Status: 500
>>> Error in request processing: '_StreamingResponse' object has no attribute 'body'
Traceback (most recent call last):
  File "C:\Users\sarkarg\OneDrive - Moody's\Desktop\ColumnMappingV2\server.py", line 60, in debug_middleware
    print(f">>> Error Response: {response.body}")
                                 ^^^^^^^^^^^^^
AttributeError: '_StreamingResponse' object has no attribute 'body'
INFO:     127.0.0.1:64956 - "POST /api/init_processing HTTP/1.1" 500 Internal Server Error
ERROR:    Exception in ASGI application
  + Exception Group Traceback (most recent call last):
  |   File "C:\Users\sarkarg\OneDrive - Moody's\Desktop\ColumnMappingV2\venv\Lib\site-packages\starlette\_utils.py", line 76, in collapse_excgroups
  |     yield
  |   File "C:\Users\sarkarg\OneDrive - Moody's\Desktop\ColumnMappingV2\venv\Lib\site-packages\starlette\middleware\base.py", line 178, in __call__
  |     async with anyio.create_task_group() as task_group:
  |                ~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "C:\Users\sarkarg\OneDrive - Moody's\Desktop\ColumnMappingV2\venv\Lib\site-packages\anyio\_backends\_asyncio.py", line 767, in __aexit__
  |     raise BaseExceptionGroup(
  |         "unhandled errors in a TaskGroup", self._exceptions
  |     )
  | ExceptionGroup: unhandled errors in a TaskGroup (1 sub-exception)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "C:\Users\sarkarg\OneDrive - Moody's\Desktop\ColumnMappingV2\venv\Lib\site-packages\uvicorn\protocols\http\h11_impl.py", line 403, in run_asgi
    |     result = await app(  # type: ignore[func-returns-value]
    |              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |         self.scope, self.receive, self.send
    |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |     )
    |     ^
    |   File "C:\Users\sarkarg\OneDrive - Moody's\Desktop\ColumnMappingV2\venv\Lib\site-packages\uvicorn\middleware\proxy_headers.py", line 60, in __call__
    |     return await self.app(scope, receive, send)
    |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "C:\Users\sarkarg\OneDrive - Moody's\Desktop\ColumnMappingV2\venv\Lib\site-packages\fastapi\applications.py", line 1054, in __call__
    |     await super().__call__(scope, receive, send)
    |   File "C:\Users\sarkarg\OneDrive - Moody's\Desktop\ColumnMappingV2\venv\Lib\site-packages\starlette\applications.py", line 112, in __call__
    |     await self.middleware_stack(scope, receive, send)
    |   File "C:\Users\sarkarg\OneDrive - Moody's\Desktop\ColumnMappingV2\venv\Lib\site-packages\starlette\middleware\errors.py", line 187, in __call__
    |     raise exc
    |   File "C:\Users\sarkarg\OneDrive - Moody's\Desktop\ColumnMappingV2\venv\Lib\site-packages\starlette\middleware\errors.py", line 165, in __call__
    |     await self.app(scope, receive, _send)
    |   File "C:\Users\sarkarg\OneDrive - Moody's\Desktop\ColumnMappingV2\venv\Lib\site-packages\starlette\middleware\base.py", line 177, in __call__
    |     with recv_stream, send_stream, collapse_excgroups():
    |                                    ~~~~~~~~~~~~~~~~~~^^
    |   File "C:\Program Files\Python313\Lib\contextlib.py", line 162, in __exit__
    |     self.gen.throw(value)
    |     ~~~~~~~~~~~~~~^^^^^^^
    |   File "C:\Users\sarkarg\OneDrive - Moody's\Desktop\ColumnMappingV2\venv\Lib\site-packages\starlette\_utils.py", line 82, in collapse_excgroups
    |     raise exc
    |   File "C:\Users\sarkarg\OneDrive - Moody's\Desktop\ColumnMappingV2\venv\Lib\site-packages\starlette\middleware\base.py", line 179, in __call__
    |     response = await self.dispatch_func(request, call_next)
    |                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "C:\Users\sarkarg\OneDrive - Moody's\Desktop\ColumnMappingV2\server.py", line 60, in debug_middleware
    |     print(f">>> Error Response: {response.body}")
    |                                  ^^^^^^^^^^^^^
    | AttributeError: '_StreamingResponse' object has no attribute 'body'
    +------------------------------------

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\Users\sarkarg\OneDrive - Moody's\Desktop\ColumnMappingV2\venv\Lib\site-packages\uvicorn\protocols\http\h11_impl.py", line 403, in run_asgi
    result = await app(  # type: ignore[func-returns-value]
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        self.scope, self.receive, self.send
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "C:\Users\sarkarg\OneDrive - Moody's\Desktop\ColumnMappingV2\venv\Lib\site-packages\uvicorn\middleware\proxy_headers.py", line 60, in __call__
    return await self.app(scope, receive, send)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\sarkarg\OneDrive - Moody's\Desktop\ColumnMappingV2\venv\Lib\site-packages\fastapi\applications.py", line 1054, in __call__
    await super().__call__(scope, receive, send)
  File "C:\Users\sarkarg\OneDrive - Moody's\Desktop\ColumnMappingV2\venv\Lib\site-packages\starlette\applications.py", line 112, in __call__
    await self.middleware_stack(scope, receive, send)
  File "C:\Users\sarkarg\OneDrive - Moody's\Desktop\ColumnMappingV2\venv\Lib\site-packages\starlette\middleware\errors.py", line 187, in __call__
    raise exc
  File "C:\Users\sarkarg\OneDrive - Moody's\Desktop\ColumnMappingV2\venv\Lib\site-packages\starlette\middleware\errors.py", line 165, in __call__
    await self.app(scope, receive, _send)
  File "C:\Users\sarkarg\OneDrive - Moody's\Desktop\ColumnMappingV2\venv\Lib\site-packages\starlette\middleware\base.py", line 177, in __call__
    with recv_stream, send_stream, collapse_excgroups():
                                   ~~~~~~~~~~~~~~~~~~^^
  File "C:\Program Files\Python313\Lib\contextlib.py", line 162, in __exit__
    self.gen.throw(value)
    ~~~~~~~~~~~~~~^^^^^^^
  File "C:\Users\sarkarg\OneDrive - Moody's\Desktop\ColumnMappingV2\venv\Lib\site-packages\starlette\_utils.py", line 82, in collapse_excgroups
    raise exc
  File "C:\Users\sarkarg\OneDrive - Moody's\Desktop\ColumnMappingV2\venv\Lib\site-packages\starlette\middleware\base.py", line 179, in __call__
    response = await self.dispatch_func(request, call_next)
  File "C:\Users\sarkarg\OneDrive - Moody's\Desktop\ColumnMappingV2\venv\Lib\site-packages\starlette\_utils.py", line 82, in collapse_excgroups
    raise exc
  File "C:\Users\sarkarg\OneDrive - Moody's\Desktop\ColumnMappingV2\venv\Lib\site-packages\starlette\middleware\base.py", line 179, in __call__
    response = await self.dispatch_func(request, call_next)
    raise exc
  File "C:\Users\sarkarg\OneDrive - Moody's\Desktop\ColumnMappingV2\venv\Lib\site-packages\starlette\middleware\base.py", line 179, in __call__
    response = await self.dispatch_func(request, call_next)
  File "C:\Users\sarkarg\OneDrive - Moody's\Desktop\ColumnMappingV2\venv\Lib\site-packages\starlette\middleware\base.py", line 179, in __call__
    response = await self.dispatch_func(request, call_next)
__call__
    response = await self.dispatch_func(request, call_next)
    response = await self.dispatch_func(request, call_next)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\sarkarg\OneDrive - Moody's\Desktop\ColumnMappingV2\server.py", line 60, in debug_middleware
    print(f">>> Error Response: {response.body}")
                                 ^^^^^^^^^^^^^
AttributeError: '_StreamingResponse' object has no attribute 'body'















