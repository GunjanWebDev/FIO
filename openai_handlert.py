import os
from openai import AzureOpenAI
from typing import Dict, List
import json
from dotenv import load_dotenv
from pathlib import Path
from .training_data_handler import TrainingDataHandler
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

class OpenAIHandler:
    def __init__(self, training_file_path: str = None):
        load_dotenv()
        
        # Azure OpenAI specific environment variables
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_KEY")
        deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        
        if not all([azure_endpoint, api_key, deployment_name]):
            raise ValueError("Missing required Azure OpenAI environment variables")
            
        self.client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version="2023-12-01-preview"
        )
        
        self.deployment_name = deployment_name
        self.temperature = 0.1
        self.max_tokens = 2000

        # Initialize embeddings
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment=embedding_deployment,
            openai_api_version="2024-08-06-preview",
            openai_api_key=api_key,
            azure_endpoint=azure_endpoint
        )

        
        # Initialize training data path
        if training_file_path is None:
            # Default to data/training_data.xlsx in the project directory
            base_dir = Path(__file__).parent.parent
            training_file_path = base_dir / 'training_data.xlsx'
        
        # Ensure the training data file exists
        if not os.path.exists(training_file_path):
            raise FileNotFoundError(f"Training data file not found at: {training_file_path}")
            
        self.training_data = TrainingDataHandler(training_file_path)

    def get_relevant_context(self, column_name: str, pdf_text: str) -> str:
        """Get relevant context for a column from PDF using semantic search"""
        search_results = self.search_handler.search(
            query=column_name,
            top_k=2  # Get top 2 most relevant chunks
        )
        return " ".join([result["text"] for result in search_results])

    def analyze_column_data(self, column_name: str, samples: List[str]) -> str:
        """Analyze column data samples to understand the data type and pattern"""
        try:
            # Convert samples to appropriate types when possible
            typed_samples = []
            for sample in samples:
                try:
                    # Try numeric conversion
                    if '.' in str(sample):
                        typed_samples.append(float(sample))
                    else:
                        typed_samples.append(int(sample))
                except ValueError:
                    # Keep as string if not numeric
                    typed_samples.append(str(sample))

            # Analyze patterns in the data
            pattern_analysis = {
                "appears_numeric": all(isinstance(x, (int, float)) for x in typed_samples),
                "appears_percentage": any('%' in str(x) for x in samples) or 
                                   all(0 <= float(str(x).replace('%', '')) <= 100 
                                   for x in samples if str(x).replace('%', '').replace('.', '').isdigit()),
                "appears_monetary": any('$' in str(x) for x in samples) or
                                  (all(isinstance(x, (int, float)) for x in typed_samples) and
                                   any(x > 100 for x in typed_samples if isinstance(x, (int, float)))),
                "appears_date": any('/' in str(x) or '-' in str(x) for x in samples),
                "max_length": max(len(str(x)) for x in samples),
                "unique_values": len(set(str(x) for x in samples)),
                "sample_count": len(samples)
            }

            return pattern_analysis
        except Exception as e:
            print(f"Error analyzing column data: {e}")
            return {}

    def map_columns(self, columns: List[str], pdf_context: str = "", column_samples: Dict[str, List[str]] = None) -> Dict[str, str]:
        """Enhanced column mapping with data analysis"""
        try:
            self.search_handler.create_index(pdf_context)  # Initialize semantic search index
            
            column_analyses = {}
            relevant_contexts = {}
            historical_mappings = {}
            
            # Get relevant context and analysis for each column
            for col in columns:
                # Get column analysis
                samples = column_samples.get(col, []) if column_samples else []
                analysis = self.analyze_column_data(col, samples)
                column_analyses[col] = analysis
                
                # Get focused context for this column
                relevant_context = self.get_relevant_context(col, pdf_context)
                if relevant_context:
                    relevant_contexts[col] = relevant_context
                
                # Get similar historical mappings
                similar_mappings = self.training_data.get_similar_mappings(col)
                if similar_mappings:
                    historical_mappings[col] = similar_mappings

            # Create optimized system prompt
            system_prompt = {
                "role": "system",
                "content": f"""You are an expert in construction Schedule of Values (SOV) data standardization.
                Use these historical mappings and contexts to map columns accurately:
                
                Historical Mappings:
                {json.dumps(historical_mappings, indent=2)}
                
                Column Patterns from Training Data:
                {json.dumps(self.training_data.column_patterns, indent=2)}
                
                Map each column based on:
                1. Historical mapping patterns
                2. Column data patterns
                3. Relevant context from PDF
                4. Industry standard terminology
                """
            }

            user_message = {
                "role": "user",
                "content": f"""Map these columns with their data patterns and contexts:
                
                Column Analyses:
                {json.dumps(column_analyses, indent=2)}
                
                Relevant Contexts:
                {json.dumps(relevant_contexts, indent=2)}
                """
            }

            # Call Azure OpenAI with optimized content
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[system_prompt, user_message],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            # Process the response and extract mappings
            response_text = response.choices[0].message.content.strip()
            try:
                start_idx = response_text.find('{')
                end_idx = response_text.rindex('}') + 1
                json_str = response_text[start_idx:end_idx]
                return json.loads(json_str)
            except:
                raise Exception(f"Invalid JSON response: {response_text[:100]}...")

        except Exception as e:
            raise Exception(f"Error mapping columns: {e}")

    async def map_single_column(
        self, 
        column_name: str, 
        analysis: Dict,
        context: str,
        historical_mappings: List,
        samples: List[str]
    ) -> Dict:
        """Map a single column using focused context"""
        try:
            system_prompt = {
                "role": "system",
                "content": f"""You are an expert in construction Schedule of Values (SOV) data standardization.
                Focus on mapping this single column accurately:
                Column Name: {column_name}
                
                Historical Similar Mappings:
                {json.dumps(historical_mappings, indent=2)}
                
                Consider:
                1. Column name semantics
                2. Data patterns
                3. Historical mappings
                4. Industry terminology
                
                Return a JSON with mapped_name and confidence."""
            }

            user_message = {
                "role": "user",
                "content": f"""Analyze and map this column:
                
                Column Analysis:
                {json.dumps(analysis, indent=2)}
                
                Context:
                {context[:500]}
                
                Sample Values:
                {json.dumps(samples[:5], indent=2)}
                """
            }

            response = await self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[system_prompt, user_message],
                temperature=self.temperature,
                max_tokens=500  # Reduced tokens for single column
            )

            result = json.loads(response.choices[0].message.content)
            return result

        except Exception as e:
            raise Exception(f"Error mapping column {column_name}: {e}")

    def validate_mapping(self, mapping: Dict[str, str], pdf_context: str) -> Dict[str, Dict]:
        """Enhanced validation with semantic search and training data"""
        try:
            validation_results = {}
            
            for original, mapped in mapping.items():
                # Get relevant context for validation
                relevant_context = self.get_relevant_context(original, pdf_context)
                
                # Get training data validation
                historical_validation = self.training_data.validate_mapping(original, mapped)
                
                # Get semantic search validation
                semantic_validation = self.search_handler.validate_mapping(original, mapped)
                
                # Combine validations with weights
                confidence = (
                    historical_validation.get('confidence', 0) * 0.4 +  # Historical data weight
                    semantic_validation.get('confidence', 0) * 0.6      # Semantic search weight
                )
                
                validation_results[mapped] = {
                    "isValid": confidence > 0.5,
                    "confidence": confidence,
                    "context": relevant_context[:200],  # Limit context length
                    "historicalMatch": bool(historical_validation.get('isValid')),
                    "semanticMatch": bool(semantic_validation.get('isValid'))
                }
            
            return validation_results

        except Exception as e:
            raise Exception(f"Error validating mapping: {e}")

    def get_embeddings(self, text: str) -> List[float]:
        """Get embeddings using Azure OpenAI"""
        try:
            return self.embeddings.embed_query(text)
        except Exception as e:
            raise Exception(f"Error getting embeddings: {e}")

    def search_context(self, query: str, context: str, top_k: int = 2) -> List[Dict]:
        """Search for relevant context using embeddings"""
        try:
            # Create temporary vector store
            texts = [context]
            docsearch = FAISS.from_texts(texts, self.embeddings)
            
            # Search
            results = docsearch.similarity_search_with_score(query, k=top_k)
            
            return [{
                "text": doc.page_content,
                "score": score,
                "metadata": doc.metadata
            } for doc, score in results]
            
        except Exception as e:
            print(f"Error searching context: {e}")
            return []
