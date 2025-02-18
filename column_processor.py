from typing import Dict, List
from .openai_handler import OpenAIHandler
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class ColumnProcessor:
    def __init__(self, openai_handler: OpenAIHandler, pdf_context: str = ""):
        self.openai_handler = openai_handler
        self.pdf_context = pdf_context
        self.processed_columns = {}
        self.current_column = None
        
        # Column type definitions
        self.column_types = {
            'text': ['description', 'name', 'title', 'comment'],
            'number': ['amount', 'quantity', 'count', 'total'],
            'percentage': ['percent', 'complete', 'progress'],
            'date': ['date', 'scheduled', 'deadline']
        }

    def analyze_column(self, name: str, samples: List[str]) -> Dict:
        """Analyze column data for type and patterns"""
        try:
            analysis = {
                "name": name,
                "samples": samples[:5],
                "type": self._detect_type(samples),
                "patterns": self._find_patterns(samples)
            }
            return analysis
        except Exception as e:
            print(f"Error analyzing column {name}: {e}")
            return {"name": name, "error": str(e)}

    def _detect_type(self, samples: List[str]) -> str:
        """Detect column data type from samples"""
        if not samples:
            return 'unknown'
            
        numeric_count = sum(1 for s in samples if str(s).replace('.', '').isdigit())
        percent_count = sum(1 for s in samples if '%' in str(s))
        date_count = sum(1 for s in samples if '/' in str(s) or '-' in str(s))
        
        total = len(samples)
        if numeric_count/total > 0.8:
            return 'number'
        elif percent_count/total > 0.8:
            return 'percentage'
        elif date_count/total > 0.8:
            return 'date'
        return 'text'

    def _find_patterns(self, samples: List[str]) -> List[str]:
        """Find common patterns in sample data"""
        patterns = []
        if not samples:
            return patterns
            
        # Check for common patterns
        first_chars = set(s[0].lower() if s else '' for s in samples)
        if len(first_chars) == 1:
            patterns.append(f"Always starts with '{list(first_chars)[0]}'")
            
        lengths = set(len(str(s)) for s in samples)
        if len(lengths) == 1:
            patterns.append(f"Fixed length: {list(lengths)[0]}")
            
        return patterns

    async def process_column(self, column_name: str, samples: List[str]) -> Dict:
        """Process a single column with analysis and mapping"""
        try:
            self.current_column = column_name
            
            # First analyze the column
            analysis = self.analyze_column(column_name, samples)
            
            # Get column mapping from OpenAI with analysis
            mapping_result = await self.openai_handler.map_single_column(
                column_name=column_name,
                context=self.pdf_context,
                samples=samples,
                analysis=analysis
            )

            # Add analysis to result
            result = {
                "original_name": column_name,
                "mapped_name": mapping_result["mapped_name"],
                "confidence": mapping_result.get("confidence", 0.0),
                "analysis": analysis,
                "samples": samples[:3],
                "reasoning": mapping_result.get("reasoning", "")
            }
            
            self.processed_columns[column_name] = result
            return result
            
        except Exception as e:
            return {
                "original_name": column_name,
                "error": str(e),
                "status": "failed"
            }

    async def get_progress(self) -> Dict:
        """Get current processing progress"""
        total_processed = len(self.processed_columns)
        return {
            'current_column': self.current_column,
            'processed_count': total_processed,
            'processed_columns': self.processed_columns
        }
