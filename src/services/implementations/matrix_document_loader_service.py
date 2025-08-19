import os
import re
import uuid
from typing import List, Dict, Any, Optional
from kbac.loaders.matrix_script_loader import MatrixScriptLoader, Document as BaseDocument
from src.schemas.models import Document
from src.config.settings import BASE_DIR
from ..interfaces.document_loader_service import DocumentLoaderService


class MatrixDocumentLoaderService(DocumentLoaderService):
    """Enhanced Matrix script loader with intelligent chunking"""
    
    def __init__(
        self, 
        script_path: Optional[str] = None,
        chunk_size: int = 1000,
        overlap: int = 200
    ):
        if script_path is None:
            script_path = os.path.join(
                BASE_DIR,
                "resources/movie-scripts/the-matrix-1999.pdf"
            )
        
        self.script_path = script_path
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.loader = MatrixScriptLoader(source_path=self.script_path)

    def load_documents(self) -> List[Document]:
        """Load and chunk Matrix script documents"""
        base_documents = self.loader.load()
        
        chunks = self._create_intelligent_chunks(base_documents)
        
        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                id=str(uuid.uuid4()),
                page_content=chunk['content'],
                metadata={
                    **chunk['metadata'],
                    'chunk_id': i,
                    'source': self.script_path}
            )
            documents.append(doc)
            
        return documents
    
    def _create_intelligent_chunks(self, raw_documents: List[BaseDocument]) -> List[Dict[str, Any]]:
        """Create intelligent chunks preserving context"""
        chunks = []
        current_chunk = {
            'content': '',
            'metadata': {
                'scenes': set(),
                'characters': set(),
                'text_types': set(),
                'page_numbers': set()
            }
        }
        
        for doc in raw_documents:
            doc_text = doc.text
            doc_metadata = doc.metadata
            
            # Check if adding this document would exceed chunk size
            if (len(current_chunk['content']) + len(doc_text) > self.chunk_size and 
                len(current_chunk['content']) > 0):
                
                # Finalize current chunk
                chunks.append(self._finalize_chunk(current_chunk))
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk['content'])
                current_chunk = {
                    'content': overlap_text,
                    'metadata': {
                        'scenes': set(),
                        'characters': set(),
                        'text_types': set(),
                        'page_numbers': set()
                    }
                }
            
            # Add document to current chunk
            if current_chunk['content']:
                current_chunk['content'] += '\n\n'
            
            # Format content with context
            formatted_content = self._format_document_content(doc)
            current_chunk['content'] += formatted_content
            
            # Update metadata
            if doc_metadata.get('location'):
                current_chunk['metadata']['scenes'].add(doc_metadata['location'])
            if doc_metadata.get('character'):
                current_chunk['metadata']['characters'].add(doc_metadata['character'])
            if doc_metadata.get('text_type'):
                current_chunk['metadata']['text_types'].add(doc_metadata['text_type'])
            if doc_metadata.get('page_number'):
                current_chunk['metadata']['page_numbers'].add(doc_metadata['page_number'])
        
        # Add final chunk if not empty
        if current_chunk['content'].strip():
            chunks.append(self._finalize_chunk(current_chunk))
        
        return chunks
    
    def _format_document_content(self, doc: BaseDocument) -> str:
        """Format document content with context markers"""
        content = doc.text
        metadata = doc.metadata
        text_type = metadata.get('text_type', 'unknown')
        
        if text_type == 'dialog':
            character = metadata.get('character', 'UNKNOWN')
            return f"{character}: {content}"
        elif text_type == 'scene_description':
            return f"[SCENE] {content}"
        else:
            return content
    
    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from the end of current chunk"""
        if len(text) <= self.overlap:
            return text
        
        # Try to find a good breaking point (sentence or paragraph)
        overlap_text = text[-self.overlap:]
        
        # Find the first sentence boundary in the overlap
        sentence_match = re.search(r'[.!?]\s+', overlap_text)
        if sentence_match:
            return overlap_text[sentence_match.end():]
        
        # Find the first paragraph boundary
        paragraph_match = re.search(r'\n\n', overlap_text)
        if paragraph_match:
            return overlap_text[paragraph_match.end():]
        
        return overlap_text
    
    def _finalize_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Convert sets to lists for serialization"""
        metadata = chunk['metadata']
        return {
            'content': chunk['content'],
            'metadata': {
                'scenes': list(metadata['scenes']),
                'characters': list(metadata['characters']),
                'text_types': list(metadata['text_types']),
                'page_numbers': list(metadata['page_numbers']),
                'scene_count': len(metadata['scenes']),
                'character_count': len(metadata['characters'])
            }
        }