import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    openai_api_key: str
    rag_top_k: int = 10
    collection_name: str = "matrix_collection"
    embedding_model: str = "text-embedding-3-large"
    embedding_dims: int = 256
    use_memory: bool = True
    qdrant_use_memory: bool = True
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    model_config = SettingsConfigDict(
        env_file=os.path.join(BASE_DIR, ".env"),
        env_file_encoding='utf-8',
        case_sensitive=False
    )
    
    def __hash__(self):
        return id(self)
    
    def __eq__(self, other):
        if not isinstance(other, Settings):
            return False
        return id(self) == id(other)

settings = Settings()
