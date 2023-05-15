import os

from handler.embedding.vectorize import Vectorize
from models.api import Settings

async def get_vectorize(settings: Settings) -> Vectorize:
    embedding_method = settings.embedding_method
    chunk_size = settings.chunk_size
    match embedding_method:
        case "openai":
            from handler.embedding.openai import OpenAIEmbeddings
            api_key = settings.embedding_api_key
            api_type = settings.embedding_api_type
            api_base = settings.embedding_api_base
            api_version = settings.embedding_api_version
            return OpenAIEmbeddings(openai_api_key=api_key,
                                    openai_api_base=api_base,
                                    openai_api_type=api_type,
                                    openai_api_version=api_version,
                                    chunk_size=chunk_size)
        case _:
            from handler.embedding.vectorize import MockVectorize
            return MockVectorize()