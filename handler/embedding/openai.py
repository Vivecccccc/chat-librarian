"""Wrapper around OpenAI embedding models."""
from __future__ import annotations

import logging
import os
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Union,
)

import numpy as np
from pydantic import BaseModel, Extra, root_validator
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from handler.embedding.vectorize import Vectorize

logger = logging.getLogger(__name__)


def _create_retry_decorator(embeddings: OpenAIEmbeddings) -> Callable[[Any], Any]:
    import openai

    min_seconds = 4
    max_seconds = 10
    # Wait 2^x * 1 second between each retry starting with
    # 4 seconds, then up to 10 seconds, then 10 seconds afterwards
    return retry(
        reraise=True,
        stop=stop_after_attempt(embeddings.max_retries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=(
            retry_if_exception_type(openai.error.Timeout)
            | retry_if_exception_type(openai.error.APIError)
            | retry_if_exception_type(openai.error.APIConnectionError)
            | retry_if_exception_type(openai.error.RateLimitError)
            | retry_if_exception_type(openai.error.ServiceUnavailableError)
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


def embed_with_retry(embeddings: OpenAIEmbeddings, **kwargs: Any) -> Any:
    """Use tenacity to retry the embedding call."""
    retry_decorator = _create_retry_decorator(embeddings)

    @retry_decorator
    def _embed_with_retry(**kwargs: Any) -> Any:
        return embeddings.client.create(**kwargs)

    return _embed_with_retry(**kwargs)


class OpenAIEmbeddings(BaseModel, Vectorize):
    client: Any  #: :meta private:
    model: str = "text-embedding-ada-002"
    deployment: str = model  # to support Azure OpenAI Service custom deployment names
    openai_api_version: str = "2022-12-01"
    # to support Azure OpenAI Service custom endpoints
    openai_api_base: Optional[str] = None
    # to support Azure OpenAI Service custom endpoints
    openai_api_type: Optional[str] = None
    embedding_ctx_length: int = 8191
    openai_api_key: Optional[str] = None
    openai_organization: Optional[str] = None
    allowed_special: Union[Literal["all"], Set[str]] = set()
    disallowed_special: Union[Literal["all"], Set[str], Tuple[()]] = "all"
    chunk_size: int = 1000
    """Maximum number of texts to embed in each batch"""
    max_retries: int = 6
    """Maximum number of retries to make when generating."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        openai_api_key = values["openai_api_key"] or os.environ.get("OPENAI_API_KEY", None)
        openai_api_base = values["openai_api_base"] or os.environ.get("OPENAI_API_BASE", "https://azure-openai-test-02.openai.azure.com")
        openai_api_type = values["openai_api_type"] or os.environ.get("OPENAI_API_TYPE", "azure")
        openai_api_version = values["openai_api_version"] or os.environ.get("OPENAI_API_VERSION", values["openai_api_version"])
        if openai_api_type == "azure":
            values["deployment"] = values["deployment"] if values["deployment"] is not None else values["model"]
        try:
            import openai
            openai.api_key = openai_api_key
            if openai_api_base:
                openai.api_base = openai_api_base
                openai.api_version = openai_api_version
            if openai_api_type:
                openai.api_type = openai_api_type
            values["client"] = openai.Embedding
        except ImportError:
            raise ValueError(
                "Could not import openai python package. "
                "Please install it with `pip install openai`."
            )
        return values

    def _embedding_func(self, text: str, *, engine: str) -> List[float]:
        """Call out to OpenAI's embedding endpoint."""
        # handle large input text
        assert len(text) <= self.embedding_ctx_length
        if self.model.endswith("001"):
            text = text.replace("\n", " ")
        return embed_with_retry(self, input=[text], engine=engine)["data"][0][
            "embedding"
        ]

    async def embed_text_bundle(
        self, 
        texts: List[str]
    ) -> List[List[float]]:
        embeddings: List[List[float]] = []
        for text in texts:
            embedding = self._embedding_func(text, engine=self.deployment)
            embeddings.append(embedding)
        return embeddings

    async def embed_text(self, text: str) -> List[float]:
        embedding = self._embedding_func(text, engine=self.deployment)
        return embedding