import os
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union
from pydantic import BaseModel, Extra, root_validator
from tenacity import retry, stop_after_attempt, wait_random_exponential


class OpenAIChatCompletion(BaseModel):
    client: Any  #: :meta private:
    model: str = "gpt-35-turbo"
    deployment: str = model
    openai_api_version: Optional[str] = "2023-03-15-preview"
    # to support Azure OpenAI Service custom endpoints
    openai_api_base: Optional[str] = None
    # to support Azure OpenAI Service custom endpoints
    openai_api_type: Optional[str] = None
    completion_ctx_length: int = 8191
    openai_api_key: Optional[str] = None
    allowed_special: Union[Literal["all"], Set[str]] = set()
    disallowed_special: Union[Literal["all"], Set[str], Tuple[()]] = "all"
    max_retries: int = 6
    """Maximum number of retries to make when generating."""

    # class Config:
    #     """Configuration for this pydantic object."""

    #     extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        openai_api_key = values["openai_api_key"] or os.environ.get("OPENAI_API_KEY", None)
        openai_api_base = values["openai_api_base"] or os.environ.get("OPENAI_API_BASE", "https://azure-openai-test-02.openai.azure.com")
        openai_api_type = values["openai_api_type"] or os.environ.get("OPENAI_API_TYPE", "azure")
        openai_api_version = values["openai_api_version"] or os.environ.get("OPENAI_API_VERSION", "2023-03-15-preview")
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
            values["client"] = openai.ChatCompletion
        except ImportError:
            raise ValueError(
                "Could not import openai python package. "
                "Please install it with `pip install openai`."
            )
        return values
    
    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
    def respond(self, msgs: List[Dict[str, str]]) -> str:
        response = self.client.create(model=self.model, messages=msgs, engine=self.deployment)
        return response["choices"][0].message.content.strip()