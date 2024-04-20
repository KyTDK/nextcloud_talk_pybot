from typing import Optional, Type, List
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_community.document_loaders.chromium import AsyncChromiumLoader
from langchain_community.document_transformers.html2text import Html2TextTransformer

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

class ScrapeInput(BaseModel):
    url: List[str] = Field(description="URL to scrape")


class ScrapeTool(BaseTool):
    name = "Scrape"
    description = "Scrape and return text for a specific url. Always return your sources and the urls scraped"
    args_schema: Type[BaseModel] = ScrapeInput
    return_direct: bool = False

    def _run(
        self, urls: List[str], run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool synchronously"""
        raise NotImplementedError("Calculator does not support async")

    async def _arun(
        self,
        urls: List[str],
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        loader = AsyncChromiumLoader(urls)
        docs = await loader.aload()
        html2text = Html2TextTransformer()
        docs_transformed = html2text.transform_documents(docs)
        return docs_transformed[0].page_content[0:500]