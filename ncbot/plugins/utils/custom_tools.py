from typing import Optional, Type, List
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_community.document_loaders.chromium import AsyncChromiumLoader
from langchain_community.document_transformers.html2text import Html2TextTransformer
from langchain_community.utilities.searx_search import SearxSearchWrapper

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

class ScrapeInput(BaseModel):
    urls: List[str] = Field(description="URL to scrape")


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
        return docs_transformed[0].page_content
    
class SearchInput(BaseModel):
    query: str = Field(description="What to lookup on the internet")
    num_results: int = Field(description="Number of search results that are returned. Minimum of 5 required.")


class SearchTool(BaseTool):
    name = "Search"
    description = "Browse the internet to return URLs and snippets of websites. Useful for when you need to answer questions about current events. You should ask targeted questions.  Always return your sources and urls of websites."
    args_schema: Type[BaseModel] = SearchInput
    return_direct: bool = False

    def _run(
        self, query: str, num_results: int, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool synchronously"""
        raise NotImplementedError("Calculator does not support async")

    async def _arun(
        self,
        query: str, 
        num_results: int,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        if num_results < 5:
            num_results=5
        search = SearxSearchWrapper(searx_host="http://localhost:8888")
        return search.results(query, num_results=num_results)
