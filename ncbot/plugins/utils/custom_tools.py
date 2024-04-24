from typing import Optional, Type, List
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_community.document_loaders.chromium import AsyncChromiumLoader
from langchain_community.document_transformers.html2text import Html2TextTransformer
from langchain_community.utilities.searx_search import SearxSearchWrapper
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

texts = text_splitter.split_text(document.page_content)
vectorstore = FAISS.from_texts(texts, embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 1}
)  # Only extract from first document

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

class ScrapeInput(BaseModel):
    url: str = Field(description="URL to scrape")
    query: str = Field(description="Type of content to extract, for example, key ideas, dates, names, etc")


class ScrapeTool(BaseTool):
    name = "Scrape"
    description = "Extract specific text from a website for a specific URL. Always return your sources and the urls scraped"
    args_schema: Type[BaseModel] = ScrapeInput
    return_direct: bool = False

    def _run(
        self, url: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool synchronously"""
        raise NotImplementedError("Calculator does not support async")

    async def _arun(
        self,
        url: str,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        loader = AsyncChromiumLoader([url])
        document = await loader.aload()[0]
        texts = text_splitter.split_text(document.page_content)
        vectorstore = FAISS.from_texts(texts, embedding=OpenAIEmbeddings())
        
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 1}
        )  # Only extract from first document
        rag_extractor = {
            "text": retriever | (lambda docs: docs[0].page_content)  # fetch content of top doc
        } | extractor
        results = rag_extractor.invoke(query)
        return results
    
class SearchInput(BaseModel):
    query: str = Field(description="What to lookup on the internet")
    num_results: int = Field(description="Number of search results that are returned. Minimum of 5 required.")


class SearchTool(BaseTool):
    name = "Search"
    description = "Browse the internet to return URLs and snippets of websites. Useful for when you need to answer questions about current events. You should ask targeted questions. Always return your sources and urls of websites. Follow up with scrape to get additional information for complex questions."
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
