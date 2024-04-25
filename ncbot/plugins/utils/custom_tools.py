from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_community.document_loaders.chromium import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_community.utilities.searx_search import SearxSearchWrapper
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter, TokenTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document

import urllib.request
from typing_extensions import Annotated
from typing import Optional, Type, List, Any
import tempfile
import re

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

class ScrapeInput(BaseModel):
    url: str = Field(description="URL to scrape")
    description: str = Field(description="Detailed and concise description of the type of information that should be extracted from the website")

def create_data_class(description):
    class Data(BaseModel):
        data: Optional[Any] = Field(None, description=description)
    
    class ExtractionData(BaseModel):
        data: List[Data]
        
    return ExtractionData

def download_file(url):
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file_path = temp_file.name
    
    try:
        # Download the file from the URL
        with urllib.request.urlopen(url) as response:
            temp_file.write(response.read())
        
        # Close the file after writing
        temp_file.close()
        
        # Return the path of the downloaded file
        return temp_file_path
    
    except Exception as e:
        # If any error occurs, delete the temporary file
        temp_file.close()
        os.unlink(temp_file_path)
        raise e
    
class ScrapeTool(BaseTool):
    name = "Scrape"
    description = "Extract specific text from a website for a specific URL. Always return your sources and the urls scraped"
    args_schema: Type[BaseModel] = ScrapeInput
    return_direct: bool = False

    def _run(
        self, url: str, description: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool synchronously"""
        raise NotImplementedError("Calculator does not support async")

    async def _arun(
        self,
        url: str,
        description: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        #Get document type
        content_subtype="None"
        with urllib.request.urlopen(url) as response:
            content_subtype = response.info().get_content_subtype()
        documents=None
        if(content_subtype=="html"):
            loader = AsyncChromiumLoader([url])
            html = await loader.aload()
            bs_transformer = BeautifulSoupTransformer()
            documents = bs_transformer.transform_documents(html, remove_lines=True, remove_comments=True)
        elif(content_subtype=="pdf"):
            downloaded_file = download_file(url)
            loader = PyPDFLoader(downloaded_file)
            data = await loader.aload()
            documents = data
        elif(content_subtype=="vnd.openxmlformats-officedocument.wordprocessingml.document"):
            downloaded_file = download_file(url)
            loader = Docx2txtLoader(downloaded_file)
            data = await loader.aload()
            documents = data
        else:
            return "Document not supported"
    
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert identifying the requested information from text."
                    "Only extract information that fits the requested description. Extract nothing if no important information can be found in the text.",
                ),
                # MessagesPlaceholder('examples'), # Keep on reading through this use case to see how to use examples to improve performance
                ("human", "{text}"),
            ]
        )
        
        
        # We will be using tool calling mode, which
        # requires a tool calling capable model.
        llm = ChatOpenAI(
            # Consider benchmarking with a good model to get
            # a sense of the best possible quality.
            model="gpt-3.5-turbo-0125",
            # Remember to set the temperature to 0 for extractions!
            temperature=0,
        )
        
        extractor = prompt | llm.with_structured_output(
            schema=create_data_class(description),
            method="function_calling",
            include_raw=False,
        )
        
        text_splitter = TokenTextSplitter(
            # Controls the size of each chunk
            chunk_size=2000,
            # Controls overlap between chunks
            chunk_overlap=20,
        )
                
        page_content=""
        for document in documents:
            page_content = page_content+document.page_content
        page_content = re.sub("\n\n+", "\n", page_content)
        texts = text_splitter.split_text(page_content)
        
        # Limit just to the first 3 chunks
        # so the code can be re-run quickly
        first_few = texts[:100]
        
        extractions = extractor.batch(
            [{"text": text} for text in first_few],
            {"max_concurrency": 5},  # limit the concurrency by passing max concurrency!
        )

        results = []

        for extraction in extractions:
            results.extend(extraction.data)
        return results
    
class SearchInput(BaseModel):
    query: str = Field(description="What to lookup on the internet")
    num_results: int = Field(description="Number of search results that are returned. Minimum of 5 required.")


class SearchTool(BaseTool):
    name = "Search"
    description = "Browse the internet to return URLs and snippets of websites, use scrape for further information. Useful for when you need to answer questions about current events. You should ask targeted questions. Always return your sources and urls of websites."
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
        if num_resukts > 20:
            num_results=20
        search = SearxSearchWrapper(searx_host="http://localhost:8888")
        return search.results(query, num_results=num_results)
