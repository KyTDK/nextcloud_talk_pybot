from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_community.document_loaders.chromium import AsyncChromiumLoader
from langchain_community.document_transformers.beautiful_soup_transformer import BeautifulSoupTransformer
from langchain_community.utilities.searx_search import SearxSearchWrapper
from langchain_text_splitters import TokenTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders.word_document import Docx2txtLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders import UnstructuredODTLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader

import urllib.request
from typing import Dict, Optional, Tuple, Type, List, Any, Union
import tempfile
import re
import os
import nc_py_api
import pathlib
import ngram

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

#Scrape tool

class ScrapeInput(BaseModel):
    url: str = Field(description="URL to scrape")
    description: str = Field(description="Detailed and concise description of the type of information that should be extracted from the website")

def create_data_class(description):
    class Data(BaseModel):
        data: Optional[Any] = Field(None, description=description)
        urls: Optional[Any] = Field(None, description="Links in page from hyperlinks etc")
    
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

def save_file(bytes):
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file_path = temp_file.name
    
    try:
        # Download the file from the URL
        temp_file.write(bytes)
        
        # Close the file after writing
        temp_file.close()
        
        # Return the path of the downloaded file
        return temp_file_path
    
    except Exception as e:
        # If any error occurs, delete the temporary file
        temp_file.close()
        os.unlink(temp_file_path)
        raise e

def ai_read_data(description, content):
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
        chunk_size=1000,
        # Controls overlap between chunks
        chunk_overlap=20,
    )
            
    texts = text_splitter.split_text(content)
    
    # Limit just to the first 3 chunks
    # so the code can be re-run quickly
    first_few = texts[:20]
    
    extractions = extractor.batch(
        [{"text": text} for text in first_few],
        {"max_concurrency": 5},  # limit the concurrency by passing max concurrency!
    )

    results = []

    for extraction in extractions:
        results.extend(extraction.data)
    return results

async def documents_to_content(documents):
    page_content=""
    for document in documents:
        page_content = page_content+document.page_content
    page_content = re.sub("\n\n+", "\n", page_content)
    return page_content

async def get_file_content(location, file_type):
    if(file_type=="pdf" or file_type==".pdf"):
        loader = PyPDFLoader(location)
        data = await loader.aload()
        data = await documents_to_content(data)
    elif(file_type=="vnd.openxmlformats-officedocument.wordprocessingml.document" or file_type==".docx"):
        loader = Docx2txtLoader(location)
        data = await loader.aload()
        data = await documents_to_content(data)
    elif(file_type==".odt"):
        loader = UnstructuredODTLoader(location, mode="elements")
        data = await loader.aload()
        data = await documents_to_content(data)
    elif(file_type==".txt"):
        # Open the file in read mode
        with open(location, 'r') as file:
            # Read the entire content of the file
            data = file.read()
    elif(file_type==".md"):
        loader = UnstructuredMarkdownLoader(location)
        data = await loader.aload()
        data = await documents_to_content(data)
    else:
        return "Document not supported"
    return data

class ScrapeTool(BaseTool):
    name = "Scrape"
    description = "Extract specific text from a website for a specific URL. Always return your sources and the urls scraped"
    args_schema: Type[BaseModel] = ScrapeInput
    return_direct: bool = False

    def _run(
        self, url: str, description: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool synchronously"""
        raise NotImplementedError("Scrape does not support sync")

    async def _arun(
        self,
        url: str,
        description: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        #Get document type
        try:
            with urllib.request.urlopen(url) as response:
                content_subtype = response.info().get_content_subtype()
            if(content_subtype=="html"):
                loader = AsyncChromiumLoader([url])
                html = await loader.aload()
                bs_transformer = BeautifulSoupTransformer()
                documents = bs_transformer.transform_documents(html, remove_lines=True, remove_comments=True)
                page_content = await documents_to_content(documents)
            else:
                downloaded_file = download_file(url)
                page_content = await get_file_content(content_subtype, downloaded_file)
        except Exception as e:
            page_content = "An error occured. Either the website doesn't allow scraping, or it is currently down."
        results = ai_read_data(description, page_content)
        
        return results
    
#Search tool

class SearchInput(BaseModel):
    query: str = Field(description="What to lookup on the internet")
    num_results: int = Field(description="Number of search results that are returned. Minimum of 5 required.")


class SearchTool(BaseTool):
    name = "Search"
    description = "Browse the internet to return URLs and snippets of websites, extract text for further information as snippets are brief. Useful for when you need to answer questions about current events. You should ask targeted questions. Always return your sources and urls of websites."
    args_schema: Type[BaseModel] = SearchInput
    return_direct: bool = False

    def _run(
        self, query: str, num_results: int, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool synchronously"""
        raise NotImplementedError("Search does not support sync")

    async def _arun(
        self,
        query: str, 
        num_results: int,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        if num_results < 5:
            num_results=5
        if num_results > 20:
            num_results=20
        search = SearxSearchWrapper(searx_host="http://localhost:8888")
        return search.results(query, num_results=num_results)

def get_shared_files_paths(username: str, nc: nc_py_api.Nextcloud):
    all_shares = nc.files.sharing.get_list(shared_with_me=True)
    user_shared_files = []
    for share in all_shares:
        if share.file_owner==username:
            user_shared_files.append(share.path)
    return user_shared_files

#File get tool

class FileGetByLocationInput(BaseModel):
    file_location: str = Field(description="File location to open")
    description: str = Field(description="Detailed and concise description of the type of information that should be extracted from the file")

class FileGetByLocationTool(BaseTool):
    nc: nc_py_api.Nextcloud = None
    username: str = None
    name = "file_read_by_location"
    description = "Get and read a file by its location, get locations with file_list"
    args_schema: Type[BaseModel] = FileGetByLocationInput
    return_direct: bool = False
    
    def _run(
        self, file_location: str, description: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool synchronously"""
        raise NotImplementedError("Files does not support sync")

    async def _arun(
        self,
        file_location: str,
        description: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        user_shared_files = []
        for share in self.nc.files.sharing.get_list(shared_with_me=True):
            if share.file_owner==self.username:
                user_shared_files.append(share)
                
        result_path = None
        max_similarity = 0.0  # Initialize maximum similarity
        for share in user_shared_files:
            similarity = ngram.NGram.compare(file_location, share.path)
            if similarity > max_similarity and similarity>0.01:
                max_similarity = similarity
                result_path = share.path
        if result_path:
            file = self.nc.files.by_path(result_path)
            data = self.nc.files.download(result_path)
            file_type = ''.join(pathlib.Path(file.name).suffixes)
            saved_file_location = save_file(data)
            content = await get_file_content(saved_file_location, file_type)
            return ai_read_data(description, content)
        return "No results found, try again with one of the following " + str(get_shared_files_paths(self.username, self.nc))
        
#File list tool

class FileListTool(BaseTool):
    nc: nc_py_api.Nextcloud = None
    username: str = None

    name = "file_list"
    description = "Get list of files shared by the user"
    return_direct: bool = False
    
    def _to_args_and_kwargs(self, tool_input: Union[str, Dict]) -> Tuple[Tuple, Dict]:
        return (), {}
    
    def _run(
        self, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool synchronously"""
        raise NotImplementedError("Files does not support sync")

    async def _arun(
        self,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        return get_shared_files_paths(self.username, self.nc)
