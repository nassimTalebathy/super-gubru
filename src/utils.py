import re
from typing import List
from langchain.schema import Document
from langchain.document_loaders import AsyncHtmlLoader
from langchain.utilities.brave_search import BraveSearchWrapper
from langchain.utilities.google_search import GoogleSearchAPIWrapper
from langchain.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
import json
import asyncio
from pydantic import BaseModel
import os
import openai


def clean_text(text: str):
    new_text = text
    new_text = re.sub(r"\n{2,}", "\n", new_text)
    new_text = re.sub(r"[\t ]{2,}", " ", new_text)
    return new_text.strip()


async def load_docs_from_urls(urls: list[str]) -> List[Document]:
    loader = AsyncHtmlLoader(urls, requests_per_second=5)
    results = await loader.fetch_all(loader.web_paths)
    docs = []
    for i, text in enumerate(results):
        metadata = {"source": loader.web_paths[i]}
        docs.append(Document(page_content=clean_text(text), metadata=metadata))
    return docs


def load_docs_from_urls_sync(urls: list[str]) -> List[Document]:
    loader = AsyncHtmlLoader(urls, requests_per_second=5)
    results = asyncio.run(loader.fetch_all(loader.web_paths))
    docs = []
    for i, text in enumerate(results):
        metadata = {"source": loader.web_paths[i]}
        docs.append(Document(page_content=clean_text(text), metadata=metadata))
    return docs


class SearchResult(BaseModel):
    snippet: str
    title: str
    link: str


SearchType = BraveSearchWrapper | GoogleSearchAPIWrapper | DuckDuckGoSearchAPIWrapper


def run_search(
    search_wrapper: SearchType, query: str, num_results: int = 3, as_dict: bool = True
) -> List[SearchResult] | List[dict]:
    if isinstance(search_wrapper, BraveSearchWrapper):
        results = search_wrapper.run(query)
        results = json.loads(results)[:num_results]
    elif isinstance(search_wrapper, DuckDuckGoSearchAPIWrapper):
        results = search_wrapper.results(query, num_results=num_results)
    elif isinstance(search_wrapper, GoogleSearchAPIWrapper):
        results = search_wrapper.results(query, num_results=num_results)
    else:
        raise ValueError(f"invalid search_wrapper {search_wrapper}")
    results = [SearchResult(**el) for el in results]
    if as_dict:
        return [el.dict() for el in results]
    else:
        return results


def shorten_str(s: str, n: int = 200):
    if len(s) <= n * 2:
        return s
    return s[:n] + "..." + s[-n:]


def get_search_engine(name: str) -> SearchType:
    if name == "ddg":
        return DuckDuckGoSearchAPIWrapper(region="za-en", time="m")
    elif name == "brave":
        return BraveSearchWrapper(
            api_key=os.environ["BRAVE_SEARCH_API_KEY"],
            search_kwargs={
                "country": "ZA",
                "search_lang": "en",
                "count": 10,
                "freshness": "pm",  # last month,
                "text_decorations": False,
            },
        )
    elif name == "google":
        return GoogleSearchAPIWrapper(
            google_api_key=os.environ["GOOGLE_CUSTOM_SEARCH_API_KEY"],
            google_cse_id=os.environ["GOOGLE_CUSTOM_SEARCH_ID"],
            k=10,
        )
    else:
        raise ValueError(f"Invalid search name {name}")


def vanilla_chatgpt_response(
    query, system_msg="You are a helpful assistant", model="gpt-3.5-turbo"
):
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_msg), ("human", "{query}")]
    )
    llm = ChatOpenAI(model=model)
    llm_chain = LLMChain(llm=llm, prompt=prompt, output_key="text")
    prediction = llm_chain({"query": query})
    return prediction["text"]
