import langchain as lc
import json
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.chains import LLMChain
from langchain.schema import Document, BaseOutputParser
from langchain.schema.language_model import BaseLanguageModel
from langchain.chat_models import ChatOpenAI
from langchain.document_transformers import Html2TextTransformer
from langchain.text_splitter import TokenTextSplitter, TextSplitter
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
)
from langchain.chains import (
    MapReduceDocumentsChain,
    ReduceDocumentsChain,
)
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser

import src.utils as utils
import src.basic_combine_docs_chain as bcdc
from langchain.chains import SequentialChain


# Pydantic classes
class MatchPrediction(BaseModel):
    reasoning: str = Field(
        ...,
        description="Logical reasoning and thought process used to predict the winner of the match",
    )
    winner: str = Field(
        ...,
        description='The team who is predicted to win the match. Use "Draw" if a draw is predicted',
    )
    points_margin: int = Field(
        ..., description="The predicted points difference for the winning team", ge=0
    )


# Environment variables
load_dotenv(".env", override=True)
# Langchain setup
lc.verbose = True

# Constants
GPT_MODEL_NAME = "gpt-3.5-turbo-16k"
NO_RELEVANT_INFO_MSG = "No relevant info"
SUMMARISE_PROMPT_TEMPLATE = (
    "You are an expert rugby pundit. Extract and summarise relevant info from the web scraped text, "
    + "only if it relates to the match between {home} and {away} in the 2023 rugby world cup (e.g. lineups, form, rankings, odds, favourites, etc...). "
    + "If there is no relevant info to the match in the entire passage, respond with `"
    + NO_RELEVANT_INFO_MSG
    + "`"
)
INITIAL_PREDICTION_FORMAT_INSTRUCTIONS = """
Answer using the following format:
- <important factor 1>
- <thought/observation>
- <another factor ...>
... (as many iterations as needed to get to a final answer)
Predicted winner: <team>
Predicted points margin: <number>
""".strip()
INITIAL_PREDICTION_PROMPT_PREFIX = (
    "You are an expert rugby pundit consulting me with my SuperBru predictions for the 2023 Rugby World Cup. "
    + "Use the information provided in the summary below (insofar as it is relevant) to predict the match outcome between {home} and {away}. "
    + "Think logically through your prediction, critically analysing relevant factors e.g. lineups, form, rankings, odds. "
    + "\n{format_instructions}\n"
)
REVIEW_PREDICTION_PROMPT_PREFIX = (
    "You are an expert rugby pundit consulting me with my SuperBru predictions for the 2023 Rugby World Cup. "
    + "Critically review the initial match prediction between {home} and {away} for reasonability, accuracy and soundness. "
    + "Make use of reasoning and the information provided in the summary below (insofar as it is relevant). "
    + "Think through your prediction. If you think it should change, do so. "
    + "Provide the final answer IN JSON FORMAT!!"
    + "\n{format_instructions}\n"
)
# Default LLMS
DEFAULT_OPENAI_KWARGS = {
    "temperature": 0,
    "max_tokens": 512,
    "max_retries": 3,
    "cache": False,
}
get_llm_small = lambda: ChatOpenAI(model="gpt-3.5-turbo", **DEFAULT_OPENAI_KWARGS)
get_llm = lambda: ChatOpenAI(model=GPT_MODEL_NAME, **DEFAULT_OPENAI_KWARGS)


# Functions
def _build_map_reduce_chain(
    llm: BaseLanguageModel = None,
    llm_summariser: BaseLanguageModel = None,
    input_key: str = "scraped_docs",
    summarise_prompt_template: str = SUMMARISE_PROMPT_TEMPLATE,
    raw_document_variable_name: str = "raw_content",
    summary_variable_name: str = "summaries",
    map_reduce_output_key: str = "reduced_summary",
) -> MapReduceDocumentsChain:
    if llm is None:
        llm = get_llm()
    if llm_summariser is None:
        llm_summariser = get_llm()
    # Chain that is mapped over each document (small LLM used)
    summarise_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate(
                prompt=PromptTemplate.from_template(summarise_prompt_template)
            ),
            HumanMessagePromptTemplate.from_template(
                "Scraped text: \n{" + raw_document_variable_name + "}\n"
            ),
        ]
    )
    summarise_chain = LLMChain(llm=llm_summariser, prompt=summarise_prompt)
    combine_documents_chain = bcdc.BasicStuffDocumentsChain(
        document_variable_name=summary_variable_name
    )
    reduce_documents_chain = ReduceDocumentsChain(
        combine_documents_chain=combine_documents_chain,
    )
    # Summarise and reduce
    map_reduce_chain = MapReduceDocumentsChain(
        llm_chain=summarise_chain,
        reduce_documents_chain=reduce_documents_chain,
        document_variable_name=raw_document_variable_name,
        input_key=input_key,
        output_key=map_reduce_output_key,
        return_intermediate_steps=True,
    )
    return map_reduce_chain


def _build_initial_prediction_chain(
    llm: BaseLanguageModel = None,
    prompt_prefix: str = INITIAL_PREDICTION_PROMPT_PREFIX,
    format_instructions: str = INITIAL_PREDICTION_FORMAT_INSTRUCTIONS,
    summary_key: str = "reduced_summary",
    output_key: str = "initial_prediction",
) -> LLMChain:
    # Use small LLM as dealing with combined summarised info
    if llm is None:
        llm = get_llm_small()
    prediction_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate(
                prompt=PromptTemplate.from_template(
                    prompt_prefix,
                    partial_variables={"format_instructions": format_instructions},
                )
            ),
            AIMessagePromptTemplate(
                prompt=PromptTemplate.from_template(
                    'Summary:\n"""\n{' + summary_key + '}\n"""\n'
                )
            ),
        ]
    )
    match_predictor_chain = LLMChain(
        llm=llm,
        prompt=prediction_prompt,
        output_key=output_key,
    )
    return match_predictor_chain


def _build_reviewer_chain(
    llm: BaseLanguageModel = None,
    prompt_prefix: str = REVIEW_PREDICTION_PROMPT_PREFIX,
    output_parser: BaseOutputParser = PydanticOutputParser(
        pydantic_object=MatchPrediction
    ),
    summary_key: str = "reduced_summary",
    initial_prediction_key: str = "initial_prediction",
    output_key: str = "revised_prediction",
) -> LLMChain:
    # Small LLM is fine here
    if llm is None:
        llm = get_llm_small()
    format_instructions = output_parser.get_format_instructions()
    prediction_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate(
                prompt=PromptTemplate.from_template(
                    prompt_prefix,
                    partial_variables={
                        "format_instructions": format_instructions,
                    },
                )
            ),
            AIMessagePromptTemplate(
                prompt=PromptTemplate.from_template(
                    'Summary:\n"""{' + summary_key + '}"""\n'
                )
            ),
            HumanMessagePromptTemplate.from_template(
                'Initial prediction: \n"""\n{' + initial_prediction_key + '}\n"""'
            ),
        ]
    )

    # Pass combined summary to final llm
    reviewer_chain = LLMChain(
        llm=llm,
        prompt=prediction_prompt,
        output_parser=output_parser,
        output_key=output_key,
    )
    return reviewer_chain


class BaseGuBru(BaseModel):
    verbose: bool = True
    search: utils.SearchType | None = None
    map_reduce_chain: MapReduceDocumentsChain = Field(
        default_factory=lambda **kw: _build_map_reduce_chain()
    )
    match_predictor_chain: LLMChain = Field(
        default_factory=lambda **kw: _build_initial_prediction_chain()
    )
    reviewer_chain: LLMChain = Field(
        default_factory=lambda **kw: _build_reviewer_chain()
    )
    text_splitter: TextSplitter = Field(
        default_factory=lambda **kw: TokenTextSplitter(
            model_name=GPT_MODEL_NAME,
            chunk_size=8000,  # max number of tokens per chunk
            chunk_overlap=400,
        )
    )
    html2text: Html2TextTransformer = Html2TextTransformer()
    verbose: bool = True

    class Config:
        arbitrary_types_allowed = True

    def search_queries(self, home: str, away: str, num_results=4) -> list[dict]:
        raise NotImplementedError()

    async def get_docs_from_search_results(
        self, search_results: list[dict], first_chunk_only: bool = True
    ) -> list[Document]:
        raise NotImplementedError()

    def run(self, home: str, away: str, docs: list[Document]) -> dict:
        raise NotImplementedError()


class SuperGuBru(BaseGuBru):
    def __init__(self, search_name: str = "ddg", **data):
        data["search"] = utils.get_search_engine(search_name)
        super().__init__(**data)

    def search_queries(self, home: str, away: str, num_results=4) -> list[dict]:
        wc_prefix = "Rugby world cup 2023"
        queries = {
            "predictions": f"{home} v {away} rugby world cup prediction",
            "lineups": f"{wc_prefix} {home} v {away} lineups",
            "news": f"News for {home} v {away} {wc_prefix}",
            "odds": f"Odds for {home} v {away} {wc_prefix}",
        }
        # Send search query
        if self.verbose:
            print(f"Running query: {queries['predictions']} with {num_results} results")
        pred_query_result = utils.run_search(
            self.search,
            queries["predictions"],
            num_results=num_results,
            as_dict=True,
        )
        if self.verbose:
            print(f"Running query: {queries['odds']} with 1 result")
        odds_query_result = utils.run_search(
            self.search,
            queries["odds"],
            num_results=1,
            as_dict=True,
        )
        return pred_query_result + odds_query_result

    async def get_docs_from_search_results(
        self, search_results: list[dict], first_chunk_only: bool = True
    ) -> list[Document]:
        # Get urls to follow
        query_urls = [el["link"] for el in search_results]
        query_urls = list(set(query_urls))  # unique
        if self.verbose:
            print("urls retrieved: ", "\n".join(query_urls), sep="\n")
        # Scrape urls
        query_docs = await utils.load_docs_from_urls(query_urls)
        if self.verbose:
            print(
                "query_docs[0]: ",
                query_docs[0].metadata,
                utils.shorten_str(query_docs[0].page_content),
            )
        # Transform to text
        query_docs_text = list(self.html2text.transform_documents(query_docs))
        if self.verbose:
            print(
                "query_docs_text[0]: ",
                query_docs_text[0].metadata,
                utils.shorten_str(query_docs_text[0].page_content),
            )
        query_docs_text_clean = [
            Document(
                page_content=utils.clean_text(el.page_content), metadata=el.metadata
            )
            for el in query_docs_text
            # Filter
            if "404 not found" not in el.page_content.lower()
            and "error 404" not in el.page_content.lower()
        ]
        if self.verbose:
            print(
                "query_docs_text[0] after cleaning: ",
                query_docs_text_clean[0].metadata,
                utils.shorten_str(query_docs_text_clean[0].page_content),
            )
        # Chunk
        if first_chunk_only:
            query_docs_text_chunked = []
            for doc in query_docs_text_clean:
                splits = self.text_splitter.split_documents([doc])
                if len(splits) > 0:
                    # First chunk
                    query_docs_text_chunked.append(splits[0])
        else:
            query_docs_text_chunked = self.text_splitter.split_documents(
                query_docs_text_clean
            )
        if self.verbose:
            print(
                f"query_docs_text_chunked[0] with first_chunk_only={first_chunk_only}: ",
                query_docs_text_chunked[0].metadata,
                utils.shorten_str(query_docs_text_chunked[0].page_content),
            )
            print(
                [
                    len(el)
                    for el in [
                        query_urls,
                        query_docs,
                        query_docs_text,
                        query_docs_text_clean,
                        query_docs_text_chunked,
                    ]
                ]
            )
        return query_docs_text_chunked

    def run(self, home: str, away: str, docs: list[Document]) -> dict:
        _home_away = dict(home=home, away=away)
        full_chain = SequentialChain(
            chains=[
                self.map_reduce_chain,
                self.match_predictor_chain,
                self.reviewer_chain,
            ],
            input_variables=list(
                set(self.map_reduce_chain.input_keys + list(_home_away.keys()))
            ),
            return_all=True,
        )
        output = full_chain(
            {**_home_away, "scraped_docs": docs},
            return_only_outputs=False,
        )
        if self.verbose:
            print(
                "full chain output",
                {k: v for k, v in output.items() if k != "scraped_docs"},
                sep="\n",
            )
        return output


#####
# MOCK Implemenation
with open("testing/mock_output.json") as f:
    MOCK_PREDICTION = json.load(f)
    MOCK_PREDICTION["scraped_docs"]: list[Document] = [
        Document(**el) for el in MOCK_PREDICTION["scraped_docs"]
    ]
    MOCK_PREDICTION["revised_prediction"]: MatchPrediction = MatchPrediction(
        **MOCK_PREDICTION["revised_prediction"]
    )


class MockSuperGuBru(BaseGuBru):
    def search_queries(self, home: str, away: str, num_results=4) -> list[dict]:
        return [
            {"link": doc.metadata["source"]} for doc in MOCK_PREDICTION["scraped_docs"]
        ]

    async def get_docs_from_search_results(
        self, search_results: list[dict], first_chunk_only: bool = True
    ) -> list[Document]:
        return MOCK_PREDICTION["scraped_docs"]

    def run(self, home: str, away: str, docs: list[Document]) -> dict:
        return MOCK_PREDICTION


def get_gubru(mock: bool = False, **kwargs) -> BaseGuBru:
    if mock:
        return MockSuperGuBru(**kwargs)
    else:
        return SuperGuBru(**kwargs)
