from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.combine_documents.base import (
    BaseCombineDocumentsChain,
)
from langchain.schema import BasePromptTemplate, Document, format_document
from langchain.prompts import PromptTemplate
from pydantic import Field, Extra
from typing import Tuple, List, Dict, Any
from langchain.callbacks.manager import Callbacks


class BasicStuffDocumentsChain(BaseCombineDocumentsChain):
    """Chain that combines documents by stuffing into context."""

    document_prompt: BasePromptTemplate = Field(
        default_factory=lambda **kw: PromptTemplate.from_template(
            'Content (from "{source}"):\n{page_content}',
        )
    )
    document_variable_name: str = "summaries"
    document_separator: str = "\n\n"
    no_relevant_info_msg: str = "No relevant info"

    class Config:
        extra = Extra.forbid
        arbitrary_types_allowed = True

    def _get_doc_string(self, docs: List[Document], **kwargs: Any) -> dict:
        """Construct inputs from kwargs and docs."""
        # Format each document according to the prompt
        eligible_docs = [
            el for el in docs if self.no_relevant_info_msg not in el.page_content
        ]
        if len(eligible_docs) == 0:
            return self.no_relevant_info_msg
        doc_strings = [
            format_document(doc, self.document_prompt) for doc in eligible_docs
        ]
        return self.document_separator.join(doc_strings)

    def combine_docs(
        self, docs: List[Document], callbacks: Callbacks = None, **kwargs: Any
    ) -> Tuple[str, dict]:
        """Stuff all documents into one prompt and pass to LLM."""
        return self._get_doc_string(docs, **kwargs), {}

    async def acombine_docs(
        self, docs: List[Document], callbacks: Callbacks = None, **kwargs: Any
    ) -> Tuple[str, dict]:
        """Async stuff all documents into one prompt and pass to LLM."""
        return self._get_doc_string(docs, **kwargs), {}

    @property
    def _chain_type(self) -> str:
        return "basic_stuff_documents_chain"
