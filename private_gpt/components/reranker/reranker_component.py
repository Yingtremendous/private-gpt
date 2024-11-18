# reranker_component.py
from typing import List, Optional
from llama_index.postprocessor.node import BaseNodePostprocessor
from llama_index.indices.response.type import RESPONSE_TEXT_TYPE, RESPONSE_TYPE
from llama_index.llms import OpenAI
from llama_index.postprocessor import LLMSimilarityReranker, SimilarityPostprocessor, HybridSearchReranker
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.output_composer.compression import ContextualCompressionReranker
from llama_index.indices.postprocessor.node import NodeWithScore
class RerankerComponent(BaseNodePostprocessor):
    def __init__(
        self,
        reranker_type: str,
        top_k: Optional[int] = None,
        similarity_cutoff: Optional[float] = None,
        llm: Optional[OpenAI] = None,
        model_name: Optional[str] = None,
        **kwargs
    ):
        self.reranker_type = reranker_type
        self.top_k = top_k
        self.similarity_cutoff = similarity_cutoff
        self.llm = llm
        self.model_name = model_name
        self.kwargs = kwargs
    def postprocess_nodes(self, nodes: List[NodeWithScore], query: str) -> List[NodeWithScore]:
        match self.reranker_type:
            case "FlagEmbeddingReranker":
                reranker = FlagEmbeddingReranker(
                    top_k=self.top_k,
                    similarity_cutoff=self.similarity_cutoff,
                    model=self.model_name,
                    **self.kwargs
                )
                return reranker.postprocess_nodes(nodes, query_str=query)
            case "LLMSimilarityReranker":
                if self.llm is None:
                    raise ValueError("llm must be provided for LLMSimilarityReranker")
                reranker = LLMSimilarityReranker(
                    top_k=self.top_k,
                    llm=self.llm,
                    similarity_threshold=self.similarity_cutoff,
                    **self.kwargs
                )
                return reranker.postprocess_nodes(nodes, query_str=query)
            case "SimilarityPostprocessor":
                reranker = SimilarityPostprocessor(
                    similarity_cutoff=self.similarity_cutoff,
                    **self.kwargs
                )
                return reranker.postprocess_nodes(nodes)
            case "ContextualCompressionReranker":
                if self.llm is None:
                    raise ValueError("llm must be provided for ContextualCompressionReranker")
                reranker = ContextualCompressionReranker(
                    llm=self.llm,
                    top_k=self.top_k,
                    **self.kwargs
                )
                return reranker.postprocess_nodes(nodes, query_str=query)
            case "HybridSearchReranker":
                reranker = HybridSearchReranker(
                    top_k=self.top_k,
                    weight_keyword=self.kwargs.get("weight_keyword", 0.5),
                )
            case _:
                raise ValueError(f"Unknown reranker_type: {self.reranker_type}")