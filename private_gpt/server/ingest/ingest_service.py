import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, AnyStr, BinaryIO

from injector import inject, singleton
from llama_index.core.node_parser import MarkdownElementNodeParser, SentenceWindowNodeParser
from llama_index.core.storage import StorageContext

from private_gpt.components.embedding.embedding_component import EmbeddingComponent
from private_gpt.components.ingest.ingest_component import get_ingestion_component
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.node_store.node_store_component import NodeStoreComponent
from private_gpt.components.vector_store.vector_store_component import (
    VectorStoreComponent,
)
from private_gpt.server.ingest.model import IngestedDoc
from private_gpt.settings.settings import settings
# from private_gpt.server.ingest.ingest_router import DocumentMetadadta

if TYPE_CHECKING:
    from llama_index.core.storage.docstore.types import RefDocInfo

logger = logging.getLogger(__name__)

#---------------test-------------------
# metadata
from llama_index.core.schema import MetadataMode

from llama_index.core.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
    KeywordExtractor,
    BaseExtractor,
)
# from llama_index.extractors.entity import EntityExtractor
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.llms.openai import OpenAI
#-----------------------------------------------
class IngestService:
    @inject
    def __init__(
        self,
        llm_component: LLMComponent,
        vector_store_component: VectorStoreComponent,
        embedding_component: EmbeddingComponent,
        node_store_component: NodeStoreComponent,
    ) -> None:
        self.llm_service = llm_component
        self.storage_context = StorageContext.from_defaults(
            vector_store=vector_store_component.vector_store,
            docstore=node_store_component.doc_store,
            index_store=node_store_component.index_store,
        )
        # node_parser = SentenceWindowNodeParser.from_defaults()
        # using markdown parse because the output format is markdown
        node_parser = MarkdownElementNodeParser(
            llm=llm_component.llm,
            num_workers=8,
        )
        extractor_summary = SummaryExtractor(summaries=[
            "prev", "self", "next",], llm = llm_component.llm)
        transformations = [node_parser,
                           embedding_component.embedding_model,
                           extractor_summary]
        
        self.ingest_component = get_ingestion_component(
            self.storage_context,
            embed_model=embedding_component.embedding_model,
            settings=settings(),
            transformations=transformations)
        
        # - `TitleExtractor`: Document title, possible inferred across multiple nodes
        # `QuestionsAnsweredExtractor`: Questions that the node can answer
        # `KeywordsExtractor`: Keywords that uniquely identify the node
        # `SummaryExtractor`: Summary of each node, and pre and post nodes
    async def _ingest_data(self, file_name: str, file_data: AnyStr, docmeta) -> list[IngestedDoc]:
        logger.debug("Got file data of size=%s to ingest", len(file_data))
        logger.info("The metadata related to the document is: %s", docmeta.dict())  
        async def async_ingest():
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                try:
                    path_to_tmp = Path(tmp.name)
                    if isinstance(file_data, bytes):
                        path_to_tmp.write_bytes(file_data)
                    else:
                        path_to_tmp.write_text(str(file_data))
                    logger.info(" xxxxxxxxxx  path_to_tmp: %s", path_to_tmp)
                    logger.info("x             file_name: %s", file_name)
                    return await self.ingest_file(file_name, path_to_tmp, docmeta)
                finally:
                    tmp.close()
                    path_to_tmp.unlink()
                    
        return await async_ingest()

    async def ingest_file(self, file_name: str, file_data: Path, docmeta) -> list[IngestedDoc]:
        logger.info("Ingesting file_name=%s", file_name)
        documents = await self.ingest_component.async_ingest(file_name, file_data, docmeta)
        logger.info("Finished ingestion file_name=%s", file_name)
        return [IngestedDoc.from_document(document) for document in documents]

    async def ingest_text(self, file_name: str, text: str) -> list[IngestedDoc]:
        logger.debug("Ingesting text data with file_name=%s", file_name)
        return await self._ingest_data(file_name, text)

    async def ingest_bin_data(
        self, file_name: str, raw_file_data: BinaryIO, docmeta
    ) -> list[IngestedDoc]:
        logger.debug(
        "Ingesting binary data with file_name=%s and metadata=%s",
        file_name,
        docmeta.dict()  # 如果需要记录元数据
        )
        logger.debug("Ingesting binary data with file_name=%s", file_name)
        file_data = raw_file_data.read()
        return await self._ingest_data(file_name, file_data, docmeta)

    async def async_bulk_ingest(self, files: list[tuple[str, Path]]) -> list[IngestedDoc]:
        logger.info("Ingesting file_names=%s", [f[0] for f in files])
        async def process_files():
            print(f"ingest_component: {self.ingest_component}")
            documents = await self.ingest_component.async_bulk_ingest(files)
            return [IngestedDoc.from_document(document) for document in documents]
            
        result = await process_files()
        logger.info("Finished ingestion file_names=%s", [f[0] for f in files])
        print(f"result: {result}")
        return result

    async def list_ingested(self) -> list[IngestedDoc]:
        ingested_docs: list[IngestedDoc] = []
        try:
            docstore = self.storage_context.docstore
            ref_docs: dict[str, RefDocInfo] | None = docstore.get_all_ref_doc_info()

            if not ref_docs:
                return ingested_docs

            for doc_id, ref_doc_info in ref_docs.items():
                doc_metadata = None
                if ref_doc_info is not None and ref_doc_info.metadata is not None:
                    doc_metadata = IngestedDoc.curate_metadata(ref_doc_info.metadata)
                ingested_docs.append(
                    IngestedDoc(
                        object="ingest.document",
                        doc_id=doc_id,
                        doc_metadata=doc_metadata,
                    )
                )
        except ValueError:
            logger.warning("Got an exception when getting list of docs", exc_info=True)
            pass
        logger.debug("Found count=%s ingested documents", len(ingested_docs))
        return ingested_docs

    async def delete(self, doc_id: str) -> None:
        """Delete an ingested document.

        :raises ValueError: if the document does not exist
        """
        logger.info(
            "Deleting the ingested document=%s in the doc and index store", doc_id
        )
        await self.ingest_component.delete(doc_id)



