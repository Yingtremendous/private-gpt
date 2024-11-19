import logging
from pathlib import Path
import os
from tqdm import tqdm
import asyncio
import shutil
import tempfile
import nest_asyncio

from private_gpt.settings.settings import settings
from llama_index.core.readers import StringIterableReader
from llama_index.core.readers.base import BaseReader
from llama_index.core.readers.json import JSONReader
from llama_index.core.schema import Document
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from dotenv import load_dotenv

# 初始化
logger = logging.getLogger(__name__)
load_dotenv()
nest_asyncio.apply()

def transform_file_into_documents(file_name: str, file_data: Path) -> list[Document]:
    """同步包装器，用于调用异步文档转换方法"""
    try:
        logger.debug("Creating new event loop")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            logger.debug("Starting async document transformation")
            documents = loop.run_until_complete(
                IngestionHelper.transform_file_into_documents(file_name, file_data)
            )
            logger.info(f"Documents processed successfully: {len(documents) if documents else 0} documents")
            return documents or []
        except Exception as e:
            logger.error(f"Error in async execution: {e}", exc_info=True)
            return []
        finally:
            logger.debug("Closing event loop")
            loop.close()
            
    except Exception as e:
        logger.error(f"Error in transform_file_into_documents: {e}", exc_info=True)
        return []

class IngestionHelper:
    """Helper class to transform a file into a list of documents."""
    
    @staticmethod
    async def transform_file_into_documents(
        file_name: str, file_data: Path
    ) -> list[Document]:
        try:
            temp_dir = Path(tempfile.gettempdir())
            temp_file = temp_dir / file_name

            shutil.copy2(file_data, temp_file)
            logger.info(f"Created temporary file with extension: {temp_file}")

            try:
                documents = await IngestionHelper._aload_file_to_documents([temp_file], 5)
                
                if not documents or not documents[0]:
                    logger.error("No documents generated from file")
                    return []

                async def process_document(doc):
                    doc.metadata["file_name"] = file_name
                    IngestionHelper._exclude_metadata([doc])
                    return doc

                logger.info("Processing documents metadata")
                processed_docs = await asyncio.gather(*[process_document(doc) for doc in documents[0]])
                return list(processed_docs) if processed_docs else []

            finally:
                if temp_file.exists():
                    temp_file.unlink()
                    logger.debug(f"Cleaned up temporary file: {temp_file}")

        except Exception as e:
            logger.exception(f"Error processing file {file_name}: {e}")
            return []

    @staticmethod
    async def _aload_file_to_documents(
        file_paths: list[Path], 
        batch_size: int
    ) -> list[Document]:
        doc_paths = ["/home/gu/Documents/private-gpt/pdf/04_20210919_ISpec_FEBI_Operations_Inventur.pdf"]
        
        with tqdm(total=len(doc_paths), desc="Parsing documents") as pbar:
            if not file_paths:
                logger.error("No files to parse")
                return []

            if settings().parse.name == "llamaparse":
                logger.info("Using LlamaParse to parse the file")
                parsing_instruction = settings().parse.parsing_instruction if settings().parse.enable_parsing_instruction else None
                
                parser = LlamaParse(
                    result_type="markdown", 
                    num_workers=8,
                    check_interval=10,
                    show_progress=True,
                    language="de",
                    # page_separator="\n== {pageNumber} ==\n",
                    # use_vendor_multimodal_model=False,
                    # vendor_multimodal_model_name="openai-gpt4o",
                    # vendor_multimodal_api_key=os.getenv("OPENAI_API"),
                    # premium_mode=False,
                    # parsing_instruction=parsing_instruction
                )

                async def safe_load(file_paths):
                    try:
                        if not file_paths.exists():
                            logger.error(f"File does not exist: {file_paths}")
                            return None
                        if file_paths.suffix.lower() not in ['.pdf', '.docx', '.txt']:
                            logger.error(f"Unsupported file type: {file_paths.suffix}")
                            return None
                        
                        logger.info(f"Parsing file: {file_paths}")
                        documents = await parser.aload_data(str(file_paths))
                        if not documents:
                            logger.warning(f"No documents generated from file: {file_paths}")
                            return []
                        
                        # for doc in documents:
                        #     doc.metadata.update({
                        #         "department": docmeta.department,
                        #         "user": docmeta.user,
                        #         "description": docmeta.description,
                        #         "tags": docmeta.tags
                        #     })
                        return list(documents)
                    except Exception as e:
                        logger.exception(f"Error parsing file {file_paths}: {e}")
                        return None

                try:
                    all_documents = []
                    if len(file_paths) < batch_size:
                        results = await asyncio.gather(*[safe_load(doc_path) for doc_path in file_paths])
                        all_documents = [doc for doc in results if doc]
                    else:
                        for i in range(0, len(file_paths), batch_size):
                            batch = file_paths[i:i+batch_size]
                            batch_tasks = [safe_load(doc_path) for doc_path in batch]
                            batch_results = await asyncio.gather(*batch_tasks)
                            all_documents.extend([doc for doc in batch_results if doc])
                            pbar.update(len(batch))

                    logger.info(f"Successfully parsed {len(all_documents)} documents")
                    return list(all_documents)
                except Exception as e:
                    logger.error(f"Error during document processing: {e}")
                    return []
            else:
                logger.error("Invalid parser name")
                return []

    @staticmethod
    def _exclude_metadata(documents: list[Document]) -> None:
        """Exclude specific metadata from documents."""
        logger.debug("Excluding metadata from count=%s documents", len(documents))
        for document in documents:
            document.metadata["doc_id"] = document.doc_id
            document.excluded_embed_metadata_keys = ["doc_id"]
            document.excluded_llm_metadata_keys = ["file_name", "doc_id", "page_label"]

