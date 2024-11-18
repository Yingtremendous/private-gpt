import logging
from pathlib import Path
import os
import asyncio
from tqdm import tqdm
import shutil
import tempfile

from private_gpt.settings.settings import settings

from llama_index.core.readers import StringIterableReader
from llama_index.core.readers.base import BaseReader
from llama_index.core.readers.json import JSONReader
from llama_index.core.schema import Document

# try llama-parse
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

logger = logging.getLogger(__name__)

# get llama-parse key
from dotenv import load_dotenv
load_dotenv()

# Inspired by the `llama_index.core.readers.file.base` module
def _try_loading_included_file_formats() -> dict[str, type[BaseReader]]:
    try:
        from llama_index.readers.file.docs import (  # type: ignore
            DocxReader,
            HWPReader,
            PDFReader,
        )
        from llama_index.readers.file.epub import EpubReader  # type: ignore
        from llama_index.readers.file.image import ImageReader  # type: ignore
        from llama_index.readers.file.ipynb import IPYNBReader  # type: ignore
        from llama_index.readers.file.markdown import MarkdownReader  # type: ignore
        from llama_index.readers.file.mbox import MboxReader  # type: ignore
        from llama_index.readers.file.slides import PptxReader  # type: ignore
        from llama_index.readers.file.tabular import PandasCSVReader  # type: ignore
        from llama_index.readers.file.video_audio import (  # type: ignore
            VideoAudioReader,
        )
    except ImportError as e:
        raise ImportError("`llama-index-readers-file` package not found") from e

    default_file_reader_cls: dict[str, type[BaseReader]] = {
        ".hwp": HWPReader,
        ".pdf": PDFReader,
        ".docx": DocxReader,
        ".pptx": PptxReader,
        ".ppt": PptxReader,
        ".pptm": PptxReader,
        ".jpg": ImageReader,
        ".png": ImageReader,
        ".jpeg": ImageReader,
        ".mp3": VideoAudioReader,
        ".mp4": VideoAudioReader,
        ".csv": PandasCSVReader,
        ".epub": EpubReader,
        ".md": MarkdownReader,
        ".mbox": MboxReader,
        ".ipynb": IPYNBReader,
    }
    return default_file_reader_cls


# Patching the default file reader to support other file types
FILE_READER_CLS = _try_loading_included_file_formats()
FILE_READER_CLS.update(
    {
        ".json": JSONReader,
    }
)

class IngestionHelper:
    """Helper class to transform a file into a list of documents.

    This class should be used to transform a file into a list of documents.
    These methods are thread-safe (and multiprocessing-safe).
    """

    @staticmethod
    async def transform_file_into_documents(
        file_name: str, file_data: Path, docmeta
        ) -> list[Document]:
        if settings().parse.async_mode:
            try:
                temp_dir = Path(tempfile.gettempdir())
                temp_file = temp_dir / file_name  # 使用原始文件名保持扩展名

                shutil.copy2(file_data, temp_file)
                logger.info(f"Created temporary file with extension: {temp_file}")

                try:
                    documents = await IngestionHelper._aload_file_to_documents([temp_file], 5, docmeta)
                    
                    if not documents or not documents[0]:
                        logger.error("No documents generated from file")
                        return []

                    async def process_document(doc):
                        doc.metadata["file_name"] = file_name
                        IngestionHelper._exclude_metadata([doc])
                        return doc

                    logger.info("Processing documents metadata")
                    processed_docs = await asyncio.gather(*[process_document(doc) for doc in documents[0]])
                    logger.info(f"Successfully processed {processed_docs} documents")
                    return processed_docs

                finally:
                    # 清理临时文件
                    if temp_file.exists():
                        temp_file.unlink()
                        logger.debug(f"Cleaned up temporary file: {temp_file}")

            except Exception as e:
                logger.exception(f"Error processing file {file_name}: {e}")
                return []

    @staticmethod
    def _load_file_to_documents(file_name: str, file_data: Path) -> list[Document]:
        logger.debug("Transforming file_name=%s into documents", file_name)
        extension = Path(file_name).suffix
        reader_cls = FILE_READER_CLS.get(extension)
        if reader_cls is None:
            logger.debug(
                "No reader found for extension=%s, using default string reader",
                extension,
            )
            # Read as a plain text
            string_reader = StringIterableReader()
            return string_reader.load_data([file_data.read_text()])
        logger.debug("Specific reader found for extension=%s", extension)
        documents = reader_cls().load_data(file_data)
        # Sanitize NUL bytes in text which can't be stored in Postgres
        for i in range(len(documents)):
            documents[i].text = documents[i].text.replace("\u0000", "")
        return documents
    
    @staticmethod
    async def _aload_file_to_documents(file_paths: list[Path], batch_size: int, docmeta) -> list[Document]:
        #todo: create a list of file paths 
        #todo: check parse using input_dir or input_file
        doc_paths= ["/home/gu/Documents/private-gpt/pdf/04_20210919_ISpec_FEBI_Operations_Inventur.pdf"]
        with tqdm(total=len(doc_paths), desc="Parsing documents") as pbar:
            if not file_paths:
                logger.error("No files to parse")
                return []
            if settings().parse.name == "llamaparse":
                logger.info("Using LlamaParse to parse the file")
                parsing_instruction = settings().parse.parsing_instruction if settings().parse.enable_parsing_instruction else None
                parser = LlamaParse(result_type="markdown", 
                                    num_workers=8, 
                                    check_interval=10, 
                                    show_progress=True,
                                    language="de",
                                    page_separator="\n== {pageNumber} ==\n",
                                    use_vendor_multimodal_model=False,
                                    vendor_multimodal_model_name="openai-gpt4o",
                                    vendor_multimodal_api_key= os.getenv("OPENAI_API"),
                                    premium_mode=True,
                                    parsing_instruction=parsing_instruction) #todo: refacor
                alldocuments = []
                errors = []
                
                async def safe_load(file_paths, docmeta):
                    try:
                        if not file_paths.exists():
                            logger.error(f"File does not exist: {file_paths}")
                            return None
                        if file_paths.suffix.lower() not in ['.pdf', '.docx', '.txt']:
                            logger.error(f"Unsupported file type: {file_paths.suffix}")
                            return None
                        
                        logger.info(f"Parsing file: {file_paths}")
                        documents =  await parser.aload_data(str(file_paths))
                        if not documents: 
                                logger.warning(f"No documents generated from file: {file_paths}")
                                return []
                        # Add metadata 
                        for doc in documents:
                            doc.metadata.update({
                                "department": docmeta.department,
                                "user": docmeta.user,
                                "description": docmeta.description,
                                "tags": docmeta.tags
                            })
                        return documents
                    except Exception as e:
                        logger.exception(f"Error parsing file {file_paths}: {e}")
                        return None

                if len(file_paths) < batch_size:
                    alldocuments = await asyncio.gather(*[safe_load(doc_path, docmeta) for doc_path in file_paths])
                else:
                    for i in range(0, len(file_paths), batch_size):
                        batch = file_paths[i:i+batch_size]
                        batch_tasks = [safe_load(doc_path, docmeta) for doc_path in batch]
                        batch_results = await asyncio.gather(*batch_tasks)
                        alldocuments.extend(batch_results)
                        pbar.update(len(batch))
                logger.info(f"Successfully parsed {alldocuments[0]} documents")
                # get text
                # logger.info(f"get detail of the parsed document documents: \n{alldocuments[0][0].text} ") 
                # logger.info(f"get detail of the parsed document documents: \n{alldocuments[0][0].metadata}") 
                # hier wird kein metadata zurückgegeben
                # logger.info(f"get detail of the parsed document documents: {alldocuments[0][0].text.split[[1]} ")
                return alldocuments
            else:
                logger.error("Invalid parser name")
                


    @staticmethod
    def _exclude_metadata(documents: list[Document]) -> None:
        logger.debug("Excluding metadata from count=%s documents", len(documents))
        for document in documents:
            document.metadata["doc_id"] = document.doc_id
            # We don't want the Embeddings search to receive this metadata
            document.excluded_embed_metadata_keys = ["doc_id"]
            # We don't want the LLM to receive these metadata in the context
            document.excluded_llm_metadata_keys = ["file_name", "doc_id", "page_label"]

