import logging
from pathlib import Path
import os
import asyncio
import tempfile
import shutil
from tqdm import tqdm

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
        file_name: str, file_data: Path
    ) -> list[Document]:
        if settings().parse.async_mode:
            
            # 🔴 添加文件验证和处理
            try:
                # 创建带有正确扩展名的临时文件
                temp_dir = Path(tempfile.gettempdir())
                temp_file = temp_dir / file_name  # 使用原始文件名保持扩展名
                
                # 复制文件内容
                shutil.copy2(file_data, temp_file)
                logger.info(f"Created temporary file with extension: {temp_file}")

                try:
                    # 使用正确的文件路径
                    documents = await IngestionHelper._aload_file_to_documents([temp_file], 5)
                    
                    if not documents or not documents[0]:
                        logger.error("No documents generated from file")
                        return []

                    # 处理文档元数据
                    async def process_document(doc):
                        doc.metadata["file_name"] = file_name
                        IngestionHelper._exclude_metadata([doc])
                        return doc

                    logger.info("Processing documents metadata")
                    processed_docs = await asyncio.gather(*[process_document(doc) for doc in documents[0]])
                    
                    return processed_docs

                finally:
                    # 清理临时文件
                    if temp_file.exists():
                        temp_file.unlink()
                        logger.debug(f"Cleaned up temporary file: {temp_file}")

            except Exception as e:
                logger.exception(f"Error processing file {file_name}: {e}")
                return []
            
            
            
        #     #todo rewrite the transform_file_into_documents method
        #     logger.info("Transforming file_name=%s into documents", file_name)
        #     the_file = Path(file_data/file_name)
        #     file_paths = [file_data]
        #     documents = await IngestionHelper._aload_file_to_documents(file_paths, 5)
        #     async def process_document(doc):
        #         doc.metadata["file_name"] = file_name
        #         IngestionHelper._exclude_metadata([doc])
        #         return doc
        #     logger.info("Processing documents metadata")
        #     documents = await asyncio.gather(*[process_document(doc) for doc in documents[0]])
        #     print(f"documents get content: {documents[0].get_content()}")
            
        #     return documents
        # else:
        #     documents = IngestionHelper._load_file_to_documents(file_name, file_data)
        #     for document in documents:
        #         document.metadata["file_name"] = file_name
        #     IngestionHelper._exclude_metadata(documents)
        #     return documents

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
    async def _aload_file_to_documents(file_paths, batch_size: int) -> list[Document]:
        #todo: create a list of file paths 
        #todo: check parse using input_dir or input_file
        doc_paths= ["/home/gu/Documents/private-gpt/pdf/04_20210919_ISpec_FEBI_Operations_Inventur.pdf"]
        with tqdm(total=len(file_paths), desc="Parsing documents") as pbar:
            if not file_paths:
                logger.error("No files to parse")
                return []
            if settings().parse.name == "llamaparse":
                logger.info("Using LlamaParse to parse the file")
                parser = LlamaParse(result_type="markdown", 
                                    num_workers=8, 
                                    check_interval=10, 
                                    show_progress=True,
                                    language="de",
                                    page_separator="\n== {pageNumber} ==\n",
                                    use_vendor_multimodal_model=False,
                                    # vendor_multimodal_model_name="openai-gpt4o",
                                    # vendor_multimodal_api_key= os.getenv("OPENAI_API"),
                                    premium_mode=True) #todo: refacor
                alldocuments = []
                errors = []
                
                async def safe_load(file_paths):
                    try:
                        if not file_paths.exists():
                            logger.error(f"File does not exist: {file_paths}")
                            return None

                        if file_paths.suffix.lower() not in ['.pdf', '.docx', '.txt']:
                            logger.error(f"Unsupported file type: {file_paths.suffix}")
                            return None

                        logger.info(f"Parsing file: {file_paths}")
                        return await parser.aload_data(str(file_paths))

                    except Exception as e:
                        logger.exception(f"Error parsing file {file_paths}: {e}")
                        return None
                    # try:
                    #     logger.info("xxxxxxxxxxxxxxxxxxxxx Parsing file=%s", file_paths)
                    #     return await parser.aload_data(file_paths)
                    # except Exception as e:
                    #     errors.append((file_paths, e))
                    #     return None
                if len(file_paths) < batch_size:
                    alldocuments = await asyncio.gather(*[safe_load(doc_path) for doc_path in file_paths])
                else:
                    for i in range(0, len(file_paths), batch_size):
                        batch = file_paths[i:i+batch_size]
                        batch_tasks = [safe_load(doc_path) for doc_path in batch]
                        batch_results = await asyncio.gather(*batch_tasks)
                        alldocuments.extend(batch_results)
                        pbar.update(len(batch))
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


        """
        import asyncio
from typing import List, Any, Optional
from contextlib import asynccontextmanager

class BatchProcessor:
    def __init__(self, batch_size: int = 5, max_concurrent: int = 3):
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.parser = LlamaParse()
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    @asynccontextmanager
    async def get_parser(self):
        async with self.semaphore:
            yield self.parser
    
    async def process_single_file(self, path: str) -> Any:
        async with self.get_parser() as parser:
            return await parser.aload_data(path)
    
    async def batch_process_files(
        self,
        file_paths: List[str],
        cancel_event: Optional[asyncio.Event] = None
    ) -> List[Any]:
        if not file_paths:
            return []
        
        results = []
        actual_batch_size = min(self.batch_size, len(file_paths))
        
        for i in range(0, len(file_paths), actual_batch_size):
            # 检查是否请求取消
            if cancel_event and cancel_event.is_set():
                break
                
            batch = file_paths[i:i + actual_batch_size]
            batch_tasks = [self.process_single_file(path) for path in batch]
            
            try:
                batch_results = await asyncio.gather(*batch_tasks)
                results.extend(batch_results)
            except Exception as e:
                print(f"Batch processing error: {str(e)}")
                continue
        
        return results

# 使用示例
async def main():
    file_paths = ["file1.pdf", "file2.pdf", "file3.pdf"]
    processor = BatchProcessor(batch_size=5, max_concurrent=3)
    
    # 创建取消事件
    cancel_event = asyncio.Event()
    
    # 启动处理
    try:
        results = await processor.batch_process_files(file_paths, cancel_event)
        print(f"Processed {len(results)} files successfully")
    except Exception as e:
        print(f"Processing failed: {str(e)}")
        # 设置取消事件
        cancel_event.set()

if __name__ == "__main__":
    asyncio.run(main())
        """