from typing import Literal, Annotated, List 

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, Form
import json
from pydantic import BaseModel, Field, field_validator

from private_gpt.server.ingest.ingest_service import IngestService
from private_gpt.server.ingest.model import IngestedDoc
from private_gpt.server.utils.auth import authenticated
from concurrent.futures import ThreadPoolExecutor
from typing import List
from fastapi import Request, UploadFile, Form, HTTPException
from fastapi.routing import APIRouter
import json
from typing_extensions import Annotated
import asyncio
import concurrent.futures
from tempfile import SpooledTemporaryFile
import logging
logger = logging.getLogger(__name__)
ingest_router = APIRouter(prefix="/v1", dependencies=[Depends(authenticated)])


class IngestTextBody(BaseModel):
    file_name: str = Field(examples=["Avatar: The Last Airbender"])
    text: str = Field(
        examples=[
            "Avatar is set in an Asian and Arctic-inspired world in which some "
            "people can telekinetically manipulate one of the four elements—water, "
            "earth, fire or air—through practices known as 'bending', inspired by "
            "Chinese martial arts."
        ]
    )


class IngestResponse(BaseModel):
    object: Literal["list"]
    model: Literal["private-gpt"]
    data: list[IngestedDoc]

class DocumentMetadadta(BaseModel):
    department: str = Field(examples=["HR"], description="Department of the document")
    user: str = Field(examples=["John Doe"], description="User who uploaded the document")
    description: str = Field(examples=["Employee Handbook"], description="Description of the document")
    tags: list[str] = Field(examples=[["HR", "Employee Handbook"]], description="Tags for the document")
    @field_validator("department")
    def validate_department(cls, v):
        allowed_departments = ["HR", "Finance", "Legal", "ICC"]
        if v not in allowed_departments:
            raise ValueError(f"Department must be one of {allowed_departments}")
        if len(v) == 0:
            raise ValueError("Department cannot be empty")
        return v



        
@ingest_router.post("/ingest/mfiles", tags=["Ingestion"])
def ingest_files(request: Request, files: List[UploadFile]) -> IngestResponse:
    service = request.state.injector.get(IngestService)
    logger.info(f"Received {len(files)} files")
    
    try:
        # 1. 优化文件映射构建
        existing_files = {}
        try:
            ingested_documents = service.list_ingested()
            for doc in ingested_documents:
                # 正确访问 doc_metadata 中的 file_name
                file_name = doc.doc_metadata.get('file_name')
                if file_name:
                    if file_name not in existing_files:
                        existing_files[file_name] = []
                    existing_files[file_name].append(doc.doc_id)
            logger.info(f"existing_files: {existing_files}")
        except Exception as e:
            logger.error(f"Error listing ingested documents: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error listing ingested documents: {str(e)}")
            existing_files = {}

    # 2. 高效处理文档删除
        if existing_files:
            # 收集需要删除的文档ID
            docs_to_delete = {
                doc_id
                for file in files
                for doc_id in existing_files.get(file.filename, [])
            }
            
            if docs_to_delete:
                # 使用线程池并行删除文档
                with ThreadPoolExecutor(max_workers=min(len(docs_to_delete), 5)) as executor:
                    def delete_doc(doc_id):
                        try:
                            service.delete(doc_id)
                            logger.info(f"Deleted document {doc_id}")
                            return True
                        except Exception as e:
                            logger.error(f"Failed to delete document {doc_id}: {str(e)}")
                            return False

                    # 并行执行删除操作
                    deletion_results = list(executor.map(delete_doc, docs_to_delete))
                    logger.info(f"Deleted {sum(deletion_results)} documents out of {len(docs_to_delete)}")

        # 3. 直接使用原始文件列表
        files_to_process = files
        logger.info(f"Processing {len(files_to_process)} files")

    
        if not files:
            raise HTTPException(400, "No files provided")
        
        all_ingested_documents = []
        
        def process_single_file(file: UploadFile):
            try:
                # 读取文件内容
                contents = file.file.read()
                # 创建新的临时文件
                temp_file = SpooledTemporaryFile()
                temp_file.write(contents)
                temp_file.seek(0)
                
                # 处理文件
                result = service.ingest_bin_data(file.filename, temp_file)
                
                temp_file.close()
                file.file.seek(0)  # 重置文件指针
                return result
                
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error processing file {file.filename}: {str(e)}"
                )

        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=min(len(files), 5)) as executor:
            logger.info("Starting parallel processing")
            futures = [executor.submit(process_single_file, file) for file in files]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        all_ingested_documents.extend(result)
                except Exception as e:
                    raise e

        return IngestResponse(
            object="list",
            model="private-gpt",
            data=all_ingested_documents
        )
        
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Invalid JSON format in docmeta"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid metadata: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred: {str(e)}"
        )
    finally:
        # 确保所有文件都被关闭
        for file in files:
            file.file.close()
            
           
@ingest_router.post("/ingest/file", tags=["Ingestion"])
def ingest_file(request: Request, file: UploadFile) -> IngestResponse:
    """Ingests and processes a file, storing its chunks to be used as context.

    The context obtained from files is later used in
    `/chat/completions`, `/completions`, and `/chunks` APIs.

    Most common document
    formats are supported, but you may be prompted to install an extra dependency to
    manage a specific file type.

    A file can generate different Documents (for example a PDF generates one Document
    per page). All Documents IDs are returned in the response, together with the
    extracted Metadata (which is later used to improve context retrieval). Those IDs
    can be used to filter the context used to create responses in
    `/chat/completions`, `/completions`, and `/chunks` APIs.
    """
    service = request.state.injector.get(IngestService)
    try: 
        
        if file.filename is None:
            raise HTTPException(400, "No file name provided")

        ingested_documents = service.ingest_bin_data(file.filename, file.file)
        return IngestResponse(object="list", model="private-gpt", data=ingested_documents)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Invalid JSON format in docmeta"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid metadata: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred: {str(e)}" 
        )


@ingest_router.post("/ingest/text", tags=["Ingestion"])
def ingest_text(request: Request, body: IngestTextBody) -> IngestResponse:
    """Ingests and processes a text, storing its chunks to be used as context.

    The context obtained from files is later used in
    `/chat/completions`, `/completions`, and `/chunks` APIs.

    A Document will be generated with the given text. The Document
    ID is returned in the response, together with the
    extracted Metadata (which is later used to improve context retrieval). That ID
    can be used to filter the context used to create responses in
    `/chat/completions`, `/completions`, and `/chunks` APIs.
    """
    service = request.state.injector.get(IngestService)
    if len(body.file_name) == 0:
        raise HTTPException(400, "No file name provided")
    ingested_documents = service.ingest_text(body.file_name, body.text)
    return IngestResponse(object="list", model="private-gpt", data=ingested_documents)


@ingest_router.get("/ingest/list", tags=["Ingestion"])
def list_ingested(request: Request) -> IngestResponse:
    """Lists already ingested Documents including their Document ID and metadata.

    Those IDs can be used to filter the context used to create responses
    in `/chat/completions`, `/completions`, and `/chunks` APIs.
    """
    service = request.state.injector.get(IngestService)
    ingested_documents = service.list_ingested()
    return IngestResponse(object="list", model="private-gpt", data=ingested_documents)


@ingest_router.delete("/ingest/{doc_id}", tags=["Ingestion"])
def delete_ingested(request: Request, doc_id: str) -> None:
    """Delete the specified ingested Document.

    The `doc_id` can be obtained from the `GET /ingest/list` endpoint.
    The document will be effectively deleted from your storage context.
    """
    service = request.state.injector.get(IngestService)
    service.delete(doc_id)