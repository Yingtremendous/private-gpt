from typing import Literal, Annotated, List 

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, Form
import json
from pydantic import BaseModel, Field, field_validator

from private_gpt.server.ingest.ingest_service import IngestService
from private_gpt.server.ingest.model import IngestedDoc
from private_gpt.server.utils.auth import authenticated

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

# @ingest_router.post("/ingest", tags=["Ingestion"], deprecated=True)
# def ingest(request: Request, files: UploadFile) -> IngestResponse:
#     """Ingests and processes a file.

#     Deprecated. Use ingest/file instead.
#     """
#     return ingest_file(request, file)


@ingest_router.post("/ingest/file", tags=["Ingestion"])
def ingest_file(request: Request, file: UploadFile, docmeta: Annotated[str, Form()]) -> IngestResponse:
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
        docmeta_dict = json.loads(docmeta)
        docmeta_obj = DocumentMetadadta(**docmeta_dict)
        
        if file.filename is None:
            raise HTTPException(400, "No file name provided")

        ingested_documents = service.ingest_bin_data(file.filename, file.file, docmeta_obj)
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

@ingest_router.post("/ingest/mfiles", tags=["Ingestion"])
def ingest_file(request: Request, file: List[UploadFile], docmeta: Annotated[str, Form()]) -> IngestResponse:
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
        docmeta_dict = json.loads(docmeta)
        docmeta_obj = DocumentMetadadta(**docmeta_dict)
        
        if file.filename is None:
            raise HTTPException(400, "No file name provided")

        ingested_documents = service.ingest_bin_data(file.filename, file.file, docmeta_obj)
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