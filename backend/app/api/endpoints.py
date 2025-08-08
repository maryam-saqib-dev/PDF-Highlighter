from fastapi import APIRouter, UploadFile, File, HTTPException, Form, BackgroundTasks, status
from fastapi.responses import JSONResponse, FileResponse
import os
import logging
from ..core import processor 

# --- Router and Logging Setup ---
router = APIRouter()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Endpoints ---

@router.post(
    "/upload/",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload and process a PDF in the background",
    responses={
        202: {"description": "File accepted for background processing"},
        400: {"description": "Invalid file type or content"},
        500: {"description": "Internal server error"}
    }
)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF file to upload and process")
):
    """
    Accepts a PDF, validates it, and adds the OCR and embedding process to a background queue.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are allowed."
        )

    try:
        file_contents = await file.read()
        
        # Basic validation to check if it's a PDF
        if not file_contents.startswith(b'%PDF'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or corrupted PDF file."
            )
        
        # Add the slow processing task to the background
        background_tasks.add_task(
            processor.process_and_embed_pdf,
            file_contents,
            file.filename
        )
        
        logger.info(f"Accepted file for processing: {file.filename}")
        
        return {
            "filename": file.filename,
            "message": "File is being processed in the background."
        }
    except Exception as e:
        logger.error(f"Error during file upload for {file.filename}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )

@router.post(
    "/ask/",
    summary="Ask a question and get a highlighted PDF",
    responses={
        200: {"description": "Successful response with answer and highlight URL"},
        404: {"description": "Document not found or no answer available"},
        500: {"description": "Internal server error"}
    }
)
async def ask_question(
    filename: str = Form(..., description="Name of the processed PDF file"),
    question: str = Form(..., description="Question about the document content")
):
    """
    Gets a conversational answer and attempts to highlight the source quote.
    Returns a JSON response with the answer and a URL to the highlighted PDF if successful.
    """
    try:
        rag_result = await processor.query_rag_pipeline(filename, question)
        
        if not rag_result or rag_result.get("error"):
            error_msg = rag_result.get("error", "No relevant answer found in the document.")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=error_msg
            )
        
        conversational_answer = rag_result.get("answer")
        text_to_highlight = rag_result.get("quote_for_highlighting")
        
        highlighted_url = None
        if text_to_highlight:
            highlighted_pdf_path = processor.create_highlighted_pdf(
                filename,
                text_to_highlight
            )
            if highlighted_pdf_path:
                # Construct the URL the frontend can use to fetch the file
                highlighted_filename = os.path.basename(highlighted_pdf_path)
                highlighted_url = f"/files/{highlighted_filename}"
        
        return JSONResponse(content={
            "answer": conversational_answer,
            "highlighted_pdf_url": highlighted_url
        })
    except HTTPException:
        raise # Re-raise exceptions from the RAG pipeline
    except Exception as e:
        logger.error(f"Error answering question for file {filename}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your question."
        )

@router.get(
    "/files/{filename}",
    summary="Download a highlighted PDF",
    responses={
        200: {"description": "Returns the highlighted PDF file"},
        404: {"description": "File not found"}
    }
)
async def get_file(filename: str):
    """Serve highlighted PDF files from the output directory."""
    try:
        # Security check to prevent directory traversal attacks
        if ".." in filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid filename."
            )
        
        file_path = os.path.join(processor.HIGHLIGHTED_OUTPUT_DIRECTORY, filename)
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Highlighted file not found."
            )
        
        return FileResponse(
            path=file_path,
            media_type='application/pdf',
            filename=filename
        )
    except Exception as e:
        logger.error(f"Error serving file {filename}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving file."
        )
