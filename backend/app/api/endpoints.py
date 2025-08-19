from fastapi import APIRouter, UploadFile, File, HTTPException, Form, BackgroundTasks, status
from fastapi.responses import JSONResponse, FileResponse
import os
import logging
import uuid
from ..core import processor

# --- Router and Logging Setup ---
router = APIRouter()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- In-Memory Task Store ---
tasks = {}

# --- Helper for Background Task ---
def process_pdf_and_update_status(task_id: str, file_contents: bytes, filename: str):
    """
    A wrapper function that runs the PDF processing and updates the task's
    status in our in-memory store upon completion or failure.
    """
    try:
        logger.info(f"Starting processing for task_id: {task_id}")
        processor.process_and_embed_pdf(file_contents, filename)
        tasks[task_id]["status"] = "completed"
        logger.info(f"Successfully completed processing for task_id: {task_id}")
    except Exception as e:
        logger.error(f"Processing failed for task {task_id}: {str(e)}")
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)


# --- Endpoints ---

@router.post(
    "/upload/",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload and process a PDF in the background",
)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF file to upload and process")
):
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are allowed."
        )

    try:
        file_contents = await file.read()
        if not file_contents.startswith(b'%PDF'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or corrupted PDF file."
            )
        
        task_id = str(uuid.uuid4())
        tasks[task_id] = {"status": "processing", "filename": file.filename}

        background_tasks.add_task(
            process_pdf_and_update_status,
            task_id,
            file_contents,
            file.filename
        )

        logger.info(f"Task {task_id} created for file: {file.filename}")

        return {
            "message": "File is being processed in the background.",
            "task_id": task_id,
            "filename": file.filename
        }
    except Exception as e:
        logger.error(f"Error during file upload for {file.filename}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )


@router.get(
    "/status/{task_id}",
    summary="Check the status of a background processing task",
)
async def get_task_status(task_id: str):
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found.")
    return {"status": task["status"], "filename": task.get("filename"), "error": task.get("error")}


@router.post(
    "/ask/",
    summary="Ask a question and get a highlighted PDF",
)
async def ask_question(
    filename: str = Form(..., description="Name of the processed PDF file"),
    question: str = Form(..., description="Question about the document content")
):
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
                highlighted_filename = os.path.basename(highlighted_pdf_path)
                # --- FIX ---
                # The URL must now include the /api prefix to match the router
                highlighted_url = f"/api/files/{highlighted_filename}"

        return JSONResponse(content={
            "answer": conversational_answer,
            "highlighted_pdf_url": highlighted_url
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error answering question for file {filename}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your question."
        )

@router.get(
    "/files/{filename}",
    summary="Download a highlighted PDF",
)
async def get_file(filename: str):
    """Serve highlighted PDF files from the output directory."""
    try:
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
