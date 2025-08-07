from fastapi import APIRouter, UploadFile, File, HTTPException, Form, BackgroundTasks
from fastapi.responses import JSONResponse
import os
from ..core import processor 
from fastapi.responses import FileResponse


router = APIRouter()

@router.post("/upload/", status_code=202, summary="Upload and process a PDF in the background")
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDFs are allowed.")

    file_contents = await file.read()
    background_tasks.add_task(processor.process_and_embed_pdf, file_contents, file.filename)
    
    return {"filename": file.filename, "message": "File upload successful. Processing has started in the background."}


@router.post("/ask/", summary="Ask a question and get a text answer and highlighted PDF")
async def ask_question(filename: str = Form(...), question: str = Form(...)):
    """
    Gets a conversational answer and attempts to highlight the source quote.
    Returns a JSON response with the answer and a URL to the highlighted PDF if successful.
    """
    rag_result = await processor.query_rag_pipeline(filename, question)

    if rag_result is None or rag_result.get("error"):
        error_message = rag_result.get("error") if rag_result else "Could not find a relevant answer."
        raise HTTPException(status_code=404, detail=error_message)

    conversational_answer = rag_result.get("answer")
    text_to_highlight = rag_result.get("quote_for_highlighting")
    
    highlighted_pdf_path = None
    if text_to_highlight:
        highlighted_pdf_path = processor.create_highlighted_pdf(filename, text_to_highlight)

    # The URL that the frontend will use to fetch the highlighted PDF
    highlighted_url = f"/files/highlighted_{filename}" if highlighted_pdf_path else None

    return JSONResponse(content={
        "answer": conversational_answer,
        "highlighted_pdf_url": highlighted_url
    })

# Endpoint to serve the static highlighted PDF files
@router.get("/files/{filename}")
async def get_file(filename: str):
    file_path = os.path.join(processor.HIGHLIGHTED_OUTPUT_DIRECTORY, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(path=file_path, media_type='application/pdf')
