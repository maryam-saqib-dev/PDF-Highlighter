import os
import fitz  # PyMuPDF
from dotenv import load_dotenv
import ocrmypdf
import pytesseract
from PIL import Image
import io
import numpy as np
import layoutparser as lp
from thefuzz import fuzz
import asyncio
import re
from typing import Optional, Dict, Union

import time


# Langchain Imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Initialization ---
load_dotenv()

# Cache for vector stores with TTL (time-to-live) consideration
vector_store_cache: Dict[str, FAISS] = {}
CACHE_TTL_HOURS = 24  # Consider adding cache expiration logic

# --- Directory Setup ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIRECTORY = os.path.join(BASE_DIR, "data/uploads")
PROCESSED_DIRECTORY = os.path.join(BASE_DIR, "data/processed_pdfs")
HIGHLIGHTED_OUTPUT_DIRECTORY = os.path.join(BASE_DIR, "data/highlighted")
OCRMYPDF_TEXT_DIRECTORY = os.path.join(BASE_DIR, "data/ocrmypdf_text")
LAYOUTPARSER_TEXT_DIRECTORY = os.path.join(BASE_DIR, "data/layoutparser_text")
MODELS_DIRECTORY = os.path.join(BASE_DIR, "models")

# Ensure all directories exist
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
os.makedirs(PROCESSED_DIRECTORY, exist_ok=True)
os.makedirs(HIGHLIGHTED_OUTPUT_DIRECTORY, exist_ok=True)
os.makedirs(OCRMYPDF_TEXT_DIRECTORY, exist_ok=True)
os.makedirs(LAYOUTPARSER_TEXT_DIRECTORY, exist_ok=True)

# --- Initialize Models ---
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.environ.get("GEMINI_API_KEY"),
    task_type="RETRIEVAL_DOCUMENT"
)

# Initialize LayoutParser model for debugging comparison
layout_model = lp.Detectron2LayoutModel(
    config_path=os.path.join(MODELS_DIRECTORY, 'config.yml'),
    model_path=os.path.join(MODELS_DIRECTORY, 'model_final.pth'),
    label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8]
)

# --- Helper Functions ---

def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace and normalizing."""
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def create_ocr_layered_pdf(original_path: str, processed_path: str) -> str:
    """Create searchable PDF with optimized OCR settings."""
    try:
        ocrmypdf.ocr(
            input_file=original_path,
            output_file=processed_path,
            force_ocr=True,
            clean=True,
            deskew=True,
            optimize=1,  # Mild optimization (0-3)
            progress_bar=False,
            skip_big=30,  # Skip very large images
            jobs=4,  # Use 4 CPU cores
            # Safe parameters that work across versions:
            rotate_pages=True,
            remove_vectors=False,
            # For difficult text:
            tesseract_timeout=180,
            tesseract_oem=3,  # LSTM + Legacy engine
            tesseract_config=[
                'preserve_interword_spaces=1',
                'tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?()[]{}<>:;-/\\@#$%^&*_+=|"\''
            ]
        )
        return processed_path
    except ocrmypdf.exceptions.PriorOcrFoundError:
        print("PDF already contains text layer, creating copy...")
        os.replace(original_path, processed_path)
        return processed_path
    except Exception as e:
        print(f"OCR processing failed: {str(e)}")
        raise RuntimeError(f"Failed to process PDF: {str(e)}")

def debug_save_layoutparser_ocr_text(file_contents: bytes, filename: str):
    """Perform OCR with LayoutParser for debugging."""
    print(f"Starting LayoutParser OCR debug for '{filename}'...")
    try:
        doc = fitz.open(stream=file_contents, filetype="pdf")
        full_text = ""
        for page in doc:
            pix = page.get_pixmap(dpi=300)
            image = np.array(Image.open(io.BytesIO(pix.tobytes("png"))))
            layout = layout_model.detect(image)
            text_blocks = lp.Layout([b for b in layout if b.type in ['Text', 'Title', 'List']])
            for block in text_blocks:
                segment_image = block.pad(left=5, right=5, top=5, bottom=5).crop_image(image)
                full_text += pytesseract.image_to_string(Image.fromarray(segment_image), lang='eng') + "\n"
        
        output_path = os.path.join(LAYOUTPARSER_TEXT_DIRECTORY, f"layoutparser_text_{filename.replace('.pdf', '.txt')}")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(clean_text(full_text))
        print(f"Saved LayoutParser OCR text to: {output_path}")
    except Exception as e:
        print(f"Error in LayoutParser debug: {str(e)}")

# --- MAIN PROCESSING FUNCTIONS ---

def process_and_embed_pdf(file_contents: bytes, filename: str):
    """
    Main background task. Saves the file, runs ocrmypdf, and builds the vector store.
    """
    # Save the original uploaded file first
    original_pdf_path = os.path.join(UPLOAD_DIRECTORY, filename)
    with open(original_pdf_path, "wb+") as file_object:
        file_object.write(file_contents)

    # Define the path for the new, processed PDF
    processed_pdf_path = os.path.join(PROCESSED_DIRECTORY, filename)

    # 1. Create the "remastered" PDF with a perfect text layer using ocrmypdf
    create_ocr_layered_pdf(original_pdf_path, processed_pdf_path)

    # 2. Now, use this new, high-quality PDF as the source of truth
    doc = fitz.open(processed_pdf_path)
    full_text = "\n".join([page.get_text() for page in doc])
    doc.close()

    if not full_text.strip():
        print(f"Warning: No text could be extracted from the ocrmypdf-processed file: {filename}")
        return

    # 3. Chunk and embed the text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    chunks = text_splitter.split_text(full_text)
    
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store_cache[filename] = vector_store
    
    print(f"Successfully processed and embedded '{filename}'. Found {len(chunks)} chunks.")


async def query_rag_pipeline(filename: str, question: str) -> Optional[Dict[str, str]]:
    """Enhanced RAG pipeline with hybrid search and better prompting."""
    if filename not in vector_store_cache:
        return {"error": "Document not found. Please upload and process first."}

    vector_store = vector_store_cache[filename]
    
    # Hybrid search with MMR for diversity
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "lambda_mult": 0.5,
            "score_threshold": 0.7
        }
    )
    
    # Get relevant context
    context_docs = await retriever.aget_relevant_documents(question)
    context_text = "\n---\n".join([clean_text(doc.page_content) for doc in context_docs])

    # Initialize LLM with optimized settings
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.environ.get("GEMINI_API_KEY"),
        temperature=0.3,
        max_output_tokens=1000
    )
    
    # Improved prompt templates
    answer_prompt_template = """You are an expert assistant analyzing documents. 
    Answer the question based ONLY on the following context. Be precise and factual.
    If the answer isn't in the context, say "I couldn't find that information in the document."
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:"""
    
    quote_prompt_template = """Identify the most relevant exact quote from the context that answers this question:
    Question: {question}
    
    Context:
    {context}
    
    The quote must be verbatim from the context. If no direct quote exists, say "QUOTE_NOT_FOUND".
    Exact quote:"""
    
    # Create and run chains in parallel
    answer_chain = (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
        | ChatPromptTemplate.from_template(answer_prompt_template)
        | llm
        | StrOutputParser()
    )
    
    quote_chain = (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
        | ChatPromptTemplate.from_template(quote_prompt_template)
        | llm
        | StrOutputParser()
    )
    
    conversational_answer, exact_quote = await asyncio.gather(
        answer_chain.ainvoke({"context": context_text, "question": question}),
        quote_chain.ainvoke({"context": context_text, "question": question})
    )
    
    return {
        "answer": clean_text(conversational_answer),
        "quote_for_highlighting": None if "QUOTE_NOT_FOUND" in exact_quote else clean_text(exact_quote)
    }

def create_highlighted_pdf(original_filename: str, text_to_highlight: str, question_id: str = None) -> Optional[str]:
    """
    Create a new highlighted PDF for each question, starting from the clean processed PDF.
    
    Args:
        original_filename: Name of the original PDF file
        text_to_highlight: Text to be highlighted
        question_id: Unique identifier for the question (optional)
        
    Returns:
        Path to the new highlighted PDF or None if no matches found
    """
    processed_path = os.path.join(PROCESSED_DIRECTORY, original_filename)
    if not os.path.exists(processed_path):
        return None
        
    # Open the clean processed PDF (not previously highlighted)
    doc = fitz.open(processed_path)
    found = False
    search_text = clean_text(text_to_highlight.lower())
    
    # Highlight all matching instances
    for page in doc:
        text_instances = page.get_text("words")
        for inst in text_instances:
            word_text = clean_text(inst[4].lower())
            if (search_text in word_text or 
                fuzz.ratio(search_text, word_text) > 85 or
                fuzz.partial_ratio(search_text, word_text) > 90):
                
                rect = fitz.Rect(inst[:4])
                highlight = page.add_highlight_annot(rect)
                highlight.set_colors({"stroke": (1, 1, 0)})  # Yellow highlight
                highlight.update()
                found = True
    
    if not found:
        doc.close()
        return None

    # Generate unique filename for each highlight
    if question_id:
        highlighted_filename = f"highlighted_{question_id}_{original_filename}"
    else:
        highlighted_filename = f"highlighted_{int(time.time())}_{original_filename}"
    
    output_path = os.path.join(HIGHLIGHTED_OUTPUT_DIRECTORY, highlighted_filename)
    doc.save(output_path, garbage=4, deflate=True)
    doc.close()
    
    return output_path
