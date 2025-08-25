import os
import fitz  # PyMuPDF
from thefuzz import fuzz
import asyncio
import re
import time
from typing import Optional, Dict

# Google Document AI Imports
from google.api_core.client_options import ClientOptions
from google.cloud import documentai

# Langchain Imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Initialization & Config ---
# No longer using dotenv; Render handles environment variables directly.
GOOGLE_PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID")
GOOGLE_PROCESSOR_ID = os.getenv("GOOGLE_PROCESSOR_ID")
GOOGLE_PROCESSOR_LOCATION = os.getenv("GOOGLE_PROCESSOR_LOCATION")

# --- Cache & Directory Setup ---
vector_store_cache: Dict[str, FAISS] = {}
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIRECTORY = os.path.join(BASE_DIR, "data/uploads")
PROCESSED_DIRECTORY = os.path.join(BASE_DIR, "data/processed_pdfs")
HIGHLIGHTED_OUTPUT_DIRECTORY = os.path.join(BASE_DIR, "data/highlighted")
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
os.makedirs(PROCESSED_DIRECTORY, exist_ok=True)
os.makedirs(HIGHLIGHTED_OUTPUT_DIRECTORY, exist_ok=True)

# --- Initialize Models ---
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    task_type="RETRIEVAL_DOCUMENT"
)

def clean_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

def _get_text_from_anchor(text_anchor, full_text: str) -> str:
    if not text_anchor.text_segments: return ""
    return "".join(full_text[segment.start_index:segment.end_index] for segment in text_anchor.text_segments)

def run_document_ai_ocr(file_contents: bytes):
    if not all([GOOGLE_PROJECT_ID, GOOGLE_PROCESSOR_ID, GOOGLE_PROCESSOR_LOCATION]):
        raise ValueError("Google Cloud credentials for Document AI are not set.")
    
    # Use ClientOptions to set the API endpoint based on location
    opts = ClientOptions(api_endpoint=f"{GOOGLE_PROCESSOR_LOCATION}-documentai.googleapis.com")
    
    # The client will automatically find the credentials from GOOGLE_APPLICATION_CREDENTIALS
    client = documentai.DocumentProcessorServiceClient(client_options=opts)
    
    name = client.processor_path(GOOGLE_PROJECT_ID, GOOGLE_PROCESSOR_LOCATION, GOOGLE_PROCESSOR_ID)
    raw_document = documentai.RawDocument(content=file_contents, mime_type="application/pdf")
    request = documentai.ProcessRequest(name=name, raw_document=raw_document)
    result = client.process_document(request=request)
    print("Google Document AI OCR processing complete.")
    return result.document

def reconstruct_searchable_pdf(file_contents: bytes, google_doc, output_path: str):
    original_pdf = fitz.open(stream=file_contents, filetype="pdf")
    new_pdf = fitz.open()
    for i, page in enumerate(google_doc.pages):
        original_page = original_pdf[i]
        new_page = new_pdf.new_page(width=original_page.rect.width, height=original_page.rect.height)
        pix = original_page.get_pixmap(dpi=200)
        new_page.insert_image(new_page.rect, pixmap=pix)
        tw = fitz.TextWriter(new_page.rect, opacity=0)
        for token in page.tokens:
            token_text = _get_text_from_anchor(token.layout.text_anchor, google_doc.text)
            vertices = token.layout.bounding_poly.normalized_vertices
            if not vertices: continue
            x0, y0 = vertices[0].x * new_page.rect.width, vertices[0].y * new_page.rect.height
            x1, y1 = vertices[2].x * new_page.rect.width, vertices[2].y * new_page.rect.height
            rect = fitz.Rect(x0, y0, x1, y1)
            fontsize = rect.height * 0.8
            if rect.width > 0:
                text_width = fitz.get_text_length(token_text, fontname="helv", fontsize=fontsize)
                if text_width > rect.width:
                    fontsize *= rect.width / text_width
            tw.fill_textbox(rect, token_text, font=fitz.Font("helv"), fontsize=fontsize)
        tw.write_text(new_page)
    new_pdf.save(output_path, garbage=4, deflate=True, clean=True)
    new_pdf.close()
    original_pdf.close()
    print(f"Successfully reconstructed searchable PDF at: {output_path}")

def process_and_embed_pdf(file_contents: bytes, filename: str):
    original_pdf_path = os.path.join(UPLOAD_DIRECTORY, filename)
    with open(original_pdf_path, "wb") as f:
        f.write(file_contents)
    try:
        google_doc = run_document_ai_ocr(file_contents)
        full_text = google_doc.text
        if not full_text.strip(): raise RuntimeError("Google Document AI returned empty text.")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        chunks = text_splitter.split_text(full_text)
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        vector_store_cache[filename] = vector_store
        print(f"Successfully built vector store for '{filename}'.")
        processed_pdf_path = os.path.join(PROCESSED_DIRECTORY, filename)
        reconstruct_searchable_pdf(file_contents, google_doc, processed_pdf_path)
    except Exception as e:
        print(f"Error during document processing: {e}")
        raise

async def query_rag_pipeline(filename: str, question: str) -> Optional[Dict[str, str]]:
    if filename not in vector_store_cache:
        return {"error": "Document not processed. Please upload the file again."}
    
    vector_store = vector_store_cache[filename]
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 7})
    context_docs = await retriever.ainvoke(question)
    context_text = "\n---\n".join([clean_text(doc.page_content) for doc in context_docs])

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
    
    prompt_template = """
    Based ONLY on the provided context, perform the following two steps:
    1. First, find the single, most relevant, and verbatim quote from the context that directly answers the question.
    2. Second, provide a concise, conversational answer to the question based on that quote.

    Use the following format, separating the quote and answer with '|||':
    QUOTE: [Your extracted quote here]
    |||
    ANSWER: [Your conversational answer here]

    If no relevant quote is found in the context, use this exact format:
    QUOTE: QUOTE_NOT_FOUND
    |||
    ANSWER: I could not find an answer to that question in the provided document.

    CONTEXT:
    {context}

    QUESTION:
    {question}
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm | StrOutputParser()
    
    response = await chain.ainvoke({"context": context_text, "question": question})
    
    try:
        quote_part, answer_part = response.split("|||")
        quote = quote_part.replace("QUOTE:", "").strip()
        answer = answer_part.replace("ANSWER:", "").strip()
        
        return {
            "answer": answer,
            "quote_for_highlighting": None if "QUOTE_NOT_FOUND" in quote else quote
        }
    except ValueError:
        return {"answer": response, "quote_for_highlighting": None}

def create_highlighted_pdf(original_filename: str, text_to_highlight: str) -> Optional[str]:
    processed_path = os.path.join(PROCESSED_DIRECTORY, original_filename)
    if not os.path.exists(processed_path): 
        print(f"Error: Processed file not found at {processed_path}")
        return None
        
    doc = fitz.open(processed_path)
    found = False
    search_text = clean_text(text_to_highlight)
    search_words = search_text.split()
    search_word_count = len(search_words)
    
    SIMILARITY_THRESHOLD = 75 

    best_overall_score = 0
    best_overall_match = None

    for page in doc:
        words_on_page = page.get_text("words")
        if not words_on_page:
            continue

        for window_size in range(max(1, search_word_count - 1), search_word_count + 2):
            if window_size > len(words_on_page): continue
            
            for i in range(len(words_on_page) - window_size + 1):
                window_words = words_on_page[i : i + window_size]
                window_phrase = " ".join([w[4] for w in window_words])
                
                score = fuzz.ratio(search_text.lower(), window_phrase.lower())
                
                if score > best_overall_score:
                    best_overall_score = score
                    best_overall_match = {"page": page, "words": window_words}

    if best_overall_score >= SIMILARITY_THRESHOLD:
        found = True
        page_to_highlight = best_overall_match["page"]
        words_to_highlight = best_overall_match["words"]
        
        for word_info in words_to_highlight:
            rect = fitz.Rect(word_info[:4])
            highlight = page_to_highlight.add_highlight_annot(rect)
            highlight.set_colors(stroke=(1, 0.6, 0.6))
            highlight.update()
        
        print(f"Found and highlighted best match with score: {best_overall_score}")

    if not found:
        print(f"Could not find a suitable match for '{search_text[:50]}...' to highlight.")
        doc.close()
        return None

    highlighted_filename = f"highlighted_{int(time.time())}_{original_filename}"
    output_path = os.path.join(HIGHLIGHTED_OUTPUT_DIRECTORY, highlighted_filename)
    doc.save(output_path, garbage=4, deflate=True, clean=True)
    doc.close()
    
    return output_path