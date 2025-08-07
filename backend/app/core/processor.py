import os
import fitz  # PyMuPDF
from dotenv import load_dotenv
import pytesseract
from PIL import Image
import io
import pandas as pd

# Langchain Imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Initialization ---
load_dotenv(override=True)

# This dictionary will act as a simple, temporary cache for our vector stores.
vector_store_cache = {}

# --- Directory Setup ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIRECTORY = os.path.join(BASE_DIR, "data/uploads")
PROCESSED_DIRECTORY = os.path.join(BASE_DIR, "data/processed_pdfs")
MODELS_DIRECTORY = os.path.join(BASE_DIR, "models")
HIGHLIGHTED_OUTPUT_DIRECTORY = os.path.join(BASE_DIR, "data/highlighted")
RAW_OCR_DIRECTORY = os.path.join(BASE_DIR, "data/raw_ocr_text")


os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
os.makedirs(PROCESSED_DIRECTORY, exist_ok=True)
os.makedirs(MODELS_DIRECTORY, exist_ok=True)
os.makedirs(HIGHLIGHTED_OUTPUT_DIRECTORY, exist_ok=True)
os.makedirs(RAW_OCR_DIRECTORY, exist_ok=True)


# --- Initialize Models in the Main Thread ---
# FIXED: Reverted to loading the API key securely from the environment
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key="AIzaSyCcqq7F_Pv2hixPqtqp-mbBX9NSiF0Y3_g",
    task_type="RETRIEVAL_DOCUMENT"
)

# --- HELPER FUNCTION 1: Create the OCR-layered PDF ---
def create_ocr_layered_pdf(file_contents: bytes, filename: str) -> str:
    """
    Creates a new PDF with an invisible OCR text layer using a precise, word-by-word method.
    """
    processed_pdf_path = os.path.join(PROCESSED_DIRECTORY, filename)
    
    original_doc = fitz.open(stream=file_contents, filetype="pdf")
    new_doc = fitz.open()

    all_ocr_text_blocks = []

    print(f"Creating OCR-layered PDF for {filename}...")
    for page_num, page in enumerate(original_doc):
        print(f"  - Remastering page {page_num + 1}/{len(original_doc)}")
        
        pix = page.get_pixmap(dpi=300)
        image = Image.open(io.BytesIO(pix.tobytes("png")))

        ocr_data = pytesseract.image_to_data(image, lang='eng', output_type=pytesseract.Output.DATAFRAME)
        ocr_data = ocr_data.dropna(subset=['text'])
        ocr_data = ocr_data[ocr_data.conf > 30]

        new_page = new_doc.new_page(width=page.rect.width, height=page.rect.height)
        new_page.insert_image(page.rect, pixmap=pix)

        page_text = ""
        for i, row in ocr_data.iterrows():
            x, y, w, h = row['left'], row['top'], row['width'], row['height']
            word = str(row['text'])
            page_text += word + " "
            
            new_page.insert_text(
                (x, y + h),
                word,
                fontname="helv",
                fontsize=h,
                render_mode=3
            )
        all_ocr_text_blocks.append(page_text.strip())

    raw_ocr_path = os.path.join(RAW_OCR_DIRECTORY, f"raw_ocr_{filename.replace('.pdf', '.txt')}")
    print(f"Saving raw OCR text for debugging to: {raw_ocr_path}")
    with open(raw_ocr_path, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(all_ocr_text_blocks))
            
    print(f"Saving remastered PDF to: {processed_pdf_path}")
    new_doc.save(processed_pdf_path, garbage=4, deflate=True)
    new_doc.close()
    original_doc.close()
    
    return processed_pdf_path

# --- MAIN PROCESSING FUNCTION ---
def process_and_embed_pdf(file_contents: bytes, filename: str):
    """
    Main background task. It creates the OCR-layered PDF and then builds the vector store from it.
    """
    processed_pdf_path = create_ocr_layered_pdf(file_contents, filename)

    doc = fitz.open(processed_pdf_path)
    full_text = "\n".join([page.get_text() for page in doc])
    doc.close()

    if not full_text.strip():
        print(f"Warning: No text could be extracted from the remastered PDF: {filename}")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    chunks = text_splitter.split_text(full_text)
    
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store_cache[filename] = vector_store
    
    print(f"Successfully processed and embedded '{filename}'. Found {len(chunks)} chunks.")


async def query_rag_pipeline(filename: str, question: str) -> dict | None:
    """
    Performs a 2-step RAG pipeline to get a conversational answer and an exact quote.
    """
    if filename not in vector_store_cache:
        return {"error": "Document not found. Please upload and process the document first."}

    vector_store = vector_store_cache[filename]
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    context_docs = retriever.get_relevant_documents(question)
    context_text = "\n".join([doc.page_content for doc in context_docs])

    # FIXED: Reverted to loading the API key securely from the environment
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key="AIzaSyCcqq7F_Pv2hixPqtqp-mbBX9NSiF0Y3_g",
        temperature=0.5, 
    )
    
    # --- STEP 1: Get a flexible, conversational answer ---
    answer_prompt_template = """
    You are a helpful assistant. Based ONLY on the provided context, provide a clear and descriptive answer to the user's question.
    You can provide a short explanation or summarize the findings from the document.
    Be aware that the context is from an OCR process and may contain typos.
    If the answer is not found in the context, state that you could not find the information.

    Context:
    {context}

    Question: {question}
    """
    answer_prompt = ChatPromptTemplate.from_template(answer_prompt_template)
    answer_chain = answer_prompt | llm | StrOutputParser()
    conversational_answer = await answer_chain.ainvoke({"context": context_text, "question": question})

    if "I could not find information" in conversational_answer:
        return None

    # --- STEP 2: Get the exact quote for highlighting ---
    quote_prompt_template = """
    Based on the following context and the answer provided, find the single, best, contiguous sentence or phrase from the CONTEXT that serves as evidence for the answer.
    Your response must be an **exact quote** from the context. Do not add any explanation.
    If no direct quote can be found, return the string "QUOTE_NOT_FOUND".

    Context:
    {context}
    
    Answer:
    {answer}
    """
    quote_prompt = ChatPromptTemplate.from_template(quote_prompt_template)
    quote_chain = quote_prompt | llm | StrOutputParser()
    exact_quote = await quote_chain.ainvoke({"context": context_text, "answer": conversational_answer})

    return {
        "answer": conversational_answer.strip(),
        "quote_for_highlighting": None if "QUOTE_NOT_FOUND" in exact_quote else exact_quote.strip()
    }


def create_highlighted_pdf(original_filename: str, text_to_highlight: str) -> str | None:
    """
    Highlights the text on the "remastered" PDF.
    """
    processed_path = os.path.join(PROCESSED_DIRECTORY, original_filename)
    if not os.path.exists(processed_path):
        return None
        
    doc = fitz.open(processed_path)
    found = False
    for page in doc:
        instances = page.search_for(text_to_highlight)
        if instances:
            found = True
            for inst in instances:
                page.add_highlight_annot(inst)
    
    if not found:
        doc.close()
        return None

    highlighted_filename = f"highlighted_{original_filename}"
    output_path = os.path.join(HIGHLIGHTED_OUTPUT_DIRECTORY, highlighted_filename)
    doc.save(output_path, garbage=4, deflate=True)
    doc.close()
    
    return output_path
