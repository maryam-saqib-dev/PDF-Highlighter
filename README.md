
# AI Document Highlighter

This project is an intelligent web application that allows users to upload scanned PDF documents, ask questions in natural language, and receive a highlighted version of the PDF showing the location of the answer.

---

## 🧠 Project Overview

The application is built with a modern tech stack:

- **Backend**: FastAPI (Python)  
- **Frontend**: React (Vite)  
- **OCR Engine**: `ocrmypdf` with the Tesseract engine  
- **AI/RAG Pipeline**: Langchain with Google Gemini and a FAISS vector store  

---

## 🧱 1. Prerequisites (Linux Only)

Before you begin, you must install the **Tesseract-OCR engine** on your system. This is a required dependency for the backend to perform Optical Character Recognition.

### 📦 For Ubuntu/Debian:

```bash
sudo apt update
sudo apt install tesseract-ocr
````

To verify the installation:

```bash
tesseract --version
```

---

## 🚀 2. Backend Setup (Linux)

### ✅ Check Python Version

This project requires **Python 3.10.12** or newer:

```bash
python3 --version
```

### 📁 Navigate to the Backend Directory

```bash
cd path/to/your/project/backend/
```

### 🐍 Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

### 🔐 Set Up Environment Variables

1. Create a file named `.env` in the backend directory.
2. Add your Google Gemini API key:

```
GEMINI_API_KEY="YOUR_API_KEY_HERE"
```

### ▶️ Run the Backend Server

```bash
uvicorn app.main:app --reload --port 8001
```

The backend will now be running at:
👉 [http://127.0.0.1:8001](http://127.0.0.1:8001)

---

## 🎨 3. Frontend Setup

### 📁 Navigate to the Frontend Directory

In a new terminal:

```bash
cd path/to/your/project/frontend/
```

### 📦 Install Dependencies

```bash
npm install
```

### ▶️ Run the Frontend Development Server

```bash
npm start
```

The frontend will now be running at:
👉 [http://localhost:3000](http://localhost:3000)

---

## 📚 4. How to Use the Application

### 1. **Open the Application**

Go to: [http://localhost:3000](http://localhost:3000)

---

### 2. **Upload a Document**

* Click **"Choose File"** and select a scanned PDF.
* Click **"Upload & Process"**.
* Wait for the progress bar to complete (OCR runs in the background).
* You’ll see a success message when it’s done.

---

### 3. **Ask a Question**

* Once processing finishes, the input field becomes active.
* Type your question about the document’s content.
* Press **Enter** or click **"Find & Highlight"**.

---

### 4. **View the Result**

* The system displays a **text answer** below the form.
* If it finds the source, the PDF viewer auto-updates to show the **highlighted version**.


