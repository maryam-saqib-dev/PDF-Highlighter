import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Document, Page, pdfjs } from 'react-pdf';
import 'react-pdf/dist/esm/Page/AnnotationLayer.css';
import 'react-pdf/dist/esm/Page/TextLayer.css';
import './styles/App.css'; // This line imports the CSS and makes the styles work

// --- PDF.js Worker Setup ---
pdfjs.GlobalWorkerOptions.workerSrc = `//cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjs.version}/pdf.worker.min.js`;

// --- Configuration ---
const API_URL = 'http://127.0.0.1:8001'; 

// --- Main App Component ---
function App() {
  // State for file handling
  const [file, setFile] = useState(null);
  const [fileUrl, setFileUrl] = useState('');
  const [processedFilename, setProcessedFilename] = useState('');
  const [numPages, setNumPages] = useState(null);

  // State for Q&A
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  
  // State for UI feedback
  const [isLoadingUpload, setIsLoadingUpload] = useState(false);
  const [isLoadingAsk, setIsLoadingAsk] = useState(false);
  const [status, setStatus] = useState({ message: '', type: 'info' });
  const [progress, setProgress] = useState(0); // State for progress bar
  const progressInterval = useRef(null); // Ref to hold the interval ID

  // Handler for file input changes
  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile && selectedFile.type === "application/pdf") {
      setFile(selectedFile);
      setFileUrl(URL.createObjectURL(selectedFile));
      setProcessedFilename('');
      setAnswer('');
      setProgress(0);
      setStatus({ message: `Selected file: ${selectedFile.name}`, type: 'info' });
    } else {
      setFile(null);
      setFileUrl('');
      setStatus({ message: 'Please select a valid PDF file.', type: 'error' });
    }
  };

  // Handler for uploading and processing the file
  const handleUpload = async () => {
    if (!file) {
      setStatus({ message: 'Please select a file first.', type: 'error' });
      return;
    }
    setIsLoadingUpload(true);
    setAnswer('');
    setStatus({ message: 'Uploading and processing...', type: 'info' });
    setProgress(0);

    // Simulate progress
    progressInterval.current = setInterval(() => {
      setProgress(prev => {
        if (prev >= 95) {
          clearInterval(progressInterval.current);
          return 95; // Stop just before 100
        }
        return prev + 5;
      });
    }, 1500); // Increment progress every 1.5 seconds

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${API_URL}/upload/`, formData);
      setProcessedFilename(response.data.filename);
      setStatus({ message: response.data.message, type: 'success' });
    } catch (err) {
      // FIXED: Added robust error message parsing to handle complex error objects
      let errorMessage = 'An unexpected upload error occurred.';
      if (err.response && err.response.data && err.response.data.detail) {
        const detail = err.response.data.detail;
        if (Array.isArray(detail)) {
          errorMessage = detail.map((d) => d.msg).join(', ');
        } else if (typeof detail === 'string') {
          errorMessage = detail;
        } else {
          errorMessage = JSON.stringify(detail);
        }
      }
      setStatus({ message: errorMessage, type: 'error' });
    } finally {
      clearInterval(progressInterval.current); // Ensure interval is cleared
      setProgress(100); // Finish progress
      setIsLoadingUpload(false);
    }
  };

  // Handler for asking a question
  const handleAsk = async () => {
    if (isLoadingAsk || !processedFilename || !question) return;

    setIsLoadingAsk(true);
    setAnswer('');
    setStatus({ message: 'Thinking...', type: 'info' });
    
    const formData = new FormData();
    formData.append('filename', processedFilename);
    formData.append('question', question);

    try {
      const response = await axios.post(`${API_URL}/ask/`, formData);
      const { answer, highlighted_pdf_url } = response.data;
      
      setAnswer(answer); 

      if (highlighted_pdf_url) {
        setFileUrl(API_URL + highlighted_pdf_url);
        setStatus({ message: 'Answer found and highlighted!', type: 'success' });
      } else {
        setStatus({ message: 'Answer found (highlighting not possible for this quote).', type: 'info' });
      }

    } catch (err) {
      const errorMessage = err.response?.data?.detail || 'An unexpected error occurred while asking.';
      setStatus({ message: errorMessage, type: 'error' });
    } finally {
      setIsLoadingAsk(false);
    }
  };

  return (
    <div className="container">
      <header className="header">
        <h1 className="title">AI Document Highlighter</h1>
        <p className="subtitle">Upload a PDF and ask Gemini questions about its content.</p>
      </header>

      <div className="main-content">
        
        <div className="controls-column">
          <div className="section">
            <h2>1. Upload Document</h2>
            <input type="file" accept=".pdf" onChange={handleFileChange} className="input" />
            
            {/* Progress Bar */}
            {isLoadingUpload && (
              <div className="progress-bar-container">
                <div 
                  className="progress-bar" 
                  style={{ width: `${progress}%` }}
                >
                  {progress > 0 && `${progress}%`}
                </div>
              </div>
            )}

            <button 
              type="button"
              onClick={handleUpload} 
              disabled={isLoadingUpload || !file} 
              className="button"
            >
              {isLoadingUpload ? 'Processing...' : 'Upload & Process'}
            </button>
          </div>

          <div className="section">
            <h2>2. Ask a Question</h2>
            <input
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="e.g., What is the main conclusion?"
              className="input"
              disabled={!processedFilename || isLoadingUpload}
              onKeyPress={(e) => { if (e.key === 'Enter') handleAsk(); }}
            />
            <button 
              type="button"
              onClick={handleAsk} 
              disabled={isLoadingAsk || !processedFilename || !question || isLoadingUpload} 
              className="button"
            >
              {isLoadingAsk ? 'Asking...' : 'Find & Highlight'}
            </button>
          </div>
          
          {status.message && (
            <div className={`status-message ${status.type}-message`}>
              {status.message}
            </div>
          )}

          {answer && (
            <div className="answer-section">
              <h3>Answer:</h3>
              <div className="answer-box">
                {answer}
              </div>
            </div>
          )}
        </div>

        <div className="section">
          <h2>Document Preview</h2>
          {fileUrl ? (
            <div className="document-viewer">
              <Document
                file={fileUrl}
                onLoadSuccess={({ numPages }) => setNumPages(numPages)}
                onLoadError={(error) => setStatus({ message: `Failed to load PDF preview: ${error.message}`, type: 'error' })}
              >
                {Array.from(new Array(numPages || 0), (el, index) => (
                  <Page 
                    key={`page_${index + 1}`} 
                    pageNumber={index + 1}
                  />
                ))}
              </Document>
            </div>
          ) : (
            <div className="status-message info-message">
              Your PDF will be displayed here after you select it.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
