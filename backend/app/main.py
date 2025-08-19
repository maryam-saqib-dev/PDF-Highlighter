from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from .api import endpoints
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Initialize the FastAPI application
app = FastAPI(title="PDF Highlighter Backend")

# --- CORS Configuration ---
# This is still useful for local development with `npm start`
origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Router ---
# We add a prefix to all API routes to distinguish them from frontend routes.
# All your endpoints (/upload, /ask, etc.) will now be under /api
app.include_router(endpoints.router, prefix="/api")


# --- Static Files Mount ---
# This is the crucial part. It tells FastAPI to serve the static files
# (HTML, CSS, JS) from your React app's 'build' directory.

# Construct the path to the frontend build directory
# Assumes the backend is in 'PDF-Highlighter/backend' and frontend is in 'PDF-Highlighter/frontend'
frontend_build_path = os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "build")

# Check if the directory exists before mounting
if os.path.exists(frontend_build_path):
    # The `html=True` argument ensures index.html is served for any path not
    # found, which is essential for single-page applications like React.
    app.mount("/", StaticFiles(directory=frontend_build_path, html=True), name="static")
else:
    print(f"Warning: Frontend build directory not found at '{frontend_build_path}'. Static file serving is disabled.")
    @app.get("/", summary="Root endpoint")
    async def read_root():
        """Confirms that the API server is running."""
        return {"message": "Welcome to the PDF Highlighter API! (Frontend not found)"}
