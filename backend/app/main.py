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
# MODIFIED: Changed origins to a wildcard to allow the live frontend to connect.
# For a production application with stricter security, you would replace "*"
# with the specific URL of your deployed frontend.
origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Router ---
# The /api prefix is correctly configured here.
app.include_router(endpoints.router, prefix="/api")


# --- Static Files Mount ---
# This configuration is designed to serve the frontend and backend from the same server.
# This will work on Render if you deploy as a single Web Service.
frontend_build_path = os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "build")

if os.path.exists(frontend_build_path):
    app.mount("/", StaticFiles(directory=frontend_build_path, html=True), name="static")
else:
    print(f"Warning: Frontend build directory not found at '{frontend_build_path}'. Static file serving is disabled.")
    @app.get("/", summary="Root endpoint")
    async def read_root():
        """Confirms that the API server is running."""
        return {"message": "Welcome to the PDF Highlighter API! (Frontend not found)"}
