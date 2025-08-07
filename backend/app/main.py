from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api import endpoints
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the FastAPI application
app = FastAPI(title="PDF Highlighter Backend")

# Configure CORS to allow the frontend to communicate with this backend.
origins = [
    "http://localhost:3001",
    "http://localhost:5173",
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the API endpoints from the endpoints.py file
app.include_router(endpoints.router)

@app.get("/", summary="Root endpoint")
async def read_root():
    """Confirms that the API server is running."""
    return {"message": "Welcome to the PDF Highlighter API!"}