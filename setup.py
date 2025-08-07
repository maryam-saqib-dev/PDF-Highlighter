import os
from pathlib import Path

# This script generates a simplified directory and file structure for the PDF-Highlighter application.
# It creates only the essential folders and empty files to begin development.

def create_project_structure():
    """Creates a streamlined folder and file structure for the PDF-Highlighter project."""
    print("Creating the PDF-Highlighter application structure...")

    # --- Base Directory ---
    base_dir = Path("PDF-Highlighter")
    base_dir.mkdir(exist_ok=True)

    # --- Simplified Backend Structure ---
    print("Setting up backend directories and files...")
    backend_dirs = [
        "backend/app/api",
        "backend/data/uploads",
    ]
    for dir_path in backend_dirs:
        (base_dir / dir_path).mkdir(parents=True, exist_ok=True)

    backend_files = [
        "backend/app/__init__.py",
        "backend/app/main.py",
        "backend/app/api/__init__.py",
        "backend/app/api/endpoints.py",
        "backend/requirements.txt",
    ]
    for file_path in backend_files:
        (base_dir / file_path).touch()

    # --- Simplified Frontend Structure ---
    print("Setting up frontend directories and files...")
    frontend_dirs = [
        "frontend/public",
        "frontend/src/components",
        "frontend/src/styles",
    ]
    for dir_path in frontend_dirs:
        (base_dir / dir_path).mkdir(parents=True, exist_ok=True)

    frontend_files = [
        "frontend/public/index.html",
        "frontend/src/App.js",
        "frontend/src/index.js",
        "frontend/src/components/DocumentViewer.js",
        "frontend/src/styles/App.css",
        "frontend/package.json",
        "frontend/README.md",
    ]
    for file_path in frontend_files:
        (base_dir / file_path).touch()

    print("Project structure for 'PDF-Highlighter' created successfully!")

if __name__ == "__main__":
    create_project_structure()

