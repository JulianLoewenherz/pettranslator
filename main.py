from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import shutil
from pathlib import Path
import subprocess
import tempfile

# Create FastAPI app instance
app = FastAPI(title="Pet Video Interpreter", description="Upload videos of your pets to discover what they're thinking!")

# Create necessary directories
Path("uploads").mkdir(exist_ok=True)
Path("static").mkdir(exist_ok=True)
Path("templates").mkdir(exist_ok=True)

# Mount static files (CSS, JS, images)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 templates for HTML rendering
templates = Jinja2Templates(directory="templates")

# Allowed video file extensions
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB max file size

def get_video_duration(file_path: str) -> float:
    """Get video duration using ffprobe (part of ffmpeg)"""
    try:
        result = subprocess.run([
            "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
            "-of", "csv=p=0", file_path
        ], capture_output=True, text=True)
        return float(result.stdout.strip())
    except Exception as e:
        print(f"Error getting video duration: {e}")
        return 0.0

def validate_video_file(file: UploadFile) -> None:
    """Validate uploaded video file"""
    # Check file extension
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type {file_extension} not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Check file size 
    if file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
        )

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main upload page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """Handle video file upload and validation"""
    
    # Validate the uploaded file
    validate_video_file(file)
    
    # Create a temporary file path
    file_extension = Path(file.filename).suffix.lower()
    temp_filename = f"temp_video_{file.filename}"
    temp_path = Path("uploads") / temp_filename
    
    try:
        # Save the uploaded file
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Check video duration
        duration = get_video_duration(str(temp_path))
        
        if duration > 10.0:  # 10 seconds limit
            # Clean up the file
            os.remove(temp_path)
            raise HTTPException(
                status_code=400,
                detail=f"Video too long ({duration:.1f} seconds). Maximum allowed: 10 seconds"
            )
        
        # For now, just return success info
        # In later stages, we'll process the video here
        return {
            "message": "Video uploaded successfully!",
            "filename": file.filename,
            "duration": f"{duration:.1f} seconds",
            "size": f"{file.size / (1024*1024):.1f}MB" if file.size else "Unknown",
            "status": "ready_for_processing"
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions (our validation errors)
        raise
    except Exception as e:
        # Clean up file if something went wrong
        if temp_path.exists():
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
    
    finally:
        # Clean up the temporary file (for now)
        # In later stages, we'll keep it for processing
        if temp_path.exists():
            os.remove(temp_path)

@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy", "message": "Pet Video Interpreter is running!"}

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 