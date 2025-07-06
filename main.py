from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import shutil
from pathlib import Path
import subprocess
import tempfile
import uuid
import asyncio
from datetime import datetime, timedelta
from typing import Optional

# Google Generative AI imports
import google.generativeai as genai
from dotenv import load_dotenv

# Import our pet behavior knowledge base
from pet_behaviors import get_behavior_context, get_pet_thoughts_examples

# Load environment variables
load_dotenv()

# Configure Google Generative AI
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please check your .env file.")

genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Gemini model (using Flash for better free tier quotas)
try:
    model = genai.GenerativeModel('gemini-1.5-flash')
    print("✅ Gemini Flash model initialized successfully!")
except Exception as e:
    print(f"❌ Error initializing Gemini model: {e}")
    model = None

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

async def analyze_pet_video(video_path: str, pet_type: str = "dog") -> dict:
    """
    🎯 CORE FUNCTION: Analyze pet video using Gemini AI
    
    This function:
    1. Uploads video to Gemini (supports video + audio)
    2. Creates smart prompt with pet behavior context
    3. Gets AI interpretation of pet's thoughts
    4. Returns formatted response
    
    Args:
        video_path (str): Path to the video file
        pet_type (str): Type of pet ("dog" or "cat")
    
    Returns:
        dict: Analysis results with pet thoughts and metadata
    """
    try:
        # 📤 Step 1: Upload video to Gemini
        print(f"🔄 Uploading video to Gemini AI...")
        video_file = genai.upload_file(video_path)
        print(f"✅ Video uploaded successfully: {video_file.name}")
        
        # ⏳ Step 2: Wait for file to become ACTIVE
        print(f"⏳ Waiting for video to be processed...")
        max_wait_time = 60  # Maximum wait time in seconds
        wait_interval = 2   # Check every 2 seconds
        total_waited = 0
        
        while total_waited < max_wait_time:
            # Check file state
            file_info = genai.get_file(video_file.name)
            print(f"📊 File state: {file_info.state.name}")
            
            if file_info.state.name == "ACTIVE":
                print(f"✅ Video is ready for analysis!")
                break
            elif file_info.state.name == "FAILED":
                raise Exception(f"Video processing failed: {file_info.error}")
            
            # Wait before checking again
            await asyncio.sleep(wait_interval)
            total_waited += wait_interval
        
        if total_waited >= max_wait_time:
            raise Exception("Video processing timed out. Please try with a smaller video.")
        
        # 📚 Step 3: Get pet behavior context from our knowledge base
        behavior_context = get_behavior_context(pet_type)
        thought_examples = get_pet_thoughts_examples(pet_type)
        
        # 🎭 Step 4: Create smart prompt for AI analysis
        prompt = f"""
You are a professional pet behavior expert and animal psychologist with years of experience.

ANALYZE THIS {pet_type.upper()} VIDEO:
- Watch the entire video carefully
- Pay attention to BOTH visual behavior AND audio (barks, meows, purrs, etc.)
- Consider body language, facial expressions, and movements
- Listen for vocalizations and their emotional tone

{behavior_context}

RESPONSE INSTRUCTIONS:
- Respond in FIRST PERSON as if you are the {pet_type}
- Keep response to 2-3 sentences maximum
- Make it warm, fun, and conversational
- Focus on what the {pet_type} is likely thinking/feeling
- Consider both what you SEE and what you HEAR

Example {pet_type} thoughts:
{chr(10).join(f"- {example}" for example in thought_examples[:3])}

Based on everything you observe in this video (visual + audio), what is this {pet_type} thinking?
"""
        
        # 🤖 Step 5: Send to AI for analysis
        print(f"🧠 Analyzing {pet_type} behavior with AI...")
        response = await asyncio.to_thread(
            model.generate_content,
            [video_file, prompt]
        )
        
        # 📝 Step 6: Extract and format the response
        pet_thoughts = response.text.strip()
        
        # 🧹 Step 7: Clean up - delete video from Gemini
        try:
            genai.delete_file(video_file.name)
            print(f"🗑️ Cleaned up video from Gemini")
        except Exception as cleanup_error:
            print(f"⚠️ Warning: Could not clean up video: {cleanup_error}")
        
        # 📊 Step 8: Return formatted results
        return {
            "success": True,
            "pet_thoughts": pet_thoughts,
            "pet_type": pet_type,
            "analysis_complete": True,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        # 🚨 Error handling
        print(f"❌ Error analyzing video: {e}")
        return {
            "success": False,
            "error": str(e),
            "pet_thoughts": f"Oops! I'm having trouble understanding what I'm thinking right now. Maybe try again?",
            "pet_type": pet_type,
            "analysis_complete": False
        }

def detect_pet_type(filename: str) -> str:
    """
    🐾 Simple pet type detection based on filename
    
    In the future, this could use AI to detect pet type from video,
    but for now, we'll use simple keyword detection or default to dog.
    """
    filename_lower = filename.lower()
    
    # Check for cat keywords
    cat_keywords = ['cat', 'kitten', 'feline', 'meow', 'purr']
    if any(keyword in filename_lower for keyword in cat_keywords):
        return "cat"
    
    # Check for dog keywords
    dog_keywords = ['dog', 'puppy', 'canine', 'bark', 'woof']
    if any(keyword in filename_lower for keyword in dog_keywords):
        return "dog"
    
    # Default to dog (most common)
    return "dog"

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main upload page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    🎬 MAIN ENDPOINT: Handle video upload and AI analysis
    
    Process:
    1. Validate uploaded video file
    2. Save temporarily for processing
    3. Analyze with AI to get pet thoughts
    4. Clean up temporary file
    5. Return AI interpretation
    """
    
    # 🔍 Step 1: Validate the uploaded file
    validate_video_file(file)
    
    # 📁 Step 2: Create unique temporary file path
    file_extension = Path(file.filename).suffix.lower()
    unique_id = str(uuid.uuid4())[:8]  # Short unique identifier
    temp_filename = f"temp_video_{unique_id}_{file.filename}"
    temp_path = Path("uploads") / temp_filename
    
    try:
        # 💾 Step 3: Save the uploaded file temporarily
        print(f"📥 Saving video: {file.filename}")
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # ⏱️ Step 4: Check video duration
        duration = get_video_duration(str(temp_path))
        
        if duration > 10.0:  # 10 seconds limit
            # Clean up the file
            os.remove(temp_path)
            raise HTTPException(
                status_code=400,
                detail=f"Video too long ({duration:.1f} seconds). Maximum allowed: 10 seconds"
            )
        
        # 🐾 Step 5: Detect pet type from filename
        pet_type = detect_pet_type(file.filename)
        print(f"🔍 Detected pet type: {pet_type}")
        
        # 🧠 Step 6: Analyze video with AI (THE MAGIC HAPPENS HERE!)
        print(f"🚀 Starting AI analysis...")
        analysis_result = await analyze_pet_video(str(temp_path), pet_type)
        
        # 📊 Step 7: Prepare response with video metadata + AI results
        response = {
            "success": analysis_result["success"],
            "pet_thoughts": analysis_result["pet_thoughts"],
            "pet_type": analysis_result["pet_type"],
            "video_info": {
                "filename": file.filename,
                "duration": f"{duration:.1f} seconds",
                "size": f"{file.size / (1024*1024):.1f}MB" if file.size else "Unknown",
            },
            "analysis_complete": analysis_result["analysis_complete"],
            "timestamp": analysis_result.get("timestamp")
        }
        
        # Add error info if analysis failed
        if not analysis_result["success"]:
            response["error"] = analysis_result.get("error", "Unknown error")
        
        print(f"✅ Analysis complete! Pet thoughts: {analysis_result['pet_thoughts'][:50]}...")
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions (our validation errors)
        raise
    except Exception as e:
        # Clean up file if something went wrong
        if temp_path.exists():
            os.remove(temp_path)
        print(f"❌ Error in upload endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
    
    finally:
        # 🧹 Step 8: Always clean up the temporary file
        if temp_path.exists():
            try:
                os.remove(temp_path)
                print(f"🗑️ Cleaned up temporary file: {temp_filename}")
            except Exception as cleanup_error:
                print(f"⚠️ Warning: Could not clean up file: {cleanup_error}")

@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy", "message": "Pet Video Interpreter is running!"}

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 