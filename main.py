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

def generate_pet_analysis_prompt(pet_type: str = "pet") -> str:
    """
    Generate a comprehensive prompt for pet behavior analysis
    
    Args:
        pet_type: Type of pet ("dog", "cat", or "pet" for unknown)
    
    Returns:
        Formatted prompt string for Gemini AI
    """
    # Get behavior context from our knowledge base
    behavior_context = get_behavior_context(pet_type)
    
    # Get example thoughts for this pet type
    thought_examples = get_pet_thoughts_examples(pet_type)
    example_thoughts = "\n".join([f"- {thought}" for thought in thought_examples[:3]])
    
    # Create comprehensive prompt
    prompt = f"""You are a professional pet behavior expert and animal psychologist. 

Analyze this {pet_type} video carefully, paying attention to BOTH visual and audio cues.

{behavior_context}

ANALYSIS INSTRUCTIONS:
1. Watch the entire video from start to finish
2. Listen to any sounds the pet makes (barking, meowing, purring, etc.)
3. Observe body language, facial expressions, and movement patterns
4. Note how the pet's behavior changes throughout the video
5. Consider the context and environment shown

RESPONSE FORMAT:
- Respond in first person as if you ARE the pet
- Keep it warm, conversational, and engaging
- Focus on positive interpretations when possible
- Be specific about what you observe
- Limit response to 2-3 sentences maximum
- Make it fun and relatable for pet owners

EXAMPLE THOUGHTS:
{example_thoughts}

Based on what you see AND hear in this video, what is this {pet_type} thinking or feeling?"""

    return prompt

async def analyze_pet_video(video_path: str, pet_type: str = "pet") -> dict:
    """
    Analyze a pet video using Gemini Pro Vision
    
    Args:
        video_path: Path to the video file
        pet_type: Type of pet ("dog", "cat", or "pet")
    
    Returns:
        Dictionary with analysis results or error information
    """
    if not model:
        return {
            "success": False,
            "error": "AI model not initialized",
            "pet_thoughts": "I'm sorry, but I can't analyze videos right now. Please try again later!"
        }
    
    try:
        # Upload video file to Gemini
        print(f"🎬 Uploading video to Gemini for analysis...")
        video_file = genai.upload_file(video_path)
        
        # Wait for file to be processed
        print(f"⏳ Waiting for video processing...")
        while video_file.state.name == "PROCESSING":
            await asyncio.sleep(1)
            video_file = genai.get_file(video_file.name)
        
        # Check if processing was successful
        if video_file.state.name == "FAILED":
            return {
                "success": False,
                "error": "Video processing failed",
                "pet_thoughts": "I had trouble analyzing your video. Please try with a different video!"
            }
        
        # Generate analysis prompt
        prompt = generate_pet_analysis_prompt(pet_type)
        
        # Send video and prompt to Gemini for analysis
        print(f"🧠 Analyzing pet behavior with AI...")
        response = await asyncio.to_thread(
            model.generate_content,
            [video_file, prompt]
        )
        
        # Clean up the uploaded file from Gemini
        try:
            genai.delete_file(video_file.name)
        except:
            pass  # Don't fail if cleanup doesn't work
        
        # Extract and format the response
        if response and response.text:
            pet_thoughts = response.text.strip()
            
            # Basic response validation
            if len(pet_thoughts) < 10:
                pet_thoughts = "I'm feeling great right now! Thanks for sharing this moment with me!"
            
            return {
                "success": True,
                "pet_thoughts": pet_thoughts,
                "pet_type": pet_type,
                "analysis_model": "gemini-1.5-flash"
            }
        else:
            return {
                "success": False,
                "error": "No response from AI",
                "pet_thoughts": "I'm feeling a bit shy and don't have much to say right now!"
            }
            
    except Exception as e:
        print(f"❌ Error in video analysis: {e}")
        return {
            "success": False,
            "error": str(e),
            "pet_thoughts": "I'm having trouble expressing my thoughts right now. Please try again!"
        }

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main upload page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """Handle video file upload, validation, and AI analysis"""
    
    # Validate the uploaded file
    validate_video_file(file)
    
    # Create a unique temporary file path
    file_extension = Path(file.filename).suffix.lower()
    unique_id = str(uuid.uuid4())[:8]
    temp_filename = f"pet_video_{unique_id}{file_extension}"
    temp_path = Path("uploads") / temp_filename
    
    try:
        # Save the uploaded file temporarily
        print(f"📁 Saving video: {temp_filename}")
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
        
        # Detect pet type from filename (simple heuristic)
        pet_type = "pet"  # Default
        filename_lower = file.filename.lower()
        if "dog" in filename_lower or "puppy" in filename_lower:
            pet_type = "dog"
        elif "cat" in filename_lower or "kitten" in filename_lower:
            pet_type = "cat"
        
        print(f"🐾 Detected pet type: {pet_type}")
        
        # Analyze the video with AI
        print(f"🤖 Starting AI analysis...")
        analysis_result = await analyze_pet_video(str(temp_path), pet_type)
        
        # Return the AI analysis result
        return {
            "message": "Video analyzed successfully!",
            "filename": file.filename,
            "duration": f"{duration:.1f} seconds",
            "size": f"{file.size / (1024*1024):.1f}MB" if file.size else "Unknown",
            "pet_type": pet_type,
            "analysis": analysis_result,
            "pet_thoughts": analysis_result.get("pet_thoughts", "I'm feeling great!"),
            "success": analysis_result.get("success", False)
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
        # Always clean up the temporary file after processing
        if temp_path.exists():
            try:
                os.remove(temp_path)
                print(f"🗑️ Cleaned up temporary file: {temp_filename}")
            except:
                pass  # Don't fail if cleanup doesn't work

@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy", "message": "Pet Video Interpreter is running!"}

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 