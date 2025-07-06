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

# RAG System Integration
from rag_interface import get_behavior_insights, format_insights_for_prompt

# Pet behavior knowledge comes from RAG system with research-backed insights

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
    🎯 CORE FUNCTION: Two-Stage Pet Video Analysis using Gemini AI + RAG
    
    STAGE 1: Video Observation
    - Gemini watches video and describes specific behaviors observed
    - Creates detailed behavioral observation report
    
    STAGE 2: RAG-Enhanced Clinical Analysis  
    - Uses specific observations to query research database
    - Provides clinician-style behavioral interpretation
    
    Args:
        video_path (str): Path to the video file
        pet_type (str): Type of pet ("dog" or "cat")
    
    Returns:
        dict: Analysis results with observations, research context, and clinical interpretation
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
        
        # 🎯 STAGE 1: Video Observation
        print(f"🔍 Stage 1: Analyzing video for behavioral observations...")
        observation_result = await get_video_observations(video_file, pet_type)
        
        # 🎯 STAGE 2: RAG-Enhanced Clinical Analysis
        print(f"🧠 Stage 2: Performing clinical analysis with research insights...")
        clinical_analysis = await get_clinical_analysis(observation_result["observations"], pet_type)
        
        # 🧹 Step 3: Clean up - delete video from Gemini
        try:
            genai.delete_file(video_file.name)
            print(f"🗑️ Cleaned up video from Gemini")
        except Exception as cleanup_error:
            print(f"⚠️ Warning: Could not clean up video: {cleanup_error}")
        
        # 📊 Step 4: Return combined results
        return {
            "success": True,
            "pet_type": pet_type,
            "stage1_observations": observation_result["observations"],
            "stage1_description": observation_result["description"],
            "stage2_clinical_analysis": clinical_analysis["clinical_response"],
            "stage2_research_used": clinical_analysis["research_insights"],
            "analysis_complete": True,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        # 🚨 Error handling
        print(f"❌ Error analyzing video: {e}")
        return {
            "success": False,
            "error": str(e),
            "stage1_observations": [],
            "stage1_description": f"I'm having trouble analyzing this {pet_type} video right now. Please try again.",
            "stage2_clinical_analysis": "Unable to provide clinical analysis due to processing error.",
            "stage2_research_used": [],
            "pet_type": pet_type,
            "analysis_complete": False
        }

async def get_video_observations(video_file, pet_type: str) -> dict:
    """
    🔍 STAGE 1: Get detailed behavioral observations from video
    
    This function focuses purely on observation - what specific behaviors
    are visible in the video without interpretation.
    
    Args:
        video_file: Gemini uploaded video file
        pet_type: Type of pet for context
        
    Returns:
        dict: Detailed observations and video description
    """
    
    observation_prompt = f"""
You are a veterinary behaviorist conducting a systematic observation of this {pet_type} video.

OBSERVATION PROTOCOL:
Your task is to document SPECIFIC, OBSERVABLE behaviors without interpretation. Focus on factual descriptions of what you see.

SYSTEMATIC OBSERVATION CHECKLIST:

BODY LANGUAGE:
- Overall posture (standing, sitting, lying, crouched, etc.)
- Back position (straight, arched, lowered, etc.)
- Head position and orientation
- Limb positioning

FACIAL EXPRESSIONS:
- Ear position and movement (upright, forward, back, flattened, etc.)
- Eye appearance (wide, narrow, dilated pupils, half-closed, etc.)
- Mouth and jaw position (closed, open, panting, etc.)
- Facial muscle tension

TAIL BEHAVIOR:
- Tail position (high, low, tucked, straight out, etc.)
- Tail movement (still, wagging, twitching, thrashing, etc.)
- Speed and pattern of tail movement

MOVEMENT PATTERNS:
- Gait and walking style
- Speed of movement
- Direction changes
- Specific actions or behaviors

VOCALIZATIONS:
- Type of sounds made (if any)
- Frequency and intensity
- Context of vocalizations

ENVIRONMENTAL INTERACTIONS:
- How the pet interacts with surroundings
- Attention focus and gaze direction
- Responses to stimuli

RESPONSE FORMAT:
1. First, provide a brief video description (1-2 sentences)
2. Then list specific observations in this format:

OBSERVED BEHAVIORS:
- [Specific behavior 1]
- [Specific behavior 2]
- [Specific behavior 3]
... etc.

Example format:
OBSERVED BEHAVIORS:
- Ears are flattened against head
- Tail is low and tucked between legs
- Body is in crouched position close to ground
- Eyes appear wide with dilated pupils
- Moving slowly with hesitant steps

Be specific, factual, and thorough. Do not interpret what these behaviors mean - just describe what you observe.
"""
    
    # Send observation request to Gemini
    response = await asyncio.to_thread(
        model.generate_content,
        [video_file, observation_prompt]
    )
    
    observation_text = response.text.strip()
    
    # Parse the response to extract video description and specific behaviors
    lines = observation_text.split('\n')
    
    # Extract video description (usually first few lines before "OBSERVED BEHAVIORS:")
    description = ""
    observations = []
    
    found_behaviors_section = False
    for line in lines:
        line = line.strip()
        if "OBSERVED BEHAVIORS:" in line.upper():
            found_behaviors_section = True
            continue
        
        if not found_behaviors_section and line:
            # This is part of the video description
            description += line + " "
        elif found_behaviors_section and line.startswith('-'):
            # This is a specific behavior observation
            behavior = line.lstrip('- ').strip()
            if behavior:
                observations.append(behavior)
    
    # If we didn't find the structured format, use the whole response as description
    if not observations:
        description = observation_text
        # Try to extract bullet points anyway
        for line in lines:
            if line.strip().startswith('-'):
                behavior = line.lstrip('- ').strip()
                if behavior:
                    observations.append(behavior)
    
    return {
        "description": description.strip(),
        "observations": observations
    }

async def get_clinical_analysis(observations: list, pet_type: str) -> dict:
    """
    🧠 STAGE 2: Perform clinical analysis using RAG-enhanced research
    
    This function takes specific observations and uses them to:
    1. Query the research database for relevant behavioral insights
    2. Provide clinician-style interpretation and recommendations
    
    Args:
        observations: List of specific behaviors observed in the video
        pet_type: Type of pet
        
    Returns:
        dict: Clinical analysis and research insights used
    """
    
    # 🔍 Step 1: Query RAG system for each specific observation
    print(f"🔍 Querying research database for {len(observations)} specific behaviors...")
    
    all_research_insights = []
    
    # Query for each specific behavior
    for observation in observations:
        # Clean up the observation for better RAG querying
        clean_query = observation.lower().replace('the cat', '').replace('the dog', '').strip()
        
        # Query RAG system for this specific behavior
        insights = get_behavior_insights(
            query=f"{pet_type} {clean_query}",
            pet_type=pet_type,
            top_k=3  # Get top 3 research insights for each behavior
        )
        
        if insights:
            all_research_insights.extend(insights)
    
    # Also do a general query for overall context
    general_insights = get_behavior_insights(
        query=f"{pet_type} behavior analysis interpretation",
        pet_type=pet_type,
        top_k=5
    )
    
    all_research_insights.extend(general_insights)
    
    # Remove duplicates and limit total research insights
    unique_insights = []
    seen_behaviors = set()
    
    for insight in all_research_insights:
        behavior_key = insight['behavior'].lower()
        if behavior_key not in seen_behaviors:
            unique_insights.append(insight)
            seen_behaviors.add(behavior_key)
            
        # Limit to top 10 most relevant insights
        if len(unique_insights) >= 10:
            break
    
    # 🔬 Step 2: Format research context for clinical analysis
    research_context = format_insights_for_prompt(unique_insights)
    
    # 🩺 Step 3: Generate clinician-style analysis
    print(f"🩺 Generating clinical behavioral analysis...")
    
    clinical_prompt = f"""
You are a veterinary behaviorist providing a professional clinical assessment.

PATIENT INFORMATION:
- Species: {pet_type}
- Behavioral observations from video analysis

OBSERVED BEHAVIORS:
{chr(10).join(f"• {obs}" for obs in observations)}

RELEVANT RESEARCH CONTEXT:
{research_context}

CLINICAL ASSESSMENT PROTOCOL:
Provide a professional behavioral assessment in this format:

BEHAVIORAL ANALYSIS:
Based on my observation of your {pet_type}, I noticed several key behavioral indicators:

[For each significant behavior, provide clinical interpretation]
- "I observed [specific behavior]. This typically indicates [meaning based on research] [confidence level]."

EMOTIONAL STATE ASSESSMENT:
Your {pet_type} appears to be experiencing [emotional state] based on the combination of behaviors observed.

CLINICAL INTERPRETATION:
[Provide professional interpretation of what these behaviors mean for the pet's wellbeing]

RECOMMENDATIONS:
[If applicable, provide any care recommendations or notes about the pet's condition]

IMPORTANT NOTES:
- Use clinical language but keep it accessible to pet owners
- Reference specific behaviors you observed
- Base interpretations on the research context provided
- If behaviors indicate potential issues, mention them professionally
- Always include confidence levels (high/medium/low confidence)
- Be specific about which behaviors led to which conclusions

Provide your assessment now:
"""
    
    # Generate clinical analysis
    clinical_response = await asyncio.to_thread(
        model.generate_content,
        clinical_prompt
    )
    
    return {
        "clinical_response": clinical_response.text.strip(),
        "research_insights": unique_insights
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
            "pet_type": analysis_result["pet_type"],
            "stage1_observations": analysis_result["stage1_observations"],
            "stage1_description": analysis_result["stage1_description"],
            "stage2_clinical_analysis": analysis_result["stage2_clinical_analysis"],
            "stage2_research_used": analysis_result["stage2_research_used"],
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
        
        print(f"✅ Analysis complete! Pet thoughts: {analysis_result['stage2_clinical_analysis'][:50]}...")
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