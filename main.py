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
import logging

# Google Generative AI imports
import google.generativeai as genai
from dotenv import load_dotenv

# RAG System Integration
from rag_interface import get_behavior_insights, format_insights_for_prompt

# Pet behavior knowledge comes from RAG system with research-backed insights

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        clinical_result = await get_clinical_analysis(
            observations=observation_result["observations"],
            pet_type=pet_type,
            search_terms=observation_result["search_terms"]
        )
        
        clinical_response = clinical_result["clinical_response"]
        research_insights = clinical_result["research_insights"]
        searchable_terms = clinical_result.get("searchable_terms", [])
        
        logger.info(f"✅ Analysis complete! Pet thoughts: {clinical_response[:50]}...")
        
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
            "stage2_clinical_analysis": clinical_response,
            "stage2_research_used": research_insights,
            "analysis_complete": True,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        # 🚨 Error handling with quota check
        print(f"❌ Error analyzing video: {e}")
        error_message = str(e)
        
        # Check for quota exceeded errors
        if any(keyword in error_message.lower() for keyword in ['quota', 'rate limit', '429', 'exhausted', 'exceeded']):
            return {
                "success": False,
                "error": "quota_exceeded",
                "stage1_observations": [],
                "stage1_description": "🚫 Daily request limit reached! Our free AI analysis quota has been used up for today. Please try again tomorrow when the quota resets.",
                "stage2_clinical_analysis": "The daily request limit has been reached. Please check back tomorrow to analyze your pet's video!",
                "stage2_research_used": [],
                "pet_type": pet_type,
                "analysis_complete": False
            }
        
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
    🎯 STAGE 1: Get detailed behavioral observations AND search terms from video
    
    OPTIMIZED: Single call to get both observations and search terms
    This reduces API calls from 3 to 2 total, saving time and cost.
    
    Args:
        video_file: Gemini uploaded video file
        pet_type: Type of pet being analyzed
        
    Returns:
        dict: Structured observations with search terms for RAG queries
    """
    logger.info("🔍 Stage 1: Analyzing video for behavioral observations and generating search terms...")
    
    observation_prompt = f"""
You are observing this {pet_type} video to document what you see AND generate search terms for research.

TASK: Provide a detailed description of what happens in the video AND extract search terms.

WHAT TO OBSERVE:
- Body posture and position
- Ear and tail position/movement  
- Head orientation and eye appearance
- Movement patterns and speed
- Any sounds or vocalizations
- How they interact with their environment
- Facial expressions and mouth position
- Actions such as retreating or jumping

RESPONSE FORMAT - Return valid JSON with these two keys:

{{
  "observations": "Write a detailed description of what you observe. Be specific about body language, movements, and positioning. Focus on observable facts, not interpretations. Example: The cat is sitting in a crouched position with ears flattened against its head. Its tail is low and twitching slightly. The cat moves slowly while sniffing various objects in the room, keeping its head low. Its eyes appear alert and focused on the environment.",
  "search_terms": ["term1", "term2", "term3", "term4", "term5"]
}}

For search_terms, generate 3-5 specific behavioral research queries for behaviors you actually observed:

PURPOSE: These terms will query a scientific database of animal behavioral research to find exact behavioral indicators and their meanings.

REQUIREMENTS:
- ONLY behaviors actually observed in the video
- Use precise behavioral terminology (2-4 words)
- Focus on specific body language, not general activities
- Think like a veterinary behaviorist writing research terms

GOOD examples: ["ears flattened back", "tail twitching rapidly", "crouched low posture", "pupils dilated", "mouth open panting", "nose licking", "head lowered", "ears forward alert", "retreating quickly"]

BAD examples: ["cat sniffing object", "cat on table", "soft vocalization", "cats interacting"] - these are too general

Be specific to exact body positions, ear/tail positions, facial expressions, and precise movements you observed.

CRITICAL: Return ONLY the raw JSON object. Do NOT wrap it in markdown code blocks or any other formatting. No ```json or ``` markers.
"""
    
    try:
        # Get both observations and search terms in one call
        response = model.generate_content([observation_prompt, video_file])
        response_text = response.text.strip()
        
        # Parse JSON response
        try:
            import json
            
            # Strip markdown code blocks if present
            clean_response = response_text.strip()
            
            # Handle various markdown wrapper formats
            if "```json" in clean_response:
                # Extract JSON from markdown code block
                start = clean_response.find("```json") + 7
                end = clean_response.find("```", start)
                if end != -1:
                    clean_response = clean_response[start:end].strip()
            elif clean_response.startswith("```") and clean_response.endswith("```"):
                # Generic code block wrapper
                clean_response = clean_response[3:-3].strip()
            
            parsed_response = json.loads(clean_response)
            observations = parsed_response.get("observations", "")
            search_terms = parsed_response.get("search_terms", [])
            
            # Validate we got both pieces
            if not observations or not search_terms:
                raise ValueError("Missing observations or search_terms in response")
            
            logger.info(f"✅ Generated {len(search_terms)} search terms: {search_terms}")
            
            return {
                "observations": observations,
                "description": observations,  # Keep for compatibility
                "search_terms": search_terms,
                "success": True
            }
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"⚠️ JSON parsing failed: {e}")
            logger.warning(f"Raw response: {response_text[:200]}...")
            
            # Fallback: treat as plain text and generate basic search terms
            fallback_terms = [f"{pet_type} behavior", "body language", "posture", "movement", "facial expression"]
            
            return {
                "observations": response_text,
                "description": response_text,
                "search_terms": fallback_terms,
                "success": True
            }
            
    except Exception as e:
        logger.error(f"❌ Error during video observation: {e}")
        error_message = str(e)
        
        # Check for quota exceeded errors
        if any(keyword in error_message.lower() for keyword in ['quota', 'rate limit', '429', 'exhausted', 'exceeded']):
            return {
                "observations": "🚫 Daily request limit reached! Our free AI analysis quota has been used up for today. Please try again tomorrow when the quota resets.",
                "description": "🚫 Daily request limit reached! Our free AI analysis quota has been used up for today. Please try again tomorrow when the quota resets.",
                "search_terms": [f"{pet_type} behavior", "body language"],
                "success": False
            }
        
        return {
            "observations": f"I had trouble analyzing the video, but I can see your {pet_type} looks healthy and active!",
            "description": f"Video analysis encountered an error for this {pet_type}.",
            "search_terms": [f"{pet_type} behavior", "body language"],
            "success": False
        }

async def get_clinical_analysis(observations: str, pet_type: str, search_terms: list) -> dict:
    """
    🎯 STAGE 2: Perform clinical behavioral analysis using pre-generated search terms
    
    OPTIMIZED: Uses search terms from Stage 1 instead of generating new ones
    This eliminates the second Gemini call, reducing total calls from 3 to 2.
    
    Args:
        observations: Detailed behavioral observations from Stage 1
        pet_type: Type of pet being analyzed
        search_terms: Pre-generated search terms from Stage 1
        
    Returns:
        dict: Clinical analysis with research insights
    """
    logger.info("🧠 Stage 2: Performing clinical analysis with research insights...")
    
    # Use the pre-generated search terms (no additional Gemini call needed!)
    logger.info(f"🔍 Using {len(search_terms)} pre-generated search terms: {search_terms}")
    
    try:
        # Query RAG system for each search term
        all_research_insights = []
        
        for query in search_terms:
            logger.info(f"🔍 Querying RAG for: '{query}'")
            insights = get_behavior_insights(query, pet_type=pet_type, top_k=3)
            all_research_insights.extend(insights)
    
        
        # Remove duplicates while preserving order
        unique_insights = []
        seen_behaviors = set()
        
        for insight in all_research_insights:
            behavior_key = f"{insight['behavior']}_{insight['pet_type']}"
            if behavior_key not in seen_behaviors:
                unique_insights.append(insight)
                seen_behaviors.add(behavior_key)
        
        # Filter out low-similarity results (< 0.30) for better quality
        filtered_insights = [
            insight for insight in unique_insights 
            if insight.get('similarity_score', 0) >= 0.30
        ]
        
        logger.info(f"📊 Total unique research insights found: {len(unique_insights)}")
        logger.info(f"🎯 High-quality insights (≥0.30 similarity): {len(filtered_insights)}")
        
        # Format insights for the prompt (use only top 3 for clinical analysis)
        top_insights_for_analysis = sorted(filtered_insights, key=lambda x: x.get('similarity_score', 0), reverse=True)[:3]
        research_context = format_insights_for_prompt(top_insights_for_analysis)
        
        # Simple terminal output for debugging
        print(f"\n🔍 Search Queries: {search_terms}")
        print(f"📚 All Research Insights Found ({len(filtered_insights)}) - shown in technical panel:")
        for i, insight in enumerate(filtered_insights, 1):
            print(f"   {i}. {insight['behavior']} → {insight['indicates']} (similarity: {insight.get('similarity_score', 0):.2f})")
        print(f"\n🎯 Top 3 Insights Used in Clinical Analysis:")
        for i, insight in enumerate(top_insights_for_analysis, 1):
            print(f"   {i}. {insight['behavior']} → {insight['indicates']} (similarity: {insight.get('similarity_score', 0):.2f})")
        print()
        
        # Create clinical analysis prompt
        clinical_prompt = f"""
You are a friendly pet behavior expert writing a casual, fun paragraph for a pet owner.

OBSERVATIONS FROM VIDEO:
{observations}

RESEARCH INSIGHTS:
{research_context}

TASK: Write a short, casual paragraph (max 130 words) explaining what the pet is thinking/feeling based on the observations and research. 
Base your interpretation a large amount on the research insights above. These are scientifically-backed behavioral indicators - trust them over general assumptions.

If research shows behaviors indicating stress/anxiety/tension → mention the pet might be feeling unsure or alert
If research shows behaviors indicating aggression → mention the pet might be feeling defensive 
If research shows behaviors indicating contentment → mention the pet seems relaxed

TONE: educational, and casual but also insightful - like you're a relaxed veterinarian to a friend about their pet
STYLE: One flowing paragraph, no bullet points or lists
AVOID: overly technical terms

Focus on translating the behaviors into what the pet might be "thinking" using the research insights.
"""
        
        logger.info("🩺 Generating clinical behavioral analysis...")
        
        # Final Gemini call for clinical analysis
        clinical_response = model.generate_content(clinical_prompt)
        clinical_text = clinical_response.text.strip()
        
        return {
            "clinical_response": clinical_text,
            "research_insights": filtered_insights,
            "searchable_terms": search_terms  # Return the pre-generated terms
        }
    except Exception as e:
        logger.error(f"❌ Error during clinical analysis: {e}")
        error_message = str(e)
        
        # Check for quota exceeded errors
        if any(keyword in error_message.lower() for keyword in ['quota', 'rate limit', '429', 'exhausted', 'exceeded']):
            return {
                "clinical_response": "🚫 Daily request limit reached! Our free AI analysis quota has been used up for today. Please try again tomorrow when the quota resets.",
                "research_insights": [],
                "searchable_terms": search_terms or []
            }
        
        return {
            "clinical_response": "I had trouble analyzing the research insights, but your pet looks happy and healthy!",
            "research_insights": [],
            "searchable_terms": search_terms or []
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