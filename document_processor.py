"""
🎯 Intelligent Document Processing Pipeline for Pet Behavior Research Papers

This module uses LLM (Gemini) for smart extraction instead of crude chunking:
1. PDF text extraction from research papers
2. LLM-based intelligent behavioral data extraction
3. Structured behavioral insights preparation
4. Quality filtering and organization

Tools used:
- PyPDF2: Extract text from PDF files
- Gemini: Intelligent behavioral data extraction
- JSON: Structure the extracted behavioral insights
"""

import os
import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import PyPDF2
from datetime import datetime
import asyncio
import base64
import io

# PDF image extraction
import fitz  # PyMuPDF

# Google Generative AI imports
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Google Generative AI
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please check your .env file.")

genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Gemini model
try:
    model = genai.GenerativeModel('gemini-1.5-flash')
    print("✅ Gemini Flash model initialized for intelligent extraction!")
except Exception as e:
    print(f"❌ Error initializing Gemini model: {e}")
    model = None

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntelligentPetBehaviorProcessor:
    """
    🧠 MAIN CLASS: Uses LLM for intelligent behavioral data extraction
    
    This class:
    1. Extracts text from PDF research papers
    2. Uses Gemini to intelligently extract behavioral insights
    3. Structures data for vector database storage
    4. Filters out irrelevant content automatically
    """
    
    def __init__(self):
        """Initialize the intelligent document processor"""
        if not model:
            raise ValueError("Gemini model not initialized. Check your API key.")
        
        self.model = model
        logger.info("🧠 Intelligent Pet Behavior Processor initialized with Gemini")
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        📖 Extract text from PDF file
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            str: Extracted text content
        """
        try:
            logger.info(f"📖 Extracting text from: {pdf_path}")
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Extract text from all pages
                text = ""
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        text += page_text + "\n"
                        logger.debug(f"   Page {page_num + 1}: {len(page_text)} characters")
                    except Exception as e:
                        logger.warning(f"   ⚠️ Error extracting page {page_num + 1}: {e}")
                        continue
                
                logger.info(f"✅ Extracted {len(text)} characters from {len(pdf_reader.pages)} pages")
                return text
                
        except Exception as e:
            logger.error(f"❌ Error extracting text from {pdf_path}: {e}")
            return ""
    
    def extract_images_from_pdf(self, pdf_path: str) -> List[bytes]:
        """
        🖼️ Extract images from PDF file
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List[bytes]: List of image bytes (JPEG/PNG format)
        """
        try:
            logger.info(f"🖼️ Extracting images from: {pdf_path}")
            
            pdf_document = fitz.open(pdf_path)
            images = []
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    # Get image data
                    xref = img[0]
                    pix = fitz.Pixmap(pdf_document, xref)
                    
                    # Skip small images (likely decorative)
                    if pix.width < 100 or pix.height < 100:
                        pix = None
                        continue
                    
                    # Convert to bytes
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        img_data = pix.tobytes("png")
                        images.append(img_data)
                        logger.debug(f"   📷 Extracted image {img_index + 1} from page {page_num + 1}")
                    
                    pix = None
            
            pdf_document.close()
            logger.info(f"✅ Extracted {len(images)} images from PDF")
            return images
            
        except Exception as e:
            logger.error(f"❌ Error extracting images from {pdf_path}: {e}")
            return []
    
    def create_extraction_prompt(self, text: str, filename: str) -> str:
        """
        🎯 Create intelligent extraction prompt for Gemini
        
        Args:
            text: Full document text
            filename: Original filename
            
        Returns:
            str: Optimized prompt for behavioral data extraction
        """
        
        prompt = f"""
You are extracting cat and dog behaviors from research papers for video analysis.

GOAL: Extract specific, observable behaviors that can be identified in videos and looked up later.

ANALYZE BOTH TEXT AND IMAGES:
- Extract behaviors from text descriptions
- Extract behaviors from photos, diagrams, charts, and illustrations
- Look for body language examples, posture guides, and behavioral demonstrations in images

REQUIRED JSON FORMAT:
{{
  "behaviors": [
    {{
      "behavior": "exact behavior name (e.g., 'dilated pupils', 'approaches humans with positive expressions')",
      "pet_type": "cat/dog/both", 
      "indicates": "what this behavior means with intensity (e.g., 'moderate fear or stress', 'high excitement')",
      "confidence": "high/medium/low - how confident you are in this extraction",
      "source": "text/image/both - where you found this behavior"
    }}
  ]
}}

EXTRACT BOTH PHYSICAL AND SOCIAL BEHAVIORS:

✅ **Physical behaviors**: "dilated pupils", "ears flattened", "tail puffed up", "crouched low", "arched back"
✅ **Vocalizations**: "loud meowing", "purring", "hissing", "chirping", "excessive barking"  
✅ **Body positions**: "hiding", "pacing", "trembling", "excessive grooming"
✅ **Social behaviors**: "approaches humans with positive expressions", "follows pointing gestures from smiling humans", "avoids humans with angry expressions", "seeks attention from humans"

EXAMPLES OF GOOD EXTRACTIONS:
- "dilated pupils" → "moderate to high fear or stress" → "high" → "text"
- "tail held high with curve" → "high confidence or happiness" → "high" → "text"
- "ears flattened against head" → "moderate to high fear or aggression" → "high" → "both"
- "excessive meowing" → "moderate attention seeking or distress" → "medium" → "text"
- "follows pointing gestures from smiling humans" → "high social responsiveness and trust" → "medium" → "text"
- "half-blinking" → "low to moderate fear" → "medium" → "image"
- "nose-licking" → "mild to moderate frustration" → "high" → "image"

SOURCE FIELD INSTRUCTIONS:
- "text" = behavior described in written text
- "image" = behavior shown in charts, diagrams, photos, or illustrations
- "both" = behavior mentioned in both text AND shown in images

IGNORE:
❌ Methodology, statistics, references
❌ Abstract concepts without observable behaviors
❌ Research procedures

DOCUMENT: {filename}

TEXT:
{text}

IMAGES: If images are included, analyze them for behavioral examples, body language demonstrations, posture guides, and emotional state illustrations.

Return ONLY the JSON object.
"""
        
        return prompt
    
    async def extract_behavioral_insights(self, text: str, filename: str, images: List[bytes] = None) -> Dict[str, Any]:
        """
        🧠 Use Gemini to intelligently extract behavioral insights from text and images
        
        Args:
            text: Full document text
            filename: Original filename
            images: Optional list of image bytes from the PDF
            
        Returns:
            Dict: Structured behavioral insights
        """
        try:
            logger.info(f"🧠 Extracting behavioral insights from {filename} using Gemini...")
            
            # Create extraction prompt
            prompt = self.create_extraction_prompt(text, filename)
            
            # Prepare content for Gemini
            content = [prompt]
            
            # Add images if available
            if images:
                logger.info(f"📷 Including {len(images)} images in analysis")
                for i, img_bytes in enumerate(images[:5]):  # Limit to 5 images to avoid token limits
                    # Convert image bytes to format Gemini can understand
                    content.append({
                        "mime_type": "image/png",
                        "data": img_bytes
                    })
            
            # Send to Gemini for analysis
            response = await asyncio.to_thread(
                self.model.generate_content,
                content
            )
            
            # Parse JSON response
            response_text = response.text.strip()
            
            # Clean up response (remove markdown formatting if present)
            if response_text.startswith("```json"):
                response_text = response_text[7:-3].strip()
            elif response_text.startswith("```"):
                response_text = response_text[3:-3].strip()
            
            # Parse JSON with better error handling
            try:
                behavioral_data = json.loads(response_text)
                logger.info(f"✅ Extracted {len(behavioral_data.get('behaviors', []))} behavioral indicators")
                return behavioral_data
            except json.JSONDecodeError as e:
                logger.error(f"❌ JSON parsing error: {e}")
                logger.error(f"Response length: {len(response_text)} characters")
                logger.error(f"Response text preview: {response_text[:1000]}...")
                logger.error(f"Response text end: ...{response_text[-500:]}")
                
                # Try to fix common JSON issues
                fixed_response = self._attempt_json_fix(response_text)
                if fixed_response:
                    try:
                        behavioral_data = json.loads(fixed_response)
                        logger.info(f"✅ JSON fixed! Extracted {len(behavioral_data.get('behaviors', []))} behavioral indicators")
                        return behavioral_data
                    except json.JSONDecodeError:
                        logger.error("❌ JSON fix attempt failed")
                
                return self._create_fallback_response(filename)
                
        except Exception as e:
            logger.error(f"❌ Error extracting behavioral insights: {e}")
            return self._create_fallback_response(filename)
    
    def _attempt_json_fix(self, response_text: str) -> Optional[str]:
        """Attempt to fix common JSON formatting issues"""
        try:
            # Remove any trailing commas before closing brackets/braces
            fixed_text = response_text
            
            # Fix incomplete strings at the end
            if fixed_text.endswith('..."'):
                # Find the last complete field and truncate there
                last_complete_field = fixed_text.rfind('",\n')
                if last_complete_field > -1:
                    fixed_text = fixed_text[:last_complete_field + 1]
            
            # Ensure proper closing of arrays and objects
            open_braces = fixed_text.count('{') - fixed_text.count('}')
            open_brackets = fixed_text.count('[') - fixed_text.count(']')
            
            # Add missing closing brackets and braces
            for _ in range(open_brackets):
                fixed_text += ']'
            for _ in range(open_braces):
                fixed_text += '}'
            
            # Test if the fix worked
            json.loads(fixed_text)
            logger.info("🔧 Successfully fixed JSON formatting")
            return fixed_text
            
        except Exception as e:
            logger.error(f"❌ JSON fix attempt failed: {e}")
            return None
    
    def _extract_relevant_sections(self, text: str) -> str:
        """Extract most relevant sections from large documents"""
        # Split text into sections
        sections = text.split('\n\n')
        relevant_sections = []
        
        # Keywords that indicate relevant behavioral content
        behavioral_keywords = [
            'behavio', 'emotion', 'stress', 'fear', 'anxiety', 'tail', 'ear', 'eye', 'pupil',
            'vocal', 'meow', 'bark', 'purr', 'hiss', 'body language', 'posture', 'movement',
            'expression', 'indicator', 'sign', 'signal', 'communication', 'welfare',
            'hunger', 'thirst', 'comfort', 'need', 'seeking', 'elimination', 'social'
        ]
        
        # Score each section based on relevance
        scored_sections = []
        for section in sections:
            section_lower = section.lower()
            score = sum(1 for keyword in behavioral_keywords if keyword in section_lower)
            
            # Bonus for sections with actual behavioral descriptions
            if any(word in section_lower for word in ['indicates', 'shows', 'displays', 'demonstrates']):
                score += 2
            
            # Penalty for methodology/statistical sections
            if any(word in section_lower for word in ['methodology', 'statistical', 'references', 'citation']):
                score -= 1
            
            scored_sections.append((score, section))
        
        # Sort by score and take top sections
        scored_sections.sort(key=lambda x: x[0], reverse=True)
        
        # Take top sections until we reach a reasonable length
        selected_text = ""
        for score, section in scored_sections:
            if len(selected_text) + len(section) > 25000:  # Stay under limit
                break
            if score > 0:  # Only include sections with positive relevance
                selected_text += section + "\n\n"
        
        # If we didn't get enough content, add some high-scoring sections anyway
        if len(selected_text) < 5000:
            for score, section in scored_sections[:10]:  # Take top 10 regardless
                selected_text += section + "\n\n"
                if len(selected_text) > 20000:
                    break
        
        return selected_text.strip()
    
    def _create_fallback_response(self, filename: str) -> Dict[str, Any]:
        """Create fallback response when extraction fails"""
        return {
            "behaviors": [],
            "extraction_error": True,
            "processed_date": datetime.now().isoformat()
        }
    
    def structure_for_vector_db(self, behavioral_data: Dict[str, Any], filename: str = "unknown") -> List[Dict[str, Any]]:
        """
        📊 Structure behavioral data for vector database storage
        
        Args:
            behavioral_data: Extracted behavioral insights
            filename: Name of the source PDF file
            
        Returns:
            List[Dict]: Structured data ready for embedding
        """
        structured_data = []
        
        behaviors = behavioral_data.get("behaviors", [])
        
        for i, behavior in enumerate(behaviors):
            # Create a comprehensive text representation for embedding
            behavior_text = self._create_behavior_text(behavior)
            
            structured_item = {
                "id": f"behavior_{i:03d}",
                "text": behavior_text,
                "metadata": {
                    "behavior": behavior.get("behavior", "unknown"),
                    "pet_type": behavior.get("pet_type", "unknown"),
                    "indicates": behavior.get("indicates", ""),
                    "confidence": behavior.get("confidence", "unknown"),
                    "source": behavior.get("source", "unknown"),
                    "source_document": filename,
                    "processed_date": datetime.now().isoformat()
                }
            }
            
            structured_data.append(structured_item)
        
        logger.info(f"📊 Structured {len(structured_data)} behavioral indicators for vector storage")
        return structured_data
    
    def _create_behavior_text(self, behavior: Dict[str, Any]) -> str:
        """Create comprehensive text representation of behavioral indicator"""
        behavior_name = behavior.get("behavior", "")
        pet_type = behavior.get("pet_type", "")
        indicates = behavior.get("indicates", "")
        confidence = behavior.get("confidence", "")
        source = behavior.get("source", "")
        
        # Create comprehensive text for embedding
        text_parts = []
        
        if behavior_name:
            text_parts.append(f"Behavior: {behavior_name}")
        
        if pet_type:
            text_parts.append(f"Pet: {pet_type}")
        
        if indicates:
            text_parts.append(f"Indicates: {indicates}")
        
        if confidence:
            text_parts.append(f"Confidence: {confidence}")
        
        if source:
            text_parts.append(f"Source: {source}")
        
        return " | ".join(text_parts)
    
    async def process_single_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        🎯 MAIN METHOD: Process a single PDF with intelligent extraction
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List[Dict]: Processed behavioral data ready for embedding
        """
        logger.info(f"🚀 Processing PDF with intelligent extraction: {pdf_path}")
        
        # Step 1: Extract text
        text = self.extract_text_from_pdf(pdf_path)
        if not text.strip():
            logger.warning(f"⚠️ No text extracted from {pdf_path}")
            return []
        
        # Step 2: Extract images from PDF
        images = self.extract_images_from_pdf(pdf_path)
        
        # Step 3: Extract behavioral insights using Gemini
        filename = Path(pdf_path).stem
        
        # Handle large documents by processing in chunks if needed
        if len(text) > 30000:  # If document is very large
            logger.info(f"📄 Large document detected ({len(text)} chars). Processing in optimized mode...")
            # For large documents, we can either:
            # 1. Truncate to most relevant sections
            # 2. Process in chunks and combine results
            # For now, let's truncate to keep the most relevant content
            text = self._extract_relevant_sections(text)
            logger.info(f"📄 Optimized text length: {len(text)} characters")
        
        behavioral_data = await self.extract_behavioral_insights(text, filename, images)
        
        # Step 4: Structure for vector database
        structured_data = self.structure_for_vector_db(behavioral_data, filename)
        
        logger.info(f"✅ Successfully processed {pdf_path} → {len(structured_data)} behavioral indicators")
        return structured_data
    
    async def process_directory(self, directory_path: str, output_file: str = "behavioral_insights.json") -> List[Dict[str, Any]]:
        """
        📁 Process all PDF files in a directory with intelligent extraction
        
        Args:
            directory_path: Path to directory containing PDFs
            output_file: Optional output file to save results
            
        Returns:
            List[Dict]: All processed behavioral data
        """
        directory = Path(directory_path)
        if not directory.exists():
            logger.error(f"❌ Directory not found: {directory_path}")
            return []
        
        # Find all PDF files
        pdf_files = list(directory.glob("**/*.pdf"))  # Recursive search
        logger.info(f"📁 Found {len(pdf_files)} PDF files in {directory_path}")
        
        if not pdf_files:
            logger.warning(f"⚠️ No PDF files found in {directory_path}")
            return []
        
        # Process each PDF
        all_behavioral_data = []
        for pdf_file in pdf_files:
            try:
                behavioral_data = await self.process_single_pdf(str(pdf_file))
                all_behavioral_data.extend(behavioral_data)
            except Exception as e:
                logger.error(f"❌ Error processing {pdf_file}: {e}")
                continue
        
        # Save results
        if output_file and all_behavioral_data:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(all_behavioral_data, f, indent=2, ensure_ascii=False)
                logger.info(f"💾 Saved {len(all_behavioral_data)} behavioral indicators to {output_file}")
            except Exception as e:
                logger.error(f"❌ Error saving to {output_file}: {e}")
        
        logger.info(f"🎉 Intelligent processing complete! {len(all_behavioral_data)} behavioral indicators from {len(pdf_files)} PDFs")
        return all_behavioral_data

# 🧪 Test function
async def test_intelligent_extraction():
    """Test the intelligent extraction with sample text"""
    if not model:
        print("❌ Gemini model not available for testing")
        return
    
    processor = IntelligentPetBehaviorProcessor()
    
    # Sample research text
    sample_text = """
    ABSTRACT
    This study examines feline stress indicators in domestic cats. We analyzed 150 cats across various environments to identify reliable behavioral markers of stress and anxiety.

    INTRODUCTION
    Cats exhibit complex behavioral patterns that indicate their emotional states. Understanding these behaviors is crucial for pet welfare and human-cat relationships.

    METHODS
    We observed cats in controlled environments using video analysis and physiological measurements including cortisol levels.

    RESULTS
    Key findings include:
    
    Dilated pupils: When pupils dilate beyond normal light response (>6mm diameter), this indicates sympathetic nervous system activation associated with stress, fear, or excitement. This was observed in 89% of stressed cats (p<0.001).
    
    Ear position: Ears flattened against the head (airplane ears) indicate defensive fear or anxiety. This position was observed in 94% of cats experiencing stress stimuli.
    
    Tail positioning: A puffed tail (piloerection) indicates defensive arousal or fear. Tail tucked under body indicates submission or fear. High tail position with slight curve indicates confidence and comfort.
    
    Vocalization patterns: Excessive meowing, especially high-pitched calls, indicates distress or attention-seeking. Purring can indicate contentment but also self-soothing during stress.
    
    Body posture: Crouched low posture with tucked paws indicates fear or defensive positioning. Relaxed lying on side indicates trust and comfort.

    DISCUSSION
    These behavioral indicators provide reliable methods for assessing feline emotional states. The combination of multiple indicators increases accuracy of emotional assessment.

    CONCLUSION
    Understanding these behavioral patterns helps improve cat welfare and human-cat relationships.
    """
    
    print("🧪 Testing intelligent extraction with sample text...")
    
    # Test extraction
    behavioral_data = await processor.extract_behavioral_insights(sample_text, "sample_cat_behavior_study")
    
    # Display results
    print("\n📊 Extraction Results:")
    print("=" * 50)
    print(f"Behavioral indicators found: {len(behavioral_data.get('behaviors', []))}")
    
    # Show first few indicators
    behaviors = behavioral_data.get('behaviors', [])[:3]
    for i, behavior in enumerate(behaviors):
        print(f"\n🎯 Behavior {i+1}:")
        print(f"  Behavior: {behavior.get('behavior', 'N/A')}")
        print(f"  Pet type: {behavior.get('pet_type', 'N/A')}")
        print(f"  Indicates: {behavior.get('indicates', 'N/A')}")
        print(f"  Confidence: {behavior.get('confidence', 'N/A')}")
        print(f"  Source: {behavior.get('source', 'N/A')}")
    
    # Test structuring for vector DB
    structured = processor.structure_for_vector_db(behavioral_data, "sample_cat_behavior_study")
    print(f"\n📈 Structured for vector DB: {len(structured)} items")
    
    if structured:
        print(f"\n📝 Example structured item:")
        example = structured[0]
        print(f"  ID: {example['id']}")
        print(f"  Text: {example['text'][:100]}...")
        print(f"  Metadata keys: {list(example['metadata'].keys())}")

if __name__ == "__main__":
    asyncio.run(test_intelligent_extraction()) 