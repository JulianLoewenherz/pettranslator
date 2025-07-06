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
You are a veterinary behaviorist analyzing research papers to extract ONLY actionable behavioral insights for pet emotion recognition or understanding pet needs.

DOCUMENT TO ANALYZE: {filename}

EXTRACTION TASK:
Extract specific behavioral indicators that help identify pet emotions or needs. Focus ONLY on observable behaviors and what the paper tells us they mean. 

REQUIRED OUTPUT FORMAT (JSON):
{{
  "document_info": {{
    "filename": "{filename}",
    "pet_types": ["cat", "dog", "both"],
    "research_quality": "high/medium/low",
    "publication_type": "peer-reviewed/study/guide/other"
  }},
  "behavioral_indicators": [
    {{
      "behavior": "exact behavior name",
      "pet_type": "cat/dog/both",
      "body_part": "ears/tail/eyes/mouth/body/vocal",
      "emotional_states": ["primary emotion/need", "secondary emotion/need"],
      "confidence_level": "high/medium/low",
      "observable_signs": ["specific sign 1", "specific sign 2"],
      "context_factors": ["when this behavior occurs"],
      "scientific_evidence": "brief evidence summary",
      "differentiation": "how to distinguish from similar behaviors"
    }}
  ]
}}

WHAT TO EXTRACT:
✅ Specific body language indicators (tail position, ear position, eye expressions)
✅ Vocal patterns and their meanings
✅ Behavioral sequences that indicate emotions or needs
✅ Context-dependent behavioral interpretations
✅ Scientific evidence for behavioral meanings
✅ Differentiation between similar behaviors
✅ Need-related behaviors (food-seeking, water-seeking, elimination, comfort)

WHAT TO IGNORE:
❌ Methodology sections
❌ Statistical analyses
❌ Author information
❌ References and citations
❌ General background information
❌ Irrelevant medical information

EMOTIONAL STATES AND NEEDS TO FOCUS ON:
- Stress/Anxiety
- Fear/Defensive
- Happiness/Content
- Excitement/Arousal
- Aggression/Territorial
- Playfulness
- Relaxation/Calm
- Confusion/Uncertainty
- Hunger/Food-seeking
- Thirst/Water-seeking
- Bathroom/Elimination needs
- Comfort/Temperature needs
- Social/Attention needs
- Exercise/Energy needs

QUALITY CRITERIA:
- Only extract behaviors with clear emotional correlations
- Prioritize peer-reviewed findings
- Include confidence levels based on evidence strength
- Focus on observable, measurable behaviors

DOCUMENT TEXT:
{text}

IMPORTANT: Return ONLY the JSON object. No additional text or explanations.
"""
        
        return prompt
    
    async def extract_behavioral_insights(self, text: str, filename: str) -> Dict[str, Any]:
        """
        🧠 Use Gemini to intelligently extract behavioral insights
        
        Args:
            text: Full document text
            filename: Original filename
            
        Returns:
            Dict: Structured behavioral insights
        """
        try:
            logger.info(f"🧠 Extracting behavioral insights from {filename} using Gemini...")
            
            # Create extraction prompt
            prompt = self.create_extraction_prompt(text, filename)
            
            # Send to Gemini for analysis
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt
            )
            
            # Parse JSON response
            response_text = response.text.strip()
            
            # Clean up response (remove markdown formatting if present)
            if response_text.startswith("```json"):
                response_text = response_text[7:-3].strip()
            elif response_text.startswith("```"):
                response_text = response_text[3:-3].strip()
            
            # Parse JSON
            try:
                behavioral_data = json.loads(response_text)
                logger.info(f"✅ Extracted {len(behavioral_data.get('behavioral_indicators', []))} behavioral indicators")
                return behavioral_data
            except json.JSONDecodeError as e:
                logger.error(f"❌ JSON parsing error: {e}")
                logger.error(f"Response text: {response_text[:500]}...")
                return self._create_fallback_response(filename)
                
        except Exception as e:
            logger.error(f"❌ Error extracting behavioral insights: {e}")
            return self._create_fallback_response(filename)
    
    def _create_fallback_response(self, filename: str) -> Dict[str, Any]:
        """Create fallback response when extraction fails"""
        return {
            "document_info": {
                "filename": filename,
                "pet_types": ["unknown"],
                "research_quality": "unknown",
                "publication_type": "unknown"
            },
            "behavioral_indicators": [],
            "extraction_error": True,
            "processed_date": datetime.now().isoformat()
        }
    
    def structure_for_vector_db(self, behavioral_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        📊 Structure behavioral data for vector database storage
        
        Args:
            behavioral_data: Extracted behavioral insights
            
        Returns:
            List[Dict]: Structured data ready for embedding
        """
        structured_data = []
        
        doc_info = behavioral_data.get("document_info", {})
        indicators = behavioral_data.get("behavioral_indicators", [])
        
        for i, indicator in enumerate(indicators):
            # Create a comprehensive text representation for embedding
            behavior_text = self._create_behavior_text(indicator)
            
            structured_item = {
                "id": f"{doc_info.get('filename', 'unknown')}_{i:03d}",
                "text": behavior_text,
                "metadata": {
                    "behavior": indicator.get("behavior", "unknown"),
                    "pet_type": indicator.get("pet_type", "unknown"),
                    "body_part": indicator.get("body_part", "unknown"),
                    "emotional_states": indicator.get("emotional_states", []),
                    "confidence_level": indicator.get("confidence_level", "unknown"),
                    "observable_signs": indicator.get("observable_signs", []),
                    "context_factors": indicator.get("context_factors", []),
                    "scientific_evidence": indicator.get("scientific_evidence", ""),
                    "differentiation": indicator.get("differentiation", ""),
                    "source_document": doc_info.get("filename", "unknown"),
                    "research_quality": doc_info.get("research_quality", "unknown"),
                    "processed_date": datetime.now().isoformat()
                }
            }
            
            structured_data.append(structured_item)
        
        logger.info(f"📊 Structured {len(structured_data)} behavioral indicators for vector storage")
        return structured_data
    
    def _create_behavior_text(self, indicator: Dict[str, Any]) -> str:
        """Create comprehensive text representation of behavioral indicator"""
        behavior = indicator.get("behavior", "")
        pet_type = indicator.get("pet_type", "")
        emotional_states = ", ".join(indicator.get("emotional_states", []))
        observable_signs = "; ".join(indicator.get("observable_signs", []))
        context = "; ".join(indicator.get("context_factors", []))
        evidence = indicator.get("scientific_evidence", "")
        differentiation = indicator.get("differentiation", "")
        
        # Create comprehensive text for embedding
        text_parts = []
        
        if behavior:
            text_parts.append(f"Behavior: {behavior}")
        
        if pet_type:
            text_parts.append(f"Pet: {pet_type}")
        
        if emotional_states:
            text_parts.append(f"Indicates: {emotional_states}")
        
        if observable_signs:
            text_parts.append(f"Observable signs: {observable_signs}")
        
        if context:
            text_parts.append(f"Context: {context}")
        
        if evidence:
            text_parts.append(f"Evidence: {evidence}")
        
        if differentiation:
            text_parts.append(f"Differentiation: {differentiation}")
        
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
        
        # Step 2: Extract behavioral insights using Gemini
        filename = Path(pdf_path).stem
        behavioral_data = await self.extract_behavioral_insights(text, filename)
        
        # Step 3: Structure for vector database
        structured_data = self.structure_for_vector_db(behavioral_data)
        
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
    print(f"Document info: {behavioral_data.get('document_info', {})}")
    print(f"Behavioral indicators found: {len(behavioral_data.get('behavioral_indicators', []))}")
    
    # Show first few indicators
    indicators = behavioral_data.get('behavioral_indicators', [])[:3]
    for i, indicator in enumerate(indicators):
        print(f"\n🎯 Indicator {i+1}:")
        print(f"  Behavior: {indicator.get('behavior', 'N/A')}")
        print(f"  Pet type: {indicator.get('pet_type', 'N/A')}")
        print(f"  Emotional states: {indicator.get('emotional_states', [])}")
        print(f"  Observable signs: {indicator.get('observable_signs', [])}")
    
    # Test structuring for vector DB
    structured = processor.structure_for_vector_db(behavioral_data)
    print(f"\n📈 Structured for vector DB: {len(structured)} items")
    
    if structured:
        print(f"\n📝 Example structured item:")
        example = structured[0]
        print(f"  ID: {example['id']}")
        print(f"  Text: {example['text'][:100]}...")
        print(f"  Metadata keys: {list(example['metadata'].keys())}")

if __name__ == "__main__":
    asyncio.run(test_intelligent_extraction()) 