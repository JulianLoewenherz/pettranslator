#!/usr/bin/env python3
"""
🎯 STEP 1: Intelligent Research Paper Processing with LLM

This script uses Gemini for intelligent extraction instead of crude chunking:
1. Extract text from PDF research papers
2. Use Gemini to intelligently extract behavioral insights
3. Filter out irrelevant content automatically
4. Structure data for vector database storage

Usage:
1. Put your PDF research papers in the research_papers/ directory
2. Run this script: python process_research_papers.py
3. The processed behavioral insights will be saved to behavioral_insights.json

Directory Structure:
research_papers/
├── cats/        # Put cat behavior papers here
├── dogs/        # Put dog behavior papers here  
└── general/     # Put general pet behavior papers here
"""

from document_processor import IntelligentPetBehaviorProcessor
import asyncio
import json
from pathlib import Path

async def main():
    print("🧠 Step 1: Intelligent Research Paper Processing with LLM")
    print("=" * 60)
    
    # Initialize intelligent processor
    try:
        processor = IntelligentPetBehaviorProcessor()
    except Exception as e:
        print(f"❌ Error initializing processor: {e}")
        print("Make sure your GOOGLE_API_KEY is set in your .env file")
        return
    
    # Check if research papers directory exists
    research_dir = Path("research_papers")
    if not research_dir.exists():
        print("❌ research_papers directory not found!")
        print("Please create it and add your PDF files:")
        print("  mkdir research_papers")
        print("  # Then copy your PDF files there")
        return
    
    # Process all PDF files with intelligent extraction
    print(f"📁 Processing PDFs in {research_dir}/ with intelligent extraction...")
    
    behavioral_data = await processor.process_directory(
        directory_path=str(research_dir),
        output_file="behavioral_insights.json"
    )
    
    if not behavioral_data:
        print("\n⚠️ No behavioral data was extracted!")
        print("Make sure you have PDF files in the research_papers directory:")
        print("  research_papers/")
        print("  ├── your_cat_paper.pdf")
        print("  ├── your_dog_paper.pdf")
        print("  └── ...")
        return
    
    # Display intelligent extraction summary
    print("\n🧠 Intelligent Extraction Summary:")
    print("=" * 40)
    print(f"Total behavioral indicators: {len(behavioral_data)}")
    
    # Group by pet type
    pet_types = {}
    emotional_states = {}
    body_parts = {}
    confidence_levels = {}
    
    for item in behavioral_data:
        metadata = item.get('metadata', {})
        
        # Pet types
        pet_type = metadata.get('pet_type', 'unknown')
        pet_types[pet_type] = pet_types.get(pet_type, 0) + 1
        
        # Emotional states
        for emotion in metadata.get('emotional_states', []):
            emotional_states[emotion] = emotional_states.get(emotion, 0) + 1
        
        # Body parts
        body_part = metadata.get('body_part', 'unknown')
        body_parts[body_part] = body_parts.get(body_part, 0) + 1
        
        # Confidence levels
        confidence = metadata.get('confidence_level', 'unknown')
        confidence_levels[confidence] = confidence_levels.get(confidence, 0) + 1
    
    print(f"\n🐾 Pet Types:")
    for pet_type, count in sorted(pet_types.items()):
        print(f"  {pet_type}: {count} indicators")
    
    print(f"\n😊 Emotional States:")
    for emotion, count in sorted(emotional_states.items(), key=lambda x: x[1], reverse=True)[:8]:
        print(f"  {emotion}: {count} indicators")
    
    print(f"\n🎯 Body Parts:")
    for body_part, count in sorted(body_parts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {body_part}: {count} indicators")
    
    print(f"\n📊 Confidence Levels:")
    for confidence, count in sorted(confidence_levels.items()):
        print(f"  {confidence}: {count} indicators")
    
    # Show example behavioral indicators
    print(f"\n📝 Example Behavioral Indicators:")
    print("=" * 40)
    
    for i, item in enumerate(behavioral_data[:3]):
        metadata = item.get('metadata', {})
        print(f"\n🎯 Indicator {i+1}:")
        print(f"  Behavior: {metadata.get('behavior', 'N/A')}")
        print(f"  Pet: {metadata.get('pet_type', 'N/A')}")
        print(f"  Body Part: {metadata.get('body_part', 'N/A')}")
        print(f"  Emotions: {', '.join(metadata.get('emotional_states', []))}")
        print(f"  Confidence: {metadata.get('confidence_level', 'N/A')}")
        print(f"  Observable Signs: {', '.join(metadata.get('observable_signs', [])[:2])}")
        print(f"  Text Preview: {item.get('text', '')[:100]}...")
    
    # Quality assessment
    high_confidence = sum(1 for item in behavioral_data if item.get('metadata', {}).get('confidence_level') == 'high')
    peer_reviewed = sum(1 for item in behavioral_data if 'peer-reviewed' in item.get('metadata', {}).get('source_document', '').lower())
    
    print(f"\n📈 Quality Assessment:")
    print(f"  High confidence indicators: {high_confidence}/{len(behavioral_data)} ({high_confidence/len(behavioral_data)*100:.1f}%)")
    print(f"  From peer-reviewed sources: {peer_reviewed}/{len(behavioral_data)} ({peer_reviewed/len(behavioral_data)*100:.1f}%)")
    
    print(f"\n✅ Intelligent processing complete!")
    print(f"📄 Data saved to behavioral_insights.json")
    print(f"🎯 Ready for Step 2: Vector embedding and database creation")
    
    # Show next steps
    print(f"\n🚀 Next Steps:")
    print("1. Review the behavioral_insights.json file")
    print("2. Run the embedding script to create vector database")
    print("3. Integrate with your video analysis pipeline")

if __name__ == "__main__":
    asyncio.run(main()) 