"""
🔗 RAG Interface - PRODUCTION VERSION

=== PURPOSE OF THIS MODULE ===
This module provides a SIMPLE interface to the complex RAG system for easy integration:
1. Hides the complexity of vector databases and embeddings
2. Provides simple functions that main.py can call
3. Manages a global RAG instance (singleton pattern)
4. Formats results for AI prompts

=== INTEGRATION PATTERN ===
Instead of main.py dealing with:
- ChromaDB initialization
- Embedding models
- Vector search parameters
- Result formatting

main.py just calls:
- get_behavior_insights(query, pet_type) → get research insights
- format_insights_for_prompt(insights) → format for AI prompt

=== SINGLETON PATTERN ===
We use a global RAG instance because:
- RAG initialization is expensive (loads 90MB model)
- Vector database should be shared across requests
- Avoids re-loading model for every query

Usage in main.py:
    from rag_interface import get_behavior_insights, format_insights_for_prompt
    
    # Get research insights
    insights = get_behavior_insights("flattened ears", "cat")
    
    # Format for AI prompt
    formatted_context = format_insights_for_prompt(insights)
    
    # Add to Gemini prompt
    enhanced_prompt = f"Context: {formatted_context}\n\nAnalyze this behavior..."
"""

import logging
from typing import List, Dict, Any, Optional
from rag_system import PetBehaviorRAG

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === GLOBAL RAG INSTANCE (SINGLETON PATTERN) ===
# This prevents re-initializing the expensive RAG system for every request
_global_rag = None

def _get_rag_instance() -> PetBehaviorRAG:
    """
    🔧 Get or create the global RAG instance
    
    === SINGLETON PATTERN EXPLAINED ===
    This ensures we only have ONE RAG instance in the entire application:
    - First call: Creates and initializes RAG system (expensive)
    - Subsequent calls: Returns existing instance (fast)
    
    === WHY SINGLETON? ===
    - RAG initialization loads 90MB embedding model
    - Vector database should be shared across all requests
    - Avoids memory waste and startup delays
    
    Returns:
        Initialized RAG instance
    """
    global _global_rag
    
    if _global_rag is None:
        logger.info("🚀 Initializing global RAG instance...")
        
        # === CREATE AND INITIALIZE RAG SYSTEM ===
        _global_rag = PetBehaviorRAG()
        _global_rag.initialize_database()
        _global_rag.index_behavioral_insights()
        
        logger.info("✅ Global RAG instance ready!")
    
    return _global_rag

def get_behavior_insights(
    query: str, 
    pet_type: Optional[str] = None, 
    confidence_filter: Optional[str] = None,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    🔍 Get behavioral insights from research papers
    
    === SIMPLE RESEARCH QUERY INTERFACE ===
    This function hides all the complexity of:
    - Vector embeddings and similarity search
    - ChromaDB client management
    - Error handling and logging
    - Result formatting
    
    === EXAMPLE USAGE ===
    # Get cat ear behaviors
    insights = get_behavior_insights("flattened ears", "cat")
    
    # Get high-confidence dog behaviors
    insights = get_behavior_insights("tail wagging", "dog", "high")
    
    # Get any pet behaviors
    insights = get_behavior_insights("dilated pupils")
    
    Args:
        query: Behavior description (e.g., "flattened ears", "nose licking")
        pet_type: Filter by pet type ("cat", "dog", "both") or None for all
        confidence_filter: Filter by confidence ("high", "medium", "low") or None for all
        top_k: Maximum number of results to return
        
    Returns:
        List of behavioral insights with research context
        Each insight contains:
        - behavior: The behavior name
        - pet_type: Which pet this applies to
        - indicates: What this behavior means
        - confidence: How confident the research is
        - source: Whether from text or image analysis
        - source_document: Which research paper this came from
        - similarity_score: How well it matches the query (0-1)
    """
    try:
        # === GET RAG INSTANCE ===
        # This will create and initialize if needed, or return existing instance
        rag = _get_rag_instance()
        
        # === PERFORM SEMANTIC SEARCH ===
        # The RAG system handles all the complex vector search
        insights = rag.query_behaviors(
            query=query,
            pet_type=pet_type,
            confidence_filter=confidence_filter,
            top_k=top_k
        )
        
        logger.info(f"🔍 Found {len(insights)} insights for query: '{query}'")
        return insights
        
    except Exception as e:
        logger.error(f"❌ Error getting behavior insights: {e}")
        return []

def format_insights_for_prompt(insights: List[Dict[str, Any]]) -> str:
    """
    📝 Format behavioral insights for AI prompt context
    
    === PROMPT ENGINEERING ===
    This function converts RAG results into well-formatted text that:
    - Provides clear context for the AI
    - Includes research confidence levels
    - Cites source documents
    - Uses consistent formatting
    
    === BEFORE (Raw RAG Results) ===
    [
        {
            "behavior": "flattened ears",
            "pet_type": "cat", 
            "indicates": "negative conditions",
            "confidence": "medium",
            "source_document": "Emotion Recognition in Cats.pdf",
            "similarity_score": 0.85
        }
    ]
    
    === AFTER (Formatted for AI) ===
    "Research Context:
    1. Flattened ears (cat) → negative conditions [Medium confidence] 
       Source: Emotion Recognition in Cats.pdf
    2. Ear position (cat) → stress response [High confidence]
       Source: Facial correlates of emotional behaviour.pdf"
    
    Args:
        insights: List of behavioral insights from get_behavior_insights()
        
    Returns:
        Formatted string ready for AI prompt
    """
    if not insights:
        return "No relevant research insights found."
    
    # === BUILD FORMATTED CONTEXT ===
    formatted_lines = ["Research Context:"]
    
    for i, insight in enumerate(insights, 1):
        # === EXTRACT KEY INFORMATION ===
        behavior = insight.get("behavior", "Unknown behavior")
        pet_type = insight.get("pet_type", "unknown")
        indicates = insight.get("indicates", "unknown meaning")
        confidence = insight.get("confidence", "unknown").title()
        source_doc = insight.get("source_document", "Unknown source")
        similarity = insight.get("similarity_score", 0)
        
        # === FORMAT EACH INSIGHT ===
        # Pattern: "Behavior (pet) → meaning [Confidence] Source: paper"
        line = f"{i}. {behavior.title()} ({pet_type}) → {indicates} [{confidence} confidence]"
        
        # Add source attribution
        if source_doc != "Unknown source":
            # Shorten long paper names for readability
            short_source = source_doc.replace(".pdf", "").replace("_", " ")
            if len(short_source) > 50:
                short_source = short_source[:47] + "..."
            line += f"\n   Source: {short_source}"
        
        # Add similarity score if high enough to be relevant
        if similarity >= 0.7:
            line += f" (Match: {similarity:.2f})"
        
        formatted_lines.append(line)
    
    return "\n".join(formatted_lines)

def get_collection_stats() -> Dict[str, Any]:
    """
    📊 Get statistics about the RAG database
    
    === MONITORING YOUR RAG SYSTEM ===
    This function helps you understand your RAG database:
    - How many behaviors are indexed
    - Distribution across pet types
    - Confidence levels in the research
    - Source types (text vs image derived)
    
    Returns:
        Dictionary with database statistics
    """
    try:
        rag = _get_rag_instance()
        return rag.get_collection_stats()
    except Exception as e:
        logger.error(f"❌ Error getting collection stats: {e}")
        return {"error": str(e)} 