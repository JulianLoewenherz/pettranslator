#!/usr/bin/env python3
"""
🧪 Simple RAG Test Function
Test the get_behavior_insights function easily and see results clearly
"""

from rag_interface import get_behavior_insights, format_insights_for_prompt

def rag_test(query: str, pet_type: str = None, top_k: int = 3):
    """
    🧪 Simple function to test RAG system
    
    Args:
        query: What behavior to search for (e.g., "flattened ears", "tail wagging")
        pet_type: "cat", "dog", "both", or None for all pets
        top_k: How many results to show (default: 3)
    
    Example usage:
        rag_test("flattened ears", "cat")
        rag_test("tail wagging", "dog", 5)
        rag_test("dilated pupils")  # Any pet type
    """
    
    print(f"\n🔍 Testing RAG Query:")
    print(f"   Query: '{query}'")
    print(f"   Pet Type: {pet_type or 'Any'}")
    print(f"   Max Results: {top_k}")
    print("=" * 50)
    
    # Get insights using your RAG system
    insights = get_behavior_insights(query, pet_type, top_k=top_k)
    
    if not insights:
        print("❌ No results found!")
        return
    
    print(f"✅ Found {len(insights)} results:\n")
    
    # Show each result clearly
    for i, insight in enumerate(insights, 1):
        print(f"📋 Result {i}:")
        print(f"   Behavior: {insight['behavior']}")
        print(f"   Pet: {insight['pet_type']}")
        print(f"   Meaning: {insight['indicates']}")
        print(f"   Confidence: {insight['confidence']}")
        print(f"   Similarity: {insight['similarity_score']:.3f}")
        print(f"   Source: {insight['source_document'][:60]}...")
        print()
    
    # Show formatted prompt version
    print("📝 Formatted for AI Prompt:")
    print("-" * 30)
    formatted = format_insights_for_prompt(insights)
    print(formatted)
    print("-" * 30)

def quick_tests():
    """
    🚀 Run some quick test examples
    """
    print("🧪 Running Quick RAG Tests...")
    
    test_cases = [
        ("flattened ears", "cat"),
        ("tail wagging", "dog"), 
        ("dilated pupils", None),
        ("nose licking", "dog")
    ]
    
    for query, pet_type in test_cases:
        rag_test(query, pet_type, 2)
        input("Press Enter for next test...")

if __name__ == "__main__":
    # Example usage
    print("🎯 RAG Test Function Ready!")
    print("\nExample usage:")
    print('rag_test("flattened ears", "cat")')
    print('rag_test("tail wagging", "dog", 5)')
    print('rag_test("dilated pupils")')
    print("\nOr run quick_tests() for multiple examples")
    
    # Uncomment to run a test:
    rag_test("urinating outside of litterbox", "cat") 