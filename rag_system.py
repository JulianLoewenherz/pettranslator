"""
🎯 RAG System for Pet Behavior Analysis

=== WHAT IS RAG (Retrieval-Augmented Generation)? ===
RAG combines information retrieval with AI generation:
1. RETRIEVAL: Find relevant information from a knowledge base
2. AUGMENTATION: Add that information to the AI prompt  
3. GENERATION: AI generates better responses using the extra context

=== HOW VECTOR DATABASES WORK ===
Traditional databases store exact text matches. Vector databases store "meanings":
- Text → Numbers (vectors) that represent semantic meaning
- "flattened ears" and "ears down" have similar vectors
- Search finds semantically similar content, not just exact words

=== OUR ARCHITECTURE ===
1. Research Papers → JSON behavioral insights (195 behaviors)
2. JSON → Vector embeddings (numbers representing meaning)
3. ChromaDB stores: [text, vectors, metadata]
4. Query: "dog nose licking" → find similar vectors → return research insights
5. AI gets research context → better behavioral analysis

Usage:
    from rag_system import PetBehaviorRAG
    
    rag = PetBehaviorRAG()
    rag.initialize_database()
    rag.index_behavioral_insights()
    
    results = rag.query_behaviors("flattened ears", pet_type="dog", top_k=3)
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# === CORE LIBRARIES ===
import chromadb  # Vector database - stores and searches vectors
from chromadb.config import Settings  # Configuration for ChromaDB
from sentence_transformers import SentenceTransformer  # Converts text to vectors

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PetBehaviorRAG:
    """
    🧠 Main RAG class for pet behavior analysis
    
    === WHAT THIS CLASS DOES ===
    1. Converts text to vectors using sentence-transformers
    2. Stores vectors in ChromaDB for fast semantic search
    3. Retrieves relevant research when given behavior queries
    4. Provides research context for better AI responses
    
    === KEY CONCEPTS ===
    - Embedding Model: Converts text → vectors (arrays of numbers)
    - Vector Database: Stores vectors + metadata for fast similarity search
    - Semantic Search: Find similar meanings, not just exact text matches
    """
    
    def __init__(self, db_path: str = "./chroma_db", model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize RAG system
        
        === EMBEDDING MODEL CHOICE ===
        "all-MiniLM-L6-v2" is a good balance of:
        - Speed: Fast inference for real-time queries
        - Quality: Good semantic understanding
        - Size: Small enough to run locally (90MB)
        - Free: No API costs like OpenAI embeddings
        
        Args:
            db_path: Where to store the ChromaDB database files
            model_name: Which sentence-transformer model to use for embeddings
        """
        self.db_path = Path(db_path)
        self.model_name = model_name
        
        # These will be initialized later
        self.embedding_model = None      # Converts text → vectors
        self.chroma_client = None        # ChromaDB client instance
        self.collection = None           # ChromaDB collection (like a table)
        
        logger.info(f"🚀 Initializing RAG system with model: {model_name}")
    
    def initialize_database(self) -> None:
        """
        🔧 Initialize ChromaDB and embedding model
        
        === CHROMADB CONCEPTS ===
        - Client: Connection to the database
        - Collection: Like a table in SQL, stores related documents
        - PersistentClient: Saves data to disk (vs in-memory)
        - Settings: Configuration options
        
        === EMBEDDING MODEL CONCEPTS ===
        - SentenceTransformer: Model that converts sentences to vectors
        - Vectors: Arrays of numbers (384 dimensions for MiniLM)
        - Semantic meaning: Similar sentences have similar vectors
        """
        try:
            # === STEP 1: CREATE CHROMADB CLIENT ===
            # PersistentClient saves data to disk so it persists between runs
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.db_path),  # Where to save database files
                settings=Settings(anonymized_telemetry=False)  # Disable telemetry
            )
            
            # === STEP 2: LOAD EMBEDDING MODEL ===
            # This downloads the model if not already cached (~90MB)
            logger.info(f"📥 Loading embedding model: {self.model_name}")
            self.embedding_model = SentenceTransformer(self.model_name)
            
            # === STEP 3: CREATE/GET COLLECTION ===
            # Collection is like a table - stores documents with vectors
            self.collection = self.chroma_client.get_or_create_collection(
                name="pet_behaviors",  # Collection name
                metadata={"description": "Pet behavioral insights from research papers"}
            )
            
            logger.info(f"✅ RAG system initialized successfully!")
            logger.info(f"💾 Database path: {self.db_path}")
            logger.info(f"📊 Collection count: {self.collection.count()}")
            
        except Exception as e:
            logger.error(f"❌ Error initializing RAG system: {e}")
            raise
    
    def index_behavioral_insights(self, json_file: str = "behavioral_insights.json") -> None:
        """
        📚 Index behavioral insights from JSON file into vector database
        
        === INDEXING PROCESS ===
        1. Load JSON behavioral insights (195 behaviors from research papers)
        2. Convert each behavior text to a vector using embedding model
        3. Store in ChromaDB: [original_text, vector, metadata]
        4. Now we can search by semantic similarity!
        
        === CHROMADB STORAGE FORMAT ===
        Each entry has:
        - id: Unique identifier  
        - document: Original text
        - embedding: Vector representation (384 numbers)
        - metadata: Structured data (pet_type, confidence, etc.)
        
        Args:
            json_file: Path to behavioral insights JSON file
        """
        try:
            logger.info(f"📖 Loading behavioral insights from: {json_file}")
            
            # === STEP 1: LOAD JSON DATA ===
            with open(json_file, 'r', encoding='utf-8') as f:
                insights = json.load(f)
            
            logger.info(f"📝 Found {len(insights)} behavioral insights to index")
            
            # === STEP 2: CHECK IF ALREADY INDEXED ===
            # Avoid re-indexing if data is already there
            current_count = self.collection.count()
            if current_count >= len(insights):
                logger.info(f"✅ Database already contains {current_count} entries. Skipping indexing.")
                return
            
            # === STEP 3: PREPARE DATA FOR CHROMADB ===
            documents = []  # Text that will be converted to vectors
            metadatas = []  # Structured data associated with each document
            ids = []        # Unique identifiers
            
            for i, insight in enumerate(insights):
                # === TEXT FOR EMBEDDING ===
                # Use the pre-constructed text from our document processor
                # Example: "Behavior: flattened ears | Pet: dog | Indicates: negative conditions | Confidence: medium | Source: text"
                documents.append(insight["text"])
                
                # === METADATA FOR FILTERING/RETRIEVAL ===
                # Store structured data for filtering and display
                metadatas.append({
                    "behavior": insight["metadata"]["behavior"],           # e.g., "flattened ears"
                    "pet_type": insight["metadata"]["pet_type"],           # e.g., "dog"
                    "indicates": insight["metadata"]["indicates"],         # e.g., "negative conditions"
                    "confidence": insight["metadata"]["confidence"],       # e.g., "medium"
                    "source": insight["metadata"]["source"],               # e.g., "text" or "image"
                    "source_document": insight["metadata"]["source_document"], # Research paper title
                    "processed_date": insight["metadata"]["processed_date"]
                })
                
                # === UNIQUE ID ===
                # Generate unique ID for each insight (original IDs had duplicates)
                ids.append(f"insight_{i:04d}")
            
            # === STEP 4: GENERATE EMBEDDINGS IN BATCHES ===
            # Process in batches for efficiency (avoid memory issues)
            batch_size = 32
            logger.info(f"🔄 Generating embeddings in batches of {batch_size}...")
            
            for i in range(0, len(documents), batch_size):
                # Get current batch
                batch_docs = documents[i:i + batch_size]
                batch_metas = metadatas[i:i + batch_size]
                batch_ids = ids[i:i + batch_size]
                
                # === CONVERT TEXT TO VECTORS ===
                # This is where the magic happens!
                # Each text becomes a 384-dimensional vector representing its meaning
                embeddings = self.embedding_model.encode(batch_docs).tolist()
                
                # === STORE IN CHROMADB ===
                # Add documents, vectors, and metadata to the collection
                self.collection.add(
                    documents=batch_docs,    # Original text
                    metadatas=batch_metas,   # Structured data
                    embeddings=embeddings,   # Vector representations
                    ids=batch_ids           # Unique identifiers
                )
                
                logger.info(f"✅ Indexed batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
            
            final_count = self.collection.count()
            logger.info(f"🎉 Indexing complete! Total entries: {final_count}")
            
        except Exception as e:
            logger.error(f"❌ Error indexing behavioral insights: {e}")
            raise
    
    def query_behaviors(
        self, 
        query: str, 
        pet_type: Optional[str] = None, 
        confidence_filter: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        🔍 Query behavioral insights using semantic search
        
        === HOW SEMANTIC SEARCH WORKS ===
        1. Convert query text to vector using same embedding model
        2. ChromaDB compares query vector to all stored vectors
        3. Returns most similar vectors (cosine similarity)
        4. Even different words with similar meanings match!
           Example: "ears down" matches "flattened ears"
        
        === FILTERING ===
        ChromaDB allows filtering by metadata:
        - pet_type: Only show dog/cat/both behaviors
        - confidence: Only show high/medium/low confidence research
        
        Args:
            query: Behavior description to search for (e.g., "flattened ears")
            pet_type: Filter by pet type ("cat", "dog", "both") or None for all
            confidence_filter: Filter by confidence ("high", "medium", "low") or None for all
            top_k: Number of results to return
            
        Returns:
            List of matching behavioral insights with metadata and similarity scores
        """
        try:
            if not self.collection:
                raise ValueError("Database not initialized. Call initialize_database() first.")
            
            logger.info(f"🔍 Querying behaviors: '{query}' (pet_type: {pet_type}, confidence: {confidence_filter})")
            
            # === STEP 1: CONVERT QUERY TO VECTOR ===
            # Use same embedding model to convert query text to vector
            query_embedding = self.embedding_model.encode([query]).tolist()[0]
            
            # === STEP 2: BUILD FILTER CONDITIONS ===
            # ChromaDB WHERE clause syntax for filtering metadata
            where_clause = {}
            if pet_type and pet_type.lower() in ["cat", "dog", "both"]:
                where_clause["pet_type"] = pet_type.lower()
            
            if confidence_filter and confidence_filter.lower() in ["high", "medium", "low"]:
                where_clause["confidence"] = confidence_filter.lower()
            
            # === STEP 3: PERFORM VECTOR SEARCH ===
            # ChromaDB finds most similar vectors using cosine similarity
            results = self.collection.query(
                query_embeddings=[query_embedding],  # Query vector
                n_results=top_k,                     # How many results to return
                where=where_clause if where_clause else None  # Metadata filters
            )
            
            # === STEP 4: FORMAT RESULTS ===
            # Convert ChromaDB results to user-friendly format
            formatted_results = []
            if results["ids"] and results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    result = {
                        "id": results["ids"][0][i],
                        "behavior": results["metadatas"][0][i]["behavior"],
                        "pet_type": results["metadatas"][0][i]["pet_type"],
                        "indicates": results["metadatas"][0][i]["indicates"],
                        "confidence": results["metadatas"][0][i]["confidence"],
                        "source": results["metadatas"][0][i]["source"],
                        "source_document": results["metadatas"][0][i]["source_document"],
                        # Convert distance to similarity score (1 = perfect match, 0 = no match)
                        # ChromaDB uses cosine distance (0-2), so similarity = (2 - distance) / 2
                        "similarity_score": round(max(0, (2 - results["distances"][0][i]) / 2), 3),
                        "text": results["documents"][0][i]
                    }
                    formatted_results.append(result)
            
            logger.info(f"✅ Found {len(formatted_results)} matching behaviors")
            return formatted_results
            
        except Exception as e:
            logger.error(f"❌ Error querying behaviors: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        📊 Get statistics about the behavioral insights collection
        
        === ANALYTICS ON YOUR DATA ===
        This function analyzes your vector database to show:
        - How many behaviors for each pet type
        - Confidence distribution 
        - Source distribution (text vs image derived)
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            if not self.collection:
                return {"error": "Database not initialized"}
            
            # === GET ALL METADATA FOR ANALYSIS ===
            # ChromaDB .get() retrieves all documents and metadata
            results = self.collection.get()
            
            total_count = len(results["ids"])
            
            # === COUNT BY CATEGORIES ===
            pet_types = {}
            confidence_levels = {}
            sources = {}
            
            for metadata in results["metadatas"]:
                # Count pet types
                pet_type = metadata.get("pet_type", "unknown")
                pet_types[pet_type] = pet_types.get(pet_type, 0) + 1
                
                # Count confidence levels
                confidence = metadata.get("confidence", "unknown")
                confidence_levels[confidence] = confidence_levels.get(confidence, 0) + 1
                
                # Count sources (text vs image)
                source = metadata.get("source", "unknown")
                sources[source] = sources.get(source, 0) + 1
            
            return {
                "total_behaviors": total_count,
                "pet_types": pet_types,
                "confidence_levels": confidence_levels,
                "sources": sources,
                "database_path": str(self.db_path)
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting collection stats: {e}")
            return {"error": str(e)} 