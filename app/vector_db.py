import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from typing import List, Optional, Tuple, Dict, Any
import logging
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)


class VectorDatabase:
    """Handles vector database operations for face embeddings using Qdrant"""
    
    def __init__(self, host: str = "localhost", port: int = 6333, collection_name: str = "face_embeddings"):
        """
        Initialize the vector database connection
        
        Args:
            host: Qdrant server host
            port: Qdrant server port
            collection_name: Name of the collection to store embeddings
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.client = None
        self.embedding_size = 512  # InsightFace embedding size
        
        self._connect()
        self._ensure_collection_exists()
    
    def _connect(self):
        """Establish connection to Qdrant"""
        try:
            self.client = QdrantClient(host=self.host, port=self.port)
            logger.info(f"Connected to Qdrant at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise
    
    def _ensure_collection_exists(self):
        """Ensure the collection exists, create if it doesn't"""
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                self._create_collection()
                logger.info(f"Created collection: {self.collection_name}")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
                
        except Exception as e:
            logger.error(f"Error checking/creating collection: {e}")
            raise
    
    def _create_collection(self):
        """Create the face embeddings collection"""
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_size,
                    distance=Distance.COSINE
                )
            )
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise
    
    def store_embedding(self, embedding: np.ndarray, person_name: str, 
                       description: Optional[str] = None) -> str:
        """
        Store a face embedding in the vector database
        
        Args:
            embedding: Face embedding vector
            person_name: Name of the person
            description: Optional description
            
        Returns:
            face_id: Unique identifier for the stored face
        """
        try:
            # Generate unique ID
            face_id = str(uuid.uuid4())
            
            # Convert embedding to list
            embedding_list = embedding.tolist()
            
            # Create point with metadata
            point = PointStruct(
                id=face_id,
                vector=embedding_list,
                payload={
                    "person_name": person_name,
                    "description": description,
                    "created_at": datetime.utcnow().isoformat(),
                    "embedding_size": len(embedding_list)
                }
            )
            
            # Insert point
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            logger.info(f"Stored embedding for {person_name} with ID: {face_id}")
            return face_id
            
        except Exception as e:
            logger.error(f"Error storing embedding: {e}")
            raise
    
    def search_similar_faces(self, query_embedding: np.ndarray, limit: int = 10, 
                           score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search for similar faces in the database
        
        Args:
            query_embedding: Query face embedding
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of similar faces with metadata
        """
        try:
            # Convert embedding to list
            query_vector = query_embedding.tolist()
            
            # Search for similar vectors
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold
            )
            
            # Format results
            results = []
            for point in search_result:
                results.append({
                    "face_id": point.id,
                    "similarity_score": point.score,
                    "person_name": point.payload.get("person_name"),
                    "description": point.payload.get("description"),
                    "created_at": point.payload.get("created_at"),
                    "embedding_size": point.payload.get("embedding_size")
                })
            
            logger.info(f"Found {len(results)} similar faces")
            return results
            
        except Exception as e:
            logger.error(f"Error searching similar faces: {e}")
            raise
    
    def get_face_by_id(self, face_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve face information by ID
        
        Args:
            face_id: Unique identifier of the face
            
        Returns:
            Face information if found, None otherwise
        """
        try:
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[face_id]
            )
            
            if not points:
                return None
            
            point = points[0]
            return {
                "face_id": point.id,
                "person_name": point.payload.get("person_name"),
                "description": point.payload.get("description"),
                "created_at": point.payload.get("created_at"),
                "embedding_size": point.payload.get("embedding_size"),
                "embedding": point.vector
            }
            
        except Exception as e:
            logger.error(f"Error retrieving face by ID: {e}")
            raise
    
    def list_all_faces(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        List all stored faces
        
        Args:
            limit: Maximum number of faces to return
            
        Returns:
            List of all faces with metadata
        """
        try:
            # Get all points from collection
            points = self.client.scroll(
                collection_name=self.collection_name,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )[0]
            
            # Format results
            faces = []
            for point in points:
                faces.append({
                    "face_id": point.id,
                    "person_name": point.payload.get("person_name"),
                    "description": point.payload.get("description"),
                    "created_at": point.payload.get("created_at"),
                    "embedding_size": point.payload.get("embedding_size")
                })
            
            logger.info(f"Retrieved {len(faces)} faces from database")
            return faces
            
        except Exception as e:
            logger.error(f"Error listing faces: {e}")
            raise
    
    def delete_face(self, face_id: str) -> bool:
        """
        Delete a face from the database
        
        Args:
            face_id: Unique identifier of the face to delete
            
        Returns:
            True if deleted successfully, False if not found
        """
        try:
            # Check if face exists
            face_info = self.get_face_by_id(face_id)
            if not face_info:
                logger.warning(f"Face with ID {face_id} not found")
                return False
            
            # Delete the point
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=[face_id]
            )
            
            logger.info(f"Deleted face with ID: {face_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting face: {e}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get collection statistics
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)
            
            stats = {
                "collection_name": self.collection_name,
                "vector_size": collection_info.config.params.vectors.size,
                "distance": str(collection_info.config.params.vectors.distance),
                "points_count": collection_info.points_count,
                "segments_count": collection_info.segments_count,
                "status": collection_info.status
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            # Return basic stats if detailed stats fail
            return {
                "collection_name": self.collection_name,
                "vector_size": self.embedding_size,
                "distance": "COSINE",
                "points_count": 0,
                "segments_count": 0,
                "status": "unknown",
                "error": str(e)
            }
    
    def clear_collection(self) -> bool:
        """
        Clear all faces from the collection
        
        Returns:
            True if cleared successfully
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector="*"
            )
            
            logger.info(f"Cleared collection: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            raise 