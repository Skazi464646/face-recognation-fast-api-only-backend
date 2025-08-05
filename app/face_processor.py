import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from PIL import Image
import io
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class FaceProcessor:
    """Handles face detection and embedding extraction using InsightFace"""
    
    def __init__(self, detection_confidence: float = 0.5):
        """
        Initialize the face processor
        
        Args:
            detection_confidence: Minimum confidence for face detection
        """
        self.detection_confidence = detection_confidence
        self.app = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the InsightFace model"""
        try:
            self.app = FaceAnalysis(name='buffalo_l')
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("InsightFace model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize InsightFace model: {e}")
            raise
    
    def image_to_numpy(self, image_data: bytes) -> np.ndarray:
        """
        Convert image bytes to numpy array
        
        Args:
            image_data: Image data as bytes
            
        Returns:
            numpy.ndarray: Image as numpy array in BGR format
        """
        try:
            # Convert bytes to PIL Image
            pil_image = Image.open(io.BytesIO(image_data))
            
            # Convert PIL Image to numpy array (BGR format for OpenCV)
            numpy_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            return numpy_image
        except Exception as e:
            logger.error(f"Error converting image to numpy array: {e}")
            raise ValueError(f"Invalid image format: {e}")
    
    def detect_faces(self, image: np.ndarray) -> List[dict]:
        """
        Detect faces in the image
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of detected faces with bounding boxes and landmarks
        """
        try:
            faces = self.app.get(image)
            
            # Filter faces based on confidence
            valid_faces = [
                face for face in faces 
                if face.det_score >= self.detection_confidence
            ]
            
            if not valid_faces:
                logger.warning("No faces detected with sufficient confidence")
                return []
            
            logger.info(f"Detected {len(valid_faces)} faces")
            return valid_faces
            
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            raise
    
    def extract_embedding(self, face: dict) -> Tuple[np.ndarray, float]:
        """
        Extract embedding from a detected face
        
        Args:
            face: Detected face object from InsightFace
            
        Returns:
            Tuple of (embedding_vector, confidence_score)
        """
        try:
            embedding = face.embedding
            confidence = face.det_score
            
            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding, confidence
            
        except Exception as e:
            logger.error(f"Error extracting face embedding: {e}")
            raise
    
    def process_image(self, image_data: bytes) -> Tuple[np.ndarray, float]:
        """
        Process image to extract face embedding
        
        Args:
            image_data: Image data as bytes
            
        Returns:
            Tuple of (embedding_vector, confidence_score)
        """
        try:
            # Convert image to numpy array
            image = self.image_to_numpy(image_data)
            
            # Detect faces
            faces = self.detect_faces(image)
            
            if not faces:
                raise ValueError("No faces detected in the image")
            
            # Use the first detected face (you might want to implement logic for multiple faces)
            if len(faces) > 1:
                logger.warning(f"Multiple faces detected ({len(faces)}), using the first one")
            
            face = faces[0]
            
            # Extract embedding
            embedding, confidence = self.extract_embedding(face)
            
            logger.info(f"Successfully extracted face embedding with confidence: {confidence:.3f}")
            
            return embedding, confidence
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            # Ensure embeddings are normalized
            emb1_norm = embedding1 / np.linalg.norm(embedding1)
            emb2_norm = embedding2 / np.linalg.norm(embedding2)
            
            # Calculate cosine similarity
            similarity = np.dot(emb1_norm, emb2_norm)
            
            # Ensure similarity is between 0 and 1
            similarity = max(0.0, min(1.0, similarity))
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            raise
    
    def verify_face(self, query_embedding: np.ndarray, stored_embeddings: List[np.ndarray], 
                   threshold: float = 0.6) -> Tuple[bool, float, int]:
        """
        Verify a face against stored embeddings
        
        Args:
            query_embedding: Embedding of the face to verify
            stored_embeddings: List of stored embeddings to compare against
            threshold: Similarity threshold for verification
            
        Returns:
            Tuple of (is_match, best_similarity, best_match_index)
        """
        try:
            if not stored_embeddings:
                return False, 0.0, -1
            
            similarities = []
            for i, stored_embedding in enumerate(stored_embeddings):
                similarity = self.calculate_similarity(query_embedding, stored_embedding)
                similarities.append((similarity, i))
            
            # Find the best match
            best_similarity, best_match_index = max(similarities, key=lambda x: x[0])
            
            is_match = best_similarity >= threshold
            
            logger.info(f"Best similarity: {best_similarity:.3f}, threshold: {threshold}, match: {is_match}")
            
            return is_match, best_similarity, best_match_index
            
        except Exception as e:
            logger.error(f"Error verifying face: {e}")
            raise 