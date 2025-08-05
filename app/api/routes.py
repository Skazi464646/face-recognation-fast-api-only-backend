from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from fastapi.responses import JSONResponse
from typing import Optional
import logging

from ..models import (
    HealthResponse, FaceRegisterResponse, FaceVerifyResponse, 
    FaceListResponse, FaceDeleteResponse, ErrorResponse
)
from ..face_processor import FaceProcessor
from ..vector_db import VectorDatabase

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()

# Initialize services (these would typically be injected via dependency injection)
face_processor = FaceProcessor(detection_confidence=0.5)
vector_db = VectorDatabase(host="localhost", port=6333, collection_name="face_embeddings")


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint to verify API status
    """
    try:
        # Check if vector database is accessible
        stats = vector_db.get_collection_stats()
        
        return HealthResponse(
            status="healthy",
            version="1.0.0"
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@router.post("/faces/register", response_model=FaceRegisterResponse, tags=["Faces"])
async def register_face(
    image: UploadFile = File(..., description="Face image file"),
    person_name: str = Form(..., description="Name of the person"),
    description: Optional[str] = Form(None, description="Optional description")
):
    """
    Register a new face in the system
    
    - **image**: Face image file (JPEG, PNG, etc.)
    - **person_name**: Name of the person
    - **description**: Optional description
    """
    try:
        # Validate file type
        if not image.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400, 
                detail="File must be an image"
            )
        
        # Read image data
        image_data = await image.read()
        
        # Process image and extract embedding
        embedding, confidence = face_processor.process_image(image_data)
        
        # Store embedding in vector database
        face_id = vector_db.store_embedding(
            embedding=embedding,
            person_name=person_name,
            description=description
        )
        
        logger.info(f"Successfully registered face for {person_name} with ID: {face_id}")
        
        return FaceRegisterResponse(
            face_id=face_id,
            person_name=person_name,
            embedding_size=len(embedding),
            confidence=confidence
        )
        
    except ValueError as e:
        logger.error(f"Validation error during face registration: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error registering face: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/faces/verify", response_model=FaceVerifyResponse, tags=["Faces"])
async def verify_face(
    image: UploadFile = File(..., description="Face image to verify"),
    threshold: Optional[float] = Form(0.6, description="Similarity threshold")
):
    """
    Verify a face against stored embeddings
    
    - **image**: Face image file to verify
    - **threshold**: Similarity threshold (default: 0.6)
    """
    try:
        # Validate file type
        if not image.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400, 
                detail="File must be an image"
            )
        
        # Validate threshold
        if threshold < 0.0 or threshold > 1.0:
            raise HTTPException(
                status_code=400,
                detail="Threshold must be between 0.0 and 1.0"
            )
        
        # Read image data
        image_data = await image.read()
        
        # Process image and extract embedding
        query_embedding, confidence = face_processor.process_image(image_data)
        
        # Search for similar faces
        similar_faces = vector_db.search_similar_faces(
            query_embedding=query_embedding,
            limit=1,
            score_threshold=threshold
        )
        
        # Check if we found a match
        if similar_faces:
            best_match = similar_faces[0]
            is_match = best_match["similarity_score"] >= threshold
            
            return FaceVerifyResponse(
                is_match=is_match,
                matched_face_id=best_match["face_id"] if is_match else None,
                matched_person_name=best_match["person_name"] if is_match else None,
                similarity_score=best_match["similarity_score"],
                confidence=confidence,
                threshold_used=threshold
            )
        else:
            return FaceVerifyResponse(
                is_match=False,
                matched_face_id=None,
                matched_person_name=None,
                similarity_score=0.0,
                confidence=confidence,
                threshold_used=threshold
            )
            
    except ValueError as e:
        logger.error(f"Validation error during face verification: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error verifying face: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/faces/list", response_model=FaceListResponse, tags=["Faces"])
async def list_faces(limit: int = 100):
    """
    List all registered faces
    
    - **limit**: Maximum number of faces to return (default: 100)
    """
    try:
        # Validate limit
        if limit < 1 or limit > 1000:
            raise HTTPException(
                status_code=400,
                detail="Limit must be between 1 and 1000"
            )
        
        # Get all faces
        faces_data = vector_db.list_all_faces(limit=limit)
        
        # Convert to response format
        from ..models import FaceInfo
        from datetime import datetime
        
        faces = []
        for face_data in faces_data:
            faces.append(FaceInfo(
                face_id=face_data["face_id"],
                person_name=face_data["person_name"],
                description=face_data["description"],
                created_at=datetime.fromisoformat(face_data["created_at"]),
                embedding_size=face_data["embedding_size"]
            ))
        
        return FaceListResponse(
            faces=faces,
            total_count=len(faces)
        )
        
    except Exception as e:
        logger.error(f"Error listing faces: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/faces/{face_id}", tags=["Faces"])
async def get_face(face_id: str):
    """
    Get information about a specific face
    
    - **face_id**: Unique identifier of the face
    """
    try:
        face_info = vector_db.get_face_by_id(face_id)
        
        if not face_info:
            raise HTTPException(
                status_code=404,
                detail="Face not found"
            )
        
        return {
            "face_id": face_info["face_id"],
            "person_name": face_info["person_name"],
            "description": face_info["description"],
            "created_at": face_info["created_at"],
            "embedding_size": face_info["embedding_size"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting face: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/faces/{face_id}", response_model=FaceDeleteResponse, tags=["Faces"])
async def delete_face(face_id: str):
    """
    Delete a registered face
    
    - **face_id**: Unique identifier of the face to delete
    """
    try:
        success = vector_db.delete_face(face_id)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail="Face not found"
            )
        
        return FaceDeleteResponse(face_id=face_id)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting face: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/stats", tags=["System"])
async def get_system_stats():
    """
    Get system statistics including database info
    """
    try:
        db_stats = vector_db.get_collection_stats()
        
        return {
            "database": db_stats,
            "face_processor": {
                "detection_confidence": face_processor.detection_confidence
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/faces", tags=["System"])
async def clear_all_faces():
    """
    Clear all registered faces (DANGEROUS - use with caution)
    """
    try:
        success = vector_db.clear_collection()
        
        if success:
            return {"message": "All faces cleared successfully"}
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to clear faces"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing faces: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") 