from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import uuid


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = "healthy"
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    version: str = "1.0.0"


class FaceRegisterRequest(BaseModel):
    """Request model for face registration"""
    person_name: str = Field(..., description="Name of the person", min_length=1, max_length=100)
    description: Optional[str] = Field(None, description="Optional description", max_length=500)


class FaceRegisterResponse(BaseModel):
    """Response model for face registration"""
    face_id: str = Field(..., description="Unique identifier for the registered face")
    person_name: str = Field(..., description="Name of the person")
    embedding_size: int = Field(..., description="Size of the face embedding vector")
    confidence: float = Field(..., description="Confidence score of face detection")
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    message: str = "Face registered successfully"


class FaceVerifyRequest(BaseModel):
    """Request model for face verification"""
    threshold: Optional[float] = Field(0.6, description="Similarity threshold for verification", ge=0.0, le=1.0)


class FaceVerifyResponse(BaseModel):
    """Response model for face verification"""
    is_match: bool = Field(..., description="Whether the face matches any registered face")
    matched_face_id: Optional[str] = Field(None, description="ID of the matched face if found")
    matched_person_name: Optional[str] = Field(None, description="Name of the matched person if found")
    similarity_score: float = Field(..., description="Similarity score with the best match")
    confidence: float = Field(..., description="Confidence score of face detection")
    threshold_used: float = Field(..., description="Threshold used for verification")


class FaceInfo(BaseModel):
    """Model for face information"""
    face_id: str = Field(..., description="Unique identifier for the face")
    person_name: str = Field(..., description="Name of the person")
    description: Optional[str] = Field(None, description="Optional description")
    created_at: str = Field(..., description="When the face was registered")
    embedding_size: int = Field(..., description="Size of the face embedding vector")


class FaceListResponse(BaseModel):
    """Response model for listing faces"""
    faces: List[FaceInfo] = Field(..., description="List of registered faces")
    total_count: int = Field(..., description="Total number of registered faces")


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class FaceDeleteResponse(BaseModel):
    """Response model for face deletion"""
    face_id: str = Field(..., description="ID of the deleted face")
    message: str = "Face deleted successfully"
    deleted_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat()) 