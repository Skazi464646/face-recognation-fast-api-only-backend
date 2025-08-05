# Face Recognition API

A robust face recognition API built with FastAPI that provides face embedding extraction, storage, and verification capabilities.

## Features

- **Face Embedding Extraction**: Extract high-quality face embeddings using InsightFace
- **Vector Database Storage**: Store embeddings in Qdrant vector database
- **Face Verification**: Compare face embeddings for identity verification
- **RESTful API**: Clean FastAPI endpoints for all operations
- **Health Monitoring**: Built-in health check endpoint

## Project Structure

```
facerecognitionfastapi/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── models.py            # Pydantic models
│   ├── face_processor.py    # Face processing logic
│   ├── vector_db.py         # Vector database operations
│   └── api/
│       ├── __init__.py
│       └── routes.py        # API endpoints
├── requirements.txt
├── .env.example
└── README.md
```

## Setup Instructions

1. **Create Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Environment Variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Run the Application**:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

## API Endpoints

- `GET /health` - Health check endpoint
- `POST /api/faces/register` - Register a new face
- `POST /api/faces/verify` - Verify a face against stored embeddings
- `GET /api/faces/list` - List all registered faces
- `DELETE /api/faces/{face_id}` - Delete a registered face

## Usage Examples

### Register a Face
```bash
curl -X POST "http://localhost:8000/api/faces/register" \
     -H "Content-Type: multipart/form-data" \
     -F "image=@path/to/face.jpg" \
     -F "person_name=John Doe"
```

### Verify a Face
```bash
curl -X POST "http://localhost:8000/api/faces/verify" \
     -H "Content-Type: multipart/form-data" \
     -F "image=@path/to/face.jpg"
```

## Technologies Used

- **FastAPI**: Modern web framework for building APIs
- **InsightFace**: State-of-the-art face recognition model
- **Qdrant**: Vector database for storing embeddings
- **OpenCV**: Image processing and face detection
- **Pydantic**: Data validation and serialization 