# Face Recognition API Setup Guide

This guide provides step-by-step instructions for setting up the Face Recognition API using different deployment methods.

## Prerequisites

- Python 3.8 or higher
- Docker and Docker Compose (for containerized deployment)
- At least 4GB RAM (for InsightFace model loading)

## Option 1: Local Development Setup

### 1. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt
```

### 3. Start Qdrant Vector Database

#### Using Docker (Recommended)
```bash
# Start Qdrant container
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  qdrant/qdrant
```

#### Using Docker Compose
```bash
# Start only Qdrant
docker-compose up -d qdrant
```

### 4. Run the API

```bash
# Start the FastAPI server
python start_server.py

# Or manually:
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Test the API

```bash
# Run the test script
python test_api.py
```

## Option 2: Docker Compose Setup (Recommended for Production)

### 1. Build and Start Services

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### 2. Test the API

```bash
# Test health endpoint
curl http://localhost:8000/api/health

# Run the test script
python test_api.py
```

## Option 3: Manual Docker Setup

### 1. Build the API Image

```bash
# Build the Docker image
docker build -t face-recognition-api .
```

### 2. Start Qdrant

```bash
# Start Qdrant container
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  qdrant/qdrant
```

### 3. Start the API

```bash
# Run the API container
docker run -d \
  --name face-recognition-api \
  -p 8000:8000 \
  --link qdrant \
  -e QDRANT_HOST=qdrant \
  -e QDRANT_PORT=6333 \
  face-recognition-api
```

## API Endpoints

Once the API is running, you can access:

- **API Documentation**: http://localhost:8000/docs
- **ReDoc Documentation**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/api/health

### Available Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| POST | `/api/faces/register` | Register a new face |
| POST | `/api/faces/verify` | Verify a face |
| GET | `/api/faces/list` | List all faces |
| GET | `/api/faces/{face_id}` | Get face details |
| DELETE | `/api/faces/{face_id}` | Delete a face |
| GET | `/api/stats` | System statistics |

## Usage Examples

### Register a Face

```bash
curl -X POST "http://localhost:8000/api/faces/register" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@path/to/face.jpg" \
  -F "person_name=John Doe" \
  -F "description=Test person"
```

### Verify a Face

```bash
curl -X POST "http://localhost:8000/api/faces/verify" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@path/to/face.jpg" \
  -F "threshold=0.6"
```

### List All Faces

```bash
curl -X GET "http://localhost:8000/api/faces/list"
```

## Configuration

### Environment Variables

Create a `.env` file based on `env.example`:

```bash
# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=face_embeddings

# Face Recognition Configuration
FACE_RECOGNITION_THRESHOLD=0.6
FACE_DETECTION_CONFIDENCE=0.5

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True
```

### Performance Tuning

- **Face Detection Confidence**: Adjust `FACE_DETECTION_CONFIDENCE` (0.0-1.0)
- **Recognition Threshold**: Adjust `FACE_RECOGNITION_THRESHOLD` (0.0-1.0)
- **Qdrant Configuration**: Modify collection parameters in `vector_db.py`

## Troubleshooting

### Common Issues

#### 1. InsightFace Model Download Issues

If the InsightFace model fails to download:

```bash
# Clear InsightFace cache
rm -rf ~/.insightface/

# Or set custom model path
export INSIGHTFACE_HOME=/path/to/models
```

#### 2. Qdrant Connection Issues

```bash
# Check if Qdrant is running
curl http://localhost:6333/collections

# Restart Qdrant container
docker restart qdrant
```

#### 3. Memory Issues

If you encounter memory issues:

```bash
# Increase Docker memory limit
docker run -d \
  --name qdrant \
  --memory=2g \
  -p 6333:6333 \
  qdrant/qdrant
```

#### 4. Port Conflicts

If ports are already in use:

```bash
# Check what's using the ports
lsof -i :8000
lsof -i :6333

# Use different ports
docker run -d \
  --name qdrant \
  -p 6334:6333 \
  qdrant/qdrant
```

### Logs and Debugging

```bash
# View API logs
docker-compose logs -f face_recognition_api

# View Qdrant logs
docker-compose logs -f qdrant

# Check system stats
curl http://localhost:8000/api/stats
```

## Development

### Hot Reload

The API supports hot reload during development:

```bash
# Start with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Testing

```bash
# Run tests
python test_api.py

# Run specific test
python -c "
import test_api
test_api.test_health_check()
"
```

### Code Structure

```
facerecognitionfastapi/
├── app/
│   ├── main.py              # FastAPI application
│   ├── models.py            # Pydantic models
│   ├── face_processor.py    # Face processing logic
│   ├── vector_db.py         # Vector database operations
│   └── api/
│       └── routes.py        # API endpoints
├── requirements.txt         # Python dependencies
├── docker-compose.yml       # Docker services
├── Dockerfile              # API container
├── test_api.py             # Test script
└── start_server.py         # Startup script
```

## Production Deployment

### Security Considerations

1. **Environment Variables**: Use proper secrets management
2. **CORS**: Configure CORS properly for production
3. **Authentication**: Add authentication/authorization
4. **HTTPS**: Use HTTPS in production
5. **Rate Limiting**: Implement rate limiting

### Scaling

1. **Load Balancer**: Use a load balancer for multiple API instances
2. **Qdrant Clustering**: Configure Qdrant clustering for high availability
3. **Monitoring**: Add monitoring and alerting
4. **Backup**: Implement regular backups of the vector database

## Support

For issues and questions:

1. Check the troubleshooting section above
2. Review the API documentation at `/docs`
3. Check the logs for error messages
4. Verify all dependencies are installed correctly 