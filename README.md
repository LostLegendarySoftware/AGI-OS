# MachineGod OS Backend System

## Overview

The MachineGod OS Backend System provides the server-side infrastructure for the interactive 3D OpenGL-based GUI application with ARIEL avatar, Spectro-Emotive HUD, and Tri-Tier logic stack integration. This backend implements:

- **NSFW Toggle System**: Role-based content filtering with user permissions
- **Avatar Pack Framework**: Modular avatar system with dynamic loading and behavior profiles
- **Benchmark Logic Hooks**: Performance tracking aligned with MachineGod whitepaper requirements
- **Real-time WebSocket Communication**: Low-latency integration with the 3D GUI

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    MachineGod OS Backend                        │
├─────────────────────────────────────────────────────────────────┤
│  API Layer (FastAPI)                                           │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │   REST API      │ │   WebSocket     │ │   Authentication│   │
│  │   Endpoints     │ │   Manager       │ │   & Security    │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│  Service Layer                                                  │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │   NSFW          │ │   Avatar        │ │   Benchmark     │   │
│  │   Service       │ │   Service       │ │   Service       │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│  Data Models                                                    │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │   User Models   │ │   Avatar Models │ │ Benchmark Models│   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Performance Targets

Based on MachineGod whitepaper specifications:

- **Frame Rate**: 60 FPS minimum, 120 FPS preferred
- **Latency Requirements**:
  - Voice input to visual response: <100ms
  - Gesture recognition to avatar animation: <50ms
  - Emotional state change to visual update: <33ms
  - Panel interaction response: <16ms
- **Resource Usage**:
  - CPU utilization: <25% on Intel Xeon 12-core
  - GPU utilization: <80% on RTX 3060
  - Memory usage: <512MB RAM total

## Installation

### Prerequisites

- Python 3.8+
- Redis (for background tasks)
- PostgreSQL (optional, for persistent storage)

### Setup

1. **Clone and navigate to backend directory**:
   ```bash
   cd final/backend
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Run the server**:
   ```bash
   python main.py
   ```

   Or with uvicorn:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

## Configuration

### System Configuration

Configuration is managed through `../config/system_config.json`:

```json
{
  "system_config": {
    "server": {
      "host": "0.0.0.0",
      "port": 8000,
      "debug": false
    },
    "performance_targets": {
      "target_fps": 60,
      "preferred_fps": 120,
      "max_gpu_utilization": 0.8,
      "max_cpu_utilization": 0.25
    },
    "features": {
      "nsfw_filtering": true,
      "avatar_packs": true,
      "benchmark_tracking": true,
      "real_time_updates": true
    }
  }
}
```

### Environment Variables

Create a `.env` file with:

```env
# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=false

# Security
JWT_SECRET=your-secret-key-change-in-production
JWT_EXPIRATION=3600

# Redis Configuration
REDIS_URL=redis://localhost:6379

# Database Configuration (optional)
DATABASE_URL=postgresql://user:password@localhost/machinegodbos

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/backend.log
```

## API Documentation

### Authentication

All protected endpoints require a Bearer token:

```http
Authorization: Bearer <jwt_token>
```

### NSFW Toggle System

#### Toggle NSFW Filtering
```http
POST /api/nsfw/toggle
Content-Type: application/json
Authorization: Bearer <token>

{
  "enabled": true
}
```

#### Get NSFW Status
```http
GET /api/nsfw/status
Authorization: Bearer <token>
```

Response:
```json
{
  "nsfw_enabled": true,
  "user_permissions": ["basic_access"],
  "last_modified": "2024-01-01T12:00:00Z"
}
```

### Avatar Pack Management

#### Get Available Avatar Packs
```http
GET /api/avatars/packs
Authorization: Bearer <token>
```

Response:
```json
{
  "avatar_packs": [
    {
      "id": "default_ariel",
      "name": "ARIEL Default",
      "description": "Default ARIEL avatar with consciousness visualization",
      "type": "basic",
      "version": "1.0.0",
      "features": ["consciousness_visualization", "emotional_expressions"],
      "is_premium": false,
      "is_nsfw": false
    }
  ]
}
```

#### Switch Avatar Pack
```http
POST /api/avatars/pack/switch
Content-Type: application/json
Authorization: Bearer <token>

{
  "pack_id": "ariel_premium"
}
```

#### Get Behavior Profiles
```http
GET /api/avatars/behavior-profiles
Authorization: Bearer <token>
```

### Benchmark System

#### Submit Benchmark Data
```http
POST /api/benchmarks/submit
Content-Type: application/json
Authorization: Bearer <token>

{
  "type": "performance",
  "metrics": {
    "frame_rate": 120,
    "latency": 15,
    "cpu_usage": 0.2,
    "gpu_usage": 0.7
  },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

#### Get Leaderboard
```http
GET /api/benchmarks/leaderboard?benchmark_type=performance&limit=100
```

#### Get Performance Metrics
```http
GET /api/benchmarks/performance
Authorization: Bearer <token>
```

### WebSocket Communication

#### Connection Endpoint
```
ws://localhost:8000/ws/{client_id}
```

#### Message Format
```json
{
  "type": "consciousness_update",
  "timestamp": "2024-01-01T12:00:00Z",
  "data": {
    "consciousness_level": 0.85,
    "emotional_state": {
      "primary": "focused",
      "intensity": 0.7
    }
  }
}
```

#### Message Types

- `consciousness_update`: Consciousness state changes
- `emotional_trigger`: Emotional trigger events
- `avatar_interaction`: Avatar interaction events
- `benchmark_data`: Real-time benchmark data
- `performance_update`: System performance updates

## Services

### NSFW Service

Handles content filtering with role-based access control:

- **Filter Levels**: Disabled, Basic, Moderate, Strict
- **Content Types**: Image, Video, Text, Audio, 3D Models
- **Permissions**: User role-based filtering permissions
- **Audit Logging**: Complete audit trail of filtering actions

### Avatar Service

Manages avatar packs and behavior profiles:

- **Pack Types**: Basic, Premium, Custom, NSFW
- **Behavior Profiles**: Neutral, Analytical, Creative, etc.
- **Dynamic Loading**: Real-time avatar pack switching
- **State Management**: Avatar appearance and animation state

### Benchmark Service

Tracks performance benchmarks aligned with MachineGod whitepaper:

- **Benchmark Types**:
  - Turing Test Evaluation
  - AGI Evaluation Suite
  - HumanEval Code Generation
  - Zero-shot Logic Matching
  - System Performance
  - Consciousness Visualization

- **Metrics Tracking**:
  - Real-time performance monitoring
  - Latency measurements
  - Resource usage tracking
  - Quality assessments

### WebSocket Manager

Handles real-time communication:

- **Connection Management**: Persistent WebSocket connections
- **Message Broadcasting**: Topic-based message distribution
- **Performance Optimization**: Message queuing and prioritization
- **Health Monitoring**: Connection health checks and cleanup

## Benchmark Framework

### Performance Tracker

Located in `../../benchmarks/performance/performance_tracker.py`:

- Real-time system performance monitoring
- Compliance checking against performance targets
- Historical data analysis
- Alert generation for performance issues

### Logic Coverage Tracker

Located in `../../benchmarks/logic/logic_coverage_tracker.py`:

- Tri-Tier logic stack monitoring
- Token visualization generation
- Logic flow analysis
- Benchmark session management

## Integration with GUI

### WebSocket Integration

The backend integrates with the 3D GUI through WebSocket connections:

1. **GUI Connection**: GUI connects to `/ws/{client_id}`
2. **Real-time Updates**: Backend sends consciousness and performance updates
3. **User Interactions**: GUI sends user interaction events
4. **State Synchronization**: Avatar and HUD state kept in sync

### API Integration

GUI can make REST API calls for:

- User authentication and authorization
- Avatar pack management
- NSFW settings configuration
- Benchmark data submission

## Development

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
# Format code
black .
isort .

# Lint code
flake8 .
mypy .
```

### Adding New Services

1. Create service class in `services/`
2. Add data models in `models/`
3. Register service in `main.py`
4. Add API endpoints
5. Update configuration

## Deployment

### Production Setup

1. **Environment Configuration**:
   ```bash
   export ENVIRONMENT=production
   export DEBUG=false
   export JWT_SECRET=<secure-random-key>
   ```

2. **Database Setup**:
   ```bash
   alembic upgrade head
   ```

3. **Run with Gunicorn**:
   ```bash
   gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker
   ```

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Health Checks

The backend provides health check endpoints:

- `GET /api/system/status`: System operational status
- `GET /health`: Basic health check
- `GET /metrics`: Prometheus-compatible metrics

## Monitoring

### Performance Monitoring

- Real-time performance metrics collection
- Resource usage tracking
- Latency monitoring
- Alert generation

### Logging

Structured logging with configurable levels:

```python
import logging
logger = logging.getLogger(__name__)
logger.info("Service started", extra={"service": "avatar_service"})
```

### Metrics

Key metrics tracked:

- Request/response times
- WebSocket connection counts
- Benchmark submission rates
- Error rates and types
- Resource utilization

## Troubleshooting

### Common Issues

1. **WebSocket Connection Failures**:
   - Check firewall settings
   - Verify CORS configuration
   - Check client connection logic

2. **High Latency**:
   - Monitor system resources
   - Check network connectivity
   - Review performance metrics

3. **Authentication Errors**:
   - Verify JWT secret configuration
   - Check token expiration
   - Review user permissions

### Debug Mode

Enable debug mode for detailed logging:

```bash
export DEBUG=true
python main.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run code quality checks
5. Submit a pull request

## License

This project is part of the MachineGod OS system. See the main project license for details.