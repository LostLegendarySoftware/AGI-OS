"""
MachineGod OS Backend Server
FastAPI-based backend with WebSocket support for real-time GUI integration
Implements NSFW toggle system, avatar pack management, and benchmark tracking
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
import uuid

from services.nsfw_service import NSFWService
from services.avatar_service import AvatarService
from services.benchmark_service import BenchmarkService
from services.websocket_manager import WebSocketManager
from models.user_models import User, UserRole
from models.system_models import SystemState, PerformanceMetrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="MachineGod OS Backend",
    description="Backend API for 3D GUI with ARIEL avatar and Spectro-Emotive HUD",
    version="1.0.0"
)

# CORS middleware for GUI integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Initialize services
nsfw_service = NSFWService()
avatar_service = AvatarService()
benchmark_service = BenchmarkService()
websocket_manager = WebSocketManager()

# Global system state
system_state = SystemState()

@app.on_event("startup")
async def startup_event():
    """Initialize backend services on startup"""
    logger.info("Starting MachineGod OS Backend...")
    
    # Initialize services
    await nsfw_service.initialize()
    await avatar_service.initialize()
    await benchmark_service.initialize()
    
    # Start background tasks
    asyncio.create_task(performance_monitoring_task())
    asyncio.create_task(consciousness_processing_task())
    
    logger.info("Backend services initialized successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down MachineGod OS Backend...")
    await websocket_manager.disconnect_all()

# WebSocket endpoint for real-time GUI communication
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    Real-time WebSocket connection for GUI integration
    Handles consciousness updates, emotional triggers, and system state
    """
    await websocket_manager.connect(websocket, client_id)
    
    try:
        while True:
            # Receive data from GUI
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Process different message types
            await process_websocket_message(client_id, message)
            
    except WebSocketDisconnect:
        await websocket_manager.disconnect(client_id)
        logger.info(f"Client {client_id} disconnected")

async def process_websocket_message(client_id: str, message: dict):
    """Process incoming WebSocket messages from GUI"""
    message_type = message.get("type")
    
    if message_type == "consciousness_update":
        await handle_consciousness_update(client_id, message["data"])
    elif message_type == "emotional_trigger":
        await handle_emotional_trigger(client_id, message["data"])
    elif message_type == "avatar_interaction":
        await handle_avatar_interaction(client_id, message["data"])
    elif message_type == "benchmark_data":
        await handle_benchmark_data(client_id, message["data"])

# NSFW Toggle System API Endpoints
@app.post("/api/nsfw/toggle")
async def toggle_nsfw(
    enabled: bool,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Toggle NSFW content filtering with role-based access control"""
    user = await authenticate_user(credentials.credentials)
    
    if not await nsfw_service.can_modify_nsfw_settings(user):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    result = await nsfw_service.toggle_nsfw(user.id, enabled)
    
    # Broadcast update to connected clients
    await websocket_manager.broadcast({
        "type": "nsfw_status_update",
        "data": {"enabled": enabled, "user_id": user.id}
    })
    
    return {"success": True, "nsfw_enabled": enabled}

@app.get("/api/nsfw/status")
async def get_nsfw_status(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get current NSFW filtering status"""
    user = await authenticate_user(credentials.credentials)
    status = await nsfw_service.get_nsfw_status(user.id)
    
    return {
        "nsfw_enabled": status.enabled,
        "user_permissions": status.permissions,
        "last_modified": status.last_modified
    }

# Avatar Pack Management API Endpoints
@app.get("/api/avatars/packs")
async def get_avatar_packs(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get available avatar packs for user"""
    user = await authenticate_user(credentials.credentials)
    packs = await avatar_service.get_available_packs(user)
    
    return {"avatar_packs": packs}

@app.post("/api/avatars/pack/switch")
async def switch_avatar_pack(
    pack_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Switch to different avatar pack"""
    user = await authenticate_user(credentials.credentials)
    
    result = await avatar_service.switch_pack(user.id, pack_id)
    
    if result.success:
        # Broadcast avatar update to GUI
        await websocket_manager.broadcast({
            "type": "avatar_pack_changed",
            "data": {
                "pack_id": pack_id,
                "user_id": user.id,
                "avatar_config": result.config
            }
        })
    
    return {"success": result.success, "message": result.message}

@app.get("/api/avatars/behavior-profiles")
async def get_behavior_profiles(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get available behavior profiles"""
    user = await authenticate_user(credentials.credentials)
    profiles = await avatar_service.get_behavior_profiles(user)
    
    return {"behavior_profiles": profiles}

# Benchmark System API Endpoints
@app.post("/api/benchmarks/submit")
async def submit_benchmark_data(
    benchmark_data: dict,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Submit benchmark performance data"""
    user = await authenticate_user(credentials.credentials)
    
    result = await benchmark_service.submit_benchmark(user.id, benchmark_data)
    
    return {
        "success": result.success,
        "benchmark_id": result.benchmark_id,
        "score": result.score
    }

@app.get("/api/benchmarks/leaderboard")
async def get_benchmark_leaderboard(
    benchmark_type: str = "overall",
    limit: int = 100
):
    """Get benchmark leaderboard"""
    leaderboard = await benchmark_service.get_leaderboard(benchmark_type, limit)
    
    return {"leaderboard": leaderboard}

@app.get("/api/benchmarks/performance")
async def get_performance_metrics(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get real-time performance metrics"""
    user = await authenticate_user(credentials.credentials)
    metrics = await benchmark_service.get_performance_metrics(user.id)
    
    return {"performance_metrics": metrics}

# System Status and Health Endpoints
@app.get("/api/system/status")
async def get_system_status():
    """Get current system status"""
    return {
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat(),
        "connected_clients": websocket_manager.get_connection_count(),
        "system_metrics": system_state.get_metrics()
    }

@app.get("/api/system/consciousness")
async def get_consciousness_state(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get current consciousness processing state"""
    user = await authenticate_user(credentials.credentials)
    consciousness_data = await get_consciousness_data(user.id)
    
    return {"consciousness_state": consciousness_data}

# Background Tasks
async def performance_monitoring_task():
    """Background task for performance monitoring"""
    while True:
        try:
            # Collect performance metrics
            metrics = await collect_performance_metrics()
            
            # Update system state
            system_state.update_performance_metrics(metrics)
            
            # Broadcast to connected clients
            await websocket_manager.broadcast({
                "type": "performance_update",
                "data": metrics
            })
            
            await asyncio.sleep(1)  # Update every second
            
        except Exception as e:
            logger.error(f"Performance monitoring error: {e}")
            await asyncio.sleep(5)

async def consciousness_processing_task():
    """Background task for consciousness state processing"""
    while True:
        try:
            # Process consciousness updates
            consciousness_updates = await process_consciousness_state()
            
            if consciousness_updates:
                await websocket_manager.broadcast({
                    "type": "consciousness_update",
                    "data": consciousness_updates
                })
            
            await asyncio.sleep(0.033)  # ~30 FPS update rate
            
        except Exception as e:
            logger.error(f"Consciousness processing error: {e}")
            await asyncio.sleep(1)

# Helper Functions
async def authenticate_user(token: str) -> User:
    """Authenticate user from token"""
    # Implementation would verify JWT token
    # For now, return mock user
    return User(
        id=str(uuid.uuid4()),
        username="test_user",
        role=UserRole.USER,
        permissions=["basic_access"]
    )

async def handle_consciousness_update(client_id: str, data: dict):
    """Handle consciousness state updates from GUI"""
    # Process consciousness data and update system state
    pass

async def handle_emotional_trigger(client_id: str, data: dict):
    """Handle emotional trigger events from GUI"""
    # Process emotional triggers and generate responses
    pass

async def handle_avatar_interaction(client_id: str, data: dict):
    """Handle avatar interaction events"""
    # Process avatar interactions and update state
    pass

async def handle_benchmark_data(client_id: str, data: dict):
    """Handle benchmark data submission"""
    # Process benchmark data for performance tracking
    pass

async def collect_performance_metrics() -> dict:
    """Collect current system performance metrics"""
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "cpu_usage": 0.15,  # Mock data - would be real metrics
        "memory_usage": 0.45,
        "gpu_usage": 0.65,
        "frame_rate": 120,
        "latency": 12.5
    }

async def process_consciousness_state() -> dict:
    """Process and return consciousness state updates"""
    return {
        "consciousness_level": 0.85,
        "emotional_state": {
            "primary": "focused",
            "intensity": 0.7,
            "secondary": ["analytical", "curious"]
        },
        "processing_activity": {
            "active_processes": ["voice_recognition", "gesture_analysis"],
            "cpu_load": 0.23,
            "memory_usage": 0.45
        }
    }

async def get_consciousness_data(user_id: str) -> dict:
    """Get consciousness data for specific user"""
    return {
        "user_id": user_id,
        "consciousness_level": 0.85,
        "emotional_state": "focused",
        "last_updated": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)