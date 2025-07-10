import random
from flask import Flask, render_template, jsonify, request, session

from flask_socketio import SocketIO, emit
import psutil
import torch
import os
import json
import threading
import time
import secrets
from datetime import datetime

# Import your ARIEL components
from ariel_logging import ARIELLogger
from ariel_auth import ARIELAuth
from ariel_monitor import TrainingMonitor

# Create Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = secrets.token_hex(16)
socketio = SocketIO(app)

# Initialize ARIEL components
ariel_logger = ARIELLogger(log_dir="logs")
ariel_auth = ARIELAuth(ariel_logger)
ariel_monitor = TrainingMonitor(ariel_logger)

# Global state (would store your actual training state)
training_state = {
    "running": False,
    "progress": 0,
    "current_epoch": 0,
    "total_epochs": 3,
    "loss": 0,
    "loss_history": [],
    "emotions": {
        "joy": 50,
        "trust": 50,
        "fear": 50,
        "anger": 50
    },
    "console_logs": []
}

# API routes for data exchange
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/system_info')
def system_info():
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "name": torch.cuda.get_device_name(0),
            "memory_allocated": torch.cuda.memory_allocated() / (1024**3),
            "memory_reserved": torch.cuda.memory_reserved() / (1024**3),
            "memory_total": torch.cuda.get_device_properties(0).total_memory / (1024**3)
        }
    
    disks = []
    for part in psutil.disk_partitions(all=False):
        if os.name == 'nt' or part.fstype != '':
            usage = psutil.disk_usage(part.mountpoint)
            disks.append({
                "device": part.device,
                "mountpoint": part.mountpoint,
                "used": usage.used / (1024**3),
                "total": usage.total / (1024**3),
                "percent": usage.percent
            })
    
    return jsonify({
        "cpu": cpu_percent,
        "memory": {
            "used": memory.used / (1024**3),
            "total": memory.total / (1024**3),
            "percent": memory.percent
        },
        "gpu": gpu_info,
        "disks": disks
    })

@app.route('/api/training_state')
def get_training_state():
    return jsonify(training_state)

@app.route('/api/authenticate', methods=['POST'])
def authenticate():
    data = request.json
    token = ariel_auth.authenticate(
        data.get('username'),
        data.get('password'),
        data.get('totp')
    )
    if token:
        session['token'] = token
        return jsonify({"success": True})
    else:
        return jsonify({"success": False, "message": "Authentication failed"})

@app.route('/api/start_training', methods=['POST'])
def start_training():
    # Check authentication first
    if 'token' not in session:
        return jsonify({"success": False, "message": "Not authenticated"})
    
    # Get training parameters
    config = request.json
    
    # Start training in a background thread
    threading.Thread(target=simulate_training, args=(config,)).start()
    
    return jsonify({"success": True})

# WebSocket for real-time updates
@socketio.on('connect')
def handle_connect():
    print('Client connected')

# Simulate training process (replace with actual ARIEL training)
def simulate_training(config):
    global training_state
    
    # Reset training state
    training_state["running"] = True
    training_state["progress"] = 0
    training_state["current_epoch"] = 1
    training_state["total_epochs"] = config.get("epochs", 3)
    training_state["loss"] = 10.0
    training_state["loss_history"] = []
    
    # Log start of training
    log_message("Starting training with configuration: " + json.dumps(config))
    
    # Notify clients
    socketio.emit('training_started')
    
    # Simulate training steps
    total_steps = 1000
    for step in range(1, total_steps + 1):
        if not training_state["running"]:
            break
        
        # Update progress
        training_state["progress"] = int((step / total_steps) * 100)
        
        # Update loss with decay and noise
        base_loss = 10.0 * (0.998 ** step)
        noise = (random.random() - 0.5) * 0.3
        training_state["loss"] = base_loss + noise
        
        # Record loss periodically
        if step % 10 == 0:
            training_state["loss_history"].append({
                "step": step,
                "loss": training_state["loss"]
            })
        
        # Update emotions based on progress
        update_emotions(step, total_steps)
        
        # Log progress periodically
        if step % 50 == 0:
            log_message(f"Step {step}/{total_steps} - Loss: {training_state['loss']:.4f}")
        
        # Check if we need to move to the next epoch
        steps_per_epoch = total_steps / training_state["total_epochs"]
        if step > training_state["current_epoch"] * steps_per_epoch:
            training_state["current_epoch"] += 1
            log_message(f"Completed epoch {training_state['current_epoch']-1}")
        
        # Send updates to clients
        socketio.emit('training_update', training_state)
        
        # Simulate training time
        time.sleep(0.1)  # Adjust speed as needed
    
    # Training finished
    training_state["running"] = False
    log_message("Training completed successfully")
    socketio.emit('training_completed')

# Helper function to update emotional states
def update_emotions(step, total_steps):
    loss = training_state["loss"]
    emotions = training_state["emotions"]
    
    # Update based on loss
    if loss < 3:
        emotions["joy"] = min(100, emotions["joy"] + random.random() * 2)
        emotions["trust"] = min(100, emotions["trust"] + random.random() * 1.5)
        emotions["fear"] = max(0, emotions["fear"] - random.random() * 1.5)
        emotions["anger"] = max(0, emotions["anger"] - random.random() * 1.5)
    elif loss > 7:
        emotions["joy"] = max(0, emotions["joy"] - random.random() * 2)
        emotions["trust"] = max(0, emotions["trust"] - random.random() * 1.5)
        emotions["fear"] = min(100, emotions["fear"] + random.random() * 2)
        emotions["anger"] = min(100, emotions["anger"] + random.random() * 2)
    else:
        # Small random changes
        for key in emotions:
            delta = (random.random() - 0.5) * 2
            emotions[key] = max(0, min(100, emotions[key] + delta))
    
    # Round values
    for key in emotions:
        emotions[key] = round(emotions[key])

# Helper function to add log messages
def log_message(message, level="info"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = {
        "timestamp": timestamp,
        "level": level,
        "message": message
    }
    training_state["console_logs"].append(log_entry)
    
    # Keep only the latest 100 log messages
    if len(training_state["console_logs"]) > 100:
        training_state["console_logs"] = training_state["console_logs"][-100:]
    
    # Also log to ARIEL's logger
    ariel_logger.info("training", message)

if __name__ == '__main__':
    # Create default admin user if no users exist
    if not ariel_auth.users:
        admin_secret = ariel_auth.create_user("admin", "ariel_admin", "admin")
        print(f"Created admin user with 2FA secret: {admin_secret}")
    
    # Start the Flask server
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
from flask import Flask, render_template, jsonify, request
import os
import json
import psutil

app = Flask(__name__)

# Simulated data for demonstration purposes
algorithms = {
    "ariel": {"name": "ARIEL Algorithm", "description": "Quantum-inspired LLM", "status": "Ready"},
    "quantum-hyper": {"name": "Quantum-Hyper Network", "description": "Advanced transformer", "status": "Ready"},
    "emotional-llm": {"name": "Emotional LLM", "description": "Sentiment-aware model", "status": "Ready"},
    "self-heal": {"name": "Self-Healing System", "description": "Fault-tolerant architecture", "status": "Ready"},
    "custom": {"name": "Custom Algorithm", "description": "Build from scratch", "status": "Empty"}
}

files = {
    "ariel_algorithm.py": {"content": "# ARIEL Algorithm code here", "type": "python"},
    "ariel_neural_net.py": {"content": "# Neural network implementation", "type": "python"},
    "warp_system.py": {"content": "# Warp system implementation", "type": "python"},
    "memory_manager.py": {"content": "# Memory management code", "type": "python"},
    "ariel_training.py": {"content": "# Training procedures", "type": "python"},
    "ariel_monitor.py": {"content": "# Monitoring utilities", "type": "python"},
    "ds_config_3060.json": {"content": "{\"config\": \"data\"}", "type": "json"},
    "ariel_credentials.json": {"content": "{\"credentials\": \"sensitive\"}", "type": "json"}
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/algorithms')
def get_algorithms():
    return jsonify(algorithms)

@app.route('/api/files')
def get_files():
    return jsonify(list(files.keys()))

@app.route('/api/file/<filename>')
def get_file(filename):
    if filename in files:
        return jsonify(files[filename])
    return jsonify({"error": "File not found"}), 404

@app.route('/api/file/<filename>', methods=['POST'])
def save_file(filename):
    content = request.json.get('content')
    if filename in files:
        files[filename]['content'] = content
        return jsonify({"message": "File saved successfully"})
    return jsonify({"error": "File not found"}), 404

@app.route('/api/system-resources')
def get_system_resources():
    cpu_usage = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    gpu_memory = 3.2  # This is a placeholder. You'd need to use a GPU library to get actual GPU memory usage
    return jsonify({
        "cpu_usage": cpu_usage,
        "ram_usage": memory.used / (1024 * 1024 * 1024),  # Convert to GB
        "ram_total": memory.total / (1024 * 1024 * 1024),  # Convert to GB
        "gpu_memory_used": gpu_memory,
        "gpu_memory_total": 12  # Assuming a 12GB GPU for this example
    })

@app.route('/api/run-code', methods=['POST'])
def run_code():
    code = request.json.get('code')
    # In a real application, you'd want to run this code in a sandboxed environment
    # This is just a simple simulation
    return jsonify({"output": "Code executed successfully. Output: Hello, ARIEL!"})

@app.route('/api/validate-code', methods=['POST'])
def validate_code():
    code = request.json.get('code')
    # In a real application, you'd want to actually validate the code
    # This is just a simple simulation
    return jsonify({"valid": True, "message": "Code validation passed: no syntax errors"})

if __name__ == '__main__':
    app.run(debug=True)