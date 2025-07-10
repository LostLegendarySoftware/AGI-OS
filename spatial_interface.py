#!/usr/bin/env python3
"""
3D Spatial Interface for Extended MachineGod AI System
Cross-platform 3D spatial interaction capabilities with real-time visualization

This module implements a comprehensive 3D spatial interface with:
- Cross-platform 3D visualization using PyQt/PySide with OpenGL
- Spatial navigation and real-time interaction features
- Hands-free interaction and spatially-aware interface elements
- Integration with AI components for real-time data visualization
- Support for deployment across different hardware platforms

Author: AI System Architecture Task
Organization: MachineGod Systems
Version: 1.0.0
Date: July 2025
"""

import sys
import os
import asyncio
import logging
import math
import time
import threading
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np

# GUI Framework imports based on research
try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QPushButton, QSlider, QTextEdit, QGroupBox, QGridLayout,
        QSplitter, QFrame, QProgressBar, QTabWidget, QScrollArea
    )
    from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QObject
    from PyQt5.QtGui import QFont, QPalette, QColor, QPainter, QPixmap
    from PyQt5.QtOpenGL import QOpenGLWidget
    PYQT_AVAILABLE = True
    QT_VERSION = "PyQt5"
except ImportError:
    try:
        from PySide2.QtWidgets import (
            QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
            QLabel, QPushButton, QSlider, QTextEdit, QGroupBox, QGridLayout,
            QSplitter, QFrame, QProgressBar, QTabWidget, QScrollArea
        )
        from PySide2.QtCore import Qt, QTimer, QThread, Signal as pyqtSignal, QObject
        from PySide2.QtGui import QFont, QPalette, QColor, QPainter, QPixmap
        from PySide2.QtOpenGL import QOpenGLWidget
        PYQT_AVAILABLE = True
        QT_VERSION = "PySide2"
    except ImportError:
        PYQT_AVAILABLE = False
        print("Warning: PyQt5/PySide2 not available. 3D interface will use fallback mode.")

# OpenGL imports for 3D rendering
try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    import OpenGL.GL.shaders as shaders
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False
    print("Warning: PyOpenGL not available. 3D rendering will be limited.")

# Add core directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

try:
    from trainingless_nlp import EventBus, EventType
    from CoreletManager import ComponentState, HealthStatus
except ImportError as e:
    print(f"Warning: Core modules not available: {e}")
    # Define minimal interfaces for fallback
    class EventBus:
        def __init__(self):
            self.subscribers = {}
        def subscribe(self, event_type, callback):
            pass
        def publish(self, event_type, data):
            pass
    
    class EventType:
        QUERY_RECEIVED = "query_received"
        PROCESSING_STARTED = "processing_started"
        EMOTIONAL_STATE_CHANGED = "emotional_state_changed"
        CONSCIOUSNESS_EVOLVED = "consciousness_evolved"

logger = logging.getLogger('SpatialInterface')

class SpatialInteractionMode(Enum):
    """3D spatial interaction modes"""
    NAVIGATION = "navigation"
    SELECTION = "selection"
    MANIPULATION = "manipulation"
    VISUALIZATION = "visualization"
    HANDS_FREE = "hands_free"

class VisualizationMode(Enum):
    """Data visualization modes"""
    CONSCIOUSNESS_FLOW = "consciousness_flow"
    EMOTIONAL_LANDSCAPE = "emotional_landscape"
    SYMBOLIC_NETWORK = "symbolic_network"
    SYSTEM_HEALTH = "system_health"
    REAL_TIME_DATA = "real_time_data"

@dataclass
class SpatialPoint:
    """3D spatial point with metadata"""
    x: float
    y: float
    z: float
    data: Dict[str, Any] = None
    color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    size: float = 1.0

@dataclass
class SpatialObject:
    """3D spatial object for visualization"""
    id: str
    position: SpatialPoint
    vertices: List[SpatialPoint]
    object_type: str
    metadata: Dict[str, Any] = None
    visible: bool = True
    interactive: bool = True

class OpenGL3DWidget(QOpenGLWidget if PYQT_AVAILABLE and OPENGL_AVAILABLE else QWidget):
    """3D OpenGL widget for spatial visualization"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.rotation_x = 0
        self.rotation_y = 0
        self.rotation_z = 0
        self.zoom = -5.0
        self.objects = []
        self.interaction_mode = SpatialInteractionMode.NAVIGATION
        self.visualization_mode = VisualizationMode.CONSCIOUSNESS_FLOW
        self.last_mouse_pos = None
        self.animation_timer = QTimer() if PYQT_AVAILABLE else None
        
        if self.animation_timer:
            self.animation_timer.timeout.connect(self.update_animation)
            self.animation_timer.start(16)  # ~60 FPS
    
    def initializeGL(self):
        """Initialize OpenGL context"""
        if not OPENGL_AVAILABLE:
            return
            
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(0.1, 0.1, 0.2, 1.0)  # Dark blue background
        
        # Setup lighting
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, [1.0, 1.0, 1.0, 0.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
    
    def resizeGL(self, width, height):
        """Handle window resize"""
        if not OPENGL_AVAILABLE:
            return
            
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, width / height if height > 0 else 1, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
    
    def paintGL(self):
        """Render the 3D scene"""
        if not OPENGL_AVAILABLE:
            return
            
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Apply camera transformations
        glTranslatef(0.0, 0.0, self.zoom)
        glRotatef(self.rotation_x, 1.0, 0.0, 0.0)
        glRotatef(self.rotation_y, 0.0, 1.0, 0.0)
        glRotatef(self.rotation_z, 0.0, 0.0, 1.0)
        
        # Render based on visualization mode
        if self.visualization_mode == VisualizationMode.CONSCIOUSNESS_FLOW:
            self.render_consciousness_flow()
        elif self.visualization_mode == VisualizationMode.EMOTIONAL_LANDSCAPE:
            self.render_emotional_landscape()
        elif self.visualization_mode == VisualizationMode.SYMBOLIC_NETWORK:
            self.render_symbolic_network()
        elif self.visualization_mode == VisualizationMode.SYSTEM_HEALTH:
            self.render_system_health()
        else:
            self.render_default_scene()
    
    def render_consciousness_flow(self):
        """Render consciousness flow visualization"""
        if not OPENGL_AVAILABLE:
            return
            
        # Render flowing particles representing consciousness
        glDisable(GL_LIGHTING)
        glBegin(GL_POINTS)
        
        current_time = time.time()
        for i in range(100):
            # Create flowing particle effect
            t = (current_time + i * 0.1) % (2 * math.pi)
            x = math.sin(t) * 2
            y = math.cos(t * 1.5) * 1.5
            z = math.sin(t * 0.7) * 1
            
            # Color based on consciousness level
            intensity = (math.sin(t * 2) + 1) * 0.5
            glColor4f(0.3 + intensity * 0.7, 0.1 + intensity * 0.5, 0.8, 0.8)
            glVertex3f(x, y, z)
        
        glEnd()
        glEnable(GL_LIGHTING)
    
    def render_emotional_landscape(self):
        """Render emotional landscape visualization"""
        if not OPENGL_AVAILABLE:
            return
            
        # Render terrain-like emotional landscape
        glColor4f(0.8, 0.4, 0.2, 0.8)
        
        for x in range(-5, 6):
            glBegin(GL_TRIANGLE_STRIP)
            for z in range(-5, 6):
                # Create height based on emotional intensity
                height1 = math.sin(x * 0.5) * math.cos(z * 0.5) * 0.5
                height2 = math.sin((x + 1) * 0.5) * math.cos(z * 0.5) * 0.5
                
                glVertex3f(x, height1, z)
                glVertex3f(x + 1, height2, z)
            glEnd()
    
    def render_symbolic_network(self):
        """Render symbolic reasoning network"""
        if not OPENGL_AVAILABLE:
            return
            
        # Render network nodes and connections
        glDisable(GL_LIGHTING)
        
        # Draw nodes
        glColor4f(0.2, 0.8, 0.2, 0.9)
        for i in range(20):
            angle = i * 2 * math.pi / 20
            x = math.cos(angle) * 2
            y = math.sin(angle) * 2
            z = math.sin(i * 0.5) * 0.5
            
            glPushMatrix()
            glTranslatef(x, y, z)
            self.render_sphere(0.1, 8, 8)
            glPopMatrix()
        
        # Draw connections
        glBegin(GL_LINES)
        glColor4f(0.5, 0.5, 0.8, 0.6)
        for i in range(20):
            for j in range(i + 1, min(i + 4, 20)):
                angle1 = i * 2 * math.pi / 20
                angle2 = j * 2 * math.pi / 20
                
                x1 = math.cos(angle1) * 2
                y1 = math.sin(angle1) * 2
                z1 = math.sin(i * 0.5) * 0.5
                
                x2 = math.cos(angle2) * 2
                y2 = math.sin(angle2) * 2
                z2 = math.sin(j * 0.5) * 0.5
                
                glVertex3f(x1, y1, z1)
                glVertex3f(x2, y2, z2)
        glEnd()
        
        glEnable(GL_LIGHTING)
    
    def render_system_health(self):
        """Render system health visualization"""
        if not OPENGL_AVAILABLE:
            return
            
        # Render health status as colored cubes
        components = [
            ("Core", 0.9, (0, 2, 0)),
            ("Emotion", 0.8, (-2, 0, 0)),
            ("Symbolic", 0.85, (2, 0, 0)),
            ("Scheduler", 0.95, (0, -2, 0)),
            ("Memory", 0.75, (0, 0, 2))
        ]
        
        for name, health, pos in components:
            glPushMatrix()
            glTranslatef(*pos)
            
            # Color based on health
            if health > 0.8:
                glColor4f(0.2, 0.8, 0.2, 0.8)  # Green
            elif health > 0.6:
                glColor4f(0.8, 0.8, 0.2, 0.8)  # Yellow
            else:
                glColor4f(0.8, 0.2, 0.2, 0.8)  # Red
            
            self.render_cube(0.5)
            glPopMatrix()
    
    def render_default_scene(self):
        """Render default 3D scene"""
        if not OPENGL_AVAILABLE:
            return
            
        # Render a simple rotating cube
        glColor4f(0.5, 0.5, 0.8, 0.8)
        self.render_cube(1.0)
    
    def render_cube(self, size):
        """Render a cube with given size"""
        if not OPENGL_AVAILABLE:
            return
            
        s = size / 2
        glBegin(GL_QUADS)
        
        # Front face
        glNormal3f(0, 0, 1)
        glVertex3f(-s, -s, s)
        glVertex3f(s, -s, s)
        glVertex3f(s, s, s)
        glVertex3f(-s, s, s)
        
        # Back face
        glNormal3f(0, 0, -1)
        glVertex3f(-s, -s, -s)
        glVertex3f(-s, s, -s)
        glVertex3f(s, s, -s)
        glVertex3f(s, -s, -s)
        
        # Top face
        glNormal3f(0, 1, 0)
        glVertex3f(-s, s, -s)
        glVertex3f(-s, s, s)
        glVertex3f(s, s, s)
        glVertex3f(s, s, -s)
        
        # Bottom face
        glNormal3f(0, -1, 0)
        glVertex3f(-s, -s, -s)
        glVertex3f(s, -s, -s)
        glVertex3f(s, -s, s)
        glVertex3f(-s, -s, s)
        
        # Right face
        glNormal3f(1, 0, 0)
        glVertex3f(s, -s, -s)
        glVertex3f(s, s, -s)
        glVertex3f(s, s, s)
        glVertex3f(s, -s, s)
        
        # Left face
        glNormal3f(-1, 0, 0)
        glVertex3f(-s, -s, -s)
        glVertex3f(-s, -s, s)
        glVertex3f(-s, s, s)
        glVertex3f(-s, s, -s)
        
        glEnd()
    
    def render_sphere(self, radius, slices, stacks):
        """Render a sphere with given parameters"""
        if not OPENGL_AVAILABLE:
            return
            
        for i in range(stacks):
            lat0 = math.pi * (-0.5 + float(i) / stacks)
            z0 = math.sin(lat0) * radius
            zr0 = math.cos(lat0) * radius
            
            lat1 = math.pi * (-0.5 + float(i + 1) / stacks)
            z1 = math.sin(lat1) * radius
            zr1 = math.cos(lat1) * radius
            
            glBegin(GL_QUAD_STRIP)
            for j in range(slices + 1):
                lng = 2 * math.pi * float(j) / slices
                x = math.cos(lng)
                y = math.sin(lng)
                
                glNormal3f(x * zr0, y * zr0, z0)
                glVertex3f(x * zr0, y * zr0, z0)
                glNormal3f(x * zr1, y * zr1, z1)
                glVertex3f(x * zr1, y * zr1, z1)
            glEnd()
    
    def update_animation(self):
        """Update animation frame"""
        self.rotation_y += 1.0
        if self.rotation_y >= 360:
            self.rotation_y = 0
        self.update()
    
    def mousePressEvent(self, event):
        """Handle mouse press events"""
        if PYQT_AVAILABLE:
            self.last_mouse_pos = event.pos()
    
    def mouseMoveEvent(self, event):
        """Handle mouse move events for 3D navigation"""
        if not PYQT_AVAILABLE or not self.last_mouse_pos:
            return
            
        dx = event.x() - self.last_mouse_pos.x()
        dy = event.y() - self.last_mouse_pos.y()
        
        if event.buttons() & Qt.LeftButton:
            self.rotation_x += dy * 0.5
            self.rotation_y += dx * 0.5
        elif event.buttons() & Qt.RightButton:
            self.zoom += dy * 0.1
            self.zoom = max(-20, min(-1, self.zoom))
        
        self.last_mouse_pos = event.pos()
        self.update()
    
    def wheelEvent(self, event):
        """Handle mouse wheel events for zooming"""
        if PYQT_AVAILABLE:
            delta = event.angleDelta().y() / 120.0
            self.zoom += delta * 0.5
            self.zoom = max(-20, min(-1, self.zoom))
            self.update()
    
    def set_visualization_mode(self, mode: VisualizationMode):
        """Set the visualization mode"""
        self.visualization_mode = mode
        self.update()
    
    def set_interaction_mode(self, mode: SpatialInteractionMode):
        """Set the interaction mode"""
        self.interaction_mode = mode
    
    def add_spatial_object(self, obj: SpatialObject):
        """Add a spatial object to the scene"""
        self.objects.append(obj)
        self.update()
    
    def remove_spatial_object(self, obj_id: str):
        """Remove a spatial object from the scene"""
        self.objects = [obj for obj in self.objects if obj.id != obj_id]
        self.update()
    
    def update_spatial_data(self, data: Dict[str, Any]):
        """Update spatial visualization with new data"""
        # This method will be called by the main GUI to update 3D visualization
        # with real-time data from AI components
        pass

class SpatialInterface(QWidget if PYQT_AVAILABLE else object):
    """Main 3D spatial interface widget"""
    
    # Signals for Qt integration
    if PYQT_AVAILABLE:
        data_updated = pyqtSignal(dict)
        interaction_detected = pyqtSignal(str, dict)
    
    def __init__(self, event_bus: EventBus = None, parent=None):
        if PYQT_AVAILABLE:
            super().__init__(parent)
        
        self.event_bus = event_bus or EventBus()
        self.opengl_widget = None
        self.control_panel = None
        self.status_panel = None
        self.interaction_mode = SpatialInteractionMode.NAVIGATION
        self.visualization_mode = VisualizationMode.CONSCIOUSNESS_FLOW
        self.real_time_data = {}
        
        if PYQT_AVAILABLE:
            self.setup_ui()
            self.setup_event_subscriptions()
            self.setup_update_timer()
    
    def setup_ui(self):
        """Setup the user interface"""
        if not PYQT_AVAILABLE:
            return
            
        layout = QHBoxLayout(self)
        
        # Create main 3D view
        self.opengl_widget = OpenGL3DWidget()
        layout.addWidget(self.opengl_widget, 3)  # 3/4 of the space
        
        # Create control panel
        self.control_panel = self.create_control_panel()
        layout.addWidget(self.control_panel, 1)  # 1/4 of the space
    
    def create_control_panel(self):
        """Create the control panel for spatial interface"""
        if not PYQT_AVAILABLE:
            return None
            
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Visualization mode controls
        viz_group = QGroupBox("Visualization Mode")
        viz_layout = QVBoxLayout(viz_group)
        
        viz_buttons = [
            ("Consciousness Flow", VisualizationMode.CONSCIOUSNESS_FLOW),
            ("Emotional Landscape", VisualizationMode.EMOTIONAL_LANDSCAPE),
            ("Symbolic Network", VisualizationMode.SYMBOLIC_NETWORK),
            ("System Health", VisualizationMode.SYSTEM_HEALTH),
            ("Real-time Data", VisualizationMode.REAL_TIME_DATA)
        ]
        
        for name, mode in viz_buttons:
            btn = QPushButton(name)
            btn.clicked.connect(lambda checked, m=mode: self.set_visualization_mode(m))
            viz_layout.addWidget(btn)
        
        layout.addWidget(viz_group)
        
        # Interaction mode controls
        interaction_group = QGroupBox("Interaction Mode")
        interaction_layout = QVBoxLayout(interaction_group)
        
        interaction_buttons = [
            ("Navigation", SpatialInteractionMode.NAVIGATION),
            ("Selection", SpatialInteractionMode.SELECTION),
            ("Manipulation", SpatialInteractionMode.MANIPULATION),
            ("Hands-free", SpatialInteractionMode.HANDS_FREE)
        ]
        
        for name, mode in interaction_buttons:
            btn = QPushButton(name)
            btn.clicked.connect(lambda checked, m=mode: self.set_interaction_mode(m))
            interaction_layout.addWidget(btn)
        
        layout.addWidget(interaction_group)
        
        # Camera controls
        camera_group = QGroupBox("Camera Controls")
        camera_layout = QVBoxLayout(camera_group)
        
        # Zoom slider
        zoom_label = QLabel("Zoom")
        zoom_slider = QSlider(Qt.Horizontal)
        zoom_slider.setRange(-200, -10)
        zoom_slider.setValue(-50)
        zoom_slider.valueChanged.connect(self.on_zoom_changed)
        camera_layout.addWidget(zoom_label)
        camera_layout.addWidget(zoom_slider)
        
        # Reset button
        reset_btn = QPushButton("Reset View")
        reset_btn.clicked.connect(self.reset_camera)
        camera_layout.addWidget(reset_btn)
        
        layout.addWidget(camera_group)
        
        # Status display
        self.status_panel = QTextEdit()
        self.status_panel.setMaximumHeight(150)
        self.status_panel.setReadOnly(True)
        layout.addWidget(QLabel("System Status"))
        layout.addWidget(self.status_panel)
        
        layout.addStretch()
        return panel
    
    def setup_event_subscriptions(self):
        """Setup event bus subscriptions"""
        if not self.event_bus:
            return
            
        # Subscribe to relevant events
        self.event_bus.subscribe(EventType.EMOTIONAL_STATE_CHANGED, self.on_emotional_state_changed)
        self.event_bus.subscribe(EventType.CONSCIOUSNESS_EVOLVED, self.on_consciousness_evolved)
        self.event_bus.subscribe(EventType.PROCESSING_STARTED, self.on_processing_started)
    
    def setup_update_timer(self):
        """Setup timer for real-time updates"""
        if not PYQT_AVAILABLE:
            return
            
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_real_time_data)
        self.update_timer.start(100)  # Update every 100ms
    
    def set_visualization_mode(self, mode: VisualizationMode):
        """Set the visualization mode"""
        self.visualization_mode = mode
        if self.opengl_widget:
            self.opengl_widget.set_visualization_mode(mode)
        
        if self.status_panel:
            self.status_panel.append(f"Visualization mode changed to: {mode.value}")
    
    def set_interaction_mode(self, mode: SpatialInteractionMode):
        """Set the interaction mode"""
        self.interaction_mode = mode
        if self.opengl_widget:
            self.opengl_widget.set_interaction_mode(mode)
        
        if self.status_panel:
            self.status_panel.append(f"Interaction mode changed to: {mode.value}")
    
    def on_zoom_changed(self, value):
        """Handle zoom slider changes"""
        if self.opengl_widget:
            self.opengl_widget.zoom = value / 10.0
            self.opengl_widget.update()
    
    def reset_camera(self):
        """Reset camera to default position"""
        if self.opengl_widget:
            self.opengl_widget.rotation_x = 0
            self.opengl_widget.rotation_y = 0
            self.opengl_widget.rotation_z = 0
            self.opengl_widget.zoom = -5.0
            self.opengl_widget.update()
    
    def on_emotional_state_changed(self, data):
        """Handle emotional state change events"""
        self.real_time_data['emotional_state'] = data
        if self.visualization_mode == VisualizationMode.EMOTIONAL_LANDSCAPE:
            self.update_visualization()
    
    def on_consciousness_evolved(self, data):
        """Handle consciousness evolution events"""
        self.real_time_data['consciousness'] = data
        if self.visualization_mode == VisualizationMode.CONSCIOUSNESS_FLOW:
            self.update_visualization()
    
    def on_processing_started(self, data):
        """Handle processing start events"""
        if self.status_panel:
            self.status_panel.append(f"Processing started: {data.get('query', 'Unknown')}")
    
    def update_real_time_data(self):
        """Update real-time data visualization"""
        if self.opengl_widget and self.visualization_mode == VisualizationMode.REAL_TIME_DATA:
            self.opengl_widget.update_spatial_data(self.real_time_data)
    
    def update_visualization(self):
        """Update the 3D visualization"""
        if self.opengl_widget:
            self.opengl_widget.update()
    
    def add_spatial_object(self, obj: SpatialObject):
        """Add a spatial object to the visualization"""
        if self.opengl_widget:
            self.opengl_widget.add_spatial_object(obj)
    
    def remove_spatial_object(self, obj_id: str):
        """Remove a spatial object from the visualization"""
        if self.opengl_widget:
            self.opengl_widget.remove_spatial_object(obj_id)
    
    def get_interaction_data(self) -> Dict[str, Any]:
        """Get current interaction data"""
        return {
            'interaction_mode': self.interaction_mode.value,
            'visualization_mode': self.visualization_mode.value,
            'real_time_data': self.real_time_data,
            'camera_position': {
                'rotation_x': self.opengl_widget.rotation_x if self.opengl_widget else 0,
                'rotation_y': self.opengl_widget.rotation_y if self.opengl_widget else 0,
                'rotation_z': self.opengl_widget.rotation_z if self.opengl_widget else 0,
                'zoom': self.opengl_widget.zoom if self.opengl_widget else -5.0
            }
        }
    
    def set_system_data(self, data: Dict[str, Any]):
        """Set system data for visualization"""
        self.real_time_data.update(data)
        self.update_visualization()
    
    def enable_hands_free_mode(self):
        """Enable hands-free interaction mode"""
        self.set_interaction_mode(SpatialInteractionMode.HANDS_FREE)
        # Additional hands-free setup would go here
        # (e.g., voice recognition, gesture detection)
    
    def shutdown(self):
        """Shutdown the spatial interface"""
        if hasattr(self, 'update_timer') and self.update_timer:
            self.update_timer.stop()
        
        logger.info("Spatial interface shutdown complete")

# Factory function for creating spatial interface
def create_spatial_interface(event_bus: EventBus = None, parent=None) -> SpatialInterface:
    """Create and return a spatial interface instance"""
    if not PYQT_AVAILABLE:
        logger.warning("PyQt/PySide not available. Creating minimal spatial interface.")
        return SpatialInterface(event_bus)
    
    return SpatialInterface(event_bus, parent)

# Main execution for testing
if __name__ == "__main__":
    if PYQT_AVAILABLE:
        app = QApplication(sys.argv)
        
        # Create test window
        window = QMainWindow()
        window.setWindowTitle("3D Spatial Interface Test")
        window.setGeometry(100, 100, 1200, 800)
        
        # Create spatial interface
        spatial_interface = create_spatial_interface()
        window.setCentralWidget(spatial_interface)
        
        window.show()
        sys.exit(app.exec_())
    else:
        print("PyQt/PySide not available. Cannot run GUI test.")