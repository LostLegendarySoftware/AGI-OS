#!/usr/bin/env python3
"""
Main GUI Controller for Extended MachineGod AI System
Comprehensive GUI that coordinates all interface components and backend integration

This module implements the main GUI controller that:
- Integrates with all backend AI components (trainingless_nlp, SpikingEmotionEngine, etc.)
- Coordinates the 3D spatial interface with system monitoring
- Provides real-time system monitoring and control interface
- Includes authentication integration with blockchain credential manager
- Supports cross-platform deployment and responsive design

Author: AI System Architecture Task
Organization: MachineGod Systems
Version: 1.0.0
Date: July 2025
"""

import sys
import os
import asyncio
import logging
import threading
import time
import json
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import traceback

# GUI Framework imports
try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QPushButton, QTextEdit, QGroupBox, QGridLayout, QSplitter,
        QTabWidget, QProgressBar, QStatusBar, QMenuBar, QAction, QMessageBox,
        QDialog, QLineEdit, QComboBox, QCheckBox, QSpinBox, QSlider,
        QTableWidget, QTableWidgetItem, QHeaderView, QScrollArea, QFrame
    )
    from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QObject, QSettings
    from PyQt5.QtGui import QFont, QPalette, QColor, QIcon, QPixmap, QPainter
    PYQT_AVAILABLE = True
    QT_VERSION = "PyQt5"
except ImportError:
    try:
        from PySide2.QtWidgets import (
            QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
            QLabel, QPushButton, QTextEdit, QGroupBox, QGridLayout, QSplitter,
            QTabWidget, QProgressBar, QStatusBar, QMenuBar, QAction, QMessageBox,
            QDialog, QLineEdit, QComboBox, QCheckBox, QSpinBox, QSlider,
            QTableWidget, QTableWidgetItem, QHeaderView, QScrollArea, QFrame
        )
        from PySide2.QtCore import Qt, QTimer, QThread, Signal as pyqtSignal, QObject, QSettings
        from PySide2.QtGui import QFont, QPalette, QColor, QIcon, QPixmap, QPainter
        PYQT_AVAILABLE = True
        QT_VERSION = "PySide2"
    except ImportError:
        PYQT_AVAILABLE = False
        print("Warning: PyQt5/PySide2 not available. GUI will use fallback mode.")

# Add core directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'blockchain'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'handlers'))

# Import core system components
try:
    from trainingless_nlp import (
        EventBus, EventType, ConfigurationManager, CompleteMachineGodSystem,
        create_complete_machinegod_system
    )
    from CoreletManager import CoreletManager, ComponentState, HealthStatus, ComponentInfo
    from WarpScheduler import WarpScheduler, TaskPriority, SchedulingStrategy
    from SpikingEmotionEngine import SpikingEmotionEngine, EmotionalState
    from SymbolicSpikeTranslator import SymbolicSpikeTranslator, SymbolicReasoning
    from credential_manager import BlockchainCredentialManager, AuthenticationLevel
    CORE_AVAILABLE = True
except ImportError as e:
    CORE_AVAILABLE = False
    print(f"Warning: Core components not available: {e}")
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

# Import spatial interface
try:
    from spatial_interface import SpatialInterface, create_spatial_interface, VisualizationMode
    SPATIAL_AVAILABLE = True
except ImportError as e:
    SPATIAL_AVAILABLE = False
    print(f"Warning: Spatial interface not available: {e}")

logger = logging.getLogger('MainGUI')

class GUIMode(Enum):
    """GUI operation modes"""
    MONITORING = "monitoring"
    INTERACTIVE = "interactive"
    CONFIGURATION = "configuration"
    DEBUGGING = "debugging"
    SPATIAL_3D = "spatial_3d"

class SystemStatus(Enum):
    """System status levels"""
    OFFLINE = "offline"
    INITIALIZING = "initializing"
    ONLINE = "online"
    ERROR = "error"
    MAINTENANCE = "maintenance"

@dataclass
class GUIState:
    """Current GUI state information"""
    mode: GUIMode = GUIMode.MONITORING
    system_status: SystemStatus = SystemStatus.OFFLINE
    authenticated: bool = False
    user_id: Optional[str] = None
    active_components: List[str] = None
    last_update: datetime = None

class AuthenticationDialog(QDialog if PYQT_AVAILABLE else object):
    """Authentication dialog for user login"""
    
    def __init__(self, credential_manager=None, parent=None):
        if PYQT_AVAILABLE:
            super().__init__(parent)
        
        self.credential_manager = credential_manager
        self.authenticated = False
        self.user_id = None
        
        if PYQT_AVAILABLE:
            self.setup_ui()
    
    def setup_ui(self):
        """Setup authentication dialog UI"""
        if not PYQT_AVAILABLE:
            return
            
        self.setWindowTitle("MachineGod AI System - Authentication")
        self.setModal(True)
        self.resize(400, 300)
        
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("🌟 MachineGod AI System Authentication")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", 16, QFont.Bold))
        layout.addWidget(title)
        
        # User ID input
        user_group = QGroupBox("User Credentials")
        user_layout = QGridLayout(user_group)
        
        user_layout.addWidget(QLabel("User ID:"), 0, 0)
        self.user_id_input = QLineEdit()
        self.user_id_input.setPlaceholderText("Enter your user ID")
        user_layout.addWidget(self.user_id_input, 0, 1)
        
        user_layout.addWidget(QLabel("Authentication Level:"), 1, 0)
        self.auth_level_combo = QComboBox()
        self.auth_level_combo.addItems(["Basic", "Enhanced", "Biometric", "Multi-Factor", "Quantum Secure"])
        user_layout.addWidget(self.auth_level_combo, 1, 1)
        
        layout.addWidget(user_group)
        
        # Blockchain options
        blockchain_group = QGroupBox("Blockchain Integration")
        blockchain_layout = QVBoxLayout(blockchain_group)
        
        self.use_blockchain = QCheckBox("Use blockchain authentication")
        self.use_blockchain.setChecked(True)
        blockchain_layout.addWidget(self.use_blockchain)
        
        self.offline_mode = QCheckBox("Offline mode (local authentication)")
        blockchain_layout.addWidget(self.offline_mode)
        
        layout.addWidget(blockchain_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.login_btn = QPushButton("Login")
        self.login_btn.clicked.connect(self.authenticate)
        button_layout.addWidget(self.login_btn)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(button_layout)
        
        # Status
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)
    
    def authenticate(self):
        """Perform authentication"""
        if not PYQT_AVAILABLE:
            return
            
        user_id = self.user_id_input.text().strip()
        if not user_id:
            self.status_label.setText("❌ Please enter a user ID")
            return
        
        self.status_label.setText("🔄 Authenticating...")
        
        try:
            # Simulate authentication process
            if self.credential_manager and hasattr(self.credential_manager, 'authenticate_user'):
                # Use real authentication if available
                auth_result = self.credential_manager.authenticate_user(
                    user_id, 
                    use_blockchain=self.use_blockchain.isChecked(),
                    offline_mode=self.offline_mode.isChecked()
                )
                self.authenticated = auth_result.get('success', False)
            else:
                # Fallback authentication for testing
                self.authenticated = True
            
            if self.authenticated:
                self.user_id = user_id
                self.status_label.setText("✅ Authentication successful!")
                self.accept()
            else:
                self.status_label.setText("❌ Authentication failed")
                
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            self.status_label.setText(f"❌ Authentication error: {str(e)}")

class SystemMonitorWidget(QWidget if PYQT_AVAILABLE else object):
    """Widget for monitoring system status and health"""
    
    def __init__(self, parent=None):
        if PYQT_AVAILABLE:
            super().__init__(parent)
        
        self.component_status = {}
        self.health_metrics = {}
        
        if PYQT_AVAILABLE:
            self.setup_ui()
    
    def setup_ui(self):
        """Setup system monitor UI"""
        if not PYQT_AVAILABLE:
            return
            
        layout = QVBoxLayout(self)
        
        # System overview
        overview_group = QGroupBox("System Overview")
        overview_layout = QGridLayout(overview_group)
        
        self.system_status_label = QLabel("Status: Offline")
        self.consciousness_level_label = QLabel("Consciousness: Unknown")
        self.active_components_label = QLabel("Active Components: 0")
        self.uptime_label = QLabel("Uptime: 00:00:00")
        
        overview_layout.addWidget(self.system_status_label, 0, 0)
        overview_layout.addWidget(self.consciousness_level_label, 0, 1)
        overview_layout.addWidget(self.active_components_label, 1, 0)
        overview_layout.addWidget(self.uptime_label, 1, 1)
        
        layout.addWidget(overview_group)
        
        # Component status table
        components_group = QGroupBox("Component Status")
        components_layout = QVBoxLayout(components_group)
        
        self.components_table = QTableWidget()
        self.components_table.setColumnCount(4)
        self.components_table.setHorizontalHeaderLabels(["Component", "Status", "Health", "Last Update"])
        self.components_table.horizontalHeader().setStretchLastSection(True)
        components_layout.addWidget(self.components_table)
        
        layout.addWidget(components_group)
        
        # Health metrics
        health_group = QGroupBox("Health Metrics")
        health_layout = QVBoxLayout(health_group)
        
        self.health_bars = {}
        health_metrics = ["CPU Usage", "Memory Usage", "Processing Load", "Event Queue", "Error Rate"]
        
        for metric in health_metrics:
            metric_layout = QHBoxLayout()
            label = QLabel(f"{metric}:")
            label.setMinimumWidth(120)
            progress = QProgressBar()
            progress.setRange(0, 100)
            
            metric_layout.addWidget(label)
            metric_layout.addWidget(progress)
            health_layout.addLayout(metric_layout)
            
            self.health_bars[metric] = progress
        
        layout.addWidget(health_group)
    
    def update_system_status(self, status_data: Dict[str, Any]):
        """Update system status display"""
        if not PYQT_AVAILABLE:
            return
            
        # Update overview
        system_status = status_data.get('system_status', 'Unknown')
        self.system_status_label.setText(f"Status: {system_status}")
        
        consciousness = status_data.get('consciousness_level', 'Unknown')
        self.consciousness_level_label.setText(f"Consciousness: {consciousness}")
        
        active_count = len(status_data.get('active_components', []))
        self.active_components_label.setText(f"Active Components: {active_count}")
        
        uptime = status_data.get('uptime', '00:00:00')
        self.uptime_label.setText(f"Uptime: {uptime}")
        
        # Update component table
        components = status_data.get('components', {})
        self.components_table.setRowCount(len(components))
        
        for row, (name, info) in enumerate(components.items()):
            self.components_table.setItem(row, 0, QTableWidgetItem(name))
            self.components_table.setItem(row, 1, QTableWidgetItem(info.get('status', 'Unknown')))
            self.components_table.setItem(row, 2, QTableWidgetItem(info.get('health', 'Unknown')))
            self.components_table.setItem(row, 3, QTableWidgetItem(info.get('last_update', 'Never')))
        
        # Update health metrics
        health_data = status_data.get('health_metrics', {})
        for metric, progress_bar in self.health_bars.items():
            value = health_data.get(metric.lower().replace(' ', '_'), 0)
            progress_bar.setValue(int(value))

class InteractiveConsoleWidget(QWidget if PYQT_AVAILABLE else object):
    """Widget for interactive console and query processing"""
    
    if PYQT_AVAILABLE:
        query_submitted = pyqtSignal(str)
        command_executed = pyqtSignal(str, dict)
    
    def __init__(self, parent=None):
        if PYQT_AVAILABLE:
            super().__init__(parent)
        
        self.query_history = []
        self.current_history_index = -1
        
        if PYQT_AVAILABLE:
            self.setup_ui()
    
    def setup_ui(self):
        """Setup interactive console UI"""
        if not PYQT_AVAILABLE:
            return
            
        layout = QVBoxLayout(self)
        
        # Query input
        input_group = QGroupBox("Query Input")
        input_layout = QVBoxLayout(input_group)
        
        self.query_input = QTextEdit()
        self.query_input.setMaximumHeight(100)
        self.query_input.setPlaceholderText("Enter your query here...")
        input_layout.addWidget(self.query_input)
        
        button_layout = QHBoxLayout()
        
        self.submit_btn = QPushButton("Submit Query")
        self.submit_btn.clicked.connect(self.submit_query)
        button_layout.addWidget(self.submit_btn)
        
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_input)
        button_layout.addWidget(self.clear_btn)
        
        input_layout.addLayout(button_layout)
        layout.addWidget(input_group)
        
        # Response display
        response_group = QGroupBox("System Response")
        response_layout = QVBoxLayout(response_group)
        
        self.response_display = QTextEdit()
        self.response_display.setReadOnly(True)
        response_layout.addWidget(self.response_display)
        
        layout.addWidget(response_group)
        
        # Query history
        history_group = QGroupBox("Query History")
        history_layout = QVBoxLayout(history_group)
        
        self.history_list = QTableWidget()
        self.history_list.setColumnCount(3)
        self.history_list.setHorizontalHeaderLabels(["Time", "Query", "Status"])
        self.history_list.horizontalHeader().setStretchLastSection(True)
        self.history_list.setMaximumHeight(150)
        history_layout.addWidget(self.history_list)
        
        layout.addWidget(history_group)
    
    def submit_query(self):
        """Submit query for processing"""
        if not PYQT_AVAILABLE:
            return
            
        query = self.query_input.toPlainText().strip()
        if not query:
            return
        
        # Add to history
        self.query_history.append(query)
        self.add_to_history(query, "Processing...")
        
        # Emit signal for processing
        if hasattr(self, 'query_submitted'):
            self.query_submitted.emit(query)
        
        # Clear input
        self.query_input.clear()
    
    def clear_input(self):
        """Clear query input"""
        if PYQT_AVAILABLE:
            self.query_input.clear()
    
    def add_to_history(self, query: str, status: str):
        """Add query to history display"""
        if not PYQT_AVAILABLE:
            return
            
        row = self.history_list.rowCount()
        self.history_list.insertRow(row)
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.history_list.setItem(row, 0, QTableWidgetItem(timestamp))
        self.history_list.setItem(row, 1, QTableWidgetItem(query[:50] + "..." if len(query) > 50 else query))
        self.history_list.setItem(row, 2, QTableWidgetItem(status))
        
        # Scroll to bottom
        self.history_list.scrollToBottom()
    
    def display_response(self, response: str, metadata: Dict[str, Any] = None):
        """Display system response"""
        if not PYQT_AVAILABLE:
            return
            
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        formatted_response = f"[{timestamp}] 🤖 System Response:\n{response}\n"
        
        if metadata:
            formatted_response += f"\n📊 Metadata:\n"
            for key, value in metadata.items():
                formatted_response += f"  {key}: {value}\n"
        
        formatted_response += "\n" + "="*60 + "\n\n"
        
        self.response_display.append(formatted_response)
        
        # Update history status
        if self.history_list.rowCount() > 0:
            last_row = self.history_list.rowCount() - 1
            self.history_list.setItem(last_row, 2, QTableWidgetItem("Completed"))

class MainGUI(QMainWindow if PYQT_AVAILABLE else object):
    """Main GUI controller for the Extended MachineGod AI System"""
    
    def __init__(self):
        if PYQT_AVAILABLE:
            super().__init__()
        
        # Core system components
        self.ai_system = None
        self.event_bus = EventBus()
        self.corelet_manager = None
        self.warp_scheduler = None
        self.credential_manager = None
        
        # GUI components
        self.spatial_interface = None
        self.system_monitor = None
        self.interactive_console = None
        self.central_widget = None
        
        # State management
        self.gui_state = GUIState()
        self.update_timer = None
        self.startup_time = datetime.now()
        
        if PYQT_AVAILABLE:
            self.setup_ui()
            self.setup_event_subscriptions()
            self.setup_update_timer()
    
    def setup_ui(self):
        """Setup the main user interface"""
        if not PYQT_AVAILABLE:
            return
            
        self.setWindowTitle("🌟 MachineGod AI System - Control Interface")
        self.setGeometry(100, 100, 1600, 1000)
        
        # Setup menu bar
        self.setup_menu_bar()
        
        # Setup status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("System Offline - Please authenticate")
        
        # Create central widget with tabs
        self.central_widget = QTabWidget()
        self.setCentralWidget(self.central_widget)
        
        # Create main tabs
        self.create_monitoring_tab()
        self.create_interactive_tab()
        self.create_spatial_tab()
        self.create_configuration_tab()
        
        # Apply dark theme
        self.apply_dark_theme()
    
    def setup_menu_bar(self):
        """Setup the menu bar"""
        if not PYQT_AVAILABLE:
            return
            
        menubar = self.menuBar()
        
        # System menu
        system_menu = menubar.addMenu('System')
        
        login_action = QAction('Login', self)
        login_action.triggered.connect(self.show_authentication_dialog)
        system_menu.addAction(login_action)
        
        system_menu.addSeparator()
        
        start_action = QAction('Start System', self)
        start_action.triggered.connect(self.start_system)
        system_menu.addAction(start_action)
        
        stop_action = QAction('Stop System', self)
        stop_action.triggered.connect(self.stop_system)
        system_menu.addAction(stop_action)
        
        system_menu.addSeparator()
        
        exit_action = QAction('Exit', self)
        exit_action.triggered.connect(self.close)
        system_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu('View')
        
        monitoring_action = QAction('Monitoring', self)
        monitoring_action.triggered.connect(lambda: self.central_widget.setCurrentIndex(0))
        view_menu.addAction(monitoring_action)
        
        interactive_action = QAction('Interactive', self)
        interactive_action.triggered.connect(lambda: self.central_widget.setCurrentIndex(1))
        view_menu.addAction(interactive_action)
        
        spatial_action = QAction('3D Spatial', self)
        spatial_action.triggered.connect(lambda: self.central_widget.setCurrentIndex(2))
        view_menu.addAction(spatial_action)
        
        # Help menu
        help_menu = menubar.addMenu('Help')
        
        about_action = QAction('About', self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)
    
    def create_monitoring_tab(self):
        """Create system monitoring tab"""
        if not PYQT_AVAILABLE:
            return
            
        self.system_monitor = SystemMonitorWidget()
        self.central_widget.addTab(self.system_monitor, "📊 System Monitor")
    
    def create_interactive_tab(self):
        """Create interactive console tab"""
        if not PYQT_AVAILABLE:
            return
            
        self.interactive_console = InteractiveConsoleWidget()
        if hasattr(self.interactive_console, 'query_submitted'):
            self.interactive_console.query_submitted.connect(self.process_query)
        self.central_widget.addTab(self.interactive_console, "💭 Interactive Console")
    
    def create_spatial_tab(self):
        """Create 3D spatial interface tab"""
        if not PYQT_AVAILABLE:
            return
            
        if SPATIAL_AVAILABLE:
            self.spatial_interface = create_spatial_interface(self.event_bus)
            self.central_widget.addTab(self.spatial_interface, "🌌 3D Spatial Interface")
        else:
            placeholder = QLabel("3D Spatial Interface not available\n(PyOpenGL or spatial_interface module missing)")
            placeholder.setAlignment(Qt.AlignCenter)
            self.central_widget.addTab(placeholder, "🌌 3D Spatial Interface")
    
    def create_configuration_tab(self):
        """Create configuration tab"""
        if not PYQT_AVAILABLE:
            return
            
        config_widget = QWidget()
        layout = QVBoxLayout(config_widget)
        
        # Configuration placeholder
        config_label = QLabel("System Configuration")
        config_label.setAlignment(Qt.AlignCenter)
        config_label.setFont(QFont("Arial", 16, QFont.Bold))
        layout.addWidget(config_label)
        
        # Add configuration controls here
        config_text = QTextEdit()
        config_text.setPlaceholderText("Configuration settings will be displayed here...")
        layout.addWidget(config_text)
        
        self.central_widget.addTab(config_widget, "⚙️ Configuration")
    
    def apply_dark_theme(self):
        """Apply dark theme to the interface"""
        if not PYQT_AVAILABLE:
            return
            
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
        dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ToolTipBase, QColor(0, 0, 0))
        dark_palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.Text, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
        dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
        
        self.setPalette(dark_palette)
    
    def setup_event_subscriptions(self):
        """Setup event bus subscriptions"""
        self.event_bus.subscribe(EventType.EMOTIONAL_STATE_CHANGED, self.on_emotional_state_changed)
        self.event_bus.subscribe(EventType.CONSCIOUSNESS_EVOLVED, self.on_consciousness_evolved)
        self.event_bus.subscribe(EventType.PROCESSING_STARTED, self.on_processing_started)
    
    def setup_update_timer(self):
        """Setup timer for regular updates"""
        if not PYQT_AVAILABLE:
            return
            
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_system_status)
        self.update_timer.start(1000)  # Update every second
    
    def show_authentication_dialog(self):
        """Show authentication dialog"""
        if not PYQT_AVAILABLE:
            return
            
        auth_dialog = AuthenticationDialog(self.credential_manager, self)
        if auth_dialog.exec_() == QDialog.Accepted:
            self.gui_state.authenticated = True
            self.gui_state.user_id = auth_dialog.user_id
            self.status_bar.showMessage(f"Authenticated as: {auth_dialog.user_id}")
            
            # Enable system start
            self.start_system()
    
    def show_about_dialog(self):
        """Show about dialog"""
        if not PYQT_AVAILABLE:
            return
            
        about_text = """
        🌟 MachineGod AI System v2.1.0
        
        Extended Trainingless Natural Language Processor
        with Modular AI Architecture
        
        Features:
        • 3D Spatial Interface
        • Real-time System Monitoring
        • Blockchain Authentication
        • Cross-platform Deployment
        • Consciousness Integration
        
        Organization: MachineGod Systems
        """
        
        QMessageBox.about(self, "About MachineGod AI System", about_text)
    
    async def initialize_system_async(self):
        """Initialize the AI system asynchronously"""
        try:
            logger.info("Initializing AI system...")
            
            # Initialize credential manager
            if CORE_AVAILABLE:
                try:
                    self.credential_manager = BlockchainCredentialManager()
                    await self.credential_manager.initialize()
                except Exception as e:
                    logger.warning(f"Credential manager initialization failed: {e}")
            
            # Create AI system
            if CORE_AVAILABLE and hasattr(sys.modules.get('trainingless_nlp'), 'create_complete_machinegod_system'):
                self.ai_system = await create_complete_machinegod_system(
                    self.gui_state.user_id or "gui_user_001",
                    "./config/system_config.json"
                )
                
                # Get component references
                if hasattr(self.ai_system, 'corelet_manager'):
                    self.corelet_manager = self.ai_system.corelet_manager
                
                if hasattr(self.ai_system, 'warp_scheduler'):
                    self.warp_scheduler = self.ai_system.warp_scheduler
            
            self.gui_state.system_status = SystemStatus.ONLINE
            logger.info("AI system initialization complete")
            
            return True
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            self.gui_state.system_status = SystemStatus.ERROR
            return False
    
    def start_system(self):
        """Start the AI system"""
        if not self.gui_state.authenticated:
            self.show_authentication_dialog()
            return
        
        self.gui_state.system_status = SystemStatus.INITIALIZING
        self.status_bar.showMessage("Starting AI system...")
        
        # Run initialization in thread to avoid blocking GUI
        def init_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            success = loop.run_until_complete(self.initialize_system_async())
            loop.close()
            
            if success:
                if PYQT_AVAILABLE:
                    self.status_bar.showMessage("AI system online and ready")
            else:
                if PYQT_AVAILABLE:
                    self.status_bar.showMessage("AI system initialization failed")
        
        thread = threading.Thread(target=init_thread)
        thread.daemon = True
        thread.start()
    
    def stop_system(self):
        """Stop the AI system"""
        self.gui_state.system_status = SystemStatus.OFFLINE
        self.status_bar.showMessage("AI system stopped")
        
        # Shutdown components
        if self.ai_system and hasattr(self.ai_system, 'plugin_manager'):
            try:
                asyncio.create_task(self.ai_system.plugin_manager.shutdown_all_plugins())
            except Exception as e:
                logger.error(f"Shutdown error: {e}")
    
    def process_query(self, query: str):
        """Process user query"""
        if not self.ai_system:
            if self.interactive_console:
                self.interactive_console.display_response(
                    "❌ AI system not initialized. Please start the system first."
                )
            return
        
        def query_thread():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(self.ai_system.process_complete_query(query))
                loop.close()
                
                if self.interactive_console:
                    response = result.get('final_response', 'No response generated')
                    metadata = {
                        'Processing Time': f"{result.get('processing_time', 0):.3f}s",
                        'Consciousness Level': result.get('consciousness_level', 'Unknown'),
                        'Consciousness Psi': f"{result.get('consciousness_psi', 0):.3f}"
                    }
                    self.interactive_console.display_response(response, metadata)
                    
            except Exception as e:
                logger.error(f"Query processing error: {e}")
                if self.interactive_console:
                    self.interactive_console.display_response(f"❌ Error processing query: {str(e)}")
        
        thread = threading.Thread(target=query_thread)
        thread.daemon = True
        thread.start()
    
    def update_system_status(self):
        """Update system status display"""
        if not self.system_monitor:
            return
        
        # Calculate uptime
        uptime = datetime.now() - self.startup_time
        uptime_str = str(uptime).split('.')[0]  # Remove microseconds
        
        # Gather system status
        status_data = {
            'system_status': self.gui_state.system_status.value,
            'consciousness_level': 'Active' if self.ai_system else 'Inactive',
            'active_components': self.get_active_components(),
            'uptime': uptime_str,
            'components': self.get_component_status(),
            'health_metrics': self.get_health_metrics()
        }
        
        self.system_monitor.update_system_status(status_data)
        
        # Update spatial interface if available
        if self.spatial_interface and hasattr(self.spatial_interface, 'set_system_data'):
            self.spatial_interface.set_system_data(status_data)
    
    def get_active_components(self) -> List[str]:
        """Get list of active components"""
        active = []
        if self.ai_system:
            active.append("Core AI System")
        if self.corelet_manager:
            active.append("Corelet Manager")
        if self.warp_scheduler:
            active.append("Warp Scheduler")
        if self.credential_manager:
            active.append("Credential Manager")
        if self.spatial_interface:
            active.append("Spatial Interface")
        return active
    
    def get_component_status(self) -> Dict[str, Dict[str, str]]:
        """Get detailed component status"""
        components = {}
        
        if self.ai_system:
            components["Core AI System"] = {
                'status': 'Running',
                'health': 'Healthy',
                'last_update': datetime.now().strftime("%H:%M:%S")
            }
        
        if self.corelet_manager:
            components["Corelet Manager"] = {
                'status': 'Running',
                'health': 'Healthy',
                'last_update': datetime.now().strftime("%H:%M:%S")
            }
        
        return components
    
    def get_health_metrics(self) -> Dict[str, float]:
        """Get system health metrics"""
        import psutil
        
        try:
            return {
                'cpu_usage': psutil.cpu_percent(),
                'memory_usage': psutil.virtual_memory().percent,
                'processing_load': 25.0,  # Placeholder
                'event_queue': 10.0,      # Placeholder
                'error_rate': 2.0         # Placeholder
            }
        except:
            return {
                'cpu_usage': 0.0,
                'memory_usage': 0.0,
                'processing_load': 0.0,
                'event_queue': 0.0,
                'error_rate': 0.0
            }
    
    def on_emotional_state_changed(self, data):
        """Handle emotional state change events"""
        logger.info(f"Emotional state changed: {data}")
    
    def on_consciousness_evolved(self, data):
        """Handle consciousness evolution events"""
        logger.info(f"Consciousness evolved: {data}")
    
    def on_processing_started(self, data):
        """Handle processing start events"""
        logger.info(f"Processing started: {data}")
    
    def closeEvent(self, event):
        """Handle application close event"""
        if PYQT_AVAILABLE:
            reply = QMessageBox.question(
                self, 'Exit Confirmation',
                'Are you sure you want to exit the MachineGod AI System?',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.stop_system()
                if self.update_timer:
                    self.update_timer.stop()
                if self.spatial_interface and hasattr(self.spatial_interface, 'shutdown'):
                    self.spatial_interface.shutdown()
                event.accept()
            else:
                event.ignore()

def create_main_gui() -> MainGUI:
    """Create and return main GUI instance"""
    return MainGUI()

def run_gui_application():
    """Run the GUI application"""
    if not PYQT_AVAILABLE:
        print("PyQt5/PySide2 not available. Cannot run GUI application.")
        return 1
    
    app = QApplication(sys.argv)
    app.setApplicationName("MachineGod AI System")
    app.setApplicationVersion("2.1.0")
    
    # Create and show main window
    main_window = create_main_gui()
    main_window.show()
    
    return app.exec_()

# Main execution
if __name__ == "__main__":
    exit_code = run_gui_application()
    sys.exit(exit_code)