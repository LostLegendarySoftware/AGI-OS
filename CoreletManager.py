#!/usr/bin/env python3
"""
🔧 CoreletManager - Component Lifecycle Management System
Part of the Extended MachineGod AGI System

This module implements a comprehensive component lifecycle management system
for coordinating all AI subsystems including trainingless_nlp, SpikingEmotionEngine,
SymbolicSpikeTranslator, and specialized handlers.

Features:
- Component registration and dependency management
- Lifecycle coordination (initialization, startup, shutdown)
- Health monitoring and resource management
- Inter-module communication coordination
- Dynamic loading/unloading of components
- Integration with existing plugin architecture and EventBus

Author: AI System Architecture Task
Organization: MachineGod Systems
License: Proprietary
Version: 1.0.0
Date: July 2025
"""

import asyncio
import logging
import time
import threading
import weakref
import psutil
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable, Set, Union
from dataclasses import dataclass, asdict, field
from collections import defaultdict, deque
from enum import Enum, IntEnum
import uuid
import json
import traceback
from datetime import datetime, timedelta

try:
    from .trainingless_nlp import (
        PluginInterface, EventBus, EventType, ConfigurationManager,
        EmotionalProcessingHook, SymbolicReasoningHook, SchedulingHook
    )
except ImportError:
    from trainingless_nlp import (
        PluginInterface, EventBus, EventType, ConfigurationManager,
        EmotionalProcessingHook, SymbolicReasoningHook, SchedulingHook
    )

logger = logging.getLogger('CoreletManager')

# ========================================
# COMPONENT LIFECYCLE MANAGEMENT
# ========================================

class ComponentState(Enum):
    """Component lifecycle states"""
    UNREGISTERED = "unregistered"
    REGISTERED = "registered"
    INITIALIZING = "initializing"
    INITIALIZED = "initialized"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    FAILED = "failed"

class ComponentType(Enum):
    """Types of system components"""
    CORE_ENGINE = "core_engine"
    PROCESSING_ENGINE = "processing_engine"
    HANDLER = "handler"
    PLUGIN = "plugin"
    SERVICE = "service"
    INTERFACE = "interface"

class HealthStatus(Enum):
    """Component health status"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

@dataclass
class ComponentInfo:
    """Information about a registered component"""
    id: str
    name: str
    component_type: ComponentType
    version: str = "1.0.0"
    description: str = ""
    dependencies: List[str] = field(default_factory=list)
    provides: List[str] = field(default_factory=list)
    state: ComponentState = ComponentState.UNREGISTERED
    health_status: HealthStatus = HealthStatus.UNKNOWN
    
    # Lifecycle management
    instance: Any = None
    initialization_time: Optional[float] = None
    startup_time: Optional[float] = None
    last_health_check: Optional[float] = None
    
    # Resource tracking
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    thread_count: int = 0
    
    # Configuration and metadata
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Error tracking
    error_count: int = 0
    last_error: Optional[str] = None
    last_error_time: Optional[float] = None

@dataclass
class HealthMetrics:
    """System health metrics"""
    timestamp: float
    total_components: int
    healthy_components: int
    warning_components: int
    critical_components: int
    failed_components: int
    system_cpu_usage: float
    system_memory_usage: float
    system_uptime: float

class CoreletManager:
    """
    Advanced component lifecycle management system for AI subsystem coordination.
    
    Manages the complete lifecycle of all AI components including:
    - trainingless_nlp (core NLP processor)
    - SpikingEmotionEngine (emotional processing)
    - SymbolicSpikeTranslator (symbolic reasoning)
    - emotion_reflex and symbolic_mismatch handlers
    - Dynamic plugins and services
    """
    
    def __init__(self, event_bus: EventBus, config_manager: ConfigurationManager):
        self.event_bus = event_bus
        self.config_manager = config_manager
        
        # Component registry
        self.components: Dict[str, ComponentInfo] = {}
        self.component_instances: Dict[str, Any] = {}
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        
        # Lifecycle management
        self.startup_order: List[str] = []
        self.shutdown_order: List[str] = []
        self.lifecycle_lock = threading.RLock()
        
        # Health monitoring
        self.health_monitor_active = False
        self.health_check_interval = 30.0  # seconds
        self.health_history: deque = deque(maxlen=100)
        self.health_monitor_task: Optional[asyncio.Task] = None
        
        # Resource management
        self.resource_limits = {
            'max_cpu_per_component': 50.0,  # percentage
            'max_memory_per_component': 1024 * 1024 * 1024,  # 1GB
            'max_threads_per_component': 10
        }
        
        # Communication coordination
        self.message_bus: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.communication_lock = threading.RLock()
        
        # System state
        self.system_start_time = time.time()
        self.is_shutting_down = False
        
        # Subscribe to system events
        self._setup_event_subscriptions()
        
        logger.info("CoreletManager initialized")
    
    def _setup_event_subscriptions(self):
        """Setup event subscriptions for system coordination"""
        self.event_bus.subscribe(EventType.PLUGIN_LOADED, self._handle_plugin_loaded)
        self.event_bus.subscribe(EventType.PLUGIN_ERROR, self._handle_plugin_error)
        self.event_bus.subscribe(EventType.SYSTEM_HEALTH_CHECK, self._handle_health_check_request)
    
    async def _handle_plugin_loaded(self, event: Dict):
        """Handle plugin loaded events"""
        plugin_name = event['data'].get('plugin_name')
        if plugin_name and plugin_name in self.components:
            component = self.components[plugin_name]
            component.state = ComponentState.RUNNING
            logger.info(f"Component {plugin_name} transitioned to RUNNING state")
    
    async def _handle_plugin_error(self, event: Dict):
        """Handle plugin error events"""
        plugin_name = event['data'].get('plugin_name')
        error = event['data'].get('error', 'Unknown error')
        
        if plugin_name and plugin_name in self.components:
            component = self.components[plugin_name]
            component.state = ComponentState.ERROR
            component.error_count += 1
            component.last_error = error
            component.last_error_time = time.time()
            component.health_status = HealthStatus.CRITICAL
            
            logger.error(f"Component {plugin_name} encountered error: {error}")
            
            # Attempt recovery if configured
            if self.config_manager.get(f"components.{plugin_name}.auto_recovery", False):
                await self._attempt_component_recovery(plugin_name)
    
    async def _handle_health_check_request(self, event: Dict):
        """Handle health check requests"""
        await self.perform_health_check()
    
    def register_component(self, 
                          component_id: str,
                          name: str,
                          component_type: ComponentType,
                          instance: Any,
                          version: str = "1.0.0",
                          description: str = "",
                          dependencies: List[str] = None,
                          provides: List[str] = None,
                          config: Dict[str, Any] = None) -> bool:
        """
        Register a component with the lifecycle manager
        
        Args:
            component_id: Unique identifier for the component
            name: Human-readable name
            component_type: Type of component
            instance: The actual component instance
            version: Component version
            description: Component description
            dependencies: List of component IDs this component depends on
            provides: List of services this component provides
            config: Component-specific configuration
            
        Returns:
            bool: True if registration successful
        """
        try:
            with self.lifecycle_lock:
                if component_id in self.components:
                    logger.warning(f"Component {component_id} already registered, updating...")
                
                component_info = ComponentInfo(
                    id=component_id,
                    name=name,
                    component_type=component_type,
                    version=version,
                    description=description,
                    dependencies=dependencies or [],
                    provides=provides or [],
                    instance=instance,
                    config=config or {},
                    state=ComponentState.REGISTERED
                )
                
                self.components[component_id] = component_info
                self.component_instances[component_id] = instance
                
                # Update dependency graphs
                self._update_dependency_graphs(component_id, dependencies or [])
                
                # Resolve startup order
                self._resolve_startup_order()
                
                logger.info(f"Component registered: {name} ({component_id}) v{version}")
                
                # Publish registration event
                asyncio.create_task(self.event_bus.publish(EventType.PLUGIN_LOADED, {
                    'component_id': component_id,
                    'name': name,
                    'type': component_type.value,
                    'version': version
                }))
                
                return True
                
        except Exception as e:
            logger.error(f"Component registration failed for {component_id}: {e}")
            return False
    
    def _update_dependency_graphs(self, component_id: str, dependencies: List[str]):
        """Update dependency graphs for startup/shutdown ordering"""
        # Clear existing dependencies for this component
        for dep in list(self.dependency_graph[component_id]):
            self.reverse_dependency_graph[dep].discard(component_id)
        self.dependency_graph[component_id].clear()
        
        # Add new dependencies
        for dep in dependencies:
            self.dependency_graph[component_id].add(dep)
            self.reverse_dependency_graph[dep].add(component_id)
    
    def _resolve_startup_order(self):
        """Resolve component startup order using topological sort"""
        try:
            visited = set()
            temp_visited = set()
            self.startup_order = []
            
            def visit(component_id: str):
                if component_id in temp_visited:
                    raise ValueError(f"Circular dependency detected involving {component_id}")
                if component_id in visited:
                    return
                
                temp_visited.add(component_id)
                
                # Visit dependencies first
                for dep in self.dependency_graph.get(component_id, []):
                    if dep in self.components:
                        visit(dep)
                    else:
                        logger.warning(f"Missing dependency {dep} for component {component_id}")
                
                temp_visited.remove(component_id)
                visited.add(component_id)
                self.startup_order.append(component_id)
            
            # Visit all components
            for component_id in self.components:
                if component_id not in visited:
                    visit(component_id)
            
            # Shutdown order is reverse of startup order
            self.shutdown_order = list(reversed(self.startup_order))
            
            logger.info(f"Component startup order resolved: {self.startup_order}")
            
        except Exception as e:
            logger.error(f"Failed to resolve startup order: {e}")
            # Fallback to registration order
            self.startup_order = list(self.components.keys())
            self.shutdown_order = list(reversed(self.startup_order))
    
    async def initialize_all_components(self) -> bool:
        """Initialize all registered components in dependency order"""
        logger.info("Starting component initialization sequence")
        
        success_count = 0
        total_count = len(self.startup_order)
        
        for component_id in self.startup_order:
            if component_id not in self.components:
                continue
                
            component = self.components[component_id]
            
            try:
                logger.info(f"Initializing component: {component.name}")
                component.state = ComponentState.INITIALIZING
                
                start_time = time.time()
                
                # Initialize component if it has an initialize method
                if hasattr(component.instance, 'initialize'):
                    if asyncio.iscoroutinefunction(component.instance.initialize):
                        result = await component.instance.initialize(component.config)
                    else:
                        result = component.instance.initialize(component.config)
                    
                    if result is False:
                        raise Exception("Component initialization returned False")
                
                component.initialization_time = time.time() - start_time
                component.state = ComponentState.INITIALIZED
                component.health_status = HealthStatus.HEALTHY
                success_count += 1
                
                logger.info(f"Component initialized: {component.name} ({component.initialization_time:.2f}s)")
                
            except Exception as e:
                component.state = ComponentState.FAILED
                component.health_status = HealthStatus.CRITICAL
                component.error_count += 1
                component.last_error = str(e)
                component.last_error_time = time.time()
                
                logger.error(f"Component initialization failed: {component.name} - {e}")
                
                # Check if this is a critical component
                if component.component_type == ComponentType.CORE_ENGINE:
                    logger.critical(f"Critical component {component.name} failed to initialize")
                    return False
        
        logger.info(f"Component initialization complete: {success_count}/{total_count} successful")
        return success_count > 0
    
    async def start_all_components(self) -> bool:
        """Start all initialized components"""
        logger.info("Starting component startup sequence")
        
        success_count = 0
        
        for component_id in self.startup_order:
            if component_id not in self.components:
                continue
                
            component = self.components[component_id]
            
            if component.state != ComponentState.INITIALIZED:
                logger.warning(f"Skipping startup for component {component.name} - not initialized")
                continue
            
            try:
                logger.info(f"Starting component: {component.name}")
                component.state = ComponentState.STARTING
                
                start_time = time.time()
                
                # Start component if it has a start method
                if hasattr(component.instance, 'start'):
                    if asyncio.iscoroutinefunction(component.instance.start):
                        await component.instance.start()
                    else:
                        component.instance.start()
                
                component.startup_time = time.time() - start_time
                component.state = ComponentState.RUNNING
                success_count += 1
                
                logger.info(f"Component started: {component.name} ({component.startup_time:.2f}s)")
                
            except Exception as e:
                component.state = ComponentState.ERROR
                component.health_status = HealthStatus.CRITICAL
                component.error_count += 1
                component.last_error = str(e)
                component.last_error_time = time.time()
                
                logger.error(f"Component startup failed: {component.name} - {e}")
        
        # Start health monitoring
        if success_count > 0:
            await self.start_health_monitoring()
        
        logger.info(f"Component startup complete: {success_count} components running")
        return success_count > 0
    
    async def shutdown_all_components(self) -> bool:
        """Shutdown all components in reverse dependency order"""
        logger.info("Starting component shutdown sequence")
        
        self.is_shutting_down = True
        
        # Stop health monitoring
        await self.stop_health_monitoring()
        
        success_count = 0
        
        for component_id in self.shutdown_order:
            if component_id not in self.components:
                continue
                
            component = self.components[component_id]
            
            if component.state not in [ComponentState.RUNNING, ComponentState.ERROR]:
                continue
            
            try:
                logger.info(f"Shutting down component: {component.name}")
                component.state = ComponentState.STOPPING
                
                # Shutdown component if it has a shutdown method
                if hasattr(component.instance, 'shutdown'):
                    if asyncio.iscoroutinefunction(component.instance.shutdown):
                        await component.instance.shutdown()
                    else:
                        component.instance.shutdown()
                
                component.state = ComponentState.STOPPED
                success_count += 1
                
                logger.info(f"Component shutdown: {component.name}")
                
            except Exception as e:
                component.state = ComponentState.FAILED
                logger.error(f"Component shutdown failed: {component.name} - {e}")
        
        logger.info(f"Component shutdown complete: {success_count} components stopped")
        return True
    
    async def start_health_monitoring(self):
        """Start the health monitoring system"""
        if self.health_monitor_active:
            return
        
        self.health_monitor_active = True
        self.health_check_interval = self.config_manager.get(
            'components.health_check_interval', 30.0
        )
        
        self.health_monitor_task = asyncio.create_task(self._health_monitor_loop())
        logger.info("Health monitoring started")
    
    async def stop_health_monitoring(self):
        """Stop the health monitoring system"""
        self.health_monitor_active = False
        
        if self.health_monitor_task:
            self.health_monitor_task.cancel()
            try:
                await self.health_monitor_task
            except asyncio.CancelledError:
                pass
            self.health_monitor_task = None
        
        logger.info("Health monitoring stopped")
    
    async def _health_monitor_loop(self):
        """Main health monitoring loop"""
        try:
            while self.health_monitor_active and not self.is_shutting_down:
                await self.perform_health_check()
                await asyncio.sleep(self.health_check_interval)
        except asyncio.CancelledError:
            logger.info("Health monitor loop cancelled")
        except Exception as e:
            logger.error(f"Health monitor loop error: {e}")
    
    async def perform_health_check(self) -> HealthMetrics:
        """Perform comprehensive health check on all components"""
        try:
            current_time = time.time()
            healthy_count = 0
            warning_count = 0
            critical_count = 0
            failed_count = 0
            
            # Check each component
            for component_id, component in self.components.items():
                try:
                    # Update resource usage
                    await self._update_component_resources(component)
                    
                    # Perform component-specific health check
                    if hasattr(component.instance, 'health_check'):
                        if asyncio.iscoroutinefunction(component.instance.health_check):
                            health_result = await component.instance.health_check()
                        else:
                            health_result = component.instance.health_check()
                        
                        if isinstance(health_result, dict):
                            component.health_status = HealthStatus(
                                health_result.get('status', 'unknown')
                            )
                        elif isinstance(health_result, bool):
                            component.health_status = HealthStatus.HEALTHY if health_result else HealthStatus.CRITICAL
                    else:
                        # Default health check based on state
                        if component.state == ComponentState.RUNNING:
                            component.health_status = HealthStatus.HEALTHY
                        elif component.state in [ComponentState.ERROR, ComponentState.FAILED]:
                            component.health_status = HealthStatus.CRITICAL
                        else:
                            component.health_status = HealthStatus.WARNING
                    
                    component.last_health_check = current_time
                    
                    # Count health statuses
                    if component.health_status == HealthStatus.HEALTHY:
                        healthy_count += 1
                    elif component.health_status == HealthStatus.WARNING:
                        warning_count += 1
                    elif component.health_status == HealthStatus.CRITICAL:
                        critical_count += 1
                    else:
                        failed_count += 1
                        
                except Exception as e:
                    logger.error(f"Health check failed for component {component_id}: {e}")
                    component.health_status = HealthStatus.CRITICAL
                    component.error_count += 1
                    component.last_error = str(e)
                    component.last_error_time = current_time
                    critical_count += 1
            
            # System-wide metrics
            system_cpu = psutil.cpu_percent()
            system_memory = psutil.virtual_memory().percent
            system_uptime = current_time - self.system_start_time
            
            health_metrics = HealthMetrics(
                timestamp=current_time,
                total_components=len(self.components),
                healthy_components=healthy_count,
                warning_components=warning_count,
                critical_components=critical_count,
                failed_components=failed_count,
                system_cpu_usage=system_cpu,
                system_memory_usage=system_memory,
                system_uptime=system_uptime
            )
            
            self.health_history.append(health_metrics)
            
            # Publish health metrics
            await self.event_bus.publish(EventType.SYSTEM_HEALTH_CHECK, {
                'metrics': asdict(health_metrics),
                'component_details': {
                    comp_id: {
                        'name': comp.name,
                        'state': comp.state.value,
                        'health': comp.health_status.value,
                        'cpu_usage': comp.cpu_usage,
                        'memory_usage': comp.memory_usage,
                        'error_count': comp.error_count
                    }
                    for comp_id, comp in self.components.items()
                }
            })
            
            # Log health summary
            if critical_count > 0:
                logger.warning(f"Health check: {critical_count} critical components detected")
            elif warning_count > 0:
                logger.info(f"Health check: {warning_count} components with warnings")
            else:
                logger.debug(f"Health check: All {healthy_count} components healthy")
            
            return health_metrics
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return HealthMetrics(
                timestamp=time.time(),
                total_components=len(self.components),
                healthy_components=0,
                warning_components=0,
                critical_components=len(self.components),
                failed_components=0,
                system_cpu_usage=0.0,
                system_memory_usage=0.0,
                system_uptime=0.0
            )
    
    async def _update_component_resources(self, component: ComponentInfo):
        """Update resource usage for a component"""
        try:
            # This is a simplified resource tracking
            # In a real implementation, you'd track per-component resources
            component.cpu_usage = psutil.cpu_percent() / len(self.components)
            component.memory_usage = psutil.virtual_memory().used / len(self.components)
            component.thread_count = threading.active_count() // len(self.components)
            
        except Exception as e:
            logger.debug(f"Resource update failed for {component.name}: {e}")
    
    async def _attempt_component_recovery(self, component_id: str):
        """Attempt to recover a failed component"""
        if component_id not in self.components:
            return False
        
        component = self.components[component_id]
        logger.info(f"Attempting recovery for component: {component.name}")
        
        try:
            # Stop the component first
            if hasattr(component.instance, 'stop'):
                if asyncio.iscoroutinefunction(component.instance.stop):
                    await component.instance.stop()
                else:
                    component.instance.stop()
            
            # Wait a moment
            await asyncio.sleep(1.0)
            
            # Reinitialize
            if hasattr(component.instance, 'initialize'):
                if asyncio.iscoroutinefunction(component.instance.initialize):
                    result = await component.instance.initialize(component.config)
                else:
                    result = component.instance.initialize(component.config)
                
                if result is not False:
                    component.state = ComponentState.INITIALIZED
                    
                    # Restart
                    if hasattr(component.instance, 'start'):
                        if asyncio.iscoroutinefunction(component.instance.start):
                            await component.instance.start()
                        else:
                            component.instance.start()
                    
                    component.state = ComponentState.RUNNING
                    component.health_status = HealthStatus.HEALTHY
                    
                    logger.info(f"Component recovery successful: {component.name}")
                    return True
            
        except Exception as e:
            logger.error(f"Component recovery failed for {component.name}: {e}")
            component.state = ComponentState.FAILED
            component.health_status = HealthStatus.CRITICAL
        
        return False
    
    def get_component_info(self, component_id: str) -> Optional[ComponentInfo]:
        """Get information about a specific component"""
        return self.components.get(component_id)
    
    def get_all_components(self) -> Dict[str, ComponentInfo]:
        """Get information about all components"""
        return self.components.copy()
    
    def get_components_by_type(self, component_type: ComponentType) -> Dict[str, ComponentInfo]:
        """Get all components of a specific type"""
        return {
            comp_id: comp for comp_id, comp in self.components.items()
            if comp.component_type == component_type
        }
    
    def get_running_components(self) -> Dict[str, ComponentInfo]:
        """Get all currently running components"""
        return {
            comp_id: comp for comp_id, comp in self.components.items()
            if comp.state == ComponentState.RUNNING
        }
    
    async def send_message(self, from_component: str, to_component: str, message: Dict[str, Any]) -> bool:
        """Send a message between components"""
        try:
            if to_component not in self.components:
                logger.warning(f"Message destination component not found: {to_component}")
                return False
            
            message_envelope = {
                'id': str(uuid.uuid4()),
                'timestamp': time.time(),
                'from': from_component,
                'to': to_component,
                'message': message
            }
            
            with self.communication_lock:
                self.message_bus[to_component].append(message_envelope)
            
            # Notify the target component if it has a message handler
            target_component = self.components[to_component]
            if hasattr(target_component.instance, 'handle_message'):
                try:
                    if asyncio.iscoroutinefunction(target_component.instance.handle_message):
                        await target_component.instance.handle_message(message_envelope)
                    else:
                        target_component.instance.handle_message(message_envelope)
                except Exception as e:
                    logger.error(f"Message handling error in {to_component}: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Message sending failed from {from_component} to {to_component}: {e}")
            return False
    
    def get_messages(self, component_id: str) -> List[Dict[str, Any]]:
        """Get pending messages for a component"""
        with self.communication_lock:
            messages = list(self.message_bus[component_id])
            self.message_bus[component_id].clear()
            return messages
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        current_time = time.time()
        
        # Component status summary
        state_counts = defaultdict(int)
        health_counts = defaultdict(int)
        
        for component in self.components.values():
            state_counts[component.state.value] += 1
            health_counts[component.health_status.value] += 1
        
        # Recent health metrics
        recent_health = self.health_history[-1] if self.health_history else None
        
        return {
            'system': {
                'uptime': current_time - self.system_start_time,
                'total_components': len(self.components),
                'health_monitoring_active': self.health_monitor_active,
                'is_shutting_down': self.is_shutting_down
            },
            'components': {
                'by_state': dict(state_counts),
                'by_health': dict(health_counts),
                'startup_order': self.startup_order,
                'shutdown_order': self.shutdown_order
            },
            'health': asdict(recent_health) if recent_health else None,
            'resources': {
                'cpu_usage': psutil.cpu_percent(),
                'memory_usage': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent,
                'active_threads': threading.active_count()
            }
        }

# ========================================
# INTEGRATION HELPERS
# ========================================

def create_corelet_manager(event_bus: EventBus, config_manager: ConfigurationManager) -> CoreletManager:
    """Factory function to create a properly configured CoreletManager"""
    return CoreletManager(event_bus, config_manager)

# Export main classes
__all__ = [
    'CoreletManager',
    'ComponentInfo',
    'ComponentState',
    'ComponentType',
    'HealthStatus',
    'HealthMetrics',
    'create_corelet_manager'
]