#!/usr/bin/env python3
"""
⚡ WarpScheduler - Advanced Task Scheduling System
Part of the Extended MachineGod AGI System

This module implements an advanced task scheduling system for coordinating
AI subsystem operations including emotional processing, symbolic reasoning,
and NLP operations with real-time scheduling capabilities.

Features:
- Priority-based task queuing with dynamic priority adjustment
- Resource allocation and load balancing
- Concurrent task execution with dependency management
- Real-time scheduling for emotional and symbolic processing
- Cross-platform deployment optimization
- Integration with CoreletManager for system-wide coordination
- Performance optimization and adaptive scheduling

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
import heapq
import psutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable, Set, Union, Awaitable
from dataclasses import dataclass, asdict, field
from collections import defaultdict, deque
from enum import Enum, IntEnum
import uuid
import json
import traceback
from datetime import datetime, timedelta
import weakref
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import math

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

logger = logging.getLogger('WarpScheduler')

# ========================================
# TASK SCHEDULING SYSTEM
# ========================================

class TaskPriority(IntEnum):
    """Task priority levels"""
    CRITICAL = 0    # System-critical tasks
    HIGH = 1        # High priority tasks
    NORMAL = 2      # Normal priority tasks
    LOW = 3         # Low priority tasks
    BACKGROUND = 4  # Background tasks

class TaskState(Enum):
    """Task execution states"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

class TaskType(Enum):
    """Types of tasks in the system"""
    NLP_PROCESSING = "nlp_processing"
    EMOTIONAL_PROCESSING = "emotional_processing"
    SYMBOLIC_REASONING = "symbolic_reasoning"
    CONSCIOUSNESS_UPDATE = "consciousness_update"
    MEMORY_OPERATION = "memory_operation"
    HEALTH_CHECK = "health_check"
    SYSTEM_MAINTENANCE = "system_maintenance"
    USER_QUERY = "user_query"
    PLUGIN_OPERATION = "plugin_operation"

class ResourceType(Enum):
    """System resource types"""
    CPU = "cpu"
    MEMORY = "memory"
    IO = "io"
    NETWORK = "network"
    GPU = "gpu"

@dataclass
class TaskResource:
    """Resource requirements for a task"""
    cpu_cores: float = 1.0
    memory_mb: float = 100.0
    io_operations: int = 0
    network_bandwidth: float = 0.0
    gpu_memory_mb: float = 0.0
    estimated_duration: float = 1.0  # seconds

@dataclass
class Task:
    """Individual task in the scheduling system"""
    id: str
    name: str
    task_type: TaskType
    priority: TaskPriority
    
    # Task execution
    function: Callable
    args: Tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Scheduling metadata
    created_time: float = field(default_factory=time.time)
    scheduled_time: Optional[float] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    # Dependencies and constraints
    dependencies: List[str] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)
    required_components: List[str] = field(default_factory=list)
    
    # Resource requirements
    resources: TaskResource = field(default_factory=TaskResource)
    
    # Execution state
    state: TaskState = TaskState.PENDING
    result: Any = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout: float = 30.0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def __lt__(self, other):
        """Comparison for priority queue (lower priority value = higher priority)"""
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.created_time < other.created_time

@dataclass
class SchedulerMetrics:
    """Scheduler performance metrics"""
    timestamp: float
    total_tasks: int
    pending_tasks: int
    running_tasks: int
    completed_tasks: int
    failed_tasks: int
    average_wait_time: float
    average_execution_time: float
    throughput_per_second: float
    resource_utilization: Dict[str, float]
    queue_depths: Dict[str, int]

@dataclass
class WorkerPool:
    """Worker pool for task execution"""
    name: str
    max_workers: int
    current_workers: int = 0
    executor: Optional[Union[ThreadPoolExecutor, ProcessPoolExecutor]] = None
    task_types: List[TaskType] = field(default_factory=list)
    resource_limits: Dict[ResourceType, float] = field(default_factory=dict)
    active_tasks: Set[str] = field(default_factory=set)

class WarpScheduler:
    """
    Advanced task scheduling system for AI subsystem coordination.
    
    Provides:
    - Priority-based task queuing with dynamic adjustment
    - Resource allocation and load balancing
    - Concurrent task execution with dependency management
    - Real-time scheduling for emotional and symbolic processing
    - Performance optimization and adaptive scheduling
    - Cross-platform deployment optimization
    """
    
    def __init__(self, event_bus: EventBus, config_manager: ConfigurationManager, corelet_manager=None):
        self.event_bus = event_bus
        self.config_manager = config_manager
        self.corelet_manager = corelet_manager
        
        # Task management
        self.tasks: Dict[str, Task] = {}
        self.task_queue: List[Task] = []  # Priority queue
        self.running_tasks: Dict[str, Task] = {}
        self.completed_tasks: deque = deque(maxlen=1000)
        self.failed_tasks: deque = deque(maxlen=100)
        
        # Scheduling state
        self.scheduler_active = False
        self.scheduler_task: Optional[asyncio.Task] = None
        self.scheduling_lock = threading.RLock()
        
        # Worker pools
        self.worker_pools: Dict[str, WorkerPool] = {}
        self.default_pool_size = min(mp.cpu_count(), 8)
        
        # Resource management
        self.resource_monitor = ResourceMonitor()
        self.resource_limits = self._get_resource_limits()
        self.load_balancer = LoadBalancer()
        
        # Performance tracking
        self.metrics_history: deque = deque(maxlen=100)
        self.performance_optimizer = PerformanceOptimizer()
        
        # Dependency management
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        
        # Real-time scheduling
        self.real_time_queue: List[Task] = []
        self.real_time_threshold = 0.1  # 100ms
        
        # Cross-platform optimization
        self.platform_optimizer = PlatformOptimizer()
        
        # Initialize worker pools
        self._initialize_worker_pools()
        
        # Setup event subscriptions
        self._setup_event_subscriptions()
        
        logger.info("WarpScheduler initialized")
    
    def _get_resource_limits(self) -> Dict[ResourceType, float]:
        """Get system resource limits"""
        return {
            ResourceType.CPU: self.config_manager.get('scheduler.max_cpu_usage', 80.0),
            ResourceType.MEMORY: self.config_manager.get('scheduler.max_memory_usage', 80.0),
            ResourceType.IO: self.config_manager.get('scheduler.max_io_usage', 70.0),
            ResourceType.NETWORK: self.config_manager.get('scheduler.max_network_usage', 70.0),
            ResourceType.GPU: self.config_manager.get('scheduler.max_gpu_usage', 90.0)
        }
    
    def _initialize_worker_pools(self):
        """Initialize worker pools for different task types"""
        # NLP Processing Pool
        self.worker_pools['nlp'] = WorkerPool(
            name='nlp',
            max_workers=self.config_manager.get('scheduler.nlp_workers', 4),
            task_types=[TaskType.NLP_PROCESSING, TaskType.USER_QUERY],
            resource_limits={
                ResourceType.CPU: 50.0,
                ResourceType.MEMORY: 1024.0
            }
        )
        
        # Emotional Processing Pool
        self.worker_pools['emotional'] = WorkerPool(
            name='emotional',
            max_workers=self.config_manager.get('scheduler.emotional_workers', 2),
            task_types=[TaskType.EMOTIONAL_PROCESSING],
            resource_limits={
                ResourceType.CPU: 30.0,
                ResourceType.MEMORY: 512.0
            }
        )
        
        # Symbolic Reasoning Pool
        self.worker_pools['symbolic'] = WorkerPool(
            name='symbolic',
            max_workers=self.config_manager.get('scheduler.symbolic_workers', 2),
            task_types=[TaskType.SYMBOLIC_REASONING],
            resource_limits={
                ResourceType.CPU: 40.0,
                ResourceType.MEMORY: 768.0
            }
        )
        
        # System Operations Pool
        self.worker_pools['system'] = WorkerPool(
            name='system',
            max_workers=self.config_manager.get('scheduler.system_workers', 2),
            task_types=[
                TaskType.CONSCIOUSNESS_UPDATE,
                TaskType.MEMORY_OPERATION,
                TaskType.HEALTH_CHECK,
                TaskType.SYSTEM_MAINTENANCE
            ],
            resource_limits={
                ResourceType.CPU: 20.0,
                ResourceType.MEMORY: 256.0
            }
        )
        
        # Background Pool
        self.worker_pools['background'] = WorkerPool(
            name='background',
            max_workers=self.config_manager.get('scheduler.background_workers', 1),
            task_types=[TaskType.PLUGIN_OPERATION],
            resource_limits={
                ResourceType.CPU: 10.0,
                ResourceType.MEMORY: 128.0
            }
        )
        
        # Initialize executors
        for pool in self.worker_pools.values():
            if self.config_manager.get('scheduler.use_process_pool', False):
                pool.executor = ProcessPoolExecutor(max_workers=pool.max_workers)
            else:
                pool.executor = ThreadPoolExecutor(max_workers=pool.max_workers)
    
    def _setup_event_subscriptions(self):
        """Setup event subscriptions for scheduler coordination"""
        self.event_bus.subscribe(EventType.EMOTIONAL_STATE_CHANGED, self._handle_emotional_event)
        self.event_bus.subscribe(EventType.CONSCIOUSNESS_EVOLVED, self._handle_consciousness_event)
        self.event_bus.subscribe(EventType.SYSTEM_HEALTH_CHECK, self._handle_health_check_event)
    
    async def _handle_emotional_event(self, event: Dict):
        """Handle emotional state change events"""
        # Schedule emotional processing tasks with high priority
        await self.schedule_task(
            name="emotional_state_processing",
            task_type=TaskType.EMOTIONAL_PROCESSING,
            priority=TaskPriority.HIGH,
            function=self._process_emotional_event,
            args=(event,),
            timeout=5.0
        )
    
    async def _handle_consciousness_event(self, event: Dict):
        """Handle consciousness evolution events"""
        await self.schedule_task(
            name="consciousness_update",
            task_type=TaskType.CONSCIOUSNESS_UPDATE,
            priority=TaskPriority.HIGH,
            function=self._process_consciousness_event,
            args=(event,),
            timeout=10.0
        )
    
    async def _handle_health_check_event(self, event: Dict):
        """Handle health check events"""
        await self.schedule_task(
            name="health_check_processing",
            task_type=TaskType.HEALTH_CHECK,
            priority=TaskPriority.NORMAL,
            function=self._process_health_check_event,
            args=(event,),
            timeout=15.0
        )
    
    async def start_scheduler(self) -> bool:
        """Start the task scheduler"""
        if self.scheduler_active:
            logger.warning("Scheduler already active")
            return True
        
        try:
            self.scheduler_active = True
            
            # Start resource monitoring
            await self.resource_monitor.start()
            
            # Start scheduler loop
            self.scheduler_task = asyncio.create_task(self._scheduler_loop())
            
            # Start performance monitoring
            asyncio.create_task(self._performance_monitor_loop())
            
            logger.info("WarpScheduler started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start scheduler: {e}")
            self.scheduler_active = False
            return False
    
    async def stop_scheduler(self) -> bool:
        """Stop the task scheduler"""
        if not self.scheduler_active:
            return True
        
        try:
            logger.info("Stopping WarpScheduler...")
            self.scheduler_active = False
            
            # Cancel scheduler task
            if self.scheduler_task:
                self.scheduler_task.cancel()
                try:
                    await self.scheduler_task
                except asyncio.CancelledError:
                    pass
            
            # Shutdown worker pools
            for pool in self.worker_pools.values():
                if pool.executor:
                    pool.executor.shutdown(wait=True)
            
            # Stop resource monitoring
            await self.resource_monitor.stop()
            
            logger.info("WarpScheduler stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping scheduler: {e}")
            return False
    
    async def schedule_task(self,
                           name: str,
                           task_type: TaskType,
                           function: Callable,
                           priority: TaskPriority = TaskPriority.NORMAL,
                           args: Tuple = (),
                           kwargs: Dict[str, Any] = None,
                           dependencies: List[str] = None,
                           required_components: List[str] = None,
                           resources: TaskResource = None,
                           timeout: float = 30.0,
                           metadata: Dict[str, Any] = None,
                           tags: List[str] = None) -> str:
        """
        Schedule a task for execution
        
        Args:
            name: Task name
            task_type: Type of task
            function: Function to execute
            priority: Task priority
            args: Function arguments
            kwargs: Function keyword arguments
            dependencies: List of task IDs this task depends on
            required_components: List of component IDs required for this task
            resources: Resource requirements
            timeout: Task timeout in seconds
            metadata: Additional metadata
            tags: Task tags
            
        Returns:
            str: Task ID
        """
        try:
            task_id = str(uuid.uuid4())
            
            task = Task(
                id=task_id,
                name=name,
                task_type=task_type,
                priority=priority,
                function=function,
                args=args,
                kwargs=kwargs or {},
                dependencies=dependencies or [],
                required_components=required_components or [],
                resources=resources or TaskResource(),
                timeout=timeout,
                metadata=metadata or {},
                tags=tags or []
            )
            
            with self.scheduling_lock:
                self.tasks[task_id] = task
                
                # Update dependency graphs
                self._update_dependency_graphs(task_id, dependencies or [])
                
                # Check if task can be queued immediately
                if self._can_queue_task(task):
                    task.state = TaskState.QUEUED
                    task.scheduled_time = time.time()
                    
                    # Add to appropriate queue
                    if self._is_real_time_task(task):
                        heapq.heappush(self.real_time_queue, task)
                    else:
                        heapq.heappush(self.task_queue, task)
                    
                    logger.debug(f"Task queued: {name} ({task_id})")
                else:
                    logger.debug(f"Task pending dependencies: {name} ({task_id})")
            
            # Publish task scheduled event
            await self.event_bus.publish(EventType.PLUGIN_LOADED, {
                'task_id': task_id,
                'task_name': name,
                'task_type': task_type.value,
                'priority': priority.value
            })
            
            return task_id
            
        except Exception as e:
            logger.error(f"Failed to schedule task {name}: {e}")
            raise
    
    def _update_dependency_graphs(self, task_id: str, dependencies: List[str]):
        """Update task dependency graphs"""
        # Clear existing dependencies
        for dep in list(self.dependency_graph[task_id]):
            self.reverse_dependency_graph[dep].discard(task_id)
        self.dependency_graph[task_id].clear()
        
        # Add new dependencies
        for dep in dependencies:
            self.dependency_graph[task_id].add(dep)
            self.reverse_dependency_graph[dep].add(task_id)
    
    def _can_queue_task(self, task: Task) -> bool:
        """Check if a task can be queued (all dependencies satisfied)"""
        for dep_id in task.dependencies:
            if dep_id in self.tasks:
                dep_task = self.tasks[dep_id]
                if dep_task.state not in [TaskState.COMPLETED]:
                    return False
            else:
                # Dependency not found
                logger.warning(f"Task dependency not found: {dep_id}")
                return False
        
        # Check required components
        if self.corelet_manager:
            for comp_id in task.required_components:
                component = self.corelet_manager.get_component_info(comp_id)
                if not component or component.state.value != 'running':
                    return False
        
        return True
    
    def _is_real_time_task(self, task: Task) -> bool:
        """Check if a task requires real-time scheduling"""
        return (task.priority == TaskPriority.CRITICAL or
                task.task_type in [TaskType.EMOTIONAL_PROCESSING, TaskType.CONSCIOUSNESS_UPDATE] or
                task.resources.estimated_duration <= self.real_time_threshold)
    
    async def _scheduler_loop(self):
        """Main scheduler loop"""
        try:
            while self.scheduler_active:
                try:
                    # Process real-time tasks first
                    await self._process_real_time_tasks()
                    
                    # Process regular tasks
                    await self._process_regular_tasks()
                    
                    # Check for completed tasks
                    await self._check_completed_tasks()
                    
                    # Update metrics
                    await self._update_metrics()
                    
                    # Optimize performance
                    await self.performance_optimizer.optimize(self)
                    
                    # Short sleep to prevent busy waiting
                    await asyncio.sleep(0.01)  # 10ms
                    
                except Exception as e:
                    logger.error(f"Scheduler loop error: {e}")
                    await asyncio.sleep(0.1)
                    
        except asyncio.CancelledError:
            logger.info("Scheduler loop cancelled")
        except Exception as e:
            logger.error(f"Scheduler loop fatal error: {e}")
    
    async def _process_real_time_tasks(self):
        """Process real-time tasks with minimal latency"""
        while self.real_time_queue and self.scheduler_active:
            with self.scheduling_lock:
                if not self.real_time_queue:
                    break
                
                task = heapq.heappop(self.real_time_queue)
                
                # Check if we can execute this task now
                if not self._can_execute_task(task):
                    # Put it back and break
                    heapq.heappush(self.real_time_queue, task)
                    break
                
                # Execute task
                await self._execute_task(task)
    
    async def _process_regular_tasks(self):
        """Process regular priority tasks"""
        tasks_processed = 0
        max_tasks_per_cycle = 5
        
        while (self.task_queue and 
               tasks_processed < max_tasks_per_cycle and 
               self.scheduler_active):
            
            with self.scheduling_lock:
                if not self.task_queue:
                    break
                
                task = heapq.heappop(self.task_queue)
                
                # Check if we can execute this task now
                if not self._can_execute_task(task):
                    # Put it back and break
                    heapq.heappush(self.task_queue, task)
                    break
                
                # Execute task
                await self._execute_task(task)
                tasks_processed += 1
    
    def _can_execute_task(self, task: Task) -> bool:
        """Check if a task can be executed now"""
        # Check resource availability
        if not self.resource_monitor.can_allocate_resources(task.resources):
            return False
        
        # Check worker pool availability
        pool = self._get_worker_pool_for_task(task)
        if not pool or pool.current_workers >= pool.max_workers:
            return False
        
        # Check component availability
        if self.corelet_manager:
            for comp_id in task.required_components:
                component = self.corelet_manager.get_component_info(comp_id)
                if not component or component.state.value != 'running':
                    return False
        
        return True
    
    def _get_worker_pool_for_task(self, task: Task) -> Optional[WorkerPool]:
        """Get the appropriate worker pool for a task"""
        for pool in self.worker_pools.values():
            if task.task_type in pool.task_types:
                return pool
        
        # Default to background pool
        return self.worker_pools.get('background')
    
    async def _execute_task(self, task: Task):
        """Execute a task"""
        try:
            pool = self._get_worker_pool_for_task(task)
            if not pool or not pool.executor:
                raise Exception(f"No suitable worker pool for task {task.name}")
            
            # Update task state
            task.state = TaskState.RUNNING
            task.start_time = time.time()
            self.running_tasks[task.id] = task
            
            # Allocate resources
            self.resource_monitor.allocate_resources(task.id, task.resources)
            
            # Update pool state
            pool.current_workers += 1
            pool.active_tasks.add(task.id)
            
            logger.debug(f"Executing task: {task.name} ({task.id})")
            
            # Submit task to executor
            if asyncio.iscoroutinefunction(task.function):
                # Async function
                future = asyncio.create_task(
                    asyncio.wait_for(
                        task.function(*task.args, **task.kwargs),
                        timeout=task.timeout
                    )
                )
            else:
                # Sync function
                future = asyncio.get_event_loop().run_in_executor(
                    pool.executor,
                    lambda: task.function(*task.args, **task.kwargs)
                )
            
            # Store future for completion checking
            task.metadata['future'] = future
            
        except Exception as e:
            logger.error(f"Failed to execute task {task.name}: {e}")
            await self._handle_task_failure(task, str(e))
    
    async def _check_completed_tasks(self):
        """Check for completed tasks and handle results"""
        completed_task_ids = []
        
        for task_id, task in list(self.running_tasks.items()):
            future = task.metadata.get('future')
            if not future:
                continue
            
            try:
                if future.done():
                    completed_task_ids.append(task_id)
                    
                    if future.exception():
                        # Task failed
                        error = str(future.exception())
                        await self._handle_task_failure(task, error)
                    else:
                        # Task completed successfully
                        result = future.result()
                        await self._handle_task_completion(task, result)
                        
            except Exception as e:
                logger.error(f"Error checking task completion {task_id}: {e}")
                completed_task_ids.append(task_id)
                await self._handle_task_failure(task, str(e))
        
        # Clean up completed tasks
        for task_id in completed_task_ids:
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]
    
    async def _handle_task_completion(self, task: Task, result: Any):
        """Handle successful task completion"""
        try:
            task.state = TaskState.COMPLETED
            task.end_time = time.time()
            task.result = result
            
            # Release resources
            self.resource_monitor.release_resources(task.id)
            
            # Update worker pool
            pool = self._get_worker_pool_for_task(task)
            if pool:
                pool.current_workers = max(0, pool.current_workers - 1)
                pool.active_tasks.discard(task.id)
            
            # Add to completed tasks
            self.completed_tasks.append(task)
            
            # Check for dependent tasks that can now be queued
            await self._check_dependent_tasks(task.id)
            
            logger.debug(f"Task completed: {task.name} ({task.id}) in {task.end_time - task.start_time:.2f}s")
            
            # Publish completion event
            await self.event_bus.publish(EventType.PLUGIN_LOADED, {
                'task_id': task.id,
                'task_name': task.name,
                'status': 'completed',
                'execution_time': task.end_time - task.start_time
            })
            
        except Exception as e:
            logger.error(f"Error handling task completion {task.id}: {e}")
    
    async def _handle_task_failure(self, task: Task, error: str):
        """Handle task failure"""
        try:
            task.error = error
            task.end_time = time.time()
            
            # Release resources
            self.resource_monitor.release_resources(task.id)
            
            # Update worker pool
            pool = self._get_worker_pool_for_task(task)
            if pool:
                pool.current_workers = max(0, pool.current_workers - 1)
                pool.active_tasks.discard(task.id)
            
            # Check for retry
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.state = TaskState.PENDING
                task.start_time = None
                task.end_time = None
                
                # Re-queue task with delay
                await asyncio.sleep(min(2 ** task.retry_count, 30))  # Exponential backoff
                
                if self._can_queue_task(task):
                    task.state = TaskState.QUEUED
                    heapq.heappush(self.task_queue, task)
                    logger.info(f"Task retry {task.retry_count}/{task.max_retries}: {task.name}")
                
            else:
                # Max retries exceeded
                task.state = TaskState.FAILED
                self.failed_tasks.append(task)
                
                logger.error(f"Task failed permanently: {task.name} ({task.id}) - {error}")
                
                # Publish failure event
                await self.event_bus.publish(EventType.PLUGIN_ERROR, {
                    'task_id': task.id,
                    'task_name': task.name,
                    'error': error,
                    'retry_count': task.retry_count
                })
            
        except Exception as e:
            logger.error(f"Error handling task failure {task.id}: {e}")
    
    async def _check_dependent_tasks(self, completed_task_id: str):
        """Check if any pending tasks can now be queued"""
        dependent_task_ids = self.reverse_dependency_graph.get(completed_task_id, set())
        
        for task_id in dependent_task_ids:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                if task.state == TaskState.PENDING and self._can_queue_task(task):
                    task.state = TaskState.QUEUED
                    task.scheduled_time = time.time()
                    
                    if self._is_real_time_task(task):
                        heapq.heappush(self.real_time_queue, task)
                    else:
                        heapq.heappush(self.task_queue, task)
                    
                    logger.debug(f"Dependent task queued: {task.name} ({task_id})")
    
    async def _update_metrics(self):
        """Update scheduler performance metrics"""
        try:
            current_time = time.time()
            
            # Count tasks by state
            pending_count = sum(1 for t in self.tasks.values() if t.state == TaskState.PENDING)
            queued_count = len(self.task_queue) + len(self.real_time_queue)
            running_count = len(self.running_tasks)
            completed_count = len(self.completed_tasks)
            failed_count = len(self.failed_tasks)
            
            # Calculate average times
            recent_completed = [t for t in self.completed_tasks if t.end_time and t.end_time > current_time - 60]
            
            avg_wait_time = 0.0
            avg_execution_time = 0.0
            
            if recent_completed:
                wait_times = [t.start_time - t.created_time for t in recent_completed if t.start_time]
                execution_times = [t.end_time - t.start_time for t in recent_completed if t.start_time and t.end_time]
                
                avg_wait_time = sum(wait_times) / len(wait_times) if wait_times else 0.0
                avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0.0
            
            # Calculate throughput
            recent_completions = len([t for t in self.completed_tasks if t.end_time and t.end_time > current_time - 1.0])
            throughput = recent_completions
            
            # Resource utilization
            resource_util = self.resource_monitor.get_utilization()
            
            # Queue depths
            queue_depths = {
                'regular': len(self.task_queue),
                'real_time': len(self.real_time_queue),
                'running': len(self.running_tasks)
            }
            
            metrics = SchedulerMetrics(
                timestamp=current_time,
                total_tasks=len(self.tasks),
                pending_tasks=pending_count,
                running_tasks=running_count,
                completed_tasks=completed_count,
                failed_tasks=failed_count,
                average_wait_time=avg_wait_time,
                average_execution_time=avg_execution_time,
                throughput_per_second=throughput,
                resource_utilization=resource_util,
                queue_depths=queue_depths
            )
            
            self.metrics_history.append(metrics)
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    async def _performance_monitor_loop(self):
        """Performance monitoring loop"""
        try:
            while self.scheduler_active:
                await asyncio.sleep(10.0)  # Update every 10 seconds
                
                if self.metrics_history:
                    latest_metrics = self.metrics_history[-1]
                    
                    # Log performance summary
                    logger.info(
                        f"Scheduler Performance: "
                        f"Tasks(P:{latest_metrics.pending_tasks}, "
                        f"R:{latest_metrics.running_tasks}, "
                        f"C:{latest_metrics.completed_tasks}) "
                        f"Throughput:{latest_metrics.throughput_per_second:.1f}/s "
                        f"AvgWait:{latest_metrics.average_wait_time:.2f}s "
                        f"AvgExec:{latest_metrics.average_execution_time:.2f}s"
                    )
                    
                    # Check for performance issues
                    if latest_metrics.average_wait_time > 10.0:
                        logger.warning("High task wait times detected")
                    
                    if latest_metrics.pending_tasks > 100:
                        logger.warning("High number of pending tasks")
                    
        except asyncio.CancelledError:
            logger.info("Performance monitor loop cancelled")
        except Exception as e:
            logger.error(f"Performance monitor error: {e}")
    
    # Task processing methods for different event types
    async def _process_emotional_event(self, event: Dict):
        """Process emotional state change events"""
        try:
            # Extract emotional data
            emotional_data = event.get('data', {})
            
            # Process through emotional processing hook if available
            if hasattr(self, 'emotional_hook'):
                result = await self.emotional_hook.process_emotional_state(
                    emotional_data.get('emotional_state'),
                    emotional_data
                )
                return result
            
            return {'status': 'processed', 'event': event}
            
        except Exception as e:
            logger.error(f"Error processing emotional event: {e}")
            raise
    
    async def _process_consciousness_event(self, event: Dict):
        """Process consciousness evolution events"""
        try:
            # Extract consciousness data
            consciousness_data = event.get('data', {})
            
            # Update consciousness state
            # This would integrate with the consciousness system
            return {'status': 'processed', 'event': event}
            
        except Exception as e:
            logger.error(f"Error processing consciousness event: {e}")
            raise
    
    async def _process_health_check_event(self, event: Dict):
        """Process health check events"""
        try:
            # Extract health data
            health_data = event.get('data', {})
            
            # Process health metrics
            # This would integrate with the health monitoring system
            return {'status': 'processed', 'event': event}
            
        except Exception as e:
            logger.error(f"Error processing health check event: {e}")
            raise
    
    # Public API methods
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task"""
        if task_id not in self.tasks:
            return None
        
        task = self.tasks[task_id]
        return {
            'id': task.id,
            'name': task.name,
            'type': task.task_type.value,
            'priority': task.priority.value,
            'state': task.state.value,
            'created_time': task.created_time,
            'scheduled_time': task.scheduled_time,
            'start_time': task.start_time,
            'end_time': task.end_time,
            'retry_count': task.retry_count,
            'error': task.error,
            'result': task.result if task.state == TaskState.COMPLETED else None
        }
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get comprehensive scheduler status"""
        latest_metrics = self.metrics_history[-1] if self.metrics_history else None
        
        return {
            'active': self.scheduler_active,
            'metrics': asdict(latest_metrics) if latest_metrics else None,
            'worker_pools': {
                name: {
                    'max_workers': pool.max_workers,
                    'current_workers': pool.current_workers,
                    'active_tasks': len(pool.active_tasks),
                    'task_types': [t.value for t in pool.task_types]
                }
                for name, pool in self.worker_pools.items()
            },
            'queues': {
                'regular': len(self.task_queue),
                'real_time': len(self.real_time_queue),
                'running': len(self.running_tasks)
            },
            'resource_utilization': self.resource_monitor.get_utilization()
        }
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task"""
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        
        try:
            if task.state == TaskState.RUNNING:
                # Cancel running task
                future = task.metadata.get('future')
                if future:
                    future.cancel()
                
                # Release resources
                self.resource_monitor.release_resources(task_id)
                
                # Update worker pool
                pool = self._get_worker_pool_for_task(task)
                if pool:
                    pool.current_workers = max(0, pool.current_workers - 1)
                    pool.active_tasks.discard(task_id)
                
                # Remove from running tasks
                if task_id in self.running_tasks:
                    del self.running_tasks[task_id]
            
            elif task.state in [TaskState.PENDING, TaskState.QUEUED]:
                # Remove from queues
                with self.scheduling_lock:
                    # Remove from regular queue
                    self.task_queue = [t for t in self.task_queue if t.id != task_id]
                    heapq.heapify(self.task_queue)
                    
                    # Remove from real-time queue
                    self.real_time_queue = [t for t in self.real_time_queue if t.id != task_id]
                    heapq.heapify(self.real_time_queue)
            
            task.state = TaskState.CANCELLED
            task.end_time = time.time()
            
            logger.info(f"Task cancelled: {task.name} ({task_id})")
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling task {task_id}: {e}")
            return False

# ========================================
# SUPPORTING CLASSES
# ========================================

class ResourceMonitor:
    """Monitor and manage system resources"""
    
    def __init__(self):
        self.allocated_resources: Dict[str, TaskResource] = {}
        self.monitoring_active = False
        
    async def start(self):
        """Start resource monitoring"""
        self.monitoring_active = True
        
    async def stop(self):
        """Stop resource monitoring"""
        self.monitoring_active = False
        
    def can_allocate_resources(self, resources: TaskResource) -> bool:
        """Check if resources can be allocated"""
        # Simple resource check - in production this would be more sophisticated
        current_cpu = psutil.cpu_percent()
        current_memory = psutil.virtual_memory().percent
        
        return current_cpu < 80.0 and current_memory < 80.0
        
    def allocate_resources(self, task_id: str, resources: TaskResource):
        """Allocate resources for a task"""
        self.allocated_resources[task_id] = resources
        
    def release_resources(self, task_id: str):
        """Release resources for a task"""
        if task_id in self.allocated_resources:
            del self.allocated_resources[task_id]
            
    def get_utilization(self) -> Dict[str, float]:
        """Get current resource utilization"""
        return {
            'cpu': psutil.cpu_percent(),
            'memory': psutil.virtual_memory().percent,
            'disk': psutil.disk_usage('/').percent
        }

class LoadBalancer:
    """Load balancing for task distribution"""
    
    def __init__(self):
        pass
        
    def get_optimal_worker_pool(self, task: Task, pools: Dict[str, WorkerPool]) -> Optional[str]:
        """Get optimal worker pool for a task"""
        suitable_pools = []
        
        for name, pool in pools.items():
            if task.task_type in pool.task_types and pool.current_workers < pool.max_workers:
                suitable_pools.append((name, pool))
        
        if not suitable_pools:
            return None
        
        # Choose pool with lowest utilization
        return min(suitable_pools, key=lambda x: x[1].current_workers / x[1].max_workers)[0]

class PerformanceOptimizer:
    """Optimize scheduler performance"""
    
    def __init__(self):
        self.optimization_history = deque(maxlen=10)
        
    async def optimize(self, scheduler: WarpScheduler):
        """Perform performance optimizations"""
        try:
            if not scheduler.metrics_history:
                return
            
            latest_metrics = scheduler.metrics_history[-1]
            
            # Adjust worker pool sizes based on load
            await self._optimize_worker_pools(scheduler, latest_metrics)
            
            # Adjust scheduling parameters
            await self._optimize_scheduling_parameters(scheduler, latest_metrics)
            
        except Exception as e:
            logger.error(f"Performance optimization error: {e}")
    
    async def _optimize_worker_pools(self, scheduler: WarpScheduler, metrics: SchedulerMetrics):
        """Optimize worker pool configurations"""
        # Simple optimization - increase workers if queue is backing up
        for pool_name, pool in scheduler.worker_pools.items():
            queue_depth = metrics.queue_depths.get(pool_name, 0)
            
            if queue_depth > 10 and pool.current_workers < pool.max_workers:
                # Could dynamically adjust pool sizes here
                pass
    
    async def _optimize_scheduling_parameters(self, scheduler: WarpScheduler, metrics: SchedulerMetrics):
        """Optimize scheduling parameters"""
        # Adjust real-time threshold based on performance
        if metrics.average_execution_time > 1.0:
            scheduler.real_time_threshold = min(scheduler.real_time_threshold * 1.1, 1.0)
        elif metrics.average_execution_time < 0.1:
            scheduler.real_time_threshold = max(scheduler.real_time_threshold * 0.9, 0.01)

class PlatformOptimizer:
    """Cross-platform deployment optimization"""
    
    def __init__(self):
        self.platform_info = self._detect_platform()
        
    def _detect_platform(self) -> Dict[str, Any]:
        """Detect platform characteristics"""
        return {
            'cpu_count': mp.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'platform': sys.platform,
            'architecture': sys.maxsize > 2**32  # 64-bit check
        }
    
    def get_optimal_configuration(self) -> Dict[str, Any]:
        """Get optimal configuration for current platform"""
        config = {}
        
        # Raspberry Pi optimization
        if self.platform_info['cpu_count'] <= 4 and self.platform_info['memory_gb'] <= 8:
            config.update({
                'max_workers': 2,
                'use_process_pool': False,
                'max_concurrent_tasks': 5,
                'health_check_interval': 60.0
            })
        
        # Desktop/Server optimization
        elif self.platform_info['cpu_count'] >= 8 and self.platform_info['memory_gb'] >= 16:
            config.update({
                'max_workers': min(self.platform_info['cpu_count'], 16),
                'use_process_pool': True,
                'max_concurrent_tasks': 50,
                'health_check_interval': 30.0
            })
        
        # Default configuration
        else:
            config.update({
                'max_workers': min(self.platform_info['cpu_count'], 8),
                'use_process_pool': False,
                'max_concurrent_tasks': 20,
                'health_check_interval': 45.0
            })
        
        return config

# ========================================
# INTEGRATION HELPERS
# ========================================

def create_warp_scheduler(event_bus: EventBus, 
                         config_manager: ConfigurationManager,
                         corelet_manager=None) -> WarpScheduler:
    """Factory function to create a properly configured WarpScheduler"""
    return WarpScheduler(event_bus, config_manager, corelet_manager)

# Export main classes
__all__ = [
    'WarpScheduler',
    'Task',
    'TaskPriority',
    'TaskState',
    'TaskType',
    'TaskResource',
    'SchedulerMetrics',
    'ResourceMonitor',
    'LoadBalancer',
    'PerformanceOptimizer',
    'PlatformOptimizer',
    'create_warp_scheduler'
]