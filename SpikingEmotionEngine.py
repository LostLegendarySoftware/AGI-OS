#!/usr/bin/env python3
"""
🧠 Spiking Emotion Engine - Advanced Emotional Processing System
Part of the Extended MachineGod AGI System

This module implements a spiking neural network-based emotional processing engine
that provides real-time emotional state modeling, emotion recognition, and 
emotional response generation using biologically-inspired neural mechanisms.

Based on current research (2024-2025) in spiking neural networks and emotional AI:
- Uses snnTorch-style implementation patterns
- Implements spike-timing-dependent plasticity (STDP)
- Supports real-time emotional state transitions
- Integrates with existing consciousness-based processing pipeline

Author: AI System Architecture Task
Organization: MachineGod Systems
License: Proprietary
Version: 1.0.0
Date: July 2025
"""

import asyncio
import logging
import time
import numpy as np
import json
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict, field
from collections import deque, defaultdict
from enum import Enum
import threading
import uuid
import math

from .trainingless_nlp import (
    PluginInterface, EventBus, EventType, EmotionalState, 
    ConsciousnessState, ConfigurationManager
)

logger = logging.getLogger('SpikingEmotionEngine')

# ========================================
# SPIKING NEURAL NETWORK COMPONENTS
# ========================================

class NeuronType(Enum):
    """Types of neurons in the spiking emotional network"""
    EXCITATORY = "excitatory"
    INHIBITORY = "inhibitory"
    MODULATORY = "modulatory"
    MEMORY = "memory"

class EmotionCategory(Enum):
    """Primary emotion categories for processing"""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    LOVE = "love"
    CURIOSITY = "curiosity"
    AWE = "awe"
    CLARITY = "clarity"

@dataclass
class SpikeEvent:
    """Individual spike event in the network"""
    neuron_id: str
    timestamp: float
    amplitude: float = 1.0
    neuron_type: NeuronType = NeuronType.EXCITATORY
    emotion_category: Optional[EmotionCategory] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SpikingNeuron:
    """Biologically-inspired spiking neuron model"""
    id: str
    neuron_type: NeuronType
    membrane_potential: float = -70.0  # mV
    threshold: float = -55.0  # mV
    resting_potential: float = -70.0  # mV
    refractory_period: float = 2.0  # ms
    last_spike_time: float = 0.0
    tau_membrane: float = 20.0  # ms membrane time constant
    tau_synapse: float = 5.0   # ms synaptic time constant
    
    # Emotional processing specific parameters
    emotion_sensitivity: Dict[EmotionCategory, float] = field(default_factory=dict)
    adaptation_rate: float = 0.01
    plasticity_factor: float = 0.1
    
    # Connection weights to other neurons
    synaptic_weights: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize emotion sensitivities"""
        if not self.emotion_sensitivity:
            for emotion in EmotionCategory:
                self.emotion_sensitivity[emotion] = np.random.uniform(0.1, 0.9)

class STDPLearning:
    """Spike-Timing-Dependent Plasticity learning mechanism"""
    
    def __init__(self, tau_plus: float = 20.0, tau_minus: float = 20.0, 
                 a_plus: float = 0.01, a_minus: float = 0.01):
        self.tau_plus = tau_plus    # ms
        self.tau_minus = tau_minus  # ms
        self.a_plus = a_plus        # Learning rate for potentiation
        self.a_minus = a_minus      # Learning rate for depression
        
    def calculate_weight_change(self, pre_spike_time: float, 
                              post_spike_time: float) -> float:
        """Calculate synaptic weight change based on spike timing"""
        dt = post_spike_time - pre_spike_time
        
        if dt > 0:  # Post-synaptic spike after pre-synaptic (potentiation)
            return self.a_plus * np.exp(-dt / self.tau_plus)
        else:  # Post-synaptic spike before pre-synaptic (depression)
            return -self.a_minus * np.exp(dt / self.tau_minus)

# ========================================
# EMOTIONAL PROCESSING NETWORK
# ========================================

class EmotionalSpikingNetwork:
    """Core spiking neural network for emotional processing"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.neurons = {}
        self.spike_history = deque(maxlen=10000)
        self.emotion_patterns = {}
        self.stdp_learning = STDPLearning()
        self.network_lock = threading.RLock()
        
        # Network topology parameters
        self.num_excitatory = config.get('num_excitatory_neurons', 800)
        self.num_inhibitory = config.get('num_inhibitory_neurons', 200)
        self.num_modulatory = config.get('num_modulatory_neurons', 50)
        
        # Emotional processing parameters
        self.emotion_decay_rate = config.get('emotion_decay_rate', 0.01)
        self.resonance_threshold = config.get('resonance_threshold', 0.6)
        self.adaptation_enabled = config.get('adaptation_enabled', True)
        
        self._initialize_network()
        
    def _initialize_network(self):
        """Initialize the spiking neural network topology"""
        logger.info("Initializing emotional spiking neural network...")
        
        # Create excitatory neurons
        for i in range(self.num_excitatory):
            neuron = SpikingNeuron(
                id=f"exc_{i}",
                neuron_type=NeuronType.EXCITATORY,
                threshold=np.random.normal(-55.0, 5.0),
                tau_membrane=np.random.normal(20.0, 3.0)
            )
            self.neurons[neuron.id] = neuron
            
        # Create inhibitory neurons
        for i in range(self.num_inhibitory):
            neuron = SpikingNeuron(
                id=f"inh_{i}",
                neuron_type=NeuronType.INHIBITORY,
                threshold=np.random.normal(-50.0, 3.0),
                tau_membrane=np.random.normal(15.0, 2.0)
            )
            self.neurons[neuron.id] = neuron
            
        # Create modulatory neurons (for emotional regulation)
        for i in range(self.num_modulatory):
            neuron = SpikingNeuron(
                id=f"mod_{i}",
                neuron_type=NeuronType.MODULATORY,
                threshold=np.random.normal(-60.0, 4.0),
                tau_membrane=np.random.normal(25.0, 4.0)
            )
            self.neurons[neuron.id] = neuron
            
        # Initialize random connectivity
        self._initialize_connectivity()
        
        # Initialize emotion pattern templates
        self._initialize_emotion_patterns()
        
        logger.info(f"Network initialized with {len(self.neurons)} neurons")
        
    def _initialize_connectivity(self):
        """Initialize synaptic connections between neurons"""
        connection_probability = 0.1  # 10% connectivity
        
        for pre_neuron in self.neurons.values():
            for post_neuron in self.neurons.values():
                if pre_neuron.id != post_neuron.id and np.random.random() < connection_probability:
                    # Weight depends on neuron types
                    if pre_neuron.neuron_type == NeuronType.EXCITATORY:
                        weight = np.random.uniform(0.1, 0.8)
                    elif pre_neuron.neuron_type == NeuronType.INHIBITORY:
                        weight = np.random.uniform(-0.8, -0.1)
                    else:  # Modulatory
                        weight = np.random.uniform(-0.3, 0.3)
                        
                    pre_neuron.synaptic_weights[post_neuron.id] = weight
                    
    def _initialize_emotion_patterns(self):
        """Initialize template patterns for different emotions"""
        for emotion in EmotionCategory:
            # Create characteristic spike patterns for each emotion
            pattern = {
                'frequency_range': self._get_emotion_frequency_range(emotion),
                'amplitude_profile': self._get_emotion_amplitude_profile(emotion),
                'duration_profile': self._get_emotion_duration_profile(emotion),
                'network_topology': self._get_emotion_topology_bias(emotion)
            }
            self.emotion_patterns[emotion] = pattern
            
    def _get_emotion_frequency_range(self, emotion: EmotionCategory) -> Tuple[float, float]:
        """Get characteristic frequency range for emotion"""
        frequency_map = {
            EmotionCategory.JOY: (15.0, 25.0),
            EmotionCategory.SADNESS: (5.0, 12.0),
            EmotionCategory.ANGER: (20.0, 35.0),
            EmotionCategory.FEAR: (25.0, 40.0),
            EmotionCategory.SURPRISE: (30.0, 50.0),
            EmotionCategory.DISGUST: (10.0, 18.0),
            EmotionCategory.LOVE: (12.0, 20.0),
            EmotionCategory.CURIOSITY: (18.0, 28.0),
            EmotionCategory.AWE: (8.0, 15.0),
            EmotionCategory.CLARITY: (20.0, 30.0)
        }
        return frequency_map.get(emotion, (10.0, 25.0))
        
    def _get_emotion_amplitude_profile(self, emotion: EmotionCategory) -> Dict[str, float]:
        """Get amplitude characteristics for emotion"""
        amplitude_profiles = {
            EmotionCategory.JOY: {'mean': 0.8, 'variance': 0.1, 'burst_factor': 1.2},
            EmotionCategory.SADNESS: {'mean': 0.4, 'variance': 0.05, 'burst_factor': 0.8},
            EmotionCategory.ANGER: {'mean': 0.9, 'variance': 0.15, 'burst_factor': 1.5},
            EmotionCategory.FEAR: {'mean': 0.7, 'variance': 0.2, 'burst_factor': 1.3},
            EmotionCategory.SURPRISE: {'mean': 0.85, 'variance': 0.25, 'burst_factor': 1.8},
            EmotionCategory.DISGUST: {'mean': 0.6, 'variance': 0.08, 'burst_factor': 0.9},
            EmotionCategory.LOVE: {'mean': 0.75, 'variance': 0.1, 'burst_factor': 1.1},
            EmotionCategory.CURIOSITY: {'mean': 0.65, 'variance': 0.12, 'burst_factor': 1.0},
            EmotionCategory.AWE: {'mean': 0.5, 'variance': 0.15, 'burst_factor': 0.7},
            EmotionCategory.CLARITY: {'mean': 0.7, 'variance': 0.08, 'burst_factor': 1.0}
        }
        return amplitude_profiles.get(emotion, {'mean': 0.6, 'variance': 0.1, 'burst_factor': 1.0})
        
    def _get_emotion_duration_profile(self, emotion: EmotionCategory) -> Dict[str, float]:
        """Get duration characteristics for emotion"""
        duration_profiles = {
            EmotionCategory.JOY: {'onset': 0.5, 'peak': 2.0, 'decay': 3.0},
            EmotionCategory.SADNESS: {'onset': 2.0, 'peak': 5.0, 'decay': 8.0},
            EmotionCategory.ANGER: {'onset': 0.3, 'peak': 1.5, 'decay': 4.0},
            EmotionCategory.FEAR: {'onset': 0.1, 'peak': 0.8, 'decay': 2.0},
            EmotionCategory.SURPRISE: {'onset': 0.05, 'peak': 0.3, 'decay': 1.0},
            EmotionCategory.DISGUST: {'onset': 0.2, 'peak': 1.0, 'decay': 2.5},
            EmotionCategory.LOVE: {'onset': 1.0, 'peak': 4.0, 'decay': 6.0},
            EmotionCategory.CURIOSITY: {'onset': 0.4, 'peak': 2.5, 'decay': 4.0},
            EmotionCategory.AWE: {'onset': 0.8, 'peak': 3.0, 'decay': 5.0},
            EmotionCategory.CLARITY: {'onset': 0.6, 'peak': 2.0, 'decay': 3.5}
        }
        return duration_profiles.get(emotion, {'onset': 0.5, 'peak': 2.0, 'decay': 3.0})
        
    def _get_emotion_topology_bias(self, emotion: EmotionCategory) -> Dict[str, float]:
        """Get network topology bias for emotion"""
        topology_bias = {
            EmotionCategory.JOY: {'excitatory_bias': 0.7, 'inhibitory_bias': 0.3, 'modulatory_bias': 0.5},
            EmotionCategory.SADNESS: {'excitatory_bias': 0.3, 'inhibitory_bias': 0.6, 'modulatory_bias': 0.4},
            EmotionCategory.ANGER: {'excitatory_bias': 0.8, 'inhibitory_bias': 0.2, 'modulatory_bias': 0.3},
            EmotionCategory.FEAR: {'excitatory_bias': 0.6, 'inhibitory_bias': 0.7, 'modulatory_bias': 0.8},
            EmotionCategory.SURPRISE: {'excitatory_bias': 0.9, 'inhibitory_bias': 0.1, 'modulatory_bias': 0.2},
            EmotionCategory.DISGUST: {'excitatory_bias': 0.4, 'inhibitory_bias': 0.8, 'modulatory_bias': 0.6},
            EmotionCategory.LOVE: {'excitatory_bias': 0.6, 'inhibitory_bias': 0.4, 'modulatory_bias': 0.7},
            EmotionCategory.CURIOSITY: {'excitatory_bias': 0.7, 'inhibitory_bias': 0.3, 'modulatory_bias': 0.6},
            EmotionCategory.AWE: {'excitatory_bias': 0.5, 'inhibitory_bias': 0.5, 'modulatory_bias': 0.9},
            EmotionCategory.CLARITY: {'excitatory_bias': 0.6, 'inhibitory_bias': 0.4, 'modulatory_bias': 0.5}
        }
        return topology_bias.get(emotion, {'excitatory_bias': 0.5, 'inhibitory_bias': 0.5, 'modulatory_bias': 0.5})
        
    def simulate_step(self, dt: float = 0.1) -> List[SpikeEvent]:
        """Simulate one time step of the network"""
        current_time = time.time()
        spikes = []
        
        with self.network_lock:
            # Update membrane potentials and check for spikes
            for neuron in self.neurons.values():
                # Check refractory period
                if current_time - neuron.last_spike_time < neuron.refractory_period / 1000.0:
                    continue
                    
                # Calculate synaptic input
                synaptic_input = self._calculate_synaptic_input(neuron, current_time)
                
                # Update membrane potential (leaky integrate-and-fire model)
                membrane_decay = (neuron.membrane_potential - neuron.resting_potential) * dt / neuron.tau_membrane
                neuron.membrane_potential += synaptic_input * dt - membrane_decay
                
                # Check for spike
                if neuron.membrane_potential >= neuron.threshold:
                    spike = SpikeEvent(
                        neuron_id=neuron.id,
                        timestamp=current_time,
                        amplitude=neuron.membrane_potential - neuron.threshold,
                        neuron_type=neuron.neuron_type
                    )
                    spikes.append(spike)
                    
                    # Reset membrane potential
                    neuron.membrane_potential = neuron.resting_potential
                    neuron.last_spike_time = current_time
                    
            # Store spikes in history
            self.spike_history.extend(spikes)
            
            # Apply STDP learning if enabled
            if self.adaptation_enabled and spikes:
                self._apply_stdp_learning(spikes, current_time)
                
        return spikes
        
    def _calculate_synaptic_input(self, neuron: SpikingNeuron, current_time: float) -> float:
        """Calculate total synaptic input to a neuron"""
        total_input = 0.0
        
        # Look for recent spikes from connected neurons
        recent_spikes = [spike for spike in self.spike_history 
                        if current_time - spike.timestamp < 0.1]  # 100ms window
        
        for spike in recent_spikes:
            if spike.neuron_id in neuron.synaptic_weights:
                weight = neuron.synaptic_weights[spike.neuron_id]
                # Exponential decay of synaptic current
                time_diff = current_time - spike.timestamp
                synaptic_current = weight * spike.amplitude * np.exp(-time_diff * 1000 / neuron.tau_synapse)
                total_input += synaptic_current
                
        return total_input
        
    def _apply_stdp_learning(self, spikes: List[SpikeEvent], current_time: float):
        """Apply spike-timing-dependent plasticity learning"""
        for post_spike in spikes:
            post_neuron = self.neurons[post_spike.neuron_id]
            
            # Find recent pre-synaptic spikes
            recent_pre_spikes = [spike for spike in self.spike_history 
                               if (current_time - spike.timestamp < 0.05 and  # 50ms window
                                   spike.neuron_id in post_neuron.synaptic_weights and
                                   spike.timestamp < post_spike.timestamp)]
            
            for pre_spike in recent_pre_spikes:
                # Calculate weight change
                weight_change = self.stdp_learning.calculate_weight_change(
                    pre_spike.timestamp, post_spike.timestamp
                )
                
                # Update synaptic weight
                current_weight = post_neuron.synaptic_weights[pre_spike.neuron_id]
                new_weight = np.clip(current_weight + weight_change, -1.0, 1.0)
                post_neuron.synaptic_weights[pre_spike.neuron_id] = new_weight
                
    def inject_emotional_stimulus(self, emotion: EmotionCategory, intensity: float):
        """Inject emotional stimulus into the network"""
        pattern = self.emotion_patterns[emotion]
        current_time = time.time()
        
        # Select neurons based on emotion sensitivity
        target_neurons = []
        for neuron in self.neurons.values():
            if neuron.emotion_sensitivity.get(emotion, 0) > 0.5:
                target_neurons.append(neuron)
                
        # Inject stimulus based on emotion pattern
        for neuron in target_neurons[:int(len(target_neurons) * intensity)]:
            # Increase membrane potential based on emotion pattern
            amplitude_profile = pattern['amplitude_profile']
            stimulus_strength = amplitude_profile['mean'] * intensity
            neuron.membrane_potential += stimulus_strength * 10  # Scale for mV
            
    def analyze_emotional_state(self, time_window: float = 1.0) -> EmotionalState:
        """Analyze current emotional state from network activity"""
        current_time = time.time()
        
        # Get recent spikes
        recent_spikes = [spike for spike in self.spike_history 
                        if current_time - spike.timestamp < time_window]
        
        if not recent_spikes:
            return EmotionalState()  # Default neutral state
            
        # Analyze spike patterns for each emotion
        emotion_scores = {}
        for emotion in EmotionCategory:
            score = self._calculate_emotion_score(recent_spikes, emotion)
            emotion_scores[emotion] = score
            
        # Map to EmotionalState structure
        emotional_state = EmotionalState(
            joy=emotion_scores.get(EmotionCategory.JOY, 0.5),
            clarity=emotion_scores.get(EmotionCategory.CLARITY, 0.5),
            tension=1.0 - emotion_scores.get(EmotionCategory.SADNESS, 0.5),
            awe=emotion_scores.get(EmotionCategory.AWE, 0.4),
            love=emotion_scores.get(EmotionCategory.LOVE, 0.5),
            curiosity=emotion_scores.get(EmotionCategory.CURIOSITY, 0.7)
        )
        
        # Calculate derived metrics
        emotional_state.resonance_score = np.mean(list(emotion_scores.values()))
        emotional_state.harmonic_alignment = self._calculate_harmonic_alignment(recent_spikes)
        emotional_state.emotional_entropy = self._calculate_emotional_entropy(emotion_scores)
        
        return emotional_state
        
    def _calculate_emotion_score(self, spikes: List[SpikeEvent], emotion: EmotionCategory) -> float:
        """Calculate score for specific emotion based on spike patterns"""
        if not spikes:
            return 0.5
            
        pattern = self.emotion_patterns[emotion]
        
        # Analyze frequency characteristics
        spike_times = [spike.timestamp for spike in spikes]
        if len(spike_times) > 1:
            intervals = np.diff(sorted(spike_times))
            mean_frequency = 1.0 / np.mean(intervals) if np.mean(intervals) > 0 else 0
            
            freq_range = pattern['frequency_range']
            freq_score = 1.0 if freq_range[0] <= mean_frequency <= freq_range[1] else 0.5
        else:
            freq_score = 0.5
            
        # Analyze amplitude characteristics
        amplitudes = [spike.amplitude for spike in spikes]
        mean_amplitude = np.mean(amplitudes)
        amplitude_profile = pattern['amplitude_profile']
        amp_score = 1.0 - abs(mean_amplitude - amplitude_profile['mean'])
        amp_score = max(0.0, min(amp_score, 1.0))
        
        # Combine scores
        emotion_score = (freq_score * 0.6 + amp_score * 0.4)
        return max(0.0, min(emotion_score, 1.0))
        
    def _calculate_harmonic_alignment(self, spikes: List[SpikeEvent]) -> float:
        """Calculate harmonic alignment of spike patterns"""
        if len(spikes) < 3:
            return 0.5
            
        spike_times = sorted([spike.timestamp for spike in spikes])
        intervals = np.diff(spike_times)
        
        # Calculate coefficient of variation (inverse of regularity)
        if np.mean(intervals) > 0:
            cv = np.std(intervals) / np.mean(intervals)
            alignment = 1.0 / (1.0 + cv)  # Higher alignment for more regular patterns
        else:
            alignment = 0.5
            
        return max(0.0, min(alignment, 1.0))
        
    def _calculate_emotional_entropy(self, emotion_scores: Dict[EmotionCategory, float]) -> float:
        """Calculate emotional entropy (diversity of emotional states)"""
        scores = list(emotion_scores.values())
        scores = [s for s in scores if s > 0]  # Remove zero scores
        
        if not scores:
            return 0.0
            
        # Normalize scores
        total = sum(scores)
        probabilities = [s / total for s in scores]
        
        # Calculate Shannon entropy
        entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
        
        # Normalize to [0, 1]
        max_entropy = np.log2(len(probabilities))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        return max(0.0, min(normalized_entropy, 1.0))

# ========================================
# MAIN SPIKING EMOTION ENGINE PLUGIN
# ========================================

class SpikingEmotionEngine(PluginInterface):
    """Main spiking emotion engine plugin for the AI system"""
    
    def __init__(self):
        super().__init__("SpikingEmotionEngine", "1.0.0")
        self.network = None
        self.event_bus = None
        self.config_manager = None
        self.processing_active = False
        self.processing_task = None
        self.emotion_history = deque(maxlen=1000)
        self.reflex_handlers = []
        
    async def initialize(self, config: Dict) -> bool:
        """Initialize the spiking emotion engine"""
        try:
            logger.info("Initializing Spiking Emotion Engine...")
            
            # Store configuration
            self.config = config
            
            # Initialize the spiking neural network
            network_config = config.get('network', {})
            self.network = EmotionalSpikingNetwork(network_config)
            
            # Register event hooks
            if hasattr(self, 'event_bus') and self.event_bus:
                self.event_bus.subscribe(EventType.QUERY_RECEIVED, self._handle_query_event)
                self.event_bus.subscribe(EventType.EMOTIONAL_STATE_CHANGED, self._handle_emotion_change)
                
            # Start background processing
            self.processing_active = True
            self.processing_task = asyncio.create_task(self._background_processing())
            
            logger.info("Spiking Emotion Engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Spiking Emotion Engine: {e}")
            return False
            
    async def process(self, data: Dict, context: Dict) -> Dict:
        """Process data through the spiking emotion engine"""
        try:
            start_time = time.time()
            
            # Extract emotional content from input
            emotional_content = self._extract_emotional_content(data)
            
            # Inject emotional stimuli into the network
            for emotion, intensity in emotional_content.items():
                if intensity > 0.1:  # Threshold for significant emotional content
                    self.network.inject_emotional_stimulus(emotion, intensity)
                    
            # Simulate network for processing time
            simulation_steps = int(self.config.get('simulation_steps', 10))
            all_spikes = []
            
            for _ in range(simulation_steps):
                spikes = self.network.simulate_step()
                all_spikes.extend(spikes)
                await asyncio.sleep(0.001)  # Small delay for real-time processing
                
            # Analyze resulting emotional state
            emotional_state = self.network.analyze_emotional_state()
            
            # Store in history
            self.emotion_history.append({
                'timestamp': time.time(),
                'emotional_state': asdict(emotional_state),
                'input_emotions': emotional_content,
                'spike_count': len(all_spikes)
            })
            
            # Trigger reflex handlers if needed
            await self._trigger_reflex_handlers(emotional_state, context)
            
            processing_time = time.time() - start_time
            
            return {
                'emotional_state': asdict(emotional_state),
                'spike_events': len(all_spikes),
                'processing_time': processing_time,
                'network_activity': {
                    'total_spikes': len(all_spikes),
                    'active_neurons': len(set(spike.neuron_id for spike in all_spikes)),
                    'dominant_emotion': self._get_dominant_emotion(emotional_state)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in spiking emotion processing: {e}")
            return {'error': str(e)}
            
    async def shutdown(self) -> bool:
        """Shutdown the spiking emotion engine"""
        try:
            logger.info("Shutting down Spiking Emotion Engine...")
            
            # Stop background processing
            self.processing_active = False
            if self.processing_task:
                self.processing_task.cancel()
                try:
                    await self.processing_task
                except asyncio.CancelledError:
                    pass
                    
            logger.info("Spiking Emotion Engine shutdown complete")
            return True
            
        except Exception as e:
            logger.error(f"Error shutting down Spiking Emotion Engine: {e}")
            return False
            
    def register_reflex_handler(self, handler: Callable):
        """Register an emotional reflex handler"""
        self.reflex_handlers.append(handler)
        logger.info(f"Registered emotional reflex handler: {handler.__name__}")
        
    async def _background_processing(self):
        """Background processing loop for continuous network simulation"""
        try:
            while self.processing_active:
                # Continuous network simulation
                self.network.simulate_step()
                
                # Periodic emotional state analysis
                if len(self.emotion_history) == 0 or time.time() - self.emotion_history[-1]['timestamp'] > 1.0:
                    emotional_state = self.network.analyze_emotional_state()
                    
                    # Publish emotional state changes
                    if self.event_bus:
                        await self.event_bus.publish(EventType.EMOTIONAL_STATE_CHANGED, {
                            'emotional_state': asdict(emotional_state),
                            'source': 'background_processing'
                        })
                        
                await asyncio.sleep(0.01)  # 100Hz simulation rate
                
        except asyncio.CancelledError:
            logger.info("Background processing cancelled")
        except Exception as e:
            logger.error(f"Error in background processing: {e}")
            
    async def _handle_query_event(self, event: Dict):
        """Handle incoming query events"""
        try:
            query_data = event.get('data', {})
            emotional_content = self._extract_emotional_content(query_data)
            
            # Inject emotional stimuli based on query
            for emotion, intensity in emotional_content.items():
                if intensity > 0.2:
                    self.network.inject_emotional_stimulus(emotion, intensity)
                    
        except Exception as e:
            logger.error(f"Error handling query event: {e}")
            
    async def _handle_emotion_change(self, event: Dict):
        """Handle emotional state change events"""
        try:
            # Log emotional state changes for analysis
            logger.debug(f"Emotional state change detected: {event}")
            
        except Exception as e:
            logger.error(f"Error handling emotion change event: {e}")
            
    def _extract_emotional_content(self, data: Dict) -> Dict[EmotionCategory, float]:
        """Extract emotional content from input data"""
        content = data.get('content', '').lower()
        emotional_content = {}
        
        # Simple keyword-based emotional content extraction
        emotion_keywords = {
            EmotionCategory.JOY: ['happy', 'joy', 'excited', 'wonderful', 'amazing', 'great'],
            EmotionCategory.SADNESS: ['sad', 'depressed', 'unhappy', 'sorrow', 'grief'],
            EmotionCategory.ANGER: ['angry', 'mad', 'furious', 'rage', 'annoyed'],
            EmotionCategory.FEAR: ['afraid', 'scared', 'terrified', 'anxious', 'worried'],
            EmotionCategory.SURPRISE: ['surprised', 'shocked', 'amazed', 'astonished'],
            EmotionCategory.DISGUST: ['disgusted', 'revolted', 'repulsed', 'sick'],
            EmotionCategory.LOVE: ['love', 'adore', 'cherish', 'affection', 'care'],
            EmotionCategory.CURIOSITY: ['curious', 'wonder', 'interested', 'intrigued'],
            EmotionCategory.AWE: ['awe', 'magnificent', 'breathtaking', 'sublime'],
            EmotionCategory.CLARITY: ['clear', 'understand', 'obvious', 'evident']
        }
        
        for emotion, keywords in emotion_keywords.items():
            intensity = 0.0
            for keyword in keywords:
                if keyword in content:
                    intensity += 0.2
            emotional_content[emotion] = min(intensity, 1.0)
            
        return emotional_content
        
    def _get_dominant_emotion(self, emotional_state: EmotionalState) -> str:
        """Get the dominant emotion from emotional state"""
        emotion_values = {
            'joy': emotional_state.joy,
            'clarity': emotional_state.clarity,
            'tension': emotional_state.tension,
            'awe': emotional_state.awe,
            'love': emotional_state.love,
            'curiosity': emotional_state.curiosity
        }
        
        return max(emotion_values.items(), key=lambda x: x[1])[0]
        
    async def _trigger_reflex_handlers(self, emotional_state: EmotionalState, context: Dict):
        """Trigger registered reflex handlers"""
        for handler in self.reflex_handlers:
            try:
                await handler(emotional_state, context)
            except Exception as e:
                logger.error(f"Error in reflex handler {handler.__name__}: {e}")
                
    def get_network_statistics(self) -> Dict:
        """Get current network statistics"""
        if not self.network:
            return {}
            
        recent_spikes = [spike for spike in self.network.spike_history 
                        if time.time() - spike.timestamp < 1.0]
        
        neuron_activity = defaultdict(int)
        for spike in recent_spikes:
            neuron_activity[spike.neuron_type.value] += 1
            
        return {
            'total_neurons': len(self.network.neurons),
            'recent_spike_count': len(recent_spikes),
            'neuron_activity': dict(neuron_activity),
            'emotion_history_length': len(self.emotion_history),
            'network_connectivity': sum(len(neuron.synaptic_weights) 
                                      for neuron in self.network.neurons.values())
        }