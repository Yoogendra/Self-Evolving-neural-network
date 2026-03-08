import { useState, useEffect, useCallback, useRef } from 'react';

const useSimulation = (demoMode = false) => {
  const [isRunning, setIsRunning] = useState(false);
  const [generation, setGeneration] = useState(0);
  const [bestFitness, setBestFitness] = useState(0.0);
  const [activeSpecies, setActiveSpecies] = useState(1);
  const [nodes, setNodes] = useState([]);
  const [edges, setEdges] = useState([]);
  const [events, setEvents] = useState([]);
  const [fitnessHistory, setFitnessHistory] = useState([]);
  const [complexityHistory, setComplexityHistory] = useState([]);
  const [speed, setSpeed] = useState(1);
  const [mutationRate, setMutationRate] = useState(0.1);
  const [populationSize, setPopulationSize] = useState(8);
  const [speciesThreshold, setSpeciesThreshold] = useState(0.3);
  const [isConnected, setIsConnected] = useState(false);
  
  const wsRef = useRef(null);
  const nodeIdCounter = useRef(0);

  const normalizeEvents = useCallback((incoming) => {
    if (!Array.isArray(incoming)) return [];
    return incoming
      .map((e) => {
        if (typeof e === 'string') return e;
        if (e && typeof e === 'object') {
          if (typeof e.message === 'string') return e.message;
          if (typeof e.data?.message === 'string') return e.data.message;
        }
        try {
          return JSON.stringify(e);
        } catch {
          return String(e);
        }
      })
      .filter(Boolean);
  }, []);

  // Initialize with a simple network
  const initializeNetwork = useCallback(() => {
    const initialNodes = [
      { id: '0', type: 'input', position: { x: 100, y: 200 }, data: { label: 'Input', active: true } },
      { id: '1', type: 'hidden', position: { x: 300, y: 150 }, data: { label: 'H1', active: false } },
      { id: '2', type: 'hidden', position: { x: 300, y: 250 }, data: { label: 'H2', active: false } },
      { id: '3', type: 'output', position: { x: 500, y: 200 }, data: { label: 'Output', active: false } },
    ];

    const initialEdges = [
      { id: 'e0-1', source: '0', target: '1', animated: true, style: { stroke: '#a855f7', strokeWidth: 2 } },
      { id: 'e0-2', source: '0', target: '2', animated: true, style: { stroke: '#a855f7', strokeWidth: 2 } },
      { id: 'e1-3', source: '1', target: '3', animated: true, style: { stroke: '#a855f7', strokeWidth: 2 } },
      { id: 'e2-3', source: '2', target: '3', animated: true, style: { stroke: '#a855f7', strokeWidth: 2 } },
    ];

    setNodes(initialNodes);
    setEdges(initialEdges);
    nodeIdCounter.current = 4;
  }, []);

  // Generate random events for demo mode
  const generateEvent = useCallback((gen) => {
    const eventTypes = [
      `[GEN ${gen}] New node added between ${Math.floor(Math.random() * 5)} -> ${Math.floor(Math.random() * 5) + 3}`,
      `[INFO] Species ${Math.floor(Math.random() * 5) + 1} went extinct`,
      `[MUTATION] Connection weight optimized: ${Math.random().toFixed(3)}`,
      `[FITNESS] New best fitness achieved: ${(Math.random() * 0.1 + 0.9).toFixed(4)}`,
      `[STRUCTURE] Layer pruned for efficiency`,
      `[EVOLUTION] Crossover event occurred between top performers`,
    ];
    
    return eventTypes[Math.floor(Math.random() * eventTypes.length)];
  }, []);

  // Simulation step
  const simulationStep = useCallback(() => {
    setGeneration(prev => prev + 1);
    
    // Update fitness with some randomness but general upward trend
    setBestFitness(prev => {
      const improvement = (Math.random() - 0.3) * 0.02;
      const newFitness = Math.max(0, Math.min(1, prev + improvement));
      return newFitness;
    });

    // Update species count
    setActiveSpecies(prev => {
      const change = Math.random() > 0.7 ? (Math.random() > 0.5 ? 1 : -1) : 0;
      return Math.max(1, prev + change);
    });

    // Occasionally add/remove nodes (network evolution)
    if (Math.random() > 0.8) {
      setNodes(prevNodes => {
        if (Math.random() > 0.5 && prevNodes.length < 10) {
          // Add node
          const newNode = {
            id: nodeIdCounter.current.toString(),
            type: 'hidden',
            position: { 
              x: 200 + Math.random() * 200, 
              y: 100 + Math.random() * 200 
            },
            data: { label: `H${nodeIdCounter.current}`, active: Math.random() > 0.5 }
          };
          nodeIdCounter.current += 1;
          return [...prevNodes, newNode];
        } else if (prevNodes.length > 4) {
          // Remove node (keep input/output)
          const hiddenNodes = prevNodes.filter(n => n.type === 'hidden');
          if (hiddenNodes.length > 1) {
            const nodeToRemove = hiddenNodes[Math.floor(Math.random() * hiddenNodes.length)];
            return prevNodes.filter(n => n.id !== nodeToRemove.id);
          }
        }
        return prevNodes;
      });
    }

    // Add event
    setEvents(prev => {
      const newEvent = generateEvent(generation + 1);
      return [...prev.slice(-9), newEvent]; // Keep last 10 events
    });

    // Update history
    setFitnessHistory(prev => {
      const newEntry = { generation: generation + 1, fitness: bestFitness };
      return [...prev.slice(-19), newEntry]; // Keep last 20 entries
    });

    setComplexityHistory(prev => {
      const complexity = nodes.length + edges.length;
      const newEntry = { generation: generation + 1, complexity };
      return [...prev.slice(-19), newEntry];
    });
  }, [generation, bestFitness, nodes.length, edges.length, generateEvent]);

  // WebSocket connection
  const connectWebSocket = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    try {
      // Close existing connection if any
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }

      console.log('Attempting to connect to WebSocket...');
      const ws = new WebSocket('ws://127.0.0.1:8000/ws'); // Use 127.0.0.1 instead of localhost
      wsRef.current = ws;

      ws.onopen = () => {
        console.log('Connected to SENN backend');
        setIsConnected(true);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          console.log('WebSocket message received:', data);
          
          switch (data.type) {
            case 'initial_state':
              const state = data.data;
              setGeneration(state.generation);
              setBestFitness(state.best_fitness);
              setActiveSpecies(state.active_species);
              setFitnessHistory(state.fitness_history || []);
              setComplexityHistory(state.complexity_history || []);
              setEvents(normalizeEvents(state.events));
              setIsRunning(state.is_running);
              break;
              
            case 'update':
              const update = data.data;
              setGeneration(update.generation);
              setBestFitness(update.best_fitness);
              setActiveSpecies(update.active_species);
              setFitnessHistory(prev => [...prev.slice(-19), {
                generation: update.generation,
                fitness: update.best_fitness
              }]);
              setComplexityHistory(prev => [...prev.slice(-19), {
                generation: update.generation,
                complexity: update.complexity
              }]);
              if (update.event) {
                setEvents(prev => [...prev.slice(-9), update.event]);
              }
              break;
              
            case 'event':
              if (data.data && data.data.message) {
                setEvents(prev => [...prev.slice(-9), data.data.message]);
              }
              break;
              
            case 'complete':
              setIsRunning(false);
              if (data.data && data.data.message) {
                setEvents(prev => [...prev.slice(-9), data.data.message]);
              }
              break;
              
            case 'error':
              setIsRunning(false);
              if (data.data && data.data.message) {
                setEvents(prev => [...prev.slice(-9), data.data.message]);
              }
              break;
              
            case 'pong':
              // Ping response, ignore
              break;
              
            default:
              console.log('Unknown WebSocket message type:', data.type);
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setIsConnected(false);
      };

      ws.onclose = () => {
        console.log('WebSocket connection closed');
        setIsConnected(false);
        // Try to reconnect after 3 seconds
        setTimeout(() => {
          if (!demoMode) {
            connectWebSocket();
          }
        }, 3000);
      };

    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      setIsConnected(false);
      // Don't crash the app on WebSocket connection failure
    }
  }, [demoMode]);

  // API calls - Use HTTP only for now
  const startEvolution = useCallback(async () => {
    try {
      console.log('Attempting to start evolution...');
      const response = await fetch('/api/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      console.log('Start response status:', response.status);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      const result = await response.json();
      console.log('Started evolution via HTTP:', result);
      setIsConnected(true);
      setIsRunning(true);
    } catch (error) {
      console.error('Failed to start evolution:', error?.message || error, error);
      setIsConnected(false);
      // Still set running to allow UI to work
      setIsRunning(true);
    }
  }, []);

  const pauseEvolution = useCallback(async () => {
    try {
      console.log('Attempting to pause evolution...');
      const response = await fetch('/api/pause', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      console.log('Pause response status:', response.status);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      const result = await response.json();
      console.log('Paused evolution via HTTP:', result);
      setIsConnected(true);
      setIsRunning(false);
    } catch (error) {
      console.error('Failed to pause evolution:', error?.message || error, error);
      setIsConnected(false);
      setIsRunning(false);
    }
  }, []);

  const resetEvolution = useCallback(async () => {
    try {
      console.log('Attempting to reset evolution...');
      const response = await fetch('/api/reset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      console.log('Reset response status:', response.status);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      const result = await response.json();
      console.log('Reset evolution via HTTP:', result);
      setIsConnected(true);
      setGeneration(0);
      setBestFitness(0.0);
      setActiveSpecies(1);
      setEvents([]);
      setFitnessHistory([]);
      setComplexityHistory([]);
      setIsRunning(false);
    } catch (error) {
      console.error('Failed to reset evolution:', error?.message || error, error);
      setIsConnected(false);
      // Still reset the UI state
      setGeneration(0);
      setBestFitness(0.0);
      setActiveSpecies(1);
      setEvents([]);
      setFitnessHistory([]);
      setComplexityHistory([]);
      setIsRunning(false);
    }
  }, []);

  // Control functions
  const start = useCallback(() => {
    if (demoMode) {
      setIsRunning(true);
    } else {
      (async () => {
        setIsRunning(false);
        setGeneration(0);
        setBestFitness(0.0);
        setActiveSpecies(1);
        setEvents([]);
        setFitnessHistory([]);
        setComplexityHistory([]);

        await resetEvolution();
        await startEvolution();
      })();
    }
  }, [demoMode, resetEvolution, startEvolution]);

  const pause = useCallback(() => {
    if (demoMode) {
      setIsRunning(false);
    } else {
      pauseEvolution();
    }
  }, [demoMode, pauseEvolution]);

  const reset = useCallback(() => {
    if (demoMode) {
      setIsRunning(false);
      setGeneration(0);
      setBestFitness(0.0);
      setActiveSpecies(1);
      setEvents([]);
      setFitnessHistory([]);
      setComplexityHistory([]);
      initializeNetwork();
    } else {
      resetEvolution();
    }
  }, [demoMode, resetEvolution, initializeNetwork]);

  // HTTP polling as fallback when WebSocket fails
  const pollStatus = useCallback(async () => {
    try {
      const response = await fetch('/api/status', { cache: 'no-cache' });
      console.log('Status response:', response.status);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      const status = await response.json();
      console.log('Polled status via HTTP:', status);
      
      // Mark as connected when we get successful response
      setIsConnected(true);
      
      setGeneration(status.generation || 0);
      setBestFitness(status.best_fitness || 0.0);
      setActiveSpecies(status.active_species || 1);
      setFitnessHistory(status.fitness_history || []);
      setComplexityHistory(status.complexity_history || []);
      setEvents(normalizeEvents(status.events));
      setIsRunning(status.is_running || false);
    } catch (error) {
      // Mark as disconnected on error
      setIsConnected(false);
      console.error('Failed to poll status:', error?.message || error, error);
    }
  }, [normalizeEvents]);

  // Initialize connection and network
  useEffect(() => {
    if (!demoMode) {
      // Skip WebSocket for now, use HTTP polling only
      initializeNetwork();
      
      // Start HTTP polling for data
      const pollingInterval = setInterval(() => {
        pollStatus();
      }, 2000); // Poll every 2 seconds
      
      return () => clearInterval(pollingInterval);
    } else {
      initializeNetwork();
    }
  }, [demoMode, initializeNetwork, pollStatus]);

  // Demo mode effect
  useEffect(() => {
    if (demoMode && isRunning) {
      const interval = setInterval(() => {
        simulationStep();
      }, 1000 / speed);
      
      return () => clearInterval(interval);
    }
  }, [demoMode, isRunning, speed, simulationStep]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  return {
    // State
    isRunning,
    generation,
    bestFitness,
    activeSpecies,
    nodes,
    edges,
    events,
    fitnessHistory,
    complexityHistory,
    speed,
    mutationRate,
    populationSize,
    speciesThreshold,
    isConnected,
    
    // Controls
    start,
    pause,
    reset,
    setSpeed,
    setMutationRate,
    setPopulationSize,
    setSpeciesThreshold,
  };
};

export default useSimulation;
