import React, { useCallback, useMemo } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  addEdge,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

// Custom node component
const NeuralNode = ({ data, selected }) => {
  const isActive = data.active;
  const nodeType = data.nodeType || 'hidden';
  
  const getNodeColor = () => {
    if (nodeType === 'input') return '#00d4ff';
    if (nodeType === 'output') return '#00ff88';
    return isActive ? '#a855f7' : '#6b7280';
  };

  const getNodeSize = () => {
    if (nodeType === 'input' || nodeType === 'output') return 40;
    return 30;
  };

  return (
    <div
      className={`relative flex items-center justify-center rounded-full border-2 transition-all duration-300 ${
        selected ? 'ring-4 ring-electric-blue ring-opacity-50' : ''
      }`}
      style={{
        width: getNodeSize(),
        height: getNodeSize(),
        borderColor: getNodeColor(),
        backgroundColor: `${getNodeColor()}20`,
        boxShadow: isActive ? `0 0 20px ${getNodeColor()}` : 'none',
        animation: isActive ? 'neuralPulse 1.5s ease-in-out infinite' : 'none',
      }}
    >
      <div
        className="rounded-full"
        style={{
          width: getNodeSize() - 8,
          height: getNodeSize() - 8,
          backgroundColor: getNodeColor(),
          opacity: isActive ? 1 : 0.6,
        }}
      />
      <div className="absolute -bottom-6 text-xs font-mono text-electric-blue whitespace-nowrap">
        {data.label}
      </div>
    </div>
  );
};

const nodeTypes = {
  custom: NeuralNode,
};

const NetworkGraph = ({ nodes, edges, onNodesChange, onEdgesChange, onConnect }) => {
  const [internalNodes, setInternalNodes, onNodesChangeHandler] = useNodesState(nodes);
  const [internalEdges, setInternalEdges, onEdgesChangeHandler] = useEdgesState(edges);

  // Update internal state when props change
  React.useEffect(() => {
    setInternalNodes(nodes);
  }, [nodes, setInternalNodes]);

  React.useEffect(() => {
    setInternalEdges(edges);
  }, [edges, setInternalEdges]);

  const onConnectHandler = useCallback(
    (params) => {
      const newEdge = {
        ...params,
        animated: true,
        style: {
          stroke: '#a855f7',
          strokeWidth: 2,
          filter: 'drop-shadow(0 0 3px rgba(168, 85, 247, 0.8))',
        },
      };
      if (onConnect) {
        onConnect(newEdge);
      } else {
        setInternalEdges((eds) => addEdge(newEdge, eds));
      }
    },
    [onConnect, setInternalEdges]
  );

  // Process nodes to add nodeType
  const processedNodes = useMemo(() => {
    return internalNodes.map((node) => ({
      ...node,
      type: 'custom',
      data: {
        ...node.data,
        nodeType: node.type || 'hidden',
      },
    }));
  }, [internalNodes]);

  return (
    <div className="w-full h-full">
      <ReactFlow
        nodes={processedNodes}
        edges={internalEdges}
        onNodesChange={onNodesChange || onNodesChangeHandler}
        onEdgesChange={onEdgesChange || onEdgesChangeHandler}
        onConnect={onConnectHandler}
        nodeTypes={nodeTypes}
        fitView
        attributionPosition="bottom-left"
        style={{
          background: 'transparent',
        }}
      >
        <Background
          color="#1a1a2e"
          gap={16}
          size={1}
          variant="dots"
        />
        <Controls
          position="top-right"
          className="glass glass-border"
        />
        <MiniMap
          position="bottom-left"
          className="glass glass-border"
          nodeColor={(node) => {
            const nodeType = node.data?.nodeType || 'hidden';
            if (nodeType === 'input') return '#00d4ff';
            if (nodeType === 'output') return '#00ff88';
            return '#a855f7';
          }}
          maskColor="rgba(10, 10, 15, 0.8)"
        />
      </ReactFlow>
    </div>
  );
};

export default NetworkGraph;
