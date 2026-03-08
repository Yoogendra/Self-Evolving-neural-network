import React, { useState } from 'react';
import { motion } from 'framer-motion';
import ControlPanel from './ControlPanel';
import NetworkGraph from './NetworkGraph';
import MetricsPanel from './MetricsPanel';
import EventLog from './EventLog';
import useSimulation from '../hooks/useSimulation';
import { neuralBg } from '../assets/images';

const Dashboard = () => {
  console.log('Dashboard component is rendering');
  const [demoMode, setDemoMode] = useState(false); // Back to live mode
  const [isPanelCollapsed, setIsPanelCollapsed] = useState(false);
  
  const simulation = useSimulation(demoMode);
  
  return (
    <div className="min-h-screen relative" style={{
      backgroundImage: `url(${neuralBg})`,
      backgroundSize: 'cover',
      backgroundPosition: 'center',
      backgroundRepeat: 'no-repeat',
      backgroundAttachment: 'fixed'
    }}>

      {/* Main Content - Scrollable Container */}
      <div className={`transition-all duration-300 overflow-y-auto ${
        isPanelCollapsed ? 'ml-16' : 'ml-80'
      }`} style={{ height: '100vh' }}>
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="glass glass-border m-6 rounded-lg p-6"
        >
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-white mb-2">
                SENN - <span className="text-electric-blue">Self Evolving Neural Network</span>
              </h1>
              <p className="text-gray-400">
                Real-time visualization of neural architecture evolution
              </p>
            </div>
            <div className="text-right">
              <div className="text-sm text-gray-400">Status</div>
              <div className={`text-lg font-mono ${
                simulation.isConnected ? 'text-neon-green' : 'text-gray-500'
              }`}>
                {simulation.isConnected ? 'LIVE MODE' : 'CONNECTING...'}
              </div>
            </div>
          </div>
        </motion.div>

        {/* Main Content Area - Side by Side */}
        <div className="flex flex-col lg:flex-row gap-6 px-6 pb-6">
          {/* Left Section - Network Graph Area */}
          <div className="w-full lg:w-1/2">
            <div className="glass glass-border rounded-lg p-6" style={{ minHeight: '36rem' }}>
              <h3 className="text-xl font-semibold text-white mb-4">Network Visualization</h3>
              <div className="flex items-center justify-center text-gray-400" style={{ height: '20rem' }}>
                <div className="text-center">
                  <div className="text-6xl mb-4">🧠</div>
                  <p>Network Graph Component</p>
                  <p className="text-sm mt-2">Neural architecture visualization will appear here</p>
                </div>
              </div>
            </div>
          </div>

          {/* Right Section - Event Log */}
          <div className="w-full lg:w-1/2">
            <div className="glass glass-border rounded-lg" style={{ minHeight: '36rem' }}>
              <EventLog events={simulation.events} />
            </div>
          </div>
        </div>

        {/* Key Stats Section */}
        <div className="px-6 pb-6">
          <div className="glass glass-border rounded-lg p-6">
            <h3 className="text-xl font-semibold text-white mb-6">Evolution Statistics</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="glass glass-border rounded-lg p-4">
                <h4 className="text-sm text-gray-400 mb-2">Current Generation</h4>
                <p className="text-2xl font-bold text-electric-blue">{simulation.generation}</p>
              </div>
              <div className="glass glass-border rounded-lg p-4">
                <h4 className="text-sm text-gray-400 mb-2">Active Species</h4>
                <p className="text-2xl font-bold text-neon-green">{simulation.activeSpecies}</p>
              </div>
              <div className="glass glass-border rounded-lg p-4">
                <h4 className="text-sm text-gray-400 mb-2">Best Fitness</h4>
                <p className="text-2xl font-bold text-glowing-purple">{(simulation.bestFitness * 100).toFixed(1)}%</p>
              </div>
            </div>
          </div>
        </div>

        {/* Additional Content for Scroll */}
        <div className="px-6 pb-6">
          <div className="glass glass-border rounded-lg p-6">
            <h3 className="text-xl font-semibold text-white mb-4">Evolution Details</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h4 className="text-lg font-medium text-white mb-3">Configuration</h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Population Size:</span>
                    <span className="text-white">{simulation.populationSize}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Mutation Rate:</span>
                    <span className="text-white">{(simulation.mutationRate * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Species Threshold:</span>
                    <span className="text-white">{simulation.speciesThreshold}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Evolution Speed:</span>
                    <span className="text-white">{simulation.speed}x</span>
                  </div>
                </div>
              </div>
              <div>
                <h4 className="text-lg font-medium text-white mb-3">System Status</h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Connection:</span>
                    <span className={simulation.isConnected ? 'text-neon-green' : 'text-red-500'}>
                      {simulation.isConnected ? 'Connected' : 'Disconnected'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Evolution:</span>
                    <span className={simulation.isRunning ? 'text-neon-green' : 'text-gray-500'}>
                      {simulation.isRunning ? 'Running' : 'Stopped'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Total Events:</span>
                    <span className="text-white">{simulation.events?.length || 0}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Mode:</span>
                    <span className="text-electric-blue">Live</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Charts Section - Metrics Panel at the very bottom */}
        <div className="px-6 pb-12">
          <div className="glass glass-border rounded-lg p-6">
            <h3 className="text-xl font-semibold text-white mb-6">Performance Metrics</h3>
            <MetricsPanel
              generation={simulation.generation}
              bestFitness={simulation.bestFitness}
              activeSpecies={simulation.activeSpecies}
              fitnessHistory={simulation.fitnessHistory}
              complexityHistory={simulation.complexityHistory}
            />
          </div>
        </div>
      </div>

      {/* Connection Status Badge */}
      {!demoMode && (
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          className="fixed top-6 right-6 z-20"
        >
          <div className={`glass glass-border rounded-full px-4 py-2 flex items-center space-x-2 border ${
            simulation.isConnected ? 'border-electric-blue' : 'border-red-500'
          }`}>
            <div className={`w-2 h-2 rounded-full ${
              simulation.isConnected ? 'bg-neon-green' : 'bg-red-500'
            }`} />
            <span className="text-xs font-mono text-gray-300">
              {simulation.isConnected ? 'LIVE MODE' : 'CONNECTING...'}
            </span>
          </div>
        </motion.div>
      )}

      {/* Control Panel - Fixed Position */}
      <div className="fixed left-0 top-0 h-full z-10">
        <ControlPanel
          isRunning={simulation.isRunning}
          onStart={simulation.start}
          onPause={simulation.pause}
          onReset={simulation.reset}
          speed={simulation.speed}
          onSpeedChange={simulation.setSpeed}
          mutationRate={simulation.mutationRate}
          onMutationRateChange={simulation.setMutationRate}
          populationSize={simulation.populationSize}
          onPopulationSizeChange={simulation.setPopulationSize}
          speciesThreshold={simulation.speciesThreshold}
          onSpeciesThresholdChange={simulation.setSpeciesThreshold}
          demoMode={demoMode}
          onDemoModeChange={setDemoMode}
          onCollapseChange={setIsPanelCollapsed}
        />
      </div>
    </div>
  );
};

export default Dashboard;
