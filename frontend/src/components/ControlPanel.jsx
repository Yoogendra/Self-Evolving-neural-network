import React, { useState, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import { Play, Pause, RotateCcw, Zap, Settings, ChevronLeft, ChevronRight } from 'lucide-react';

const ControlPanel = ({ 
  isRunning, 
  onStart, 
  onPause, 
  onReset, 
  speed, 
  onSpeedChange,
  mutationRate,
  onMutationRateChange,
  populationSize,
  onPopulationSizeChange,
  speciesThreshold,
  onSpeciesThresholdChange,
  demoMode,
  onDemoModeChange,
  onCollapseChange
}) => {
  const [isCollapsed, setIsCollapsed] = useState(false);

  const handleCollapseToggle = () => {
    const newCollapsedState = !isCollapsed;
    setIsCollapsed(newCollapsedState);
    if (onCollapseChange) {
      onCollapseChange(newCollapsedState);
    }
  };

  const SpeedButton = ({ value, label }) => (
    <button
      onClick={() => onSpeedChange(value)}
      className={`px-3 py-1 rounded text-xs font-mono transition-all ${
        speed === value
          ? 'bg-electric-blue text-neural-bg'
          : 'glass glass-border text-electric-blue hover:bg-electric-blue hover:bg-opacity-20'
      }`}
    >
      {label}
    </button>
  );

  const SliderControl = ({ label, value, onChange, min, max, step = 0.01 }) => {
  const [localValue, setLocalValue] = useState(value);

  const handleChange = (e) => {
    const newValue = Number(e.target.value);
    const clampedValue = Math.max(min, Math.min(max, newValue));
    setLocalValue(clampedValue);
    onChange(clampedValue);
  };

  // Update local value when prop changes
  useEffect(() => {
    setLocalValue(value);
  }, [value]);

  const percentage = ((localValue - min) / (max - min)) * 100;

  return (
    <div className="space-y-2">
      <div className="flex justify-between text-xs">
        <span className="text-gray-400">{label}</span>
        <div className="flex items-center space-x-2">
          <div className="w-full bg-gray-700 rounded-lg h-2 relative">
            <div 
              className="h-full bg-electric-blue rounded-lg transition-all duration-300"
              style={{ width: `${percentage}%` }}
            />
          </div>
          <input
            type="number"
            min={min}
            max={max}
            step={step}
            value={localValue}
            onChange={handleChange}
            className="w-20 px-2 py-1 text-xs text-electric-blue font-mono bg-gray-700 border border-electric-blue rounded focus:outline-none focus:ring-2 focus:ring-electric-blue focus:ring-opacity-50"
          />
        </div>
      </div>
    </div>
  );
};

  return (
    <>
      {/* Collapse Toggle - Fixed position relative to viewport */}
      <button
        onClick={handleCollapseToggle}
        className={`fixed w-8 h-8 glass glass-border rounded-full flex items-center justify-center text-electric-blue hover:bg-electric-blue hover:bg-opacity-20 transition-all z-20 ${
          isCollapsed ? 'left-12' : 'left-80'
        } top-8`}
      >
        {isCollapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
      </button>

      <motion.div
        initial={{ x: -300 }}
        animate={{ x: isCollapsed ? -280 : 0 }}
        transition={{ type: 'spring', stiffness: 300, damping: 30 }}
        className="fixed left-0 top-0 h-full z-10"
      >
        <div className={`glass glass-border h-full transition-all duration-300 ${
          isCollapsed ? 'w-16' : 'w-80'
        }`}>

        {/* Content - Only visible when not collapsed */}
        {!isCollapsed && (
          <div className="p-6 space-y-6 h-full overflow-y-auto">
            {/* Header */}
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 rounded-full bg-gradient-to-r from-electric-blue to-glowing-purple flex items-center justify-center">
                <Zap size={20} className="text-neural-bg" />
              </div>
              <div>
                <h2 className="text-lg font-semibold text-white">SENN Control</h2>
                <p className="text-xs text-gray-400">Neural Evolution</p>
              </div>
            </div>

            {/* Demo Mode Toggle */}
            <div className="glass glass-border rounded-lg p-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <Settings size={16} className="text-electric-blue" />
                  <span className="text-sm text-gray-300">Demo Mode</span>
                </div>
                <button
                  onClick={() => onDemoModeChange && onDemoModeChange(!demoMode)}
                  className={`w-12 h-6 rounded-full transition-all ${
                    demoMode ? 'bg-electric-blue' : 'bg-gray-600'
                  }`}
                >
                  <div className={`w-5 h-5 bg-white rounded-full transition-all transform ${
                    demoMode ? 'translate-x-6' : 'translate-x-0.5'
                  }`} />
                </button>
              </div>
            </div>

            {/* Simulation Controls */}
            <div className="space-y-4">
              <h3 className="text-sm font-medium text-gray-300 flex items-center space-x-2">
                <Zap size={14} className="text-electric-blue" />
                <span>Simulation</span>
              </h3>
              
              <div className="flex space-x-2">
                <button
                  onClick={isRunning ? onPause : onStart}
                  className={`flex-1 glass glass-border rounded-lg py-2 px-4 flex items-center justify-center space-x-2 transition-all ${
                    isRunning 
                      ? 'text-neon-red hover:bg-red-500 hover:bg-opacity-20' 
                      : 'text-neon-green hover:bg-green-500 hover:bg-opacity-20'
                  }`}
                >
                  {isRunning ? <Pause size={16} /> : <Play size={16} />}
                  <span className="text-sm font-medium">{isRunning ? 'Pause' : 'Start'}</span>
                </button>
                
                <button
                  onClick={onReset}
                  className="glass glass-border rounded-lg py-2 px-4 flex items-center justify-center space-x-2 text-gray-400 hover:text-white transition-all"
                >
                  <RotateCcw size={16} />
                  <span className="text-sm font-medium">Reset</span>
                </button>
              </div>

              {/* Speed Control */}
              <div className="space-y-2">
                <label className="text-xs text-gray-400">Speed</label>
                <div className="flex space-x-2">
                  <SpeedButton value={1} label="1x" />
                  <SpeedButton value={2} label="2x" />
                  <SpeedButton value={4} label="4x" />
                  <SpeedButton value={10} label="MAX" />
                </div>
              </div>
            </div>

            {/* Hyperparameters */}
            <div className="space-y-4">
              <h3 className="text-sm font-medium text-gray-300">Hyperparameters</h3>
              
              <SliderControl
                label="Mutation Rate"
                value={mutationRate}
                onChange={onMutationRateChange}
                min={0.01}
                max={0.5}
                step={0.01}
              />
              
              <SliderControl
                label="Population Size"
                value={populationSize}
                onChange={onPopulationSizeChange}
                min={4}
                max={20}
                step={1}
              />
              
              <SliderControl
                label="Species Threshold"
                value={speciesThreshold}
                onChange={onSpeciesThresholdChange}
                min={0.1}
                max={0.9}
                step={0.05}
              />
            </div>
          </div>
        )}
      </div>
    </motion.div>
    </>
  );
};

export default ControlPanel;
