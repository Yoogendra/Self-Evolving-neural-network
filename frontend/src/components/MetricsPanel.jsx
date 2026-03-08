import React from 'react';
import { motion } from 'framer-motion';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Brain, Activity, Users, TrendingUp } from 'lucide-react';

const MetricCard = ({ title, value, icon: Icon, color = 'electric-blue' }) => (
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ duration: 0.3 }}
    className="glass glass-border rounded-lg p-4"
  >
    <div className="flex items-center justify-between">
      <div>
        <p className="text-xs text-gray-400 mb-1">{title}</p>
        <p className={`text-2xl font-bold text-${color} font-mono`}>{value}</p>
      </div>
      <div className={`w-10 h-10 rounded-full bg-${color} bg-opacity-20 flex items-center justify-center`}>
        <Icon size={20} className={`text-${color}`} />
      </div>
    </div>
  </motion.div>
);

const FitnessChart = ({ data }) => (
  <div className="glass glass-border rounded-lg p-4">
    <h3 className="text-sm font-medium text-gray-300 mb-4 flex items-center space-x-2">
      <TrendingUp size={16} className="text-electric-blue" />
      <span>Fitness vs Generation</span>
    </h3>
    <ResponsiveContainer width="100%" height={150}>
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
        <XAxis 
          dataKey="generation" 
          stroke="#9ca3af"
          tick={{ fill: '#9ca3af', fontSize: 10 }}
        />
        <YAxis 
          stroke="#9ca3af"
          tick={{ fill: '#9ca3af', fontSize: 10 }}
          domain={[0, 1]}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: 'rgba(10, 10, 15, 0.9)',
            border: '1px solid rgba(0, 212, 255, 0.3)',
            borderRadius: '8px',
          }}
          labelStyle={{ color: '#00d4ff' }}
        />
        <Line
          type="monotone"
          dataKey="fitness"
          stroke="#00d4ff"
          strokeWidth={2}
          dot={{ fill: '#00d4ff', r: 3 }}
          activeDot={{ r: 5 }}
        />
      </LineChart>
    </ResponsiveContainer>
  </div>
);

const ComplexityChart = ({ data }) => (
  <div className="glass glass-border rounded-lg p-4">
    <h3 className="text-sm font-medium text-gray-300 mb-4 flex items-center space-x-2">
      <Activity size={16} className="text-glowing-purple" />
      <span>Network Complexity</span>
    </h3>
    <ResponsiveContainer width="100%" height={150}>
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
        <XAxis 
          dataKey="generation" 
          stroke="#9ca3af"
          tick={{ fill: '#9ca3af', fontSize: 10 }}
        />
        <YAxis 
          stroke="#9ca3af"
          tick={{ fill: '#9ca3af', fontSize: 10 }}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: 'rgba(10, 10, 15, 0.9)',
            border: '1px solid rgba(168, 85, 247, 0.3)',
            borderRadius: '8px',
          }}
          labelStyle={{ color: '#a855f7' }}
        />
        <Line
          type="monotone"
          dataKey="complexity"
          stroke="#a855f7"
          strokeWidth={2}
          dot={{ fill: '#a855f7', r: 3 }}
          activeDot={{ r: 5 }}
        />
      </LineChart>
    </ResponsiveContainer>
  </div>
);

const MetricsPanel = ({ 
  generation, 
  bestFitness, 
  activeSpecies, 
  fitnessHistory, 
  complexityHistory 
}) => {
  const fitnessPercentage = (bestFitness * 100).toFixed(1);
  
  return (
    <div className="w-full">
      {/* Metric Cards */}
      <div className="grid grid-cols-4 gap-4 mb-6">
        <MetricCard
          title="Generation"
          value={generation}
          icon={Brain}
          color="electric-blue"
        />
        <MetricCard
          title="Best Fitness"
          value={`${fitnessPercentage}%`}
          icon={TrendingUp}
          color="neon-green"
        />
        <MetricCard
          title="Active Species"
          value={activeSpecies}
          icon={Users}
          color="glowing-purple"
        />
        <MetricCard
          title="Network Size"
          value={complexityHistory[complexityHistory.length - 1]?.complexity || 0}
          icon={Activity}
          color="electric-blue"
        />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <FitnessChart data={fitnessHistory} />
        <ComplexityChart data={complexityHistory} />
      </div>
    </div>
  );
};

export default MetricsPanel;
