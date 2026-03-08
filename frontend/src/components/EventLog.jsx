import React, { useEffect, useRef, useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Terminal, Info, AlertTriangle, CheckCircle } from 'lucide-react';

const EventLog = ({ events }) => {
  const scrollContainerRef = useRef(null);
  const safeEvents = Array.isArray(events) ? events : [];
  const [shouldAutoScroll, setShouldAutoScroll] = useState(true);

  const scrollToBottom = useCallback(() => {
    const el = scrollContainerRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, []);

  const handleScroll = useCallback(() => {
    const el = scrollContainerRef.current;
    if (!el) return;
    const distanceFromBottom = el.scrollHeight - (el.scrollTop + el.clientHeight);
    setShouldAutoScroll(distanceFromBottom < 24);
  }, []);

  useEffect(() => {
    if (!shouldAutoScroll) return;
    requestAnimationFrame(() => {
      scrollToBottom();
    });
  }, [safeEvents, shouldAutoScroll, scrollToBottom]);

  const toEventText = (event) => {
    if (typeof event === 'string') return event;
    if (event && typeof event === 'object') {
      if (typeof event.message === 'string') return event.message;
      if (typeof event.data?.message === 'string') return event.data.message;
    }
    try {
      return JSON.stringify(event);
    } catch {
      return String(event);
    }
  };

  const getEventIcon = (event) => {
    const text = toEventText(event);
    if (text.includes('[INFO]')) return <Info size={12} className="text-electric-blue" />;
    if (text.includes('[MUTATION]')) return <AlertTriangle size={12} className="text-glowing-purple" />;
    if (text.includes('[FITNESS]')) return <CheckCircle size={12} className="text-neon-green" />;
    return <Terminal size={12} className="text-gray-400" />;
  };

  const getEventColor = (event) => {
    const text = toEventText(event);
    if (text.includes('[INFO]')) return 'text-electric-blue';
    if (text.includes('[MUTATION]')) return 'text-glowing-purple';
    if (text.includes('[FITNESS]')) return 'text-neon-green';
    if (text.includes('[GEN')) return 'text-white';
    return 'text-gray-400';
  };

  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.3 }}
      className="h-full w-full"
    >
      <div className="glass glass-border rounded-lg h-full flex flex-col">
        {/* Header */}
        <div className="flex items-center space-x-2 p-4 border-b border-gray-700">
          <Terminal size={16} className="text-electric-blue" />
          <h3 className="text-sm font-medium text-white">Event Log</h3>
          <div className="ml-auto">
            <div className="w-2 h-2 rounded-full bg-neon-green animate-pulse" />
          </div>
        </div>

        {/* Log Content */}
        <div
          ref={scrollContainerRef}
          onScroll={handleScroll}
          className="flex-1 overflow-y-auto p-4 space-y-2 terminal-text"
        >
          <AnimatePresence>
            {safeEvents.length === 0 ? (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="text-gray-500 text-sm"
              >
                Waiting for events...
              </motion.div>
            ) : (
              safeEvents.map((event, index) => {
                const eventText = toEventText(event);
                return (
                  <motion.div
                    key={`${eventText}-${index}`}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    transition={{ duration: 0.2 }}
                    className="flex items-start space-x-2 text-sm"
                  >
                    <div className="mt-0.5 flex-shrink-0">
                      {getEventIcon(event)}
                    </div>
                    <div className={`flex-1 ${getEventColor(event)}`}>
                      {eventText}
                    </div>
                  </motion.div>
                );
              })
            )}
          </AnimatePresence>
        </div>

        {/* Footer */}
        <div className="p-3 border-t border-gray-700">
          <div className="flex items-center justify-between text-sm text-gray-400">
            <span>{safeEvents.length} events</span>
            <span>Live Feed</span>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default EventLog;
