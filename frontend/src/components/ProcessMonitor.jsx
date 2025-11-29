import { useState, useEffect, useRef } from 'react';
import './ProcessMonitor.css';

/**
 * ProcessMonitor - Side panel showing real-time council deliberation events.
 *
 * Features:
 * - Verbosity knob (0-3) to control event detail level
 * - Auto-scrolling event log
 * - Color-coded events by category
 * - Timestamps for each event
 * - Collapsible panel
 *
 * @param {Object} props
 * @param {Array} props.events - Array of process events to display
 * @param {number} props.verbosity - Current verbosity level (0-3)
 * @param {function} props.onVerbosityChange - Callback when verbosity changes
 * @param {boolean} props.isOpen - Whether the panel is open
 * @param {function} props.onToggle - Callback to toggle panel open/closed
 */
export default function ProcessMonitor({
  events = [],
  verbosity = 0,
  onVerbosityChange,
  isOpen = false,
  onToggle,
}) {
  const eventListRef = useRef(null);
  const [autoScroll, setAutoScroll] = useState(true);

  // Auto-scroll to bottom when new events arrive
  useEffect(() => {
    if (autoScroll && eventListRef.current) {
      eventListRef.current.scrollTop = eventListRef.current.scrollHeight;
    }
  }, [events, autoScroll]);

  // Handle scroll to detect if user scrolled up
  const handleScroll = () => {
    if (eventListRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = eventListRef.current;
      const isAtBottom = scrollHeight - scrollTop - clientHeight < 50;
      setAutoScroll(isAtBottom);
    }
  };

  const formatTimestamp = (isoString) => {
    const date = new Date(isoString);
    return date.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false,
    });
  };

  const getCategoryIcon = (category) => {
    switch (category) {
      case 'stage':
        return '▶';
      case 'model':
        return '🤖';
      case 'success':
        return '✓';
      case 'warning':
        return '⚠';
      case 'error':
        return '✗';
      case 'data':
        return '📊';
      case 'info':
      default:
        return '•';
    }
  };

  const verbosityLabels = ['Silent', 'Basic', 'Standard', 'Verbose'];

  if (!isOpen) {
    return (
      <button
        className="process-monitor-toggle collapsed"
        onClick={onToggle}
        title="Open Process Monitor"
      >
        <span className="toggle-icon">◀</span>
        <span className="toggle-label">Process</span>
        {events.length > 0 && (
          <span className="event-count">{events.length}</span>
        )}
      </button>
    );
  }

  return (
    <div className="process-monitor">
      <div className="process-monitor-header">
        <div className="header-title">
          <h3>Process Monitor</h3>
          <button
            className="close-btn"
            onClick={onToggle}
            title="Close Process Monitor"
          >
            ▶
          </button>
        </div>
        <div className="verbosity-control">
          <label>Verbosity:</label>
          <div className="verbosity-slider">
            <input
              type="range"
              min="0"
              max="3"
              value={verbosity}
              onChange={(e) => onVerbosityChange(parseInt(e.target.value, 10))}
            />
            <span className="verbosity-label">{verbosityLabels[verbosity]}</span>
          </div>
          <div className="verbosity-dots">
            {[0, 1, 2, 3].map((level) => (
              <button
                key={level}
                className={`verbosity-dot ${verbosity >= level ? 'active' : ''}`}
                onClick={() => onVerbosityChange(level)}
                title={verbosityLabels[level]}
              />
            ))}
          </div>
        </div>
      </div>

      <div
        className="event-list"
        ref={eventListRef}
        onScroll={handleScroll}
      >
        {verbosity === 0 ? (
          <div className="no-events-message">
            <p>Process monitoring is disabled.</p>
            <p className="hint">Increase verbosity to see events.</p>
          </div>
        ) : events.length === 0 ? (
          <div className="no-events-message">
            <p>No events yet.</p>
            <p className="hint">Send a message to see the council process.</p>
          </div>
        ) : (
          events.map((event, index) => (
            <div
              key={index}
              className={`event-item category-${event.category || 'info'}`}
            >
              <span className="event-icon">
                {getCategoryIcon(event.category)}
              </span>
              <span className="event-time">
                {formatTimestamp(event.timestamp)}
              </span>
              <span className="event-message">{event.message}</span>
            </div>
          ))
        )}
      </div>

      {events.length > 0 && (
        <div className="process-monitor-footer">
          <span className="event-count-label">
            {events.length} event{events.length !== 1 ? 's' : ''}
          </span>
          {!autoScroll && (
            <button
              className="scroll-to-bottom"
              onClick={() => {
                setAutoScroll(true);
                if (eventListRef.current) {
                  eventListRef.current.scrollTop = eventListRef.current.scrollHeight;
                }
              }}
            >
              ↓ Scroll to latest
            </button>
          )}
        </div>
      )}
    </div>
  );
}

/**
 * VerbosityControl - Standalone verbosity control component.
 *
 * Can be used separately from the ProcessMonitor panel.
 *
 * @param {Object} props
 * @param {number} props.verbosity - Current verbosity level (0-3)
 * @param {function} props.onChange - Callback when verbosity changes
 * @param {boolean} props.compact - Use compact display mode
 */
export function VerbosityControl({ verbosity, onChange, compact = false }) {
  const verbosityLabels = ['Off', 'Basic', 'Standard', 'Verbose'];

  if (compact) {
    return (
      <div className="verbosity-control-compact">
        <label title="Process Monitor Verbosity">V:</label>
        <select
          value={verbosity}
          onChange={(e) => onChange(parseInt(e.target.value, 10))}
        >
          {verbosityLabels.map((label, level) => (
            <option key={level} value={level}>
              {label}
            </option>
          ))}
        </select>
      </div>
    );
  }

  return (
    <div className="verbosity-control-inline">
      <span className="label">Verbosity:</span>
      <div className="verbosity-buttons">
        {verbosityLabels.map((label, level) => (
          <button
            key={level}
            className={`verbosity-btn ${verbosity === level ? 'active' : ''}`}
            onClick={() => onChange(level)}
            title={`${label} - Level ${level}`}
          >
            {level}
          </button>
        ))}
      </div>
    </div>
  );
}
