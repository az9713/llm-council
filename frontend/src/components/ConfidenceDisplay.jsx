import './ConfidenceDisplay.css';

/**
 * ConfidenceDisplay - Shows aggregate confidence statistics from Stage 1.
 *
 * Displays average confidence, range (min-max), and a visual distribution
 * bar showing the spread of confidence scores across models.
 *
 * @param {Object} props
 * @param {Object} props.aggregateConfidence - Aggregate confidence stats
 *   Contains: average, min, max, count, total_models, distribution
 * @param {boolean} props.compact - Whether to show compact view (default: false)
 */
export default function ConfidenceDisplay({ aggregateConfidence, compact = false }) {
  if (!aggregateConfidence || aggregateConfidence.count === 0) {
    return null;
  }

  const { average, min, max, count, total_models } = aggregateConfidence;

  // Get confidence level label and color
  const getConfidenceLevel = (score) => {
    if (score >= 9) return { label: 'Very High', className: 'confidence-very-high' };
    if (score >= 7) return { label: 'High', className: 'confidence-high' };
    if (score >= 5) return { label: 'Medium', className: 'confidence-medium' };
    if (score >= 3) return { label: 'Low', className: 'confidence-low' };
    return { label: 'Very Low', className: 'confidence-very-low' };
  };

  const level = getConfidenceLevel(average);

  // Compact view for inline display
  if (compact) {
    return (
      <div className={`confidence-display confidence-compact ${level.className}`}>
        <span className="confidence-icon">●</span>
        <span className="confidence-score">{average.toFixed(1)}</span>
        <span className="confidence-label">{level.label}</span>
        <span className="confidence-count">({count}/{total_models})</span>
      </div>
    );
  }

  // Full view with more details
  return (
    <div className={`confidence-display confidence-full ${level.className}`}>
      <div className="confidence-header">
        <span className="confidence-title">Council Confidence</span>
        <span className="confidence-badge">
          {count === total_models ? 'All Models' : `${count}/${total_models} Models`}
        </span>
      </div>

      <div className="confidence-main">
        <div className="confidence-score-container">
          <span className="confidence-score-value">{average.toFixed(1)}</span>
          <span className="confidence-score-max">/10</span>
        </div>
        <span className="confidence-level-label">{level.label}</span>
      </div>

      {min !== max && (
        <div className="confidence-range">
          <span className="confidence-range-label">Range:</span>
          <span className="confidence-range-value">{min} - {max}</span>
        </div>
      )}

      <ConfidenceBar score={average} />
    </div>
  );
}


/**
 * ConfidenceBar - Visual bar showing confidence level.
 *
 * @param {Object} props
 * @param {number} props.score - Confidence score (1-10)
 */
function ConfidenceBar({ score }) {
  const percentage = (score / 10) * 100;

  return (
    <div className="confidence-bar-container">
      <div className="confidence-bar-track">
        <div
          className="confidence-bar-fill"
          style={{ width: `${percentage}%` }}
        />
      </div>
      <div className="confidence-bar-labels">
        <span>1</span>
        <span>5</span>
        <span>10</span>
      </div>
    </div>
  );
}


/**
 * ConfidenceBadge - Inline badge showing individual model's confidence.
 *
 * Used in Stage1 tabs to show each model's confidence score.
 *
 * @param {Object} props
 * @param {number|null} props.confidence - Confidence score (1-10) or null
 */
export function ConfidenceBadge({ confidence }) {
  if (confidence === null || confidence === undefined) {
    return (
      <span className="confidence-badge-inline confidence-badge-none" title="No confidence provided">
        --
      </span>
    );
  }

  // Get color class based on score
  const getColorClass = (score) => {
    if (score >= 9) return 'confidence-badge-very-high';
    if (score >= 7) return 'confidence-badge-high';
    if (score >= 5) return 'confidence-badge-medium';
    if (score >= 3) return 'confidence-badge-low';
    return 'confidence-badge-very-low';
  };

  return (
    <span
      className={`confidence-badge-inline ${getColorClass(confidence)}`}
      title={`Confidence: ${confidence}/10`}
    >
      {confidence}
    </span>
  );
}


/**
 * AggregateConfidenceSummary - Summary for display in Stage 1 header.
 *
 * @param {Object} props
 * @param {Object} props.aggregateConfidence - Aggregate confidence stats
 */
export function AggregateConfidenceSummary({ aggregateConfidence }) {
  if (!aggregateConfidence || aggregateConfidence.count === 0) {
    return null;
  }

  const { average, count, total_models } = aggregateConfidence;

  // Get confidence level and styling
  const getLevel = (score) => {
    if (score >= 9) return { text: 'Very High', className: 'summary-very-high' };
    if (score >= 7) return { text: 'High', className: 'summary-high' };
    if (score >= 5) return { text: 'Medium', className: 'summary-medium' };
    if (score >= 3) return { text: 'Low', className: 'summary-low' };
    return { text: 'Very Low', className: 'summary-very-low' };
  };

  const level = getLevel(average);

  return (
    <div className={`aggregate-confidence-summary ${level.className}`}>
      <span className="summary-label">Avg Confidence:</span>
      <span className="summary-score">{average.toFixed(1)}</span>
      <span className="summary-level">({level.text})</span>
      {count < total_models && (
        <span className="summary-partial">{count}/{total_models} reported</span>
      )}
    </div>
  );
}
