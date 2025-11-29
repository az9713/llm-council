import './TierIndicator.css';

/**
 * TierIndicator - Displays the current tier in confidence-gated escalation
 *
 * Shows Tier 1 (cost-effective) or Tier 2 (premium) status with visual indicators.
 * Used in Stage 1 to indicate which model tier is being queried.
 */
export default function TierIndicator({ tier, escalated, escalationInfo }) {
  if (!tier) return null;

  const getTierInfo = () => {
    if (tier === 1) {
      return {
        label: 'Tier 1',
        sublabel: 'Fast',
        description: 'Cost-effective models',
        className: 'tier-1',
      };
    } else if (tier === 2) {
      return {
        label: 'Tier 2',
        sublabel: 'Premium',
        description: 'High-capability models',
        className: 'tier-2',
      };
    }
    return null;
  };

  const tierInfo = getTierInfo();
  if (!tierInfo) return null;

  return (
    <span className={`tier-indicator ${tierInfo.className}`} title={tierInfo.description}>
      <span className="tier-label">{tierInfo.label}</span>
      <span className="tier-sublabel">{tierInfo.sublabel}</span>
    </span>
  );
}

/**
 * TierBadge - Small badge showing tier number on model tabs
 */
export function TierBadge({ tier }) {
  if (!tier) return null;

  return (
    <span className={`tier-badge tier-badge-${tier}`} title={tier === 1 ? 'Tier 1 (Fast)' : 'Tier 2 (Premium)'}>
      T{tier}
    </span>
  );
}

/**
 * EscalationBanner - Shows when escalation was triggered
 */
export function EscalationBanner({ escalationInfo }) {
  if (!escalationInfo || !escalationInfo.escalated) return null;

  const reasons = escalationInfo.reasons || [];
  const metrics = escalationInfo.metrics || {};

  return (
    <div className="escalation-banner">
      <div className="escalation-header">
        <span className="escalation-icon">&#x2191;</span>
        <strong>Escalated to Premium Models</strong>
      </div>
      <div className="escalation-details">
        {reasons.length > 0 && (
          <div className="escalation-reasons">
            {reasons.map((reason, idx) => (
              <span key={idx} className="escalation-reason">{reason}</span>
            ))}
          </div>
        )}
        <div className="escalation-metrics">
          {metrics.average_confidence !== undefined && (
            <span className="escalation-metric">
              Avg Confidence: <strong>{metrics.average_confidence?.toFixed(1) || 'N/A'}</strong>
            </span>
          )}
          {escalationInfo.tier1_model_count && escalationInfo.tier2_model_count && (
            <span className="escalation-metric">
              Models: <strong>{escalationInfo.tier1_model_count} + {escalationInfo.tier2_model_count} = {escalationInfo.total_model_count}</strong>
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

/**
 * TierSummary - Compact summary showing tier status and counts
 */
export function TierSummary({ escalationInfo }) {
  if (!escalationInfo) return null;

  const { escalated, tier1_model_count, tier2_model_count, total_model_count } = escalationInfo;

  if (escalated) {
    return (
      <div className="tier-summary escalated">
        <span className="tier-summary-icon">&#x2191;</span>
        <span className="tier-summary-text">
          Escalated: {tier1_model_count} Tier 1 + {tier2_model_count} Tier 2 = {total_model_count} models
        </span>
      </div>
    );
  }

  return (
    <div className="tier-summary not-escalated">
      <span className="tier-summary-icon">&#x2713;</span>
      <span className="tier-summary-text">
        Tier 1 only: {tier1_model_count} cost-effective models (confidence sufficient)
      </span>
    </div>
  );
}

/**
 * EscalationToggle - Toggle switch for enabling/disabling escalation mode
 */
export function EscalationToggle({ enabled, onChange }) {
  return (
    <label className="escalation-toggle">
      <input
        type="checkbox"
        checked={enabled}
        onChange={(e) => onChange(e.target.checked)}
      />
      <span className="escalation-toggle-slider"></span>
      <span className="escalation-toggle-label">Confidence-Gated Escalation</span>
    </label>
  );
}
