import { useState, useEffect } from 'react';
import { api } from '../api';
import './PerformanceDashboard.css';

/**
 * PerformanceDashboard - Displays model performance analytics.
 *
 * Shows:
 * - Summary statistics (total queries, unique models, date range)
 * - Model leaderboard sorted by win rate
 * - Detailed per-model statistics (rank distribution, costs, tokens)
 * - Chairman model usage statistics
 *
 * @param {Object} props
 * @param {function} props.onClose - Callback to close the dashboard
 */
export default function PerformanceDashboard({ onClose }) {
  const [analytics, setAnalytics] = useState(null);
  const [chairmanStats, setChairmanStats] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showClearConfirm, setShowClearConfirm] = useState(false);
  const [activeTab, setActiveTab] = useState('leaderboard');

  useEffect(() => {
    loadAnalytics();
  }, []);

  const loadAnalytics = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const [analyticsData, chairmanData] = await Promise.all([
        api.getAnalytics(),
        api.getChairmanAnalytics(),
      ]);
      setAnalytics(analyticsData);
      setChairmanStats(chairmanData);
    } catch (err) {
      setError('Failed to load analytics data');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleClearAnalytics = async () => {
    try {
      await api.clearAnalytics();
      setShowClearConfirm(false);
      loadAnalytics();
    } catch (err) {
      setError('Failed to clear analytics');
      console.error(err);
    }
  };

  const formatDate = (isoString) => {
    if (!isoString) return 'N/A';
    const date = new Date(isoString);
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
    });
  };

  const formatCost = (cost) => {
    if (cost === 0 || cost === null || cost === undefined) return '$0.00';
    if (cost < 0.01) return `$${cost.toFixed(4)}`;
    return `$${cost.toFixed(2)}`;
  };

  const formatModelName = (model) => {
    // Extract model name after provider prefix (e.g., "openai/gpt-4" -> "gpt-4")
    const parts = model.split('/');
    return parts.length > 1 ? parts[1] : model;
  };

  const getWinRateColor = (winRate) => {
    if (winRate >= 50) return '#059669'; // green
    if (winRate >= 30) return '#0d9488'; // teal
    if (winRate >= 15) return '#d97706'; // amber
    return '#dc2626'; // red
  };

  const getRankColor = (rank) => {
    if (rank <= 1.5) return '#059669'; // green - top
    if (rank <= 2.5) return '#0d9488'; // teal - good
    if (rank <= 3.5) return '#d97706'; // amber - middle
    return '#dc2626'; // red - low
  };

  if (isLoading) {
    return (
      <div className="performance-dashboard">
        <div className="dashboard-header">
          <h2>Performance Dashboard</h2>
          <button className="close-btn" onClick={onClose}>×</button>
        </div>
        <div className="dashboard-loading">Loading analytics...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="performance-dashboard">
        <div className="dashboard-header">
          <h2>Performance Dashboard</h2>
          <button className="close-btn" onClick={onClose}>×</button>
        </div>
        <div className="dashboard-error">
          <p>{error}</p>
          <button onClick={loadAnalytics}>Retry</button>
        </div>
      </div>
    );
  }

  const { models, summary } = analytics || { models: {}, summary: {} };
  const modelList = Object.entries(models);
  const hasData = modelList.length > 0;

  return (
    <div className="performance-dashboard">
      <div className="dashboard-header">
        <h2>Performance Dashboard</h2>
        <button className="close-btn" onClick={onClose}>×</button>
      </div>

      {/* Summary Section */}
      <div className="dashboard-summary">
        <div className="summary-stat">
          <span className="stat-value">{summary.total_queries || 0}</span>
          <span className="stat-label">Total Queries</span>
        </div>
        <div className="summary-stat">
          <span className="stat-value">{summary.unique_models || 0}</span>
          <span className="stat-label">Unique Models</span>
        </div>
        <div className="summary-stat">
          <span className="stat-value">{chairmanStats?.total_syntheses || 0}</span>
          <span className="stat-label">Syntheses</span>
        </div>
        {summary.date_range?.start && (
          <div className="summary-stat date-range">
            <span className="stat-value">
              {formatDate(summary.date_range.start)} - {formatDate(summary.date_range.end)}
            </span>
            <span className="stat-label">Date Range</span>
          </div>
        )}
      </div>

      {/* Tabs */}
      <div className="dashboard-tabs">
        <button
          className={`tab-btn ${activeTab === 'leaderboard' ? 'active' : ''}`}
          onClick={() => setActiveTab('leaderboard')}
        >
          Leaderboard
        </button>
        <button
          className={`tab-btn ${activeTab === 'details' ? 'active' : ''}`}
          onClick={() => setActiveTab('details')}
        >
          Model Details
        </button>
        <button
          className={`tab-btn ${activeTab === 'chairman' ? 'active' : ''}`}
          onClick={() => setActiveTab('chairman')}
        >
          Chairman Stats
        </button>
      </div>

      {!hasData ? (
        <div className="no-data">
          <p>No analytics data yet.</p>
          <p className="hint">Run some council queries to see performance statistics.</p>
        </div>
      ) : (
        <div className="dashboard-content">
          {/* Leaderboard Tab */}
          {activeTab === 'leaderboard' && (
            <div className="leaderboard">
              <table className="leaderboard-table">
                <thead>
                  <tr>
                    <th className="rank-col">#</th>
                    <th className="model-col">Model</th>
                    <th className="stat-col">Win Rate</th>
                    <th className="stat-col">Avg Rank</th>
                    <th className="stat-col">Avg Conf</th>
                    <th className="stat-col">Queries</th>
                  </tr>
                </thead>
                <tbody>
                  {modelList.map(([model, stats], index) => (
                    <tr key={model} className={index < 3 ? 'top-three' : ''}>
                      <td className="rank-col">
                        {index === 0 && <span className="medal gold">🥇</span>}
                        {index === 1 && <span className="medal silver">🥈</span>}
                        {index === 2 && <span className="medal bronze">🥉</span>}
                        {index > 2 && <span className="rank-num">{index + 1}</span>}
                      </td>
                      <td className="model-col">
                        <span className="model-name" title={model}>
                          {formatModelName(model)}
                        </span>
                      </td>
                      <td className="stat-col">
                        <span
                          className="win-rate"
                          style={{ color: getWinRateColor(stats.win_rate) }}
                        >
                          {stats.win_rate}%
                        </span>
                        <span className="wins-count">({stats.wins} wins)</span>
                      </td>
                      <td className="stat-col">
                        <span
                          className="avg-rank"
                          style={{ color: getRankColor(stats.average_rank) }}
                        >
                          {stats.average_rank || 'N/A'}
                        </span>
                      </td>
                      <td className="stat-col">
                        {stats.average_confidence !== null ? (
                          <span className="confidence">{stats.average_confidence}/10</span>
                        ) : (
                          <span className="na">N/A</span>
                        )}
                      </td>
                      <td className="stat-col queries">
                        {stats.total_queries}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* Model Details Tab */}
          {activeTab === 'details' && (
            <div className="model-details">
              {modelList.map(([model, stats]) => (
                <div key={model} className="model-card">
                  <div className="model-card-header">
                    <h4 title={model}>{formatModelName(model)}</h4>
                    <span
                      className="win-badge"
                      style={{ backgroundColor: getWinRateColor(stats.win_rate) }}
                    >
                      {stats.win_rate}% win rate
                    </span>
                  </div>
                  <div className="model-card-stats">
                    <div className="stat-group">
                      <label>Performance</label>
                      <div className="stat-row">
                        <span>Wins:</span>
                        <span>{stats.wins} / {stats.total_queries}</span>
                      </div>
                      <div className="stat-row">
                        <span>Avg Rank:</span>
                        <span style={{ color: getRankColor(stats.average_rank) }}>
                          {stats.average_rank || 'N/A'}
                        </span>
                      </div>
                      <div className="stat-row">
                        <span>Avg Confidence:</span>
                        <span>{stats.average_confidence || 'N/A'}</span>
                      </div>
                    </div>
                    <div className="stat-group">
                      <label>Usage</label>
                      <div className="stat-row">
                        <span>Total Cost:</span>
                        <span>{formatCost(stats.total_cost)}</span>
                      </div>
                      <div className="stat-row">
                        <span>Total Tokens:</span>
                        <span>{stats.total_tokens?.toLocaleString() || 0}</span>
                      </div>
                    </div>
                    {stats.rank_distribution && Object.keys(stats.rank_distribution).length > 0 && (
                      <div className="stat-group">
                        <label>Rank Distribution</label>
                        <div className="rank-distribution">
                          {Object.entries(stats.rank_distribution).map(([rank, count]) => (
                            <div key={rank} className="rank-bar-container">
                              <span className="rank-label">#{rank}</span>
                              <div className="rank-bar-bg">
                                <div
                                  className="rank-bar"
                                  style={{
                                    width: `${(count / stats.total_queries) * 100}%`,
                                    backgroundColor: getRankColor(parseInt(rank)),
                                  }}
                                />
                              </div>
                              <span className="rank-count">{count}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Chairman Stats Tab */}
          {activeTab === 'chairman' && (
            <div className="chairman-stats">
              <p className="chairman-intro">
                The chairman model synthesizes the final answer from all council responses and rankings.
              </p>
              {chairmanStats && Object.keys(chairmanStats.models).length > 0 ? (
                <table className="chairman-table">
                  <thead>
                    <tr>
                      <th className="model-col">Chairman Model</th>
                      <th className="stat-col">Times Used</th>
                      <th className="stat-col">Total Cost</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(chairmanStats.models).map(([model, stats]) => (
                      <tr key={model}>
                        <td className="model-col">
                          <span className="model-name" title={model}>
                            {formatModelName(model)}
                          </span>
                        </td>
                        <td className="stat-col">{stats.times_used}</td>
                        <td className="stat-col">{formatCost(stats.total_cost)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <p className="no-chairman-data">No chairman statistics available yet.</p>
              )}
            </div>
          )}
        </div>
      )}

      {/* Footer Actions */}
      <div className="dashboard-footer">
        <button className="refresh-btn" onClick={loadAnalytics}>
          Refresh Data
        </button>
        {hasData && (
          <>
            {showClearConfirm ? (
              <div className="clear-confirm">
                <span>Clear all analytics data?</span>
                <button className="confirm-yes" onClick={handleClearAnalytics}>
                  Yes, Clear
                </button>
                <button className="confirm-no" onClick={() => setShowClearConfirm(false)}>
                  Cancel
                </button>
              </div>
            ) : (
              <button
                className="clear-btn"
                onClick={() => setShowClearConfirm(true)}
              >
                Clear Data
              </button>
            )}
          </>
        )}
      </div>
    </div>
  );
}
