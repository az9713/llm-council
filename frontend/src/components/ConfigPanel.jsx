import { useState, useEffect } from 'react';
import { api } from '../api';
import './ConfigPanel.css';

/**
 * ConfigPanel - UI for managing council and chairman model configuration.
 *
 * Features:
 * - Add/remove/reorder council models
 * - Select chairman model from council or enter custom
 * - Suggested models dropdown for quick selection
 * - Reset to defaults
 * - Validation (minimum 2 models)
 *
 * @param {Object} props
 * @param {function} props.onClose - Callback to close the panel
 */
export default function ConfigPanel({ onClose }) {
  // Current configuration state
  const [councilModels, setCouncilModels] = useState([]);
  const [chairmanModel, setChairmanModel] = useState('');

  // Available models for suggestions
  const [availableModels, setAvailableModels] = useState([]);

  // UI state
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState(null);
  const [successMessage, setSuccessMessage] = useState(null);

  // New model input
  const [newModelInput, setNewModelInput] = useState('');
  const [showModelDropdown, setShowModelDropdown] = useState(false);

  // Load configuration and available models on mount
  useEffect(() => {
    loadConfig();
    loadAvailableModels();
  }, []);

  const loadConfig = async () => {
    try {
      setIsLoading(true);
      const config = await api.getConfig();
      setCouncilModels(config.council_models);
      setChairmanModel(config.chairman_model);
      setError(null);
    } catch (err) {
      setError('Failed to load configuration');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  const loadAvailableModels = async () => {
    try {
      const result = await api.getAvailableModels();
      setAvailableModels(result.models);
    } catch (err) {
      console.error('Failed to load available models:', err);
    }
  };

  const handleSave = async () => {
    // Validate
    if (councilModels.length < 2) {
      setError('At least 2 council models are required');
      return;
    }

    if (!chairmanModel.trim()) {
      setError('Chairman model is required');
      return;
    }

    try {
      setIsSaving(true);
      setError(null);
      await api.updateConfig(councilModels, chairmanModel);
      setSuccessMessage('Configuration saved successfully!');
      setTimeout(() => setSuccessMessage(null), 3000);
    } catch (err) {
      setError(err.message || 'Failed to save configuration');
    } finally {
      setIsSaving(false);
    }
  };

  const handleReset = async () => {
    if (!window.confirm('Reset configuration to defaults?')) {
      return;
    }

    try {
      setIsSaving(true);
      setError(null);
      const config = await api.resetConfig();
      setCouncilModels(config.council_models);
      setChairmanModel(config.chairman_model);
      setSuccessMessage('Configuration reset to defaults');
      setTimeout(() => setSuccessMessage(null), 3000);
    } catch (err) {
      setError('Failed to reset configuration');
    } finally {
      setIsSaving(false);
    }
  };

  const addModel = (model) => {
    const trimmed = model.trim();
    if (!trimmed) return;

    // Don't add duplicates
    if (councilModels.includes(trimmed)) {
      setError('Model already in council');
      setTimeout(() => setError(null), 2000);
      return;
    }

    setCouncilModels([...councilModels, trimmed]);
    setNewModelInput('');
    setShowModelDropdown(false);
  };

  const removeModel = (index) => {
    if (councilModels.length <= 2) {
      setError('At least 2 council models are required');
      setTimeout(() => setError(null), 2000);
      return;
    }

    const removed = councilModels[index];
    const newModels = councilModels.filter((_, i) => i !== index);
    setCouncilModels(newModels);

    // If chairman was removed, update to first model
    if (removed === chairmanModel) {
      setChairmanModel(newModels[0] || '');
    }
  };

  const moveModel = (index, direction) => {
    const newIndex = index + direction;
    if (newIndex < 0 || newIndex >= councilModels.length) return;

    const newModels = [...councilModels];
    [newModels[index], newModels[newIndex]] = [newModels[newIndex], newModels[index]];
    setCouncilModels(newModels);
  };

  // Filter available models for dropdown (exclude already added ones)
  const filteredModels = availableModels.filter(
    (model) =>
      !councilModels.includes(model) &&
      model.toLowerCase().includes(newModelInput.toLowerCase())
  );

  if (isLoading) {
    return (
      <div className="config-panel">
        <div className="config-panel-header">
          <h2>Model Configuration</h2>
          <button className="close-btn" onClick={onClose}>
            &times;
          </button>
        </div>
        <div className="config-loading">Loading configuration...</div>
      </div>
    );
  }

  return (
    <div className="config-panel">
      <div className="config-panel-header">
        <h2>Model Configuration</h2>
        <button className="close-btn" onClick={onClose}>
          &times;
        </button>
      </div>

      {error && <div className="config-error">{error}</div>}
      {successMessage && <div className="config-success">{successMessage}</div>}

      <div className="config-section">
        <h3>Council Models</h3>
        <p className="config-help">
          Models that participate in Stage 1 (responses) and Stage 2 (rankings).
          Minimum 2 required.
        </p>

        <ul className="model-list">
          {councilModels.map((model, index) => (
            <li key={index} className="model-item">
              <div className="model-info">
                <span className="model-index">{index + 1}.</span>
                <span className="model-name">{model}</span>
                {model === chairmanModel && (
                  <span className="chairman-badge">Chairman</span>
                )}
              </div>
              <div className="model-actions">
                <button
                  className="move-btn"
                  onClick={() => moveModel(index, -1)}
                  disabled={index === 0}
                  title="Move up"
                >
                  ▲
                </button>
                <button
                  className="move-btn"
                  onClick={() => moveModel(index, 1)}
                  disabled={index === councilModels.length - 1}
                  title="Move down"
                >
                  ▼
                </button>
                <button
                  className="remove-btn"
                  onClick={() => removeModel(index)}
                  title="Remove model"
                >
                  &times;
                </button>
              </div>
            </li>
          ))}
        </ul>

        <div className="add-model-container">
          <div className="add-model-input-wrapper">
            <input
              type="text"
              className="add-model-input"
              placeholder="Enter model ID (e.g., openai/gpt-4o)"
              value={newModelInput}
              onChange={(e) => {
                setNewModelInput(e.target.value);
                setShowModelDropdown(true);
              }}
              onFocus={() => setShowModelDropdown(true)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  e.preventDefault();
                  addModel(newModelInput);
                }
              }}
            />
            {showModelDropdown && newModelInput && filteredModels.length > 0 && (
              <ul className="model-dropdown">
                {filteredModels.slice(0, 8).map((model) => (
                  <li
                    key={model}
                    onClick={() => addModel(model)}
                    className="model-dropdown-item"
                  >
                    {model}
                  </li>
                ))}
              </ul>
            )}
          </div>
          <button
            className="add-model-btn"
            onClick={() => addModel(newModelInput)}
            disabled={!newModelInput.trim()}
          >
            Add Model
          </button>
        </div>

        {availableModels.length > 0 && (
          <div className="suggested-models">
            <span className="suggested-label">Quick add:</span>
            {availableModels
              .filter((m) => !councilModels.includes(m))
              .slice(0, 5)
              .map((model) => (
                <button
                  key={model}
                  className="suggested-model-btn"
                  onClick={() => addModel(model)}
                >
                  + {model.split('/')[1] || model}
                </button>
              ))}
          </div>
        )}
      </div>

      <div className="config-section">
        <h3>Chairman Model</h3>
        <p className="config-help">
          The model that synthesizes the final response in Stage 3.
        </p>

        <select
          className="chairman-select"
          value={chairmanModel}
          onChange={(e) => setChairmanModel(e.target.value)}
        >
          {councilModels.map((model) => (
            <option key={model} value={model}>
              {model}
            </option>
          ))}
          <option value="" disabled>
            ─────────────
          </option>
          {availableModels
            .filter((m) => !councilModels.includes(m))
            .map((model) => (
              <option key={model} value={model}>
                {model} (not in council)
              </option>
            ))}
        </select>

        <p className="config-note">
          Tip: The chairman can be a council member or a different model entirely.
        </p>
      </div>

      <div className="config-actions">
        <button className="reset-btn" onClick={handleReset} disabled={isSaving}>
          Reset to Defaults
        </button>
        <div className="config-actions-right">
          <button className="cancel-btn" onClick={onClose} disabled={isSaving}>
            Cancel
          </button>
          <button
            className="save-btn"
            onClick={handleSave}
            disabled={isSaving || councilModels.length < 2}
          >
            {isSaving ? 'Saving...' : 'Save Configuration'}
          </button>
        </div>
      </div>
    </div>
  );
}
