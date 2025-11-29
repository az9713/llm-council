import { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import Stage1 from './Stage1';
import Stage2 from './Stage2';
import Stage3 from './Stage3';
import TagEditor from './TagEditor';
import CostDisplay from './CostDisplay';
import { exportToMarkdown, exportToJSON } from '../utils/export';
import './ChatInterface.css';

export default function ChatInterface({
  conversation,
  onSendMessage,
  isLoading,
  onTagsChange,
}) {
  const [input, setInput] = useState('');
  const [showTagEditor, setShowTagEditor] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [conversation]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (input.trim() && !isLoading) {
      onSendMessage(input);
      setInput('');
    }
  };

  const handleKeyDown = (e) => {
    // Submit on Enter (without Shift)
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  if (!conversation) {
    return (
      <div className="chat-interface">
        <div className="empty-state">
          <h2>Welcome to LLM Council</h2>
          <p>Create a new conversation to get started</p>
        </div>
      </div>
    );
  }

  const hasMessages = conversation.messages.length > 0;

  return (
    <div className="chat-interface">
      {/* Header with title, tags, and export options */}
      {hasMessages && (
        <div className="chat-header">
          <div className="chat-header-left">
            <h2 className="chat-title">{conversation.title || 'Conversation'}</h2>
            {conversation.tags && conversation.tags.length > 0 && !showTagEditor && (
              <div className="header-tags">
                {conversation.tags.map((tag) => (
                  <span key={tag} className="header-tag">#{tag}</span>
                ))}
              </div>
            )}
          </div>
          <div className="chat-header-actions">
            <button
              className={`action-btn ${showTagEditor ? 'active' : ''}`}
              onClick={() => setShowTagEditor(!showTagEditor)}
              title="Edit tags"
            >
              Tags
            </button>
            <button
              className="action-btn"
              onClick={() => exportToMarkdown(conversation)}
              title="Export to Markdown"
            >
              Export MD
            </button>
            <button
              className="action-btn"
              onClick={() => exportToJSON(conversation)}
              title="Export to JSON"
            >
              Export JSON
            </button>
          </div>
        </div>
      )}

      {/* Tag Editor */}
      {hasMessages && showTagEditor && (
        <div className="tag-editor-container">
          <TagEditor
            tags={conversation.tags || []}
            onTagsChange={onTagsChange}
          />
        </div>
      )}

      <div className="messages-container">
        {conversation.messages.length === 0 ? (
          <div className="empty-state">
            <h2>Start a conversation</h2>
            <p>Ask a question to consult the LLM Council</p>
          </div>
        ) : (
          conversation.messages.map((msg, index) => (
            <div key={index} className="message-group">
              {msg.role === 'user' ? (
                <div className="user-message">
                  <div className="message-label">You</div>
                  <div className="message-content">
                    <div className="markdown-content">
                      <ReactMarkdown>{msg.content}</ReactMarkdown>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="assistant-message">
                  <div className="message-label">LLM Council</div>

                  {/* Stage 1 */}
                  {/* Routing Status */}
                  {msg.loading?.routing && (
                    <div className="stage-loading routing-loading">
                      <div className="spinner"></div>
                      <span>Classifying question for dynamic routing...</span>
                    </div>
                  )}

                  {/* Tier 1 Escalation Status */}
                  {msg.loading?.tier1 && !msg.stage1Streaming && (
                    <div className="stage-loading tier-loading">
                      <div className="spinner"></div>
                      <span>Tier 1: Querying cost-effective models...</span>
                    </div>
                  )}

                  {/* Escalation Triggered - Tier 2 Status */}
                  {msg.loading?.tier2 && (
                    <div className="stage-loading tier-loading escalation">
                      <div className="spinner"></div>
                      <span>Escalating to Tier 2: Querying premium models...</span>
                    </div>
                  )}

                  {/* Stage 1 */}
                  {msg.loading?.stage1 && !msg.stage1Streaming && !msg.loading?.tier1 && !msg.loading?.tier2 && (
                    <div className="stage-loading">
                      <div className="spinner"></div>
                      <span>Running Stage 1: Collecting individual responses...</span>
                    </div>
                  )}
                  {(msg.stage1 || msg.stage1Streaming) && (
                    <Stage1
                      responses={msg.stage1 || []}
                      aggregateConfidence={msg.metadata?.aggregate_confidence}
                      streamingResponses={msg.stage1Streaming}
                      streamingReasoning={msg.stage1ReasoningStreaming}
                      isStreaming={msg.loading?.stage1}
                      routingInfo={msg.routingInfo}
                      escalationInfo={msg.escalationInfo}
                    />
                  )}

                  {/* Stage 2 */}
                  {msg.loading?.stage2 && (
                    <div className="stage-loading">
                      <div className="spinner"></div>
                      <span>Running Stage 2: Peer rankings...</span>
                    </div>
                  )}
                  {msg.stage2 && (
                    <Stage2
                      rankings={msg.stage2}
                      labelToModel={msg.metadata?.label_to_model}
                      aggregateRankings={msg.metadata?.aggregate_rankings}
                      useWeightedConsensus={msg.metadata?.use_weighted_consensus}
                      weightsInfo={msg.metadata?.weights_info}
                    />
                  )}

                  {/* Stage 3 */}
                  {msg.loading?.stage3 && !msg.stage3Streaming && !msg.multiSyntheses?.length && !msg.isConsensus && !msg.isRefining && (
                    <div className="stage-loading">
                      <div className="spinner"></div>
                      <span>Running Stage 3: {msg.useMultiChairman ? 'Multi-chairman synthesis...' : 'Final synthesis...'}</span>
                    </div>
                  )}
                  {msg.loading?.refinement && !msg.isRefining && (
                    <div className="stage-loading refinement-loading">
                      <div className="spinner"></div>
                      <span>Starting iterative refinement...</span>
                    </div>
                  )}
                  {(msg.stage3 || msg.stage3Streaming || msg.multiSyntheses?.length > 0 || msg.isConsensus || msg.isRefining || msg.refinementIterations?.length > 0 || msg.isDecomposing || msg.decompositionComplete || msg.subQuestions?.length > 0) && (
                    <Stage3
                      finalResponse={msg.stage3}
                      streamingResponse={msg.stage3Streaming}
                      streamingModel={msg.stage3StreamingModel}
                      isStreaming={msg.loading?.stage3 && !msg.useMultiChairman}
                      useMultiChairman={msg.useMultiChairman}
                      multiSyntheses={msg.multiSyntheses}
                      selectionStreaming={msg.selectionStreaming}
                      isSelecting={msg.isSelecting}
                      isConsensus={msg.isConsensus}
                      consensusInfo={msg.consensusInfo}
                      useRefinement={msg.useRefinement}
                      refinementIterations={msg.refinementIterations}
                      isRefining={msg.isRefining}
                      currentRefinementIteration={msg.currentRefinementIteration}
                      refinementCritiques={msg.refinementCritiques}
                      refinementStreaming={msg.refinementStreaming}
                      refinementMaxIterations={msg.refinementMaxIterations}
                      refinementConverged={msg.refinementConverged}
                      useDecomposition={msg.useDecomposition}
                      subQuestions={msg.subQuestions}
                      subResults={msg.subResults}
                      isDecomposing={msg.isDecomposing}
                      currentSubQuestion={msg.currentSubQuestion}
                      totalSubQuestions={msg.totalSubQuestions}
                      mergeStreaming={msg.mergeStreaming}
                      isMerging={msg.isMerging}
                      decompositionFinalResponse={msg.decompositionFinalResponse}
                      chairmanModel={msg.chairmanModel}
                      complexityInfo={msg.complexityInfo}
                      decompositionSkipped={msg.decompositionSkipped}
                      decompositionComplete={msg.decompositionComplete}
                    />
                  )}

                  {/* Cost Display */}
                  {msg.metadata?.costs && (
                    <CostDisplay costs={msg.metadata.costs} expanded={true} />
                  )}
                </div>
              )}
            </div>
          ))
        )}

        {isLoading && (
          <div className="loading-indicator">
            <div className="spinner"></div>
            <span>Consulting the council...</span>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {conversation.messages.length === 0 && (
        <form className="input-form" onSubmit={handleSubmit}>
          <textarea
            className="message-input"
            placeholder="Ask your question... (Shift+Enter for new line, Enter to send)"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isLoading}
            rows={3}
          />
          <button
            type="submit"
            className="send-button"
            disabled={!input.trim() || isLoading}
          >
            Send
          </button>
        </form>
      )}
    </div>
  );
}
