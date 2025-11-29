import './Sidebar.css';

export default function Sidebar({
  conversations,
  currentConversationId,
  onSelectConversation,
  onNewConversation,
  allTags,
  selectedTag,
  onTagFilterChange,
}) {
  return (
    <div className="sidebar">
      <div className="sidebar-header">
        <h1>LLM Council</h1>
        <button className="new-conversation-btn" onClick={onNewConversation}>
          + New Conversation
        </button>
      </div>

      {/* Tag Filter */}
      {allTags && allTags.length > 0 && (
        <div className="tag-filter">
          <select
            value={selectedTag || ''}
            onChange={(e) => onTagFilterChange(e.target.value || null)}
            className="tag-filter-select"
          >
            <option value="">All conversations</option>
            {allTags.map((tag) => (
              <option key={tag} value={tag}>
                #{tag}
              </option>
            ))}
          </select>
        </div>
      )}

      <div className="conversation-list">
        {conversations.length === 0 ? (
          <div className="no-conversations">
            {selectedTag ? `No conversations with #${selectedTag}` : 'No conversations yet'}
          </div>
        ) : (
          conversations.map((conv) => (
            <div
              key={conv.id}
              className={`conversation-item ${
                conv.id === currentConversationId ? 'active' : ''
              }`}
              onClick={() => onSelectConversation(conv.id)}
            >
              <div className="conversation-title">
                {conv.title || 'New Conversation'}
              </div>
              <div className="conversation-meta">
                {conv.message_count} messages
              </div>
              {conv.tags && conv.tags.length > 0 && (
                <div className="conversation-tags">
                  {conv.tags.map((tag) => (
                    <span key={tag} className="conversation-tag">
                      #{tag}
                    </span>
                  ))}
                </div>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
}
