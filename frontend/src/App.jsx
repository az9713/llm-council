import { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import ChatInterface from './components/ChatInterface';
import ConfigPanel from './components/ConfigPanel';
import PerformanceDashboard from './components/PerformanceDashboard';
import ProcessMonitor from './components/ProcessMonitor';
import { api } from './api';
import './App.css';

function App() {
  const [conversations, setConversations] = useState([]);
  const [currentConversationId, setCurrentConversationId] = useState(null);
  const [currentConversation, setCurrentConversation] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [systemPrompt, setSystemPrompt] = useState(
    () => localStorage.getItem('systemPrompt') || ''
  );
  const [showSettings, setShowSettings] = useState(false);
  const [allTags, setAllTags] = useState([]);
  const [selectedTag, setSelectedTag] = useState(null);
  const [showConfigPanel, setShowConfigPanel] = useState(false);
  const [showDashboard, setShowDashboard] = useState(false);

  // Process Monitor state
  const [showProcessMonitor, setShowProcessMonitor] = useState(false);
  const [processVerbosity, setProcessVerbosity] = useState(
    () => parseInt(localStorage.getItem('processVerbosity') || '0', 10)
  );
  const [processEvents, setProcessEvents] = useState([]);

  // Chain-of-Thought mode state
  const [useCot, setUseCot] = useState(
    () => localStorage.getItem('useCot') === 'true'
  );

  // Multi-Chairman mode state
  const [useMultiChairman, setUseMultiChairman] = useState(
    () => localStorage.getItem('useMultiChairman') === 'true'
  );

  // Weighted Consensus mode state (default true)
  const [useWeightedConsensus, setUseWeightedConsensus] = useState(
    () => localStorage.getItem('useWeightedConsensus') !== 'false'
  );

  // Early Consensus Exit mode state (default false)
  const [useEarlyConsensus, setUseEarlyConsensus] = useState(
    () => localStorage.getItem('useEarlyConsensus') === 'true'
  );

  // Dynamic Model Routing state (default false)
  const [useDynamicRouting, setUseDynamicRouting] = useState(
    () => localStorage.getItem('useDynamicRouting') === 'true'
  );

  // Confidence-Gated Escalation state (default false)
  const [useEscalation, setUseEscalation] = useState(
    () => localStorage.getItem('useEscalation') === 'true'
  );

  // Iterative Refinement state (default false)
  const [useRefinement, setUseRefinement] = useState(
    () => localStorage.getItem('useRefinement') === 'true'
  );
  const [refinementMaxIterations, setRefinementMaxIterations] = useState(
    () => parseInt(localStorage.getItem('refinementMaxIterations') || '2', 10)
  );

  // Adversarial Validation state (default false)
  const [useAdversary, setUseAdversary] = useState(
    () => localStorage.getItem('useAdversary') === 'true'
  );

  // Debate Mode state (default false)
  const [useDebate, setUseDebate] = useState(
    () => localStorage.getItem('useDebate') === 'true'
  );
  const [includeRebuttal, setIncludeRebuttal] = useState(
    () => localStorage.getItem('includeRebuttal') !== 'false'  // default true
  );

  // Sub-Question Decomposition state (default false)
  const [useDecomposition, setUseDecomposition] = useState(
    () => localStorage.getItem('useDecomposition') === 'true'
  );

  // Semantic Response Caching state (default false)
  const [useCache, setUseCache] = useState(
    () => localStorage.getItem('useCache') === 'true'
  );

  // Load conversations and tags on mount
  useEffect(() => {
    loadConversations();
    loadAllTags();
  }, []);

  // Reload conversations when tag filter changes
  useEffect(() => {
    loadConversations(selectedTag);
  }, [selectedTag]);

  // Load conversation details when selected
  useEffect(() => {
    if (currentConversationId) {
      loadConversation(currentConversationId);
    }
  }, [currentConversationId]);

  const loadConversations = async (tag = null) => {
    try {
      const convs = await api.listConversations(tag);
      setConversations(convs);
    } catch (error) {
      console.error('Failed to load conversations:', error);
    }
  };

  const loadAllTags = async () => {
    try {
      const result = await api.getAllTags();
      setAllTags(result.tags);
    } catch (error) {
      console.error('Failed to load tags:', error);
    }
  };

  const loadConversation = async (id) => {
    try {
      const conv = await api.getConversation(id);
      setCurrentConversation(conv);
    } catch (error) {
      console.error('Failed to load conversation:', error);
    }
  };

  const handleNewConversation = async () => {
    try {
      const newConv = await api.createConversation();
      // Clear tag filter when creating new conversation
      setSelectedTag(null);
      setConversations([
        { id: newConv.id, created_at: newConv.created_at, tags: [], message_count: 0 },
        ...conversations,
      ]);
      setCurrentConversationId(newConv.id);
    } catch (error) {
      console.error('Failed to create conversation:', error);
    }
  };

  const handleTagsChange = async (tags) => {
    if (!currentConversationId) return;

    try {
      await api.updateTags(currentConversationId, tags);
      // Update current conversation state
      setCurrentConversation((prev) => ({
        ...prev,
        tags,
      }));
      // Reload conversations list to reflect tag changes
      loadConversations(selectedTag);
      // Reload all tags in case new tags were added
      loadAllTags();
    } catch (error) {
      console.error('Failed to update tags:', error);
    }
  };

  const handleTagFilterChange = (tag) => {
    setSelectedTag(tag);
  };

  const handleSelectConversation = (id) => {
    setCurrentConversationId(id);
  };

  const handleSystemPromptChange = (value) => {
    setSystemPrompt(value);
    localStorage.setItem('systemPrompt', value);
  };

  const handleVerbosityChange = (value) => {
    setProcessVerbosity(value);
    localStorage.setItem('processVerbosity', value.toString());
  };

  const handleCotChange = (value) => {
    setUseCot(value);
    localStorage.setItem('useCot', value.toString());
  };

  const handleMultiChairmanChange = (value) => {
    setUseMultiChairman(value);
    localStorage.setItem('useMultiChairman', value.toString());
  };

  const handleWeightedConsensusChange = (value) => {
    setUseWeightedConsensus(value);
    localStorage.setItem('useWeightedConsensus', value.toString());
  };

  const handleEarlyConsensusChange = (value) => {
    setUseEarlyConsensus(value);
    localStorage.setItem('useEarlyConsensus', value.toString());
  };

  const handleDynamicRoutingChange = (value) => {
    setUseDynamicRouting(value);
    localStorage.setItem('useDynamicRouting', value.toString());
  };

  const handleEscalationChange = (value) => {
    setUseEscalation(value);
    localStorage.setItem('useEscalation', value.toString());
  };

  const handleRefinementChange = (value) => {
    setUseRefinement(value);
    localStorage.setItem('useRefinement', value.toString());
  };

  const handleRefinementMaxIterationsChange = (value) => {
    setRefinementMaxIterations(value);
    localStorage.setItem('refinementMaxIterations', value.toString());
  };

  const handleAdversaryChange = (value) => {
    setUseAdversary(value);
    localStorage.setItem('useAdversary', value.toString());
  };

  const handleDebateChange = (value) => {
    setUseDebate(value);
    localStorage.setItem('useDebate', value.toString());
  };

  const handleIncludeRebuttalChange = (value) => {
    setIncludeRebuttal(value);
    localStorage.setItem('includeRebuttal', value.toString());
  };

  const handleDecompositionChange = (value) => {
    setUseDecomposition(value);
    localStorage.setItem('useDecomposition', value.toString());
  };

  const handleCacheChange = (value) => {
    setUseCache(value);
    localStorage.setItem('useCache', value.toString());
  };

  const handleSendMessage = async (content) => {
    if (!currentConversationId) return;

    setIsLoading(true);

    // Clear process events for new query
    setProcessEvents([]);

    try {
      // Optimistically add user message to UI
      const userMessage = { role: 'user', content };
      setCurrentConversation((prev) => ({
        ...prev,
        messages: [...prev.messages, userMessage],
      }));

      // Create a partial assistant message that will be updated progressively
      const assistantMessage = {
        role: 'assistant',
        stage1: null,
        stage2: null,
        stage3: null,
        metadata: null,
        loading: {
          routing: false,
          stage1: false,
          stage2: false,
          stage3: false,
          tier1: false,
          tier2: false,
          refinement: false,
          adversary: false,
        },
        // Dynamic routing state
        useDynamicRouting: useDynamicRouting,
        routingInfo: null,
        // Multi-chairman streaming state
        useMultiChairman: useMultiChairman,
        multiSyntheses: [],
        selectionStreaming: '',
        isSelecting: false,
        // Escalation state
        useEscalation: useEscalation,
        escalationInfo: null,
        currentTier: null,
        escalated: false,
        // Refinement state
        useRefinement: useRefinement,
        refinementIterations: [],
        refinementStreaming: '',
        refinementCritiques: [],
        isRefining: false,
        currentRefinementIteration: 0,
        refinementMaxIterations: refinementMaxIterations,
        refinementConverged: false,
        // Adversary state
        useAdversary: useAdversary,
        adversaryCritique: '',
        adversaryStreaming: '',
        adversaryRevisionStreaming: '',
        isAdversaryReviewing: false,
        isAdversaryRevising: false,
        adversaryResult: null,
        // Debate state
        useDebate: useDebate,
        includeRebuttal: includeRebuttal,
        isDebating: false,
        debateRound: 0,
        debatePositions: [],
        debateCritiques: [],
        debateRebuttals: [],
        debateJudgment: '',
        debateJudgmentStreaming: '',
        isJudging: false,
        debateModelToLabel: {},
        debateLabelToModel: {},
        debateNumRounds: 3,
        // Decomposition state
        useDecomposition: useDecomposition,
        isDecomposing: false,
        decompositionSkipped: false,
        complexityInfo: null,
        subQuestions: [],
        subResults: [],
        currentSubQuestion: -1,
        totalSubQuestions: 0,
        mergeStreaming: '',
        isMerging: false,
        decompositionComplete: false,
        // Cache state
        useCache: useCache,
        cacheChecking: false,
        cacheHit: null,
        cacheStored: null,
      };

      // Add the partial assistant message
      setCurrentConversation((prev) => ({
        ...prev,
        messages: [...prev.messages, assistantMessage],
      }));

      // Send message with streaming
      await api.sendMessageStream(
        currentConversationId,
        content,
        (eventType, event) => {
        switch (eventType) {
          case 'routing_start':
            // Dynamic routing classification starting
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.loading.routing = true;
              return { ...prev, messages };
            });
            break;

          case 'routing_complete':
            // Dynamic routing classification finished
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.loading.routing = false;
              lastMsg.routingInfo = event.data;
              return { ...prev, messages };
            });
            break;

          case 'tier1_start':
            // Tier 1 escalation starting
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.loading.tier1 = true;
              lastMsg.currentTier = 1;
              return { ...prev, messages };
            });
            break;

          case 'tier1_complete':
            // Tier 1 complete, checking if escalation needed
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.loading.tier1 = false;
              return { ...prev, messages };
            });
            break;

          case 'escalation_triggered':
            // Escalation is triggered, Tier 2 will run
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.escalated = true;
              lastMsg.escalationInfo = event.data;
              return { ...prev, messages };
            });
            break;

          case 'tier2_start':
            // Tier 2 escalation starting
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.loading.tier2 = true;
              lastMsg.currentTier = 2;
              return { ...prev, messages };
            });
            break;

          case 'stage1_start':
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.loading.stage1 = true;
              // Initialize streaming state for Stage 1
              lastMsg.stage1Streaming = {};
              lastMsg.stage1ReasoningStreaming = {};
              return { ...prev, messages };
            });
            break;

          case 'stage1_token':
            // Accumulate tokens for a specific model
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              const model = event.model;
              // Initialize or append to streaming content
              if (!lastMsg.stage1Streaming) {
                lastMsg.stage1Streaming = {};
              }
              lastMsg.stage1Streaming[model] = (lastMsg.stage1Streaming[model] || '') + event.content;
              return { ...prev, messages };
            });
            break;

          case 'stage1_reasoning_token':
            // Accumulate reasoning tokens for a specific model
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              const model = event.model;
              if (!lastMsg.stage1ReasoningStreaming) {
                lastMsg.stage1ReasoningStreaming = {};
              }
              lastMsg.stage1ReasoningStreaming[model] = (lastMsg.stage1ReasoningStreaming[model] || '') + event.content;
              return { ...prev, messages };
            });
            break;

          case 'stage1_model_complete':
            // A single model has finished - update its entry in stage1
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              // Initialize stage1 array if needed
              if (!lastMsg.stage1) {
                lastMsg.stage1 = [];
              }
              // Add or update the model's response
              const existingIndex = lastMsg.stage1.findIndex(r => r.model === event.data.model);
              if (existingIndex >= 0) {
                lastMsg.stage1[existingIndex] = event.data;
              } else {
                lastMsg.stage1.push(event.data);
              }
              // Clear streaming state for this model
              if (lastMsg.stage1Streaming) {
                delete lastMsg.stage1Streaming[event.data.model];
              }
              if (lastMsg.stage1ReasoningStreaming) {
                delete lastMsg.stage1ReasoningStreaming[event.data.model];
              }
              return { ...prev, messages };
            });
            break;

          case 'stage1_error':
            // Model error during Stage 1 - log but continue
            console.warn('Stage 1 model error:', event.model, event.error);
            break;

          case 'stage1_complete':
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.stage1 = event.data;
              lastMsg.loading.stage1 = false;
              lastMsg.loading.tier1 = false;
              lastMsg.loading.tier2 = false;
              // Clear streaming state
              lastMsg.stage1Streaming = null;
              lastMsg.stage1ReasoningStreaming = null;
              // Store aggregate confidence from Stage 1 metadata
              if (event.metadata?.aggregate_confidence) {
                lastMsg.metadata = {
                  ...lastMsg.metadata,
                  aggregate_confidence: event.metadata.aggregate_confidence,
                };
              }
              // Store escalation info from Stage 1 metadata
              if (event.metadata?.escalation_info) {
                lastMsg.escalationInfo = event.metadata.escalation_info;
                lastMsg.escalated = event.metadata.escalation_info.escalated;
              }
              return { ...prev, messages };
            });
            break;

          case 'stage2_start':
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.loading.stage2 = true;
              return { ...prev, messages };
            });
            break;

          case 'stage2_complete':
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.stage2 = event.data;
              lastMsg.metadata = {
                ...lastMsg.metadata,
                ...event.metadata,
              };
              lastMsg.loading.stage2 = false;
              return { ...prev, messages };
            });
            break;

          case 'consensus_detected':
            // Early consensus was detected - Stage 3 will be skipped
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.consensusInfo = event.data;
              lastMsg.isConsensus = true;
              return { ...prev, messages };
            });
            break;

          case 'stage3_start':
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.loading.stage3 = true;
              // Check if multi-chairman mode
              if (event.use_multi_chairman) {
                lastMsg.useMultiChairman = true;
                lastMsg.multiSyntheses = [];
              } else {
                // Initialize streaming state for Stage 3
                lastMsg.stage3Streaming = '';
              }
              return { ...prev, messages };
            });
            break;

          case 'multi_synthesis_start':
            // Multi-chairman synthesis starting
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.multiSyntheses = [];
              return { ...prev, messages };
            });
            break;

          case 'synthesis_complete':
            // A chairman has finished synthesizing
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.multiSyntheses = [...(lastMsg.multiSyntheses || []), event.data];
              return { ...prev, messages };
            });
            break;

          case 'multi_synthesis_complete':
            // All chairmen have finished synthesizing
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.multiSyntheses = event.syntheses;
              return { ...prev, messages };
            });
            break;

          case 'selection_start':
            // Supreme chairman selection starting
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.isSelecting = true;
              lastMsg.selectionStreaming = '';
              return { ...prev, messages };
            });
            break;

          case 'selection_token':
            // Supreme chairman token
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.selectionStreaming = (lastMsg.selectionStreaming || '') + event.content;
              return { ...prev, messages };
            });
            break;

          case 'stage3_token':
            // Accumulate tokens for chairman response (single chairman mode)
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.stage3Streaming = (lastMsg.stage3Streaming || '') + event.content;
              // Also store the model for display
              if (!lastMsg.stage3StreamingModel) {
                lastMsg.stage3StreamingModel = event.model;
              }
              return { ...prev, messages };
            });
            break;

          case 'stage3_error':
            console.error('Stage 3 error:', event.error);
            break;

          case 'stage3_complete':
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.stage3 = event.data;
              lastMsg.loading.stage3 = false;
              // Clear streaming state
              lastMsg.stage3Streaming = null;
              lastMsg.stage3StreamingModel = null;
              // Clear multi-chairman streaming state
              lastMsg.isSelecting = false;
              lastMsg.selectionStreaming = null;
              // Mark if this was multi-chairman
              if (event.use_multi_chairman) {
                lastMsg.useMultiChairman = true;
              }
              // Mark if this was a consensus exit
              if (event.is_consensus) {
                lastMsg.isConsensus = true;
                lastMsg.consensusInfo = event.consensus_info;
              }
              return { ...prev, messages };
            });
            break;

          case 'costs_complete':
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              // Add costs to metadata
              lastMsg.metadata = {
                ...lastMsg.metadata,
                costs: event.data,
              };
              return { ...prev, messages };
            });
            break;

          case 'title_complete':
            // Reload conversations to get updated title
            loadConversations();
            break;

          case 'complete':
            // Stream complete, reload conversations list
            loadConversations();
            setIsLoading(false);
            break;

          case 'error':
            console.error('Stream error:', event.message);
            setIsLoading(false);
            break;

          case 'refinement_start':
            // Iterative refinement starting
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.loading.refinement = true;
              lastMsg.isRefining = true;
              lastMsg.refinementIterations = [];
              lastMsg.refinementCritiques = [];
              lastMsg.refinementStreaming = '';
              lastMsg.currentRefinementIteration = 0;
              if (event.max_iterations) {
                lastMsg.refinementMaxIterations = event.max_iterations;
              }
              return { ...prev, messages };
            });
            break;

          case 'iteration_start':
            // New refinement iteration starting
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.currentRefinementIteration = event.iteration;
              lastMsg.refinementCritiques = [];
              lastMsg.refinementStreaming = '';
              return { ...prev, messages };
            });
            break;

          case 'critiques_start':
            // Starting to collect critiques
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.refinementCritiques = [];
              return { ...prev, messages };
            });
            break;

          case 'critique_complete':
            // A single critique has been received
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.refinementCritiques = [...(lastMsg.refinementCritiques || []), {
                model: event.model,
                critique: event.critique,
                is_substantive: event.is_substantive,
              }];
              return { ...prev, messages };
            });
            break;

          case 'critiques_complete':
            // All critiques collected for this iteration
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.refinementCritiques = event.critiques;
              return { ...prev, messages };
            });
            break;

          case 'revision_start':
            // Chairman starting revision
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.refinementStreaming = '';
              return { ...prev, messages };
            });
            break;

          case 'revision_token':
            // Accumulate revision tokens
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.refinementStreaming = (lastMsg.refinementStreaming || '') + event.content;
              return { ...prev, messages };
            });
            break;

          case 'revision_complete':
            // Revision finished for this iteration
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.refinementStreaming = '';
              return { ...prev, messages };
            });
            break;

          case 'iteration_complete':
            // Full iteration complete, add to iterations list
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              const newIteration = {
                iteration: event.iteration,
                critiques: lastMsg.refinementCritiques || [],
                revision: event.revision,
                substantive_critique_count: (lastMsg.refinementCritiques || []).filter(c => c.is_substantive).length,
              };
              lastMsg.refinementIterations = [...(lastMsg.refinementIterations || []), newIteration];
              lastMsg.refinementCritiques = [];
              lastMsg.refinementStreaming = '';
              return { ...prev, messages };
            });
            break;

          case 'refinement_converged':
            // Refinement converged early
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.refinementConverged = true;
              // Add final iteration with convergence info
              const convergenceIteration = {
                iteration: event.iteration,
                critiques: lastMsg.refinementCritiques || [],
                stopped: true,
                stop_reason: event.reason,
                substantive_critique_count: (lastMsg.refinementCritiques || []).filter(c => c.is_substantive).length,
              };
              lastMsg.refinementIterations = [...(lastMsg.refinementIterations || []), convergenceIteration];
              return { ...prev, messages };
            });
            break;

          case 'refinement_complete':
            // Full refinement loop complete
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.loading.refinement = false;
              lastMsg.isRefining = false;
              lastMsg.refinementIterations = event.iterations;
              lastMsg.refinementConverged = event.converged;
              lastMsg.refinementCritiques = [];
              lastMsg.refinementStreaming = '';
              // Update stage3 response with refined version
              if (lastMsg.stage3 && event.final_response) {
                lastMsg.stage3.response = event.final_response;
                lastMsg.stage3.refinement_applied = true;
                lastMsg.stage3.refinement_iterations = event.total_iterations;
                lastMsg.stage3.refinement_converged = event.converged;
              }
              return { ...prev, messages };
            });
            break;

          case 'revision_error':
            // Error during revision
            console.warn('Revision error:', event.error);
            break;

          case 'adversary_start':
            // Adversarial validation starting
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.loading.adversary = true;
              lastMsg.isAdversaryReviewing = true;
              lastMsg.adversaryStreaming = '';
              lastMsg.adversaryModel = event.adversary_model;
              return { ...prev, messages };
            });
            break;

          case 'adversary_token':
            // Accumulate adversary critique tokens
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.adversaryStreaming = (lastMsg.adversaryStreaming || '') + event.content;
              return { ...prev, messages };
            });
            break;

          case 'adversary_complete':
            // Adversary review complete
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.isAdversaryReviewing = false;
              lastMsg.adversaryCritique = event.critique;
              lastMsg.adversaryHasIssues = event.has_issues;
              lastMsg.adversarySeverity = event.severity;
              lastMsg.adversaryStreaming = '';
              return { ...prev, messages };
            });
            break;

          case 'adversary_error':
            // Adversary error
            console.warn('Adversary error:', event.error);
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.isAdversaryReviewing = false;
              lastMsg.loading.adversary = false;
              return { ...prev, messages };
            });
            break;

          case 'adversary_revision_start':
            // Chairman starting revision based on adversary feedback
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.isAdversaryRevising = true;
              lastMsg.adversaryRevisionStreaming = '';
              return { ...prev, messages };
            });
            break;

          case 'adversary_revision_token':
            // Accumulate revision tokens
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.adversaryRevisionStreaming = (lastMsg.adversaryRevisionStreaming || '') + event.content;
              return { ...prev, messages };
            });
            break;

          case 'adversary_revision_complete':
            // Revision complete
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.isAdversaryRevising = false;
              lastMsg.adversaryRevision = event.response;
              lastMsg.adversaryRevisionStreaming = '';
              return { ...prev, messages };
            });
            break;

          case 'adversary_revision_error':
            // Revision error
            console.warn('Adversary revision error:', event.error);
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.isAdversaryRevising = false;
              return { ...prev, messages };
            });
            break;

          case 'adversary_validation_complete':
            // Full adversarial validation complete
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.loading.adversary = false;
              lastMsg.isAdversaryReviewing = false;
              lastMsg.isAdversaryRevising = false;
              lastMsg.adversaryResult = {
                issues_found: event.issues_found,
                severity: event.severity,
                revised: event.revised,
              };
              // Update stage3 response with validated/revised version
              if (lastMsg.stage3 && event.final_response) {
                lastMsg.stage3.response = event.final_response;
                lastMsg.stage3.adversary_applied = true;
                lastMsg.stage3.adversary_issues_found = event.issues_found;
                lastMsg.stage3.adversary_severity = event.severity;
                lastMsg.stage3.adversary_revised = event.revised;
              }
              return { ...prev, messages };
            });
            break;

          // =================================================================
          // Debate Mode Events
          // =================================================================

          case 'debate_start':
            // Debate mode starting
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.isDebating = true;
              lastMsg.debateRound = 0;
              lastMsg.debateModelToLabel = event.model_to_label || {};
              lastMsg.debateLabelToModel = event.label_to_model || {};
              lastMsg.debateNumRounds = event.num_rounds || 3;
              return { ...prev, messages };
            });
            break;

          case 'round1_start':
            // Position round starting
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.debateRound = 1;
              return { ...prev, messages };
            });
            break;

          case 'position_complete':
            // A position has been received
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.debatePositions = [...(lastMsg.debatePositions || []), {
                model: event.model,
                position: event.position,
                label: event.label,
              }];
              return { ...prev, messages };
            });
            break;

          case 'round1_complete':
            // All positions collected
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.debatePositions = event.positions || lastMsg.debatePositions;
              return { ...prev, messages };
            });
            break;

          case 'round2_start':
            // Critique round starting
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.debateRound = 2;
              return { ...prev, messages };
            });
            break;

          case 'debate_critique_complete':
            // A debate critique has been received (different from refinement critique)
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.debateCritiques = [...(lastMsg.debateCritiques || []), {
                critic: event.critic,
                target: event.target,
                critique: event.critique,
                critic_label: event.critic_label,
                target_label: event.target_label,
              }];
              return { ...prev, messages };
            });
            break;

          case 'round2_complete':
            // All critiques collected
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.debateCritiques = event.critiques || lastMsg.debateCritiques;
              return { ...prev, messages };
            });
            break;

          case 'round3_start':
            // Rebuttal round starting
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.debateRound = 3;
              return { ...prev, messages };
            });
            break;

          case 'rebuttal_complete':
            // A rebuttal has been received
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.debateRebuttals = [...(lastMsg.debateRebuttals || []), {
                model: event.model,
                rebuttal: event.rebuttal,
                label: event.label,
              }];
              return { ...prev, messages };
            });
            break;

          case 'round3_complete':
            // All rebuttals collected
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.debateRebuttals = event.rebuttals || lastMsg.debateRebuttals;
              return { ...prev, messages };
            });
            break;

          case 'judgment_start':
            // Chairman judgment starting
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.isJudging = true;
              lastMsg.debateRound = 4; // After all debate rounds
              return { ...prev, messages };
            });
            break;

          case 'judgment_token':
            // Accumulate judgment tokens
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.debateJudgmentStreaming = (lastMsg.debateJudgmentStreaming || '') + event.content;
              return { ...prev, messages };
            });
            break;

          case 'judgment_complete':
            // Judgment finished
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.isJudging = false;
              lastMsg.debateJudgment = event.judgment;
              lastMsg.debateJudgmentStreaming = '';
              return { ...prev, messages };
            });
            break;

          case 'judgment_error':
            // Judgment error
            console.warn('Judgment error:', event.error);
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.isJudging = false;
              return { ...prev, messages };
            });
            break;

          case 'debate_complete':
            // Full debate complete
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.isDebating = false;
              lastMsg.debatePositions = event.positions || lastMsg.debatePositions;
              lastMsg.debateCritiques = event.critiques || lastMsg.debateCritiques;
              lastMsg.debateRebuttals = event.rebuttals || lastMsg.debateRebuttals;
              lastMsg.debateJudgment = event.judgment || lastMsg.debateJudgment;
              lastMsg.debateModelToLabel = event.model_to_label || lastMsg.debateModelToLabel;
              lastMsg.debateLabelToModel = event.label_to_model || lastMsg.debateLabelToModel;
              lastMsg.debateNumRounds = event.num_rounds || lastMsg.debateNumRounds;
              // Set stage3 result with debate info
              lastMsg.stage3 = {
                model: event.chairman || 'Chairman',
                response: event.judgment || '',
                debate_mode: true,
                num_rounds: event.num_rounds || 3,
              };
              return { ...prev, messages };
            });
            break;

          // =================================================================
          // Sub-Question Decomposition Events
          // =================================================================

          case 'decomposition_start':
            // Decomposition mode starting
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.isDecomposing = true;
              lastMsg.subQuestions = [];
              lastMsg.subResults = [];
              return { ...prev, messages };
            });
            break;

          case 'complexity_analyzed':
            // Complexity analysis complete
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.complexityInfo = {
                is_complex: event.is_complex,
                confidence: event.confidence,
                reasoning: event.reasoning,
              };
              return { ...prev, messages };
            });
            break;

          case 'decomposition_skip':
            // Question not complex enough, falling through to normal flow
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.isDecomposing = false;
              lastMsg.decompositionSkipped = true;
              return { ...prev, messages };
            });
            break;

          case 'sub_questions_generated':
            // Sub-questions have been generated
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.subQuestions = event.sub_questions || [];
              lastMsg.totalSubQuestions = event.count || event.sub_questions?.length || 0;
              return { ...prev, messages };
            });
            break;

          case 'sub_council_start':
            // Starting to process a sub-question
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.currentSubQuestion = event.index;
              return { ...prev, messages };
            });
            break;

          case 'sub_council_response':
            // A model has responded to the current sub-question
            // This is intermediate - we wait for sub_council_complete
            break;

          case 'sub_council_complete':
            // A sub-question has been fully answered
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              const newResult = {
                index: event.index,
                sub_question: event.sub_question,
                best_answer: event.best_answer,
                best_model: event.best_model,
              };
              lastMsg.subResults = [...(lastMsg.subResults || []), newResult];
              return { ...prev, messages };
            });
            break;

          case 'all_sub_councils_complete':
            // All sub-questions have been answered
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.subResults = event.results || lastMsg.subResults;
              return { ...prev, messages };
            });
            break;

          case 'merge_start':
            // Chairman starting to merge sub-answers
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.isMerging = true;
              lastMsg.mergeStreaming = '';
              lastMsg.chairmanModel = event.chairman_model;
              return { ...prev, messages };
            });
            break;

          case 'merge_token':
            // Accumulate merge tokens
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.mergeStreaming = (lastMsg.mergeStreaming || '') + event.content;
              return { ...prev, messages };
            });
            break;

          case 'merge_complete':
            // Merge finished
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.isMerging = false;
              lastMsg.mergeStreaming = '';
              // Store the final merged response
              if (event.response) {
                lastMsg.decompositionFinalResponse = event.response;
              }
              return { ...prev, messages };
            });
            break;

          case 'decomposition_complete':
            // Full decomposition complete
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.isDecomposing = false;
              lastMsg.isMerging = false;
              lastMsg.decompositionComplete = true;
              lastMsg.subQuestions = event.sub_questions || lastMsg.subQuestions;
              lastMsg.subResults = event.sub_results || lastMsg.subResults;
              lastMsg.decompositionFinalResponse = event.final_response;
              lastMsg.chairmanModel = event.chairman_model;
              // Set stage3 result with decomposition info
              lastMsg.stage3 = {
                model: event.chairman_model || 'Chairman',
                response: event.final_response || '',
                decomposition_mode: true,
                sub_question_count: event.sub_questions?.length || lastMsg.subQuestions?.length || 0,
              };
              return { ...prev, messages };
            });
            break;

          // =================================================================
          // Semantic Response Caching Events
          // =================================================================

          case 'cache_check_start':
            // Cache check starting
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.cacheChecking = true;
              return { ...prev, messages };
            });
            break;

          case 'cache_hit':
            // Cache hit - response found
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.cacheChecking = false;
              lastMsg.cacheHit = {
                similarity: event.similarity,
                cached_query: event.cached_query,
                cache_id: event.cache_id,
                created_at: event.created_at,
                hit_count: event.hit_count,
              };
              // Set stages from cached response
              if (event.cached_response) {
                lastMsg.stage1 = event.cached_response.stage1 || null;
                lastMsg.stage2 = event.cached_response.stage2 || null;
                lastMsg.stage3 = event.cached_response.stage3 || null;
                lastMsg.metadata = event.cached_response.metadata || null;
              }
              lastMsg.loading.stage1 = false;
              lastMsg.loading.stage2 = false;
              lastMsg.loading.stage3 = false;
              return { ...prev, messages };
            });
            break;

          case 'cache_miss':
            // Cache miss - will run full council
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.cacheChecking = false;
              lastMsg.cacheHit = null;
              return { ...prev, messages };
            });
            break;

          case 'cache_stored':
            // Response stored in cache
            setCurrentConversation((prev) => {
              const messages = [...prev.messages];
              const lastMsg = messages[messages.length - 1];
              lastMsg.cacheStored = {
                cache_id: event.cache_id,
                embedding_method: event.embedding_method,
                cache_size: event.cache_size,
              };
              return { ...prev, messages };
            });
            break;

          case 'process':
            // Add process event to the list
            setProcessEvents((prev) => [...prev, event]);
            break;

          default:
            console.log('Unknown event type:', eventType);
        }
      },
        systemPrompt || null,
        processVerbosity,
        useCot,
        useMultiChairman,
        useWeightedConsensus,
        useEarlyConsensus,
        useDynamicRouting,
        useEscalation,
        useRefinement,
        refinementMaxIterations,
        useAdversary,
        useDebate,
        includeRebuttal,
        useDecomposition,
        useCache
      );
    } catch (error) {
      console.error('Failed to send message:', error);
      // Remove optimistic messages on error
      setCurrentConversation((prev) => ({
        ...prev,
        messages: prev.messages.slice(0, -2),
      }));
      setIsLoading(false);
    }
  };

  return (
    <div className="app">
      <Sidebar
        conversations={conversations}
        currentConversationId={currentConversationId}
        onSelectConversation={handleSelectConversation}
        onNewConversation={handleNewConversation}
        allTags={allTags}
        selectedTag={selectedTag}
        onTagFilterChange={handleTagFilterChange}
      />
      <div className="main-content">
        <div className="settings-bar">
          <div className="settings-bar-row">
            <button
              className="settings-toggle"
              onClick={() => setShowSettings(!showSettings)}
            >
              {showSettings ? '▼ Hide Settings' : '▶ Settings'}
              {systemPrompt && <span className="settings-active-indicator" title="System prompt active">●</span>}
              {useCot && <span className="settings-cot-indicator" title="Chain-of-Thought mode active">CoT</span>}
              {useMultiChairman && <span className="settings-multi-indicator" title="Multi-Chairman mode active">MC</span>}
              {useWeightedConsensus && <span className="settings-weighted-indicator" title="Weighted Consensus mode active">WC</span>}
              {useEarlyConsensus && <span className="settings-early-consensus-indicator" title="Early Consensus Exit mode active">EC</span>}
              {useDynamicRouting && <span className="settings-routing-indicator" title="Dynamic Model Routing active">DR</span>}
              {useEscalation && <span className="settings-escalation-indicator" title="Confidence-Gated Escalation active">CG</span>}
              {useRefinement && <span className="settings-refinement-indicator" title="Iterative Refinement active">IR</span>}
              {useAdversary && <span className="settings-adversary-indicator" title="Adversarial Validation active">AV</span>}
              {useDebate && <span className="settings-debate-indicator" title="Debate Mode active">DB</span>}
              {useDecomposition && <span className="settings-decomposition-indicator" title="Sub-Question Decomposition active">DQ</span>}
              {useCache && <span className="settings-cache-indicator" title="Semantic Response Caching active">CA</span>}
            </button>
            <div className="settings-bar-controls">
              <button
                className={`process-monitor-btn ${processVerbosity > 0 ? 'active' : ''}`}
                onClick={() => setShowProcessMonitor(!showProcessMonitor)}
                title="Toggle Process Monitor"
              >
                Process {processVerbosity > 0 ? `(${processVerbosity})` : ''}
              </button>
              <button
                className="dashboard-btn"
                onClick={() => setShowDashboard(true)}
              >
                Dashboard
              </button>
              <button
                className="config-models-btn"
                onClick={() => setShowConfigPanel(true)}
              >
                Configure Models
              </button>
            </div>
          </div>
          {showSettings && (
            <div className="settings-panel">
              <div className="settings-section">
                <label htmlFor="system-prompt">System Prompt</label>
                <textarea
                  id="system-prompt"
                  value={systemPrompt}
                  onChange={(e) => handleSystemPromptChange(e.target.value)}
                  placeholder="Enter a system prompt to customize model behavior (e.g., 'You are a helpful coding assistant. Always provide code examples.')..."
                  rows={3}
                />
                {systemPrompt && (
                  <button
                    className="clear-prompt-btn"
                    onClick={() => handleSystemPromptChange('')}
                  >
                    Clear
                  </button>
                )}
              </div>

              <div className="settings-section cot-section">
                <label className="cot-toggle-wrapper">
                  <input
                    type="checkbox"
                    checked={useCot}
                    onChange={(e) => handleCotChange(e.target.checked)}
                  />
                  <span className="cot-toggle-slider" />
                  <span className="cot-toggle-label">
                    <span className="cot-toggle-title">Chain-of-Thought Mode</span>
                    <span className="cot-toggle-description">
                      Request structured reasoning (Thinking → Analysis → Conclusion)
                    </span>
                  </span>
                </label>
              </div>

              <div className="settings-section multi-chairman-section">
                <label className="multi-chairman-toggle-wrapper">
                  <input
                    type="checkbox"
                    checked={useMultiChairman}
                    onChange={(e) => handleMultiChairmanChange(e.target.checked)}
                  />
                  <span className="multi-chairman-toggle-slider" />
                  <span className="multi-chairman-toggle-label">
                    <span className="multi-chairman-toggle-title">Multi-Chairman Mode</span>
                    <span className="multi-chairman-toggle-description">
                      Ensemble synthesis from multiple chairmen, with supreme chairman selection
                    </span>
                  </span>
                </label>
              </div>

              <div className="settings-section weighted-consensus-section">
                <label className="weighted-consensus-toggle-wrapper">
                  <input
                    type="checkbox"
                    checked={useWeightedConsensus}
                    onChange={(e) => handleWeightedConsensusChange(e.target.checked)}
                  />
                  <span className="weighted-consensus-toggle-slider" />
                  <span className="weighted-consensus-toggle-label">
                    <span className="weighted-consensus-toggle-title">Weighted Consensus</span>
                    <span className="weighted-consensus-toggle-description">
                      Weight model votes by historical performance (better models count more)
                    </span>
                  </span>
                </label>
              </div>

              <div className="settings-section early-consensus-section">
                <label className="early-consensus-toggle-wrapper">
                  <input
                    type="checkbox"
                    checked={useEarlyConsensus}
                    onChange={(e) => handleEarlyConsensusChange(e.target.checked)}
                  />
                  <span className="early-consensus-toggle-slider" />
                  <span className="early-consensus-toggle-label">
                    <span className="early-consensus-toggle-title">Early Consensus Exit</span>
                    <span className="early-consensus-toggle-description">
                      Skip Stage 3 synthesis when all models strongly agree (saves time and cost)
                    </span>
                  </span>
                </label>
              </div>

              <div className="settings-section dynamic-routing-section">
                <label className="dynamic-routing-toggle-wrapper">
                  <input
                    type="checkbox"
                    checked={useDynamicRouting}
                    onChange={(e) => handleDynamicRoutingChange(e.target.checked)}
                  />
                  <span className="dynamic-routing-toggle-slider" />
                  <span className="dynamic-routing-toggle-label">
                    <span className="dynamic-routing-toggle-title">Dynamic Model Routing</span>
                    <span className="dynamic-routing-toggle-description">
                      Classify questions and route to specialized model pools (coding, creative, factual, analysis)
                    </span>
                  </span>
                </label>
              </div>

              <div className="settings-section escalation-section">
                <label className="escalation-toggle-wrapper">
                  <input
                    type="checkbox"
                    checked={useEscalation}
                    onChange={(e) => handleEscalationChange(e.target.checked)}
                  />
                  <span className="escalation-toggle-slider" />
                  <span className="escalation-toggle-label">
                    <span className="escalation-toggle-title">Confidence-Gated Escalation</span>
                    <span className="escalation-toggle-description">
                      Start with cost-effective models, escalate to premium models if confidence is low
                    </span>
                  </span>
                </label>
              </div>

              <div className="settings-section refinement-section">
                <label className="refinement-toggle-wrapper">
                  <input
                    type="checkbox"
                    checked={useRefinement}
                    onChange={(e) => handleRefinementChange(e.target.checked)}
                  />
                  <span className="refinement-toggle-slider" />
                  <span className="refinement-toggle-label">
                    <span className="refinement-toggle-title">Iterative Refinement</span>
                    <span className="refinement-toggle-description">
                      Council critiques and chairman revises until quality converges
                    </span>
                  </span>
                </label>
                {useRefinement && (
                  <div className="refinement-options">
                    <label className="refinement-iterations-label">
                      Max iterations:
                      <select
                        value={refinementMaxIterations}
                        onChange={(e) => handleRefinementMaxIterationsChange(parseInt(e.target.value, 10))}
                      >
                        <option value="1">1</option>
                        <option value="2">2</option>
                        <option value="3">3</option>
                        <option value="4">4</option>
                        <option value="5">5</option>
                      </select>
                    </label>
                  </div>
                )}
              </div>

              <div className="settings-section adversary-section">
                <label className="adversary-toggle-wrapper">
                  <input
                    type="checkbox"
                    checked={useAdversary}
                    onChange={(e) => handleAdversaryChange(e.target.checked)}
                  />
                  <span className="adversary-toggle-slider" />
                  <span className="adversary-toggle-label">
                    <span className="adversary-toggle-title">Adversarial Validation</span>
                    <span className="adversary-toggle-description">
                      Devil's advocate review to find flaws, with revision if issues found
                    </span>
                  </span>
                </label>
              </div>

              <div className="settings-section debate-section">
                <label className="debate-toggle-wrapper">
                  <input
                    type="checkbox"
                    checked={useDebate}
                    onChange={(e) => handleDebateChange(e.target.checked)}
                  />
                  <span className="debate-toggle-slider" />
                  <span className="debate-toggle-label">
                    <span className="debate-toggle-title">Debate Mode</span>
                    <span className="debate-toggle-description">
                      Multi-round structured debate: Position → Critique → Rebuttal → Judgment
                    </span>
                  </span>
                </label>
                {useDebate && (
                  <label className="rebuttal-checkbox">
                    <input
                      type="checkbox"
                      checked={includeRebuttal}
                      onChange={(e) => handleIncludeRebuttalChange(e.target.checked)}
                    />
                    <span>Include Round 3 (Rebuttals)</span>
                  </label>
                )}
              </div>

              <div className="settings-section decomposition-section">
                <label className="decomposition-toggle-wrapper">
                  <input
                    type="checkbox"
                    checked={useDecomposition}
                    onChange={(e) => handleDecompositionChange(e.target.checked)}
                  />
                  <span className="decomposition-toggle-slider" />
                  <span className="decomposition-toggle-label">
                    <span className="decomposition-toggle-title">Sub-Question Decomposition</span>
                    <span className="decomposition-toggle-description">
                      Break complex questions into sub-questions, answer each, then merge (map-reduce)
                    </span>
                  </span>
                </label>
              </div>

              <div className="settings-section cache-section">
                <label className="cache-toggle-wrapper">
                  <input
                    type="checkbox"
                    checked={useCache}
                    onChange={(e) => handleCacheChange(e.target.checked)}
                  />
                  <span className="cache-toggle-slider" />
                  <span className="cache-toggle-label">
                    <span className="cache-toggle-title">Semantic Response Caching</span>
                    <span className="cache-toggle-description">
                      Cache responses and return similar past answers (saves time and cost)
                    </span>
                  </span>
                </label>
              </div>
            </div>
          )}
        </div>
        <ChatInterface
          conversation={currentConversation}
          onSendMessage={handleSendMessage}
          isLoading={isLoading}
          onTagsChange={handleTagsChange}
        />
      </div>

      {/* Model Configuration Panel */}
      {showConfigPanel && (
        <>
          <div
            className="config-overlay"
            onClick={() => setShowConfigPanel(false)}
          />
          <ConfigPanel onClose={() => setShowConfigPanel(false)} />
        </>
      )}

      {/* Performance Dashboard */}
      {showDashboard && (
        <>
          <div
            className="config-overlay"
            onClick={() => setShowDashboard(false)}
          />
          <PerformanceDashboard onClose={() => setShowDashboard(false)} />
        </>
      )}

      {/* Process Monitor Side Panel */}
      <ProcessMonitor
        events={processEvents}
        verbosity={processVerbosity}
        onVerbosityChange={handleVerbosityChange}
        isOpen={showProcessMonitor}
        onToggle={() => setShowProcessMonitor(!showProcessMonitor)}
      />
    </div>
  );
}

export default App;
