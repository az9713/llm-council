# Development Plan for LLM Council Features

This document provides a flexible implementation plan for all 21 features. Most features are independent and can be developed in any order. Only 2 features have hard dependencies that must be respected.

---

## Table of Contents

1. [**Sequential Development Order (1→21)**](#sequential-development-order-121)
2. [Quick Reference: What Can I Build Now?](#quick-reference-what-can-i-build-now)
3. [Hard Dependencies (Must Respect)](#hard-dependencies-must-respect)
4. [Feature Independence Map](#feature-independence-map)
5. [All Features by Complexity](#all-features-by-complexity)
6. [Suggested Tracks (Optional Paths)](#suggested-tracks-optional-paths)
7. [Complete Feature Reference](#complete-feature-reference)
8. [Package Requirements](#package-requirements)
9. [Implementation Checklists](#implementation-checklists)

---

## Sequential Development Order (1→21)

This is the **definitive step-by-step order** for developing all 21 features. The order is optimized to:
- Respect all hard dependencies
- Build prerequisites before features that benefit from them
- Start with quick wins for momentum
- Group related features logically
- Save the most complex features for when you have experience

### Complete Development Sequence

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DEVELOPMENT ORDER: STEP 1 → 21                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PHASE 1: FOUNDATION (Steps 1-3)                                            │
│  ─────────────────────────────────                                          │
│  Step 1:  #3  Custom System Prompts      Easy     2-4 hrs   [Unlocks #12]   │
│  Step 2:  #2  Conversation Export        Easy     2-4 hrs                   │
│  Step 3:  #6  Question Categories        Medium   4-6 hrs                   │
│                                                             Subtotal: 8-14h │
│                                                                             │
│  PHASE 2: CORE ENHANCEMENTS (Steps 4-7)                                     │
│  ──────────────────────────────────────                                     │
│  Step 4:  #5  Model Configuration UI     Medium   6-8 hrs   [Enables #1]    │
│  Step 5:  #7  Reasoning Model Support    Medium   4-6 hrs                   │
│  Step 6:  #8  Cost Tracking              Medium   6-8 hrs   [Unlocks #18]   │
│  Step 7:  #10 Confidence Voting          Medium   4-6 hrs   [Unlocks #18]   │
│                                                             Subtotal: 20-28h│
│                                                                             │
│  PHASE 3: REAL-TIME & OBSERVABILITY (Steps 8-10)                            │
│  ───────────────────────────────────────────────                            │
│  Step 8:  #4  Streaming Responses        Hard     8-12 hrs  [Enables #9,11] │
│  Step 9:  #1  Performance Dashboard      Medium   6-8 hrs                   │
│  Step 10: #11 Live Process Monitor       Med-Hard 8-10 hrs                  │
│                                                             Subtotal: 22-30h│
│                                                                             │
│  PHASE 4: REASONING PATTERNS (Steps 11-12)                                  │
│  ─────────────────────────────────────────                                  │
│  Step 11: #20 Chain-of-Thought           Medium   6-8 hrs   [Enables #15,17]│
│  Step 12: #21 Multi-Chairman Synthesis   Medium   6-8 hrs                   │
│                                                             Subtotal: 12-16h│
│                                                                             │
│  PHASE 5: CONSENSUS OPTIMIZATION (Steps 13-14)                              │
│  ─────────────────────────────────────────────                              │
│  Step 13: #13 Weighted Consensus         Easy     3-5 hrs                   │
│  Step 14: #14 Early Consensus Exit       Easy     2-4 hrs                   │
│                                                             Subtotal: 5-9h  │
│                                                                             │
│  PHASE 6: INTELLIGENT ROUTING (Steps 15-16)                                 │
│  ──────────────────────────────────────────                                 │
│  Step 15: #12 Dynamic Model Routing      Hard     10-14 hrs [Requires #3 ✓] │
│  Step 16: #18 Conf-Gated Escalation      Medium   6-8 hrs   [Requires #8+10]│
│                                                             Subtotal: 16-22h│
│                                                                             │
│  PHASE 7: ADVANCED ORCHESTRATION (Steps 17-19)                              │
│  ─────────────────────────────────────────────                              │
│  Step 17: #15 Iterative Refinement       Medium   6-8 hrs                   │
│  Step 18: #17 Adversarial Validation     Medium   6-8 hrs                   │
│  Step 19: #9  Debate Mode                Hard     10-14 hrs                 │
│                                                             Subtotal: 22-30h│
│                                                                             │
│  PHASE 8: COMPLEX PATTERNS (Steps 20-21)                                    │
│  ───────────────────────────────────────                                    │
│  Step 20: #16 Sub-Question Decomposition Hard     10-14 hrs                 │
│  Step 21: #19 Response Caching           Hard     12-16 hrs [New packages]  │
│                                                             Subtotal: 22-30h│
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  GRAND TOTAL: 127-179 hours for all 21 features                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Step-by-Step Details

#### Step 1: Feature #3 - Custom System Prompts ✅ COMPLETED
| Attribute | Value |
|-----------|-------|
| **Complexity** | Easy |
| **Effort** | 2-4 hours |
| **Status** | ✅ **IMPLEMENTED** |
| **Why First** | Quick win, teaches prompt engineering, **unlocks #12 later** |
| **Files** | `backend/council.py`, `backend/main.py`, `frontend/src/App.jsx`, `frontend/src/api.js`, `frontend/src/App.css` |
| **Packages** | None |

**What was built:** A collapsible settings panel with a textarea where users can customize the system prompt that all council models receive. The prompt is persisted to localStorage and indicated with a blue dot when active.

---

#### Step 2: Feature #2 - Conversation Export ✅ COMPLETED
| Attribute | Value |
|-----------|-------|
| **Complexity** | Easy |
| **Effort** | 2-4 hours |
| **Status** | ✅ **IMPLEMENTED** |
| **Why Second** | Quick win, immediately useful, builds confidence |
| **Files** | `frontend/src/utils/export.js` (new), `frontend/src/components/ChatInterface.jsx`, `frontend/src/components/ChatInterface.css` |
| **Packages** | None |

**What was built:** Header toolbar with "Export MD" and "Export JSON" buttons that download conversations using the browser Blob API. Formats all 3 stages with model names in Markdown output.

---

#### Step 3: Feature #6 - Question Categories & Tagging ✅ COMPLETED
| Attribute | Value |
|-----------|-------|
| **Complexity** | Medium |
| **Effort** | 4-6 hours |
| **Status** | ✅ **IMPLEMENTED** |
| **Why Third** | Organizational feature, full-stack practice, still relatively simple |
| **Files** | `backend/storage.py`, `backend/main.py`, `frontend/src/components/TagEditor.jsx` (new), `frontend/src/components/TagEditor.css` (new), `frontend/src/components/Sidebar.jsx`, `frontend/src/components/Sidebar.css`, `frontend/src/components/ChatInterface.jsx`, `frontend/src/components/ChatInterface.css`, `frontend/src/App.jsx`, `frontend/src/api.js` |
| **Packages** | None |

**What was built:** Full tagging system with TagEditor component, tag filter dropdown in Sidebar, tags displayed on conversations, suggested tags for quick selection, and backend API endpoints for tag CRUD and filtering.

---

#### Step 4: Feature #5 - Model Configuration UI ✅ COMPLETED
| Attribute | Value |
|-----------|-------|
| **Complexity** | Medium |
| **Effort** | 6-8 hours |
| **Status** | ✅ **IMPLEMENTED** |
| **Why Fourth** | Important foundation, **enables #1 Dashboard** to be more useful |
| **Files** | `backend/config_api.py` (new), `backend/main.py`, `backend/council.py`, `frontend/src/components/ConfigPanel.jsx` (new), `frontend/src/components/ConfigPanel.css` (new), `frontend/src/api.js`, `frontend/src/App.jsx`, `frontend/src/App.css` |
| **Packages** | None |

**What was built:** Modal configuration panel accessible via "Configure Models" button in settings bar. Features include add/remove/reorder council models, chairman selection dropdown, model autocomplete with suggestions, quick-add buttons for popular models, validation (minimum 2 models), reset to defaults, and configuration persistence to `data/council_config.json`.

---

#### Step 5: Feature #7 - Reasoning Model Support ✅ COMPLETED
| Attribute | Value |
|-----------|-------|
| **Complexity** | Medium |
| **Effort** | 4-6 hours |
| **Status** | ✅ **IMPLEMENTED** |
| **Why Fifth** | Enhances core functionality, shows extended thinking from o1/DeepSeek-R1 |
| **Files** | `backend/council.py`, `frontend/src/components/Stage1.jsx`, `frontend/src/components/Stage1.css` |
| **Packages** | None |

**What was built:** Support for reasoning models (o1, o3, DeepSeek-R1) that return `reasoning_details`. Backend captures reasoning from OpenRouter API and passes it through Stage 1 results. Frontend displays "Reasoning Model" badge, asterisk indicator on tabs, and collapsible "Show Thinking Process" section with amber/gold styling. Handles various reasoning formats (string, object, array).

---

#### Step 6: Feature #8 - Cost Tracking ✅ COMPLETED
| Attribute | Value |
|-----------|-------|
| **Complexity** | Medium |
| **Effort** | 6-8 hours |
| **Status** | ✅ **IMPLEMENTED** |
| **Why Sixth** | Practical value, **required for #18** (Confidence-Gated Escalation) |
| **Files** | `backend/pricing.py` (new), `backend/openrouter.py`, `backend/council.py`, `backend/main.py`, `frontend/src/components/CostDisplay.jsx` (new), `frontend/src/components/CostDisplay.css` (new), `frontend/src/components/ChatInterface.jsx`, `frontend/src/App.jsx` |
| **Packages** | None |

**What was built:** Real-time cost tracking showing tokens used and USD cost per model per query. Backend captures usage data from OpenRouter API responses and calculates costs using pricing data in `pricing.py`. Costs are aggregated per stage (Stage 1, 2, 3) and total. CostDisplay component shows breakdown with token counts, USD costs, and percentage distribution. Streaming endpoint emits `costs_complete` event.

---

#### Step 7: Feature #10 - Confidence Voting ✅ COMPLETED
| Attribute | Value |
|-----------|-------|
| **Complexity** | Medium |
| **Effort** | 4-6 hours |
| **Status** | ✅ **IMPLEMENTED** |
| **Why Seventh** | Adds quality signal, **required for #18**, enables better #14 |
| **Files** | `backend/council.py`, `backend/main.py`, `frontend/src/components/ConfidenceDisplay.jsx` (new), `frontend/src/components/ConfidenceDisplay.css` (new), `frontend/src/components/Stage1.jsx`, `frontend/src/components/Stage1.css`, `frontend/src/components/ChatInterface.jsx`, `frontend/src/App.jsx` |
| **Packages** | None |

**What was built:** Each council model reports a confidence score (1-10) with their response. Backend parses confidence using multiple regex patterns, cleans response text, and calculates aggregate statistics (average, min, max, count, distribution). Frontend displays individual confidence as color-coded badges on model tabs and aggregate confidence summary in the Stage 1 header. Streaming endpoint includes confidence metadata after Stage 1 completes.

---

#### Step 8: Feature #4 - Streaming Responses ✅ COMPLETED
| Attribute | Value |
|-----------|-------|
| **Complexity** | Hard |
| **Effort** | 8-12 hours |
| **Status** | ✅ **IMPLEMENTED** |
| **Why Eighth** | Major UX improvement, **enables #9 and #11** to work optimally |
| **Files** | `backend/openrouter.py`, `backend/council.py`, `backend/main.py`, `frontend/src/api.js`, `frontend/src/App.jsx`, `frontend/src/components/Stage1.jsx`, `frontend/src/components/Stage1.css`, `frontend/src/components/Stage3.jsx`, `frontend/src/components/Stage3.css`, `frontend/src/components/ChatInterface.jsx` |
| **Packages** | None |

**What was built:** Real-time token-by-token streaming using Server-Sent Events (SSE) for Stage 1 and Stage 3. Backend adds `query_model_streaming()` and `query_models_parallel_streaming()` functions that use OpenRouter's streaming API and yield token events as they arrive. Council functions `stage1_collect_responses_streaming()` and `stage3_synthesize_final_streaming()` orchestrate streaming for their respective stages. Frontend handles new event types (`stage1_token`, `stage1_reasoning_token`, `stage3_token`) to accumulate and display partial responses in real-time. Visual indicators include "Streaming..." badges, pulsing blue dots on active tabs, and blinking cursor at the end of streaming text. Stage 2 remains batch for simplicity.

---

#### Step 9: Feature #1 - Performance Dashboard ✅ COMPLETED
| Attribute | Value |
|-----------|-------|
| **Complexity** | Medium |
| **Effort** | 6-8 hours |
| **Status** | ✅ **IMPLEMENTED** |
| **Why Ninth** | After Model Config (#5), dashboard data is more meaningful |
| **Files** | `backend/analytics.py` (new), `backend/main.py`, `backend/council.py`, `frontend/src/api.js`, `frontend/src/App.jsx`, `frontend/src/App.css`, `frontend/src/components/PerformanceDashboard.jsx` (new), `frontend/src/components/PerformanceDashboard.css` (new) |
| **Packages** | None (decided not to use recharts - custom CSS-based visualizations instead) |

**What was built:**
- Backend `analytics.py` module that records query results automatically after each council query
- Analytics data stored in `data/analytics/model_stats.json` (separate from conversations)
- API endpoints: `GET /api/analytics`, `GET /api/analytics/recent`, `GET /api/analytics/chairman`, `DELETE /api/analytics`
- PerformanceDashboard modal with three tabbed views:
  - **Leaderboard**: Model ranking sorted by win rate with medal icons for top 3
  - **Model Details**: Cards with detailed stats and rank distribution bar charts
  - **Chairman Stats**: Chairman model usage and cost breakdown
- Summary statistics: total queries, unique models, syntheses count, date range
- "Dashboard" button in settings bar opens the modal
- Purple/indigo color scheme (#667eea) to distinguish from config panel

---

#### Step 10: Feature #11 - Live Process Monitor ✅ COMPLETED
| Attribute | Value |
|-----------|-------|
| **Complexity** | Medium-Hard |
| **Effort** | 8-10 hours |
| **Status** | ✅ **IMPLEMENTED** |
| **Why Tenth** | After Streaming (#4), monitor can show real-time events effectively |
| **Files** | `backend/process_logger.py` (new), `backend/main.py`, `frontend/src/api.js`, `frontend/src/components/ProcessMonitor.jsx` (new), `frontend/src/components/ProcessMonitor.css` (new), `frontend/src/App.jsx`, `frontend/src/App.css` |
| **Packages** | None |

**What was built:**
- Backend `process_logger.py` module with verbosity levels (0=Silent, 1=Basic, 2=Standard, 3=Verbose)
- Event categories for color-coding: stage, model, info, success, warning, error, data
- Pre-defined event generators for common operations (stage_start, model_query_complete, etc.)
- `should_emit()` function to filter events based on verbosity level
- Streaming endpoint accepts `verbosity` parameter and emits `process` events
- ProcessMonitor side panel component with:
  - Verbosity slider (0-3) with labeled levels
  - Auto-scrolling event log with timestamps
  - Color-coded events by category
  - Collapsible panel with toggle button
  - Event count badge when collapsed
- "Process" button in settings bar toggles monitor visibility
- Verbosity preference persisted to localStorage
- Dark theme (#1a1a2e) to distinguish from main content area

---

#### Step 11: Feature #20 - Chain-of-Thought Orchestration ✅ COMPLETED
| Attribute | Value |
|-----------|-------|
| **Complexity** | Medium |
| **Effort** | 6-8 hours |
| **Status** | ✅ **IMPLEMENTED** |
| **Why Eleventh** | Reasoning pattern foundation, **enables #15 and #17** |
| **Files** | `backend/council.py`, `backend/main.py`, `frontend/src/components/ReasoningView.jsx` (new), `frontend/src/components/ReasoningView.css` (new), `frontend/src/components/Stage1.jsx`, `frontend/src/components/Stage1.css`, `frontend/src/App.jsx`, `frontend/src/App.css`, `frontend/src/api.js` |
| **Packages** | None |

**What was built:**
- Backend `COT_PROMPT_SUFFIX` appends structured reasoning request (THINKING → ANALYSIS → CONCLUSION)
- `parse_cot_response()` extracts sections from model responses with multiple format support
- `extract_response_without_cot()` returns just the conclusion when CoT is present
- `stage1_collect_responses()` and streaming version accept `use_cot` parameter
- `stage2_collect_rankings()` accepts `use_cot` and evaluates reasoning quality when enabled
- Stage 2 CoT-aware ranking evaluates: reasoning quality, analysis depth, conclusion accuracy, overall coherence
- API endpoints accept `use_cot` boolean parameter
- ReasoningView component displays three-step reasoning:
  - Step 1 (blue): Thinking - Initial thoughts and considerations
  - Step 2 (purple): Analysis - Evaluation of approaches
  - Step 3 (green): Conclusion - Final answer
- Progress visualization in header showing completed reasoning steps
- Collapsible sections (conclusion expanded by default)
- CoT toggle in settings panel with slider switch
- "CoT" indicator badge in settings bar when enabled
- CoT badges on model tabs showing structured reasoning available
- Setting persisted to localStorage
- Distinct from native Reasoning Model support (prompt-based vs API-provided)

---

#### Step 12: Feature #21 - Multi-Chairman Synthesis ✅ COMPLETED
| Attribute | Value |
|-----------|-------|
| **Complexity** | Medium |
| **Effort** | 6-8 hours |
| **Status** | ✅ **IMPLEMENTED** |
| **Why Twelfth** | Ensemble method, natural extension of current architecture |
| **Files** | `backend/multi_chairman.py` (new), `backend/config_api.py`, `backend/main.py`, `frontend/src/components/MultiSynthesis.jsx` (new), `frontend/src/components/MultiSynthesis.css` (new), `frontend/src/components/Stage3.jsx`, `frontend/src/App.jsx`, `frontend/src/App.css`, `frontend/src/api.js` |
| **Packages** | None |

**What was built:**
- Backend `multi_chairman.py` module with two-phase synthesis:
  - Phase A: `stage3_multi_chairman_synthesis()` queries multiple chairmen in parallel
  - Phase B: `stage3_supreme_chairman_selection()` evaluates and selects best synthesis
  - `stage3_multi_chairman_streaming()` for real-time streaming of synthesis/selection
  - `calculate_multi_chairman_costs()` for cost aggregation
- Configuration: `multi_chairman_models` list in `config_api.py` and `data/council_config.json`
- `get_multi_chairman_models()` getter in `config_api.py` with validation
- API endpoint accepts `use_multi_chairman` parameter
- Streaming events: `multi_synthesis_start`, `synthesis_complete`, `multi_synthesis_complete`, `selection_start`, `selection_token`
- MultiSynthesis.jsx component:
  - Tabbed view of all chairman syntheses (labeled A, B, C...)
  - Checkmark icon on selected synthesis
  - Supreme chairman selection section with model name
  - Collapsible "Show Evaluation Details" with reasoning
  - Final response section with green border
- Multi-Chairman toggle in settings panel with green color scheme
- "MC" indicator badge in settings bar when enabled
- Stage3.jsx updated to support both single and multi-chairman modes
- Setting persisted to localStorage

---

#### Step 13: Feature #13 - Weighted Consensus Voting ✅ COMPLETED
| Attribute | Value |
|-----------|-------|
| **Complexity** | Easy |
| **Effort** | 3-5 hours |
| **Status** | ✅ **IMPLEMENTED** |
| **Why Thirteenth** | After Dashboard (#1), historical performance data makes weights meaningful |
| **Files** | `backend/weights.py` (new), `backend/council.py`, `backend/main.py`, `frontend/src/components/Stage2.jsx`, `frontend/src/components/Stage2.css`, `frontend/src/App.jsx`, `frontend/src/App.css`, `frontend/src/api.js` |
| **Packages** | None |

**What was built:**
- Backend `weights.py` module with weight calculation logic:
  - `get_model_weights()`: Calculate weights from analytics (win rate, avg rank, confidence)
  - `calculate_weighted_aggregate_rankings()`: Weighted voting for aggregate rankings
  - `get_weights_summary()`: Summary for frontend display
  - Weight formula: base (1.0) + win_rate_bonus + rank_bonus + confidence_bonus
  - Weights clamped to 0.3-2.5 range, normalized so average = 1.0
  - Minimum 2 queries required before historical data affects weights
- Backend `council.py` updated:
  - `run_full_council()` accepts `use_weighted_consensus` parameter (default True)
  - Uses `weights.calculate_weighted_aggregate_rankings()` for weighted mode
  - Returns metadata with `use_weighted_consensus` and `weights_info`
- Backend `main.py` updated:
  - API accepts `use_weighted_consensus` boolean parameter
  - New endpoints: `GET /api/weights`, `GET /api/weights/{model}`
  - `stage2_complete` event includes `use_weighted_consensus` and `weights_info`
- Frontend Stage2.jsx updated:
  - "Weighted" badge in aggregate rankings header when enabled
  - Collapsible weights summary section showing model weights
  - Displays normalized_weight (×) for each model with history
  - Shows weighted_average_rank instead of average_rank when weighted
  - Rank change indicator (positive/negative) showing impact of weighting
- Weighted Consensus toggle in settings panel with amber/gold color scheme
- "WC" indicator badge in settings bar when enabled
- Setting persisted to localStorage (default true)
- Uses analytics data from Performance Dashboard feature

---

#### Step 14: Feature #14 - Early Consensus Exit ✅ COMPLETED
| Attribute | Value |
|-----------|-------|
| **Complexity** | Easy |
| **Effort** | 2-4 hours |
| **Status** | ✅ **IMPLEMENTED** |
| **Why Fourteenth** | After Confidence (#10), consensus detection is more reliable |
| **Files** | `backend/council.py`, `backend/main.py`, `frontend/src/api.js`, `frontend/src/App.jsx`, `frontend/src/App.css`, `frontend/src/components/Stage3.jsx`, `frontend/src/components/Stage3.css`, `frontend/src/components/ChatInterface.jsx` |
| **Packages** | None |

**What was built:**
- Backend `detect_consensus()` function in `council.py` with three-criteria detection:
  - `CONSENSUS_MAX_AVG_RANK = 1.5`: Top model's average rank must be ≤ 1.5
  - `CONSENSUS_MIN_AGREEMENT = 0.8`: At least 80% of models must rank top model as #1
  - `CONSENSUS_MIN_CONFIDENCE = 7`: Average confidence must be ≥ 7
- `run_full_council()` accepts `use_early_consensus` parameter, skips Stage 3 on consensus
- Streaming endpoint checks consensus after Stage 2, emits `consensus_detected` event
- `stage3_complete` event includes `is_consensus` and `consensus_info` when consensus triggered
- API accepts `use_early_consensus` boolean parameter (default false)
- Frontend Stage3.jsx displays special consensus view:
  - Cyan/teal theme (#ecfeff background, #06b6d4 accents)
  - "Consensus" badge in header
  - Green notice box explaining Stage 3 was skipped
  - Metrics grid: winning model, average rank, first-place votes, confidence
  - Winner's response displayed prominently
- Early Consensus toggle in settings panel with cyan color scheme
- "EC" indicator badge in settings bar when enabled
- Setting persisted to localStorage (default false)
- Saves time and cost when models strongly agree on best response

---

#### Step 15: Feature #12 - Dynamic Model Routing ✅ COMPLETED
| Attribute | Value |
|-----------|-------|
| **Complexity** | Hard |
| **Effort** | 10-14 hours |
| **Status** | ✅ **IMPLEMENTED** |
| **Why Fifteenth** | **Requires #3** (done in Step 1), sophisticated orchestration |
| **Files** | `backend/router.py` (new), `backend/config_api.py`, `backend/council.py`, `backend/main.py`, `frontend/src/api.js`, `frontend/src/App.jsx`, `frontend/src/App.css`, `frontend/src/components/Stage1.jsx`, `frontend/src/components/Stage1.css`, `frontend/src/components/ChatInterface.jsx`, `frontend/src/components/ChatInterface.css` |
| **Packages** | None |

**What was built:**
- Backend `router.py` module implementing question classification and routing:
  - `QuestionCategory` enum: CODING, CREATIVE, FACTUAL, ANALYSIS, GENERAL
  - `CLASSIFIER_MODEL`: Fast model (Gemini Flash) for classification
  - `CATEGORY_KEYWORDS`: Fallback heuristic keyword lists per category
  - `DEFAULT_MODEL_POOLS`: Optimized model pools per question type
  - `classify_by_keywords()`: Quick heuristic classification using keywords
  - `classify_question()`: LLM-based classification with confidence score and reasoning
  - `get_model_pool()`: Returns appropriate models (intersection with available)
  - `route_query()`: Main entry point returning full routing info
  - `CATEGORY_INFO`: Display info per category (name, description, color)
- Backend `config_api.py` updated:
  - `routing_pools` in DEFAULT_CONFIG (optimized per category)
  - `get_routing_pools()` getter function
  - Configuration persisted to `data/council_config.json`
- Backend `council.py` updated:
  - `stage1_collect_responses()` accepts optional `models` parameter for routing
  - `stage1_collect_responses_streaming()` accepts optional `models` parameter
  - `run_full_council()` accepts `use_dynamic_routing` parameter
  - Calls `router.route_query()` before Stage 1 when routing enabled
  - Metadata includes `use_dynamic_routing` and `routing_info`
- Backend `main.py` updated:
  - API accepts `use_dynamic_routing` boolean parameter (default false)
  - Routing endpoints: `GET /api/routing/pools`, `POST /api/routing/classify`
  - SSE events: `routing_start`, `routing_complete`
  - `stage1_start` includes `routing_info` when routing applied
- Frontend `api.js` updated:
  - `sendMessageStream()` accepts `useDynamicRouting` parameter
  - New APIs: `getRoutingPools()`, `classifyQuestion(query)`
- Frontend `App.jsx` updated:
  - `useDynamicRouting` state (persisted to localStorage)
  - `handleDynamicRoutingChange()` handler
  - SSE event handling for `routing_start`, `routing_complete`
  - Message state includes `routingInfo`, `loading.routing`
  - "DR" indicator badge in settings bar when enabled
  - Dynamic Routing toggle in settings panel with orange color scheme
- Frontend `Stage1.jsx` updated:
  - Accepts `routingInfo` prop
  - Displays routing badge in header showing detected category
  - Badge color per category: green (coding), purple (creative), blue (factual), amber (analysis), gray (general)
  - Hover tooltip shows confidence percentage and reasoning
- Frontend `ChatInterface.jsx` updated:
  - Routing loading indicator with orange theme during classification
  - Passes `routingInfo` to Stage1 component

---

#### Step 16: Feature #18 - Confidence-Gated Escalation ✅ COMPLETED
| Attribute | Value |
|-----------|-------|
| **Complexity** | Medium |
| **Effort** | 6-8 hours |
| **Status** | ✅ **IMPLEMENTED** |
| **Why Sixteenth** | **Requires #8 + #10** (done in Steps 6-7), cost-aware routing |
| **Files** | `backend/escalation.py` (new), `backend/config_api.py`, `backend/main.py`, `frontend/src/components/TierIndicator.jsx` (new), `frontend/src/components/TierIndicator.css` (new), `frontend/src/api.js`, `frontend/src/App.jsx`, `frontend/src/App.css`, `frontend/src/components/Stage1.jsx`, `frontend/src/components/ChatInterface.jsx`, `frontend/src/components/ChatInterface.css` |
| **Packages** | None |

**What was built:**
- Backend `escalation.py` module implementing tiered model selection:
  - `TIER_INFO`: Display information for Tier 1 (Fast) and Tier 2 (Premium)
  - `should_escalate()`: Evaluates whether to escalate based on three triggers:
    - Low average confidence (< 6.0)
    - Low minimum confidence (any model < 4)
    - Low first-place agreement ratio (< 50%)
  - `merge_tier_results()`: Merges Tier 1 and Tier 2 results with tier markers
  - `get_tier_info()`: Returns tier configuration for display
- Backend `config_api.py` updated with tier configurations:
  - `tier1_models`: Cost-effective, fast models (Gemini Flash, GPT-4.1-mini, Claude Haiku, DeepSeek)
  - `tier2_models`: Premium, high-quality models (Claude Sonnet 4.5, GPT-5.1, Gemini 3 Pro, o3)
  - `escalation_confidence_threshold`: 6.0 (average confidence threshold)
  - `escalation_min_confidence_threshold`: 4 (minimum single-model confidence)
  - `escalation_agreement_threshold`: 0.5 (first-place agreement ratio)
  - Getter functions: `get_tier1_models()`, `get_tier2_models()`, `get_escalation_thresholds()`
- Backend `main.py` updated:
  - API accepts `use_escalation` boolean parameter (default false)
  - Escalation endpoints: `GET /api/escalation/tiers`, `GET /api/escalation/thresholds`
  - SSE events: `tier1_start`, `tier1_complete`, `escalation_triggered`, `tier2_start`
  - Full escalation flow in event_generator: Tier 1 → evaluate → conditionally Tier 2 → merge
- Frontend `TierIndicator.jsx` component:
  - `TierIndicator`: Shows "Tier 1 Fast" or "Tier 2 Premium" badge
  - `TierBadge`: Small "T1" or "T2" badge for model tabs
  - `EscalationBanner`: Detailed banner showing escalation reasons and metrics
  - `TierSummary`: Compact summary in Stage 1 header
- Frontend `TierIndicator.css`: Styling with blue (Tier 1) and amber (Tier 2) themes
- Frontend `api.js` updated:
  - `sendMessageStream()` accepts `useEscalation` parameter
  - New APIs: `getEscalationTiers()`, `getEscalationThresholds()`
- Frontend `App.jsx` updated:
  - `useEscalation` state (persisted to localStorage)
  - `handleEscalationChange()` handler
  - SSE event handling for tier and escalation events
  - Message state includes: `escalationInfo`, `currentTier`, `escalated`, `loading.tier1`, `loading.tier2`
  - "CG" indicator badge in settings bar when enabled
  - Escalation toggle in settings panel with pink theme (#ec4899)
  - Escalation disabled when Dynamic Routing is active (routing takes precedence)
- Frontend `Stage1.jsx` updated:
  - Accepts `escalationInfo` prop
  - Shows `TierSummary` in header
  - Shows `EscalationBanner` when escalation was triggered
  - Shows `TierBadge` on individual model tabs
- Frontend `ChatInterface.jsx` updated:
  - Tier 1 and Tier 2 loading indicators with themed styling
  - Passes `escalationInfo` to Stage1 component
- Process monitor events for tier transitions and escalation

---

#### Step 17: Feature #15 - Iterative Refinement Loop ✅ COMPLETED
| Attribute | Value |
|-----------|-------|
| **Complexity** | Medium |
| **Effort** | 6-8 hours |
| **Status** | ✅ **IMPLEMENTED** |
| **Why Seventeenth** | After Chain-of-Thought (#20), refinement builds on reasoning patterns |
| **Files** | `backend/refinement.py` (new), `backend/main.py`, `frontend/src/components/RefinementView.jsx` (new), `frontend/src/components/RefinementView.css` (new), `frontend/src/api.js`, `frontend/src/App.jsx`, `frontend/src/App.css`, `frontend/src/components/Stage3.jsx`, `frontend/src/components/Stage3.css`, `frontend/src/components/ChatInterface.jsx`, `frontend/src/components/ChatInterface.css` |
| **Packages** | None |

**What was built:**
- Backend `refinement.py` module implementing the full refinement loop:
  - `DEFAULT_MAX_ITERATIONS = 2`: Default number of refinement cycles
  - `DEFAULT_MIN_CRITIQUES_FOR_REVISION = 2`: Minimum substantive critiques to continue
  - `NON_SUBSTANTIVE_PHRASES`: List of phrases indicating non-actionable feedback
  - `is_substantive_critique()`: Determines if a critique contains actionable feedback
  - `count_substantive_critiques()`: Counts critiques that warrant revision
  - `create_critique_prompt()`: Generates prompt for council critique of synthesis
  - `create_revision_prompt()`: Generates prompt for chairman revision based on critiques
  - `collect_critiques()`: Collects critiques from council models in parallel
  - `generate_revision()`: Chairman generates revised synthesis based on critiques
  - `generate_revision_streaming()`: Streaming version of revision generation
  - `run_refinement_loop()`: Main orchestration (batch mode)
  - `run_refinement_loop_streaming()`: Full streaming refinement with SSE events
  - `get_refinement_config()`: Returns refinement configuration for API
- Backend `main.py` updates:
  - `use_refinement` parameter in SendMessageRequest (boolean, default false)
  - `refinement_max_iterations` parameter (integer, default 2, range 1-5)
  - Refinement endpoint: `GET /api/refinement/config`
  - SSE event handling for all refinement events in event_generator
  - Refinement runs after Stage 3 completes (if enabled and not early exit)
- Frontend `RefinementView.jsx` component:
  - `RefinementView`: Main component showing refinement iterations
    - Collapsible section with iteration count and convergence status in header
    - Iteration cards displaying critiques (amber for substantive, gray for non-substantive)
    - Revision sections with purple gradient backgrounds
    - Real-time streaming support during active refinement
    - Progress indicator showing current iteration vs max
  - `RefinementBadge`: Badge for Stage 3 header
    - Shows iteration count (e.g., "2 iterations")
    - Displays "Converged" or "Max reached" status
    - Purple/violet color theme (#7c3aed)
  - `RefinementToggle`: Settings toggle component
    - Enable/disable slider with purple theme
    - Max iterations dropdown (1-5)
- Frontend `RefinementView.css`: Purple/violet theme styling
  - Primary colors: #7c3aed, #8b5cf6, #6d28d9
  - Iteration cards with critique/revision sections
  - Progress indicators and status badges
- Frontend `api.js` updates:
  - `getRefinementConfig()`: Get refinement configuration
  - `sendMessageStream()` accepts `useRefinement` and `refinementMaxIterations` parameters
- Frontend `App.jsx` updates:
  - `useRefinement` state (persisted to localStorage)
  - `refinementMaxIterations` state (persisted to localStorage, default 2)
  - `handleRefinementChange()` handler
  - `handleRefinementMaxIterationsChange()` handler
  - SSE event handlers for all refinement events
  - Message state includes: `useRefinement`, `refinementIterations`, `isRefining`, `currentRefinementIteration`, `refinementCritiques`, `refinementStreaming`, `refinementMaxIterations`, `refinementConverged`
  - "IR" indicator badge in settings bar when enabled
  - Refinement toggle section in settings panel with max iterations dropdown
- Frontend `App.css` updates:
  - `.refinement-section` and `.refinement-toggle-wrapper` styles
  - `.settings-refinement-indicator` badge style (purple theme)
  - `.refinement-options` and `.refinement-iterations-label` styles
- Frontend `Stage3.jsx` updates:
  - Imports RefinementView and RefinementBadge
  - Accepts refinement props (useRefinement, refinementIterations, isRefining, etc.)
  - Shows RefinementBadge in header when refinement was applied
  - Shows "Refining..." streaming indicator during active refinement
  - Renders RefinementView component below final response
- Frontend `Stage3.css` updates:
  - `.streaming-indicator.refining` style with purple theme
- Frontend `ChatInterface.jsx` updates:
  - Refinement loading indicator with purple theme
  - Passes all refinement props to Stage3 component
- Frontend `ChatInterface.css` updates:
  - `.refinement-loading` style with purple gradient background
- Streaming events:
  - `refinement_start`: Refinement beginning (includes max_iterations)
  - `iteration_start`: New iteration starting (includes iteration number)
  - `critiques_start`: Critique collection beginning (includes model_count)
  - `critique_complete`: Individual critique received (includes model, critique, is_substantive)
  - `critiques_complete`: All critiques collected (includes substantive_count, critiques)
  - `revision_start`: Chairman revision beginning (includes iteration, model)
  - `revision_token`: Token from chairman during revision
  - `revision_complete`: Revision finished (includes iteration, content, model, cost)
  - `iteration_complete`: Full iteration finished (includes iteration, revision)
  - `refinement_converged`: Stopped early due to quality convergence
  - `refinement_complete`: All refinement finished (includes iterations, final_response, total_iterations, converged, total_cost)
  - `revision_error`: Error during revision
- Convergence detection: Stops when fewer than 2 substantive critiques received
- Non-substantive filtering: "no issues", "looks good", "well done", etc. are filtered
- Disabled when Early Consensus Exit triggers (no synthesis to refine)

---

#### Step 18: Feature #17 - Adversarial Validation Stage ✅ COMPLETED
| Attribute | Value |
|-----------|-------|
| **Complexity** | Medium |
| **Effort** | 6-8 hours |
| **Status** | ✅ **IMPLEMENTED** |
| **Why Eighteenth** | After Chain-of-Thought (#20), adversarial review benefits from structured reasoning |
| **Files** | `backend/adversary.py` (new), `backend/main.py`, `frontend/src/components/AdversaryReview.jsx` (new), `frontend/src/components/AdversaryReview.css` (new), `frontend/src/api.js`, `frontend/src/App.jsx`, `frontend/src/App.css`, `frontend/src/components/Stage3.jsx` |
| **Packages** | None |

**What was built:**
- Backend `adversary.py` module implementing devil's advocate review:
  - `DEFAULT_ADVERSARY_MODEL`: `google/gemini-2.0-flash-001` (fast, critical reviewer)
  - `NO_ISSUES_PHRASES`: Phrases indicating no problems found
  - `REVISION_THRESHOLD`: Severities that trigger revision (critical, major)
  - `has_genuine_issues()`: Determines if critique contains real issues vs praise
  - `parse_severity()`: Extracts severity level (critical, major, minor, none) from critique
  - `create_adversary_prompt()`: Generates prompt for adversary to review synthesis
  - `create_revision_prompt()`: Generates prompt for chairman to revise based on critique
  - `run_adversarial_validation_streaming()`: Full streaming adversary validation
  - `get_adversary_config()`: Returns adversary configuration for API
- Backend `main.py` updates:
  - `use_adversary` parameter in SendMessageRequest (boolean, default false)
  - Adversary endpoint: `GET /api/adversary/config`
  - SSE event handling for all adversary events in event_generator
  - Adversary runs after refinement (if enabled) or Stage 3
- Frontend `AdversaryReview.jsx` component:
  - `AdversaryReview`: Main component showing critique and optional revision
    - Collapsible section with severity badge and validation result
    - Adversary model information display
    - Critique section with streaming support during review
    - Revision section (only shown if critical/major triggered revision)
    - Result indicator: "Passed" (green), "Revised" (blue), "Issues Noted" (amber)
    - Real-time streaming support during active validation
  - `AdversaryBadge`: Badge showing validation result
    - Color-coded: green (passed), blue (revised), red (critical), orange (major), amber (minor)
  - `AdversaryToggle`: Toggle control for enabling/disabling adversary mode
- Frontend `AdversaryReview.css`: Red/rose theme styling
  - Primary colors: #e11d48, #be123c, #f87171, #fda4af
  - Has issues: Red border/background gradients
  - No issues: Green border/background (#86efac, #dcfce7)
  - Revision section: Blue border/background (#93c5fd, #dbeafe)
- Frontend `api.js` updates:
  - `getAdversaryConfig()`: Get adversary configuration
  - `sendMessageStream()` accepts `useAdversary` parameter
- Frontend `App.jsx` updates:
  - `useAdversary` state (persisted to localStorage)
  - `handleAdversaryChange()` handler
  - SSE event handlers for all adversary events
  - Message state includes: `useAdversary`, `adversaryCritique`, `adversaryHasIssues`, `adversarySeverity`, `adversaryRevised`, `adversaryRevision`, `adversaryModel`, `isAdversaryReviewing`, `isAdversaryRevising`, `adversaryStreaming`, `adversaryRevisionStreaming`
  - "AV" indicator badge in settings bar when enabled
  - Adversarial Validation toggle in settings panel with red/rose theme (#e11d48)
- Frontend `App.css` updates:
  - `.adversary-section` and `.adversary-toggle-wrapper` styles
  - `.settings-adversary-indicator` badge style (red/rose theme)
- Frontend `Stage3.jsx` updates:
  - Imports AdversaryReview and AdversaryBadge
  - Accepts all adversary props
  - Shows AdversaryBadge in header when adversary was applied
  - Shows "Validating..." and "Revising..." streaming indicators
  - Renders AdversaryReview component below RefinementView
- Streaming events:
  - `adversary_start`: Validation beginning (includes adversary_model)
  - `adversary_token`: Token from adversary during critique
  - `adversary_complete`: Critique finished (includes critique, has_issues, severity)
  - `adversary_revision_start`: Chairman revision beginning
  - `adversary_revision_token`: Token from chairman during revision
  - `adversary_revision_complete`: Revision finished
  - `adversary_validation_complete`: Full validation complete (includes all final state)
- Severity-based revision: Only critical and major issues trigger revision
- Runs after Iterative Refinement (if enabled), validating the final synthesis
- Disabled when Early Consensus Exit triggers (no synthesis to validate)

---

#### Step 19: Feature #9 - Debate Mode ✅ COMPLETED
| Attribute | Value |
|-----------|-------|
| **Complexity** | Hard |
| **Effort** | 10-14 hours |
| **Status** | ✅ **IMPLEMENTED** |
| **Why Nineteenth** | After Streaming (#4), debates display beautifully in real-time |
| **Files** | `backend/debate.py` (new), `backend/main.py`, `frontend/src/components/DebateView.jsx` (new), `frontend/src/components/DebateView.css` (new), `frontend/src/api.js`, `frontend/src/App.jsx`, `frontend/src/App.css`, `frontend/src/components/Stage3.jsx` |
| **Packages** | None |

**What was built:**
- Backend `debate.py` module implementing multi-round structured debates:
  - `DEFAULT_NUM_ROUNDS`: 3 (includes rebuttal round)
  - `DEFAULT_INCLUDE_REBUTTAL`: True (whether to include Round 3)
  - `POSITION_PROMPT_TEMPLATE`: Prompt for models to state initial position
  - `CRITIQUE_PROMPT_TEMPLATE`: Prompt for models to critique another's position (anonymized)
  - `REBUTTAL_PROMPT_TEMPLATE`: Prompt for models to defend their position
  - `JUDGMENT_PROMPT_TEMPLATE`: Prompt for chairman to evaluate debate and synthesize best answer
  - `get_critique_pairs()`: Rotation system pairing critics with targets (Model 0→1, 1→2, etc.)
  - `format_debate_transcript()`: Formats complete debate for chairman evaluation
  - `run_debate_streaming()`: Full streaming debate orchestration with all rounds
  - `get_debate_config()`: Returns debate configuration for API
- Backend `main.py` updates:
  - `use_debate` parameter in SendMessageRequest (boolean, default false)
  - `include_rebuttal` parameter in SendMessageRequest (boolean, default true)
  - Debate endpoint: `GET /api/debate/config`
  - SSE event handling for all debate events in event_generator
  - Debate mode is an alternative flow that bypasses normal 3-stage council process
- Frontend `DebateView.jsx` component:
  - `DebateView`: Main component showing complete debate flow
    - Collapsible round sections with progress indicators
    - Position cards showing each model's initial answer (Round 1)
    - Critique flows showing critic → target relationships (Round 2)
    - Rebuttal cards showing model defenses (Round 3, optional)
    - Judgment section with streaming support (Final round)
    - Progress bar showing debate round completion
    - Real-time streaming support during active debate
  - `DebateBadge`: Badge showing number of rounds
    - Displays "3 Rounds" or "2 Rounds" based on configuration
    - Orange/amber color theme
  - `DebateToggle`: Toggle control for enabling/disabling debate mode
    - Includes sub-toggle for including rebuttal round
    - Orange/amber theme (#d97706, #fbbf24)
- Frontend `DebateView.css`: Orange/amber theme styling
  - Primary colors: #fbbf24, #d97706, #92400e, #f59e0b
  - Positions: Light amber (#fffbeb, #fef3c7)
  - Critiques: Light orange (#fff7ed, #ffedd5)
  - Rebuttals: Light green (#f0fdf4, #dcfce7)
  - Judgment: Yellow gradient (#fef3c7 → #fcd34d)
  - Progress indicators with pulse animations
- Frontend `api.js` updates:
  - `getDebateConfig()`: Get debate configuration
  - `sendMessageStream()` accepts `useDebate` and `includeRebuttal` parameters
- Frontend `App.jsx` updates:
  - `useDebate` and `includeRebuttal` state (persisted to localStorage)
  - `handleDebateChange()` and `handleIncludeRebuttalChange()` handlers
  - SSE event handlers for all debate events (15 event types)
  - Message state includes: `useDebate`, `debatePositions`, `debateCritiques`, `debateRebuttals`, `debateJudgment`, `debateJudgmentStreaming`, `isDebating`, `isJudging`, `debateRound`, `debateModelToLabel`, `debateLabelToModel`, `debateNumRounds`
  - "DB" indicator badge in settings bar when enabled
  - Debate Mode toggle in settings panel with orange/amber theme (#d97706)
- Frontend `App.css` updates:
  - `.debate-section` and `.debate-toggle-wrapper` styles
  - `.settings-debate-indicator` badge style (orange/amber theme)
- Frontend `Stage3.jsx` updates:
  - Imports DebateView and DebateBadge
  - Accepts all debate props
  - Shows "Council Debate" title instead of "Stage 3: Final Council Answer" when debate mode
  - Shows DebateBadge in header with round count
  - Shows "Round N..." and "Judging..." streaming indicators
  - Renders DebateView component with full debate transcript
- Streaming events:
  - `debate_start`: Debate beginning (includes models, num_rounds, include_rebuttal, model_to_label, label_to_model)
  - `round1_start`: Position round beginning
  - `position_complete`: Model finished stating position (includes label, model, position)
  - `round1_complete`: All positions collected (includes positions array)
  - `round2_start`: Critique round beginning
  - `debate_critique_complete`: Model finished critiquing (includes critic_label, target_label, critique)
  - `round2_complete`: All critiques collected (includes critiques array)
  - `round3_start`: Rebuttal round beginning (only if include_rebuttal=True)
  - `rebuttal_complete`: Model finished rebuttal (includes label, model, rebuttal)
  - `round3_complete`: All rebuttals collected (includes rebuttals array)
  - `judgment_start`: Chairman evaluation beginning (includes model)
  - `judgment_token`: Token from chairman during judgment
  - `judgment_complete`: Judgment finished (includes judgment, model, cost)
  - `debate_complete`: Full debate process complete (includes all data)
- **Alternative Flow**: Debate mode completely bypasses normal 3-stage council process
- Anonymous labels: Models are labeled as "Position A", "Position B", etc.
- Critique pairing uses rotation: Model 0 critiques Model 1, Model 1 critiques Model 2, etc.
- Include rebuttal toggle allows skipping Round 3 for faster debates

---

#### Step 20: Feature #16 - Sub-Question Decomposition ✅ COMPLETE
| Attribute | Value |
|-----------|-------|
| **Complexity** | Hard |
| **Effort** | 10-14 hours |
| **Why Twentieth** | Complex map-reduce pattern, benefits from all prior experience |
| **Files** | `backend/decompose.py` (new), `backend/main.py`, `frontend/src/components/DecomposedView.jsx` (new), `frontend/src/components/DecomposedView.css` (new), `frontend/src/api.js`, `frontend/src/App.jsx`, `frontend/src/App.css`, `frontend/src/components/Stage3.jsx`, `frontend/src/components/Stage3.css`, `frontend/src/components/ChatInterface.jsx` |
| **Packages** | None |

**What you'll build:** Complex questions split into sub-questions, each answered by separate councils, then merged.

**Implementation Notes:**
- Backend `decompose.py` module (new) with complete decomposition orchestration:
  - `DECOMPOSER_MODEL`: Fast model for complexity detection (Gemini Flash)
  - `DEFAULT_COMPLEXITY_THRESHOLD`: 0.6 (60% confidence required)
  - `DEFAULT_MAX_SUB_QUESTIONS`: 5 (maximum sub-questions to generate)
  - `COMPLEXITY_KEYWORDS`: Multi-part indicators ("compare and contrast", "step by step", etc.)
  - `assess_complexity()`: LLM + keyword heuristics determine if question is complex
  - `generate_sub_questions()`: Chairman generates 2-5 focused sub-questions
  - `run_mini_council()`: Parallel query all council models for a single sub-question
  - `select_best_answer()`: Pick highest confidence answer from mini-council results
  - `merge_sub_answers()`: Chairman synthesizes all best sub-answers into final response
  - `run_decomposition_streaming()`: Full streaming decomposition orchestration
  - `get_decomposition_config()`: Get configuration for API endpoint
- Backend `main.py` updates:
  - `use_decomposition` parameter on `/message/stream` endpoint
  - Decomposition processing before normal council flow
  - Fallthrough behavior when question not complex enough
- Frontend `DecomposedView.jsx` component (new):
  - `DecomposedView`: Main visualization with collapsible sections
    - Sub-questions section with status indicators (processing, answered)
    - Sub-answers section showing best response and model for each
    - Progress bar during sub-council processing
    - Merge section with streaming support
    - Skip state display when question not complex enough
  - `DecompositionBadge`: Small badge for Stage 3 header (sub-question count)
  - `DecompositionToggle`: Toggle control for settings panel
- Frontend `DecomposedView.css` styling (new):
  - Teal/cyan color scheme (#22d3d1, #14b8a6, #0891b2, #0e7490)
  - Progress bar with gradient fill
  - Sub-question cards with active/answered states
  - Sub-answer cards with model attribution
  - Merge section with streaming cursor
  - Skip notice styling
- Frontend `api.js` updates:
  - `getDecompositionConfig()`: Get decomposition configuration
  - `sendMessageStream()` accepts `useDecomposition` parameter
- Frontend `App.jsx` updates:
  - `useDecomposition` state (persisted to localStorage)
  - `handleDecompositionChange()` handler
  - SSE event handlers for all decomposition events (13 event types)
  - Message state includes: `useDecomposition`, `isDecomposing`, `decompositionSkipped`, `complexityInfo`, `subQuestions`, `subResults`, `currentSubQuestion`, `totalSubQuestions`, `mergeStreaming`, `isMerging`, `decompositionComplete`
  - "DQ" indicator badge in settings bar when enabled
  - Sub-Question Decomposition toggle in settings panel with teal/cyan theme (#14b8a6)
- Frontend `App.css` updates:
  - `.decomposition-section` and `.decomposition-toggle-wrapper` styles
  - `.settings-decomposition-indicator` badge style (teal/cyan theme)
- Frontend `Stage3.jsx` updates:
  - Imports DecomposedView and DecompositionBadge
  - Accepts 13 decomposition props
  - Shows "Stage 3: Final Council Answer" title with DecompositionBadge
  - Shows "Processing N/M..." and "Merging..." streaming indicators
  - Renders DecomposedView component with full decomposition visualization
  - Skip mode displays "Skipped" state and falls through to normal council
- Frontend `Stage3.css` updates:
  - `.stage3.decomposition-mode` background gradient
  - `.streaming-indicator.decomposition` and `.streaming-indicator.decomposition.merging` styles
- Streaming events:
  - `decomposition_start`: Decomposition beginning
  - `complexity_analyzed`: Complexity detection complete (includes is_complex, confidence, reasoning)
  - `decomposition_skip`: Question not complex, falling through to normal council
  - `sub_questions_generated`: Sub-questions created (includes sub_questions array, count)
  - `sub_council_start`: Beginning to process a sub-question (includes index)
  - `sub_council_response`: A model responded to sub-question (includes index, model, response)
  - `sub_council_complete`: Sub-question fully answered (includes index, sub_question, best_answer, best_model)
  - `all_sub_councils_complete`: All sub-questions answered (includes results array)
  - `merge_start`: Chairman beginning merge (includes chairman_model)
  - `merge_token`: Token from chairman during merge (includes content)
  - `merge_complete`: Merge finished (includes response)
  - `decomposition_complete`: Full decomposition complete (includes all data)
- **Alternative Flow**: Decomposition mode replaces normal 3-stage council for complex questions
- **Fallthrough Behavior**: If question not complex enough, shows "Skipped" and falls through to normal council
- Complexity detection uses both LLM classification and keyword heuristics
- Mini-councils query all council models for each sub-question
- Best answer selection uses highest confidence from mini-council results

---

#### Step 21: Feature #19 - Response Caching Layer ✅ COMPLETED
| Attribute | Value |
|-----------|-------|
| **Complexity** | Hard |
| **Effort** | 12-16 hours |
| **Why Last** | Requires new packages, benefits from #18, production optimization |
| **Files** | `backend/cache.py` (new), `backend/embeddings.py` (new), `backend/main.py`, `frontend/src/api.js`, `frontend/src/App.jsx` |
| **Packages** | Uses OpenAI embeddings API via OpenRouter (no additional packages needed beyond existing httpx) |

**What you'll build:** Semantic cache that returns similar past answers without re-querying models.

**Implementation Notes:**
- Created `backend/embeddings.py` for text embedding generation using OpenAI's text-embedding-3-small via OpenRouter API
- Created `backend/cache.py` for semantic caching with similarity search (default threshold 92%)
- Hash-based fallback when API embeddings unavailable
- System prompt included in cache key for isolation
- Statistics tracking (hits, misses, cost/time saved)
- Cache stored in `data/cache/semantic_cache.json`, stats in `data/cache/cache_stats.json`
- Added 8 cache endpoints + 1 embeddings endpoint to `backend/main.py`
- Added SSE events: `cache_check_start`, `cache_hit`, `cache_miss`, `cache_stored`
- Added cache toggle to frontend with green theme, "CA" indicator badge
- Full documentation in CLAUDE.md

---

### Progress Tracker

Use this checklist to track your progress:

```
SEQUENTIAL DEVELOPMENT PROGRESS
═══════════════════════════════

Phase 1: Foundation
  [x] Step 1:  #3  Custom System Prompts      DONE ✅
  [x] Step 2:  #2  Conversation Export        DONE ✅
  [x] Step 3:  #6  Question Categories        DONE ✅

Phase 2: Core Enhancements
  [x] Step 4:  #5  Model Configuration UI     DONE ✅
  [x] Step 5:  #7  Reasoning Model Support    DONE ✅
  [x] Step 6:  #8  Cost Tracking              DONE ✅
  [x] Step 7:  #10 Confidence Voting          DONE ✅

Phase 3: Real-Time & Observability
  [x] Step 8:  #4  Streaming Responses        DONE ✅
  [x] Step 9:  #1  Performance Dashboard      DONE ✅
  [x] Step 10: #11 Live Process Monitor       DONE ✅

Phase 4: Reasoning Patterns
  [x] Step 11: #20 Chain-of-Thought           DONE ✅
  [x] Step 12: #21 Multi-Chairman Synthesis   DONE ✅

Phase 5: Consensus Optimization
  [x] Step 13: #13 Weighted Consensus         DONE ✅
  [x] Step 14: #14 Early Consensus Exit       DONE ✅

Phase 6: Intelligent Routing
  [x] Step 15: #12 Dynamic Model Routing      DONE ✅
  [x] Step 16: #18 Conf-Gated Escalation      DONE ✅

Phase 7: Advanced Orchestration
  [x] Step 17: #15 Iterative Refinement       DONE ✅
  [x] Step 18: #17 Adversarial Validation     DONE ✅
  [x] Step 19: #9  Debate Mode                DONE ✅

Phase 8: Complex Patterns
  [x] Step 20: #16 Sub-Question Decomposition DONE ✅
  [x] Step 21: #19 Response Caching           DONE ✅

─────────────────────────────────────────────────
ALL 21 FEATURES COMPLETE! 🎉
```

---

### Milestone Summary

| Milestone | After Step | Features Done | Hours Invested |
|-----------|------------|---------------|----------------|
| **Foundation Complete** | 3 | 3 features | 8-14 hrs |
| **Core Complete** | 7 | 7 features | 28-42 hrs |
| **Observability Complete** | 10 | 10 features | 50-72 hrs |
| **Reasoning Complete** | 12 | 12 features | 62-88 hrs |
| **Consensus Complete** | 14 | 14 features | 67-97 hrs |
| **Routing Complete** | 16 | 16 features | 83-119 hrs |
| **Advanced Complete** | 19 | 19 features | 105-149 hrs |
| **ALL COMPLETE** | 21 | 21 features | 127-179 hrs |

---

## Quick Reference: What Can I Build Now?

### Start Anywhere (11 Fully Independent Features)

These features have **zero dependencies**. Pick any based on your interest:

| # | Feature | Complexity | Effort | Status |
|---|---------|------------|--------|--------|
| 2 | Conversation Export | Easy | 2-4 hrs | ✅ DONE |
| 3 | Custom System Prompts | Easy | 2-4 hrs | ✅ DONE |
| 4 | Streaming Responses | Hard | 8-12 hrs | ✅ DONE |
| 5 | Model Configuration UI | Medium | 6-8 hrs | ✅ DONE |
| 6 | Question Categories | Medium | 4-6 hrs | ✅ DONE |
| 7 | Reasoning Model Support | Medium | 4-6 hrs | ✅ DONE |
| 8 | Cost Tracking | Medium | 6-8 hrs | ✅ DONE |
| 10 | Confidence Voting | Medium | 4-6 hrs | ✅ DONE |
| 16 | Sub-Question Decomposition | Hard | 10-14 hrs | |
| 20 | Chain-of-Thought | Medium | 6-8 hrs | ✅ DONE |
| 21 | Multi-Chairman | Medium | 6-8 hrs | ✅ DONE |

### Features with Soft Dependencies (8 Features)

These **work without prerequisites** but are slightly better with them:

| # | Feature | Complexity | Works Better After | Status |
|---|---------|------------|-------------------|--------|
| 1 | Performance Dashboard | Medium | #5 (Model Config) | ✅ DONE |
| 9 | Debate Mode | Hard | #4 (Streaming) | ✅ DONE |
| 11 | Live Process Monitor | Medium-Hard | #4 (Streaming) | ✅ DONE |
| 13 | Weighted Consensus | Easy | #1 (Dashboard) | ✅ DONE |
| 14 | Early Consensus Exit | Easy | #10 (Confidence) | ✅ DONE |
| 15 | Iterative Refinement | Medium | #20 (Chain-of-Thought) | ✅ DONE |
| 17 | Adversarial Validation | Medium | #20 (Chain-of-Thought) | ✅ DONE |
| 19 | Response Caching | Hard | #18 (Escalation) | ✅ PREREQ DONE |

### Features with Hard Dependencies (2 Features)

**MUST complete prerequisites first:**

| # | Feature | REQUIRES | Why | Status |
|---|---------|----------|-----|--------|
| 12 | Dynamic Model Routing | **#3 first** | Uses system prompt patterns for classification | ✅ DONE |
| 18 | Confidence-Gated Escalation | **#8 AND #10 first** | Needs cost data + confidence scores to decide | ✅ DONE |

---

## Hard Dependencies (Must Respect)

```
┌─────────────────────────────────────────────────────────────────┐
│                    HARD DEPENDENCY RULES                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   #3 Custom System Prompts                                      │
│           │                                                     │
│           ▼                                                     │
│   #12 Dynamic Model Routing                                     │
│                                                                 │
│   ─────────────────────────────────────────────────────────    │
│                                                                 │
│   #8 Cost Tracking  ────┐                                       │
│                         ├──▶  #18 Confidence-Gated Escalation   │
│   #10 Confidence Voting ┘                                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**That's it.** These are the only ordering constraints in the entire project.

---

## Feature Independence Map

```
FULLY INDEPENDENT - Build in any order:
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│ #2  │ #3  │ #4  │ #5  │ #6  │ #7  │ #8  │ #10 │ #16 │ #20 │ #21 │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘

SOFT DEPENDENCIES - Build anytime, but slightly better after prereq:
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│ #1  │ #9  │ #11 │ #13 │ #14 │ #15 │ #17 │ #19 │
│     │     │     │     │     │     │     │     │
│ ↑   │ ↑   │ ↑   │ ↑   │ ↑   │ ↑   │ ↑   │ ↑   │
│ #5  │ #4  │ #4  │ #1  │ #10 │ #20 │ #20 │ #18 │
│soft │soft │soft │soft │soft │soft │soft │soft │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘

HARD DEPENDENCIES - Must do prereq first:
┌─────┬─────┐
│ #12 │ #18 │
│     │     │
│ ▲   │ ▲   │
│ #3  │#8+10│
│HARD │HARD │
└─────┴─────┘
```

---

## All Features by Complexity

### Easy (4 features) - 2-5 hours each

| # | Feature | Dependencies | Effort |
|---|---------|--------------|--------|
| 2 | Conversation Export | None | 2-4 hrs |
| 3 | Custom System Prompts | None | 2-4 hrs |
| 13 | Weighted Consensus | Soft: #1 | 3-5 hrs |
| 14 | Early Consensus Exit | Soft: #10 | 2-4 hrs |

**Recommended first feature:** #3 (teaches prompt engineering, unlocks #12 later)

---

### Medium (10 features) - 4-8 hours each

| # | Feature | Dependencies | Effort |
|---|---------|--------------|--------|
| 1 | Performance Dashboard | Soft: #5 | 6-8 hrs |
| 5 | Model Configuration UI | None | 6-8 hrs |
| 6 | Question Categories | None | 4-6 hrs |
| 7 | Reasoning Model Support | None | 4-6 hrs |
| 8 | Cost Tracking | None | 6-8 hrs |
| 10 | Confidence Voting | None | 4-6 hrs |
| 15 | Iterative Refinement | Soft: #20 | 6-8 hrs | ✅ DONE |
| 17 | Adversarial Validation | Soft: #20 | 6-8 hrs | ✅ DONE |
| 20 | Chain-of-Thought | None | 6-8 hrs |
| 21 | Multi-Chairman | None | 6-8 hrs |

---

### Medium-Hard (1 feature) - 8-10 hours

| # | Feature | Dependencies | Effort |
|---|---------|--------------|--------|
| 11 | Live Process Monitor | Soft: #4 | 8-10 hrs |

---

### Hard (6 features) - 8-16 hours each

| # | Feature | Dependencies | Effort | Status |
|---|---------|--------------|--------|--------|
| 4 | Streaming Responses | None | 8-12 hrs | ✅ DONE |
| 9 | Debate Mode | Soft: #4 | 10-14 hrs | |
| 12 | Dynamic Model Routing | **HARD: #3** | 10-14 hrs | ✅ DONE |
| 16 | Sub-Question Decomposition | None | 10-14 hrs | |
| 18 | Confidence-Gated Escalation | **HARD: #8 + #10** | 6-8 hrs | ✅ DONE |
| 19 | Response Caching | Soft: #18 | 12-16 hrs | |

---

## Suggested Tracks (Optional Paths)

You don't have to follow these, but they provide logical groupings if you want guidance.

### Track A: "Quick Wins First"
**Goal**: Build confidence, see immediate results

```
#2 Conversation Export (2-4 hrs)
    ↓
#3 Custom System Prompts (2-4 hrs)
    ↓
#6 Question Categories (4-6 hrs)
    ↓
#14 Early Consensus Exit (2-4 hrs)
    ↓
#13 Weighted Consensus (3-5 hrs)

Total: 13-23 hours for 5 features
```

---

### Track B: "User Experience Focus"
**Goal**: Make the app pleasant to use

```
#5 Model Configuration UI (6-8 hrs)
    ↓
#7 Reasoning Model Support (4-6 hrs)
    ↓
#2 Conversation Export (2-4 hrs)
    ↓
#6 Question Categories (4-6 hrs)
    ↓
#4 Streaming Responses (8-12 hrs)

Total: 24-36 hours for 5 features
```

---

### Track C: "Observability & Debugging"
**Goal**: See what's happening inside the system

```
#8 Cost Tracking (6-8 hrs)
    ↓
#4 Streaming Responses (8-12 hrs)
    ↓
#11 Live Process Monitor (8-10 hrs)
    ↓
#1 Performance Dashboard (6-8 hrs)

Total: 28-38 hours for 4 features
```

---

### Track D: "Orchestration Mastery"
**Goal**: Learn advanced multi-model coordination

```
#3 Custom System Prompts (2-4 hrs)      ← Unlocks #12
    ↓
#10 Confidence Voting (4-6 hrs)         ← Unlocks #18
    ↓
#8 Cost Tracking (6-8 hrs)              ← Unlocks #18
    ↓
#20 Chain-of-Thought (6-8 hrs)
    ↓
#12 Dynamic Model Routing (10-14 hrs)
    ↓
#18 Confidence-Gated Escalation (6-8 hrs)
    ↓
#17 Adversarial Validation (6-8 hrs)
    ↓
#15 Iterative Refinement (6-8 hrs)

Total: 48-64 hours for 8 features
```

---

### Track E: "Advanced Patterns"
**Goal**: Implement sophisticated AI orchestration

```
#20 Chain-of-Thought (6-8 hrs)
    ↓
#21 Multi-Chairman (6-8 hrs)
    ↓
#16 Sub-Question Decomposition (10-14 hrs)
    ↓
#9 Debate Mode (10-14 hrs)
    ↓
#19 Response Caching (12-16 hrs)

Total: 44-60 hours for 5 features
```

---

### Track F: "Complete Implementation"
**Goal**: Build everything in optimal order

```
Phase 1 - Foundation (any order within phase):
├── #2 Conversation Export
├── #3 Custom System Prompts        ← Do early (unlocks #12)
└── #6 Question Categories

Phase 2 - Core Enhancements (any order):
├── #5 Model Configuration UI
├── #7 Reasoning Model Support
├── #8 Cost Tracking               ← Do early (unlocks #18)
└── #10 Confidence Voting          ← Do early (unlocks #18)

Phase 3 - Observability (any order):
├── #4 Streaming Responses
├── #11 Live Process Monitor
└── #1 Performance Dashboard

Phase 4 - Basic Orchestration (any order):
├── #13 Weighted Consensus
├── #14 Early Consensus Exit
├── #20 Chain-of-Thought
└── #21 Multi-Chairman

Phase 5 - Advanced Orchestration (respect dependencies):
├── #12 Dynamic Model Routing      ← Requires #3
├── #18 Confidence-Gated Escalation ← Requires #8 + #10
├── #15 Iterative Refinement
├── #17 Adversarial Validation
├── #9 Debate Mode
└── #16 Sub-Question Decomposition

Phase 6 - Production:
└── #19 Response Caching

Total: 127-179 hours for all 21 features
```

---

## Complete Feature Reference

### Feature #1: Performance Dashboard ✅ COMPLETED

| Attribute | Value |
|-----------|-------|
| **Complexity** | Medium |
| **Effort** | 6-8 hours |
| **Status** | ✅ **IMPLEMENTED** |
| **Dependencies** | Soft: #5 (Model Config UI) |
| **Unlocks** | Soft: #13 (Weighted Consensus) |

**New Files:**
```
backend/analytics.py
frontend/src/components/PerformanceDashboard.jsx
frontend/src/components/PerformanceDashboard.css
```

**Modified Files:**
```
backend/main.py          # Add analytics endpoints
backend/council.py       # Record analytics after queries
frontend/src/api.js      # Add analytics API calls
frontend/src/App.jsx     # Add dashboard button and modal
frontend/src/App.css     # Add dashboard button styling
```

**Packages:** None (custom CSS visualizations instead of recharts)

**What was built:**
- `analytics.py`: Storage and aggregation module for performance data
  - `record_query_result()`: Records model performances, ranks, costs
  - `get_model_statistics()`: Calculates win rates, averages, distributions
  - `get_recent_queries()`: Returns recent query records
  - `get_chairman_statistics()`: Returns chairman usage stats
  - Data stored in `data/analytics/model_stats.json`
- API endpoints: `/api/analytics`, `/api/analytics/recent`, `/api/analytics/chairman`, `DELETE /api/analytics`
- PerformanceDashboard component with Leaderboard, Model Details, and Chairman Stats tabs
- "Dashboard" button in settings bar with purple/indigo color scheme

---

### Feature #2: Conversation Export

| Attribute | Value |
|-----------|-------|
| **Complexity** | Easy |
| **Effort** | 2-4 hours |
| **Dependencies** | None |
| **Unlocks** | None |

**New Files:**
```
frontend/src/utils/export.js
```

**Modified Files:**
```
frontend/src/components/ChatInterface.jsx  # Add export button
```

**Packages:** None

**Steps:**
1. Create `generateMarkdown()` function
2. Create `downloadFile()` using Blob API
3. Add export button to UI
4. Test with various conversations

---

### Feature #3: Custom System Prompts

| Attribute | Value |
|-----------|-------|
| **Complexity** | Easy |
| **Effort** | 2-4 hours |
| **Dependencies** | None |
| **Unlocks** | **HARD: #12** (Dynamic Model Routing) |

**New Files:**
```
frontend/src/components/Settings.jsx  # Optional
```

**Modified Files:**
```
backend/council.py       # Add system_prompt parameter
backend/main.py          # Accept in request
frontend/src/api.js      # Pass to API
frontend/src/App.jsx     # Add input UI
```

**Packages:** None

**Steps:**
1. Add `system_prompt` param to `stage1_collect_responses()`
2. Update `MessageRequest` model
3. Add textarea in frontend
4. Persist in localStorage
5. Test with various personas

---

### Feature #4: Streaming Responses ✅ COMPLETED

| Attribute | Value |
|-----------|-------|
| **Complexity** | Hard |
| **Effort** | 8-12 hours |
| **Status** | ✅ **IMPLEMENTED** |
| **Dependencies** | None |
| **Unlocks** | Soft: #9, #11 |

**New Files:** None (refactoring existing)

**Modified Files:**
```
backend/openrouter.py    # Add streaming query functions
backend/council.py       # Add streaming stage functions
backend/main.py          # Update stream endpoint for token events
frontend/src/api.js      # Handle token events (already supports SSE)
frontend/src/App.jsx     # Handle token events, track streaming state
frontend/src/components/Stage1.jsx   # Display partial responses
frontend/src/components/Stage1.css   # Streaming visual indicators
frontend/src/components/Stage3.jsx   # Display partial chairman response
frontend/src/components/Stage3.css   # Streaming visual indicators
frontend/src/components/ChatInterface.jsx  # Pass streaming props to Stage components
```

**Packages:** None

**What was built:**
- `query_model_streaming()`: Single model streaming with SSE
- `query_models_parallel_streaming()`: Parallel streaming from multiple models
- `stage1_collect_responses_streaming()`: Streaming Stage 1 orchestration
- `stage3_synthesize_final_streaming()`: Streaming Stage 3 orchestration
- Frontend handles `stage1_token`, `stage1_reasoning_token`, `stage3_token` events
- Visual indicators: "Streaming..." badges, pulsing dots, blinking cursor

---

### Feature #5: Model Configuration UI

| Attribute | Value |
|-----------|-------|
| **Complexity** | Medium |
| **Effort** | 6-8 hours |
| **Dependencies** | None |
| **Unlocks** | Soft: #1, #13 |

**New Files:**
```
backend/config_api.py
frontend/src/components/ConfigPanel.jsx
frontend/src/components/ConfigPanel.css
```

**Modified Files:**
```
backend/main.py          # Add config endpoints
backend/council.py       # Use dynamic config
frontend/src/api.js      # Add config API
frontend/src/App.jsx     # Add config panel
```

**Packages:** None

**Steps:**
1. Create config load/save in `config_api.py`
2. Add GET/PUT config endpoints
3. Update council.py to use dynamic config
4. Build config panel UI
5. Add validation (min 2 models)

---

### Feature #6: Question Categories & Tagging

| Attribute | Value |
|-----------|-------|
| **Complexity** | Medium |
| **Effort** | 4-6 hours |
| **Dependencies** | None |
| **Unlocks** | None |

**New Files:**
```
frontend/src/components/TagEditor.jsx
frontend/src/components/TagEditor.css
```

**Modified Files:**
```
backend/storage.py       # Add tags to schema
backend/main.py          # Add tag endpoints
frontend/src/api.js      # Add tag API
frontend/src/components/Sidebar.jsx  # Add filter
```

**Packages:** None

**Steps:**
1. Update conversation schema with `tags` field
2. Add tag CRUD endpoints
3. Add filter by tag endpoint
4. Build tag editor component
5. Add filter dropdown to sidebar

---

### Feature #7: Reasoning Model Support

| Attribute | Value |
|-----------|-------|
| **Complexity** | Medium |
| **Effort** | 4-6 hours |
| **Dependencies** | None |
| **Unlocks** | None |

**New Files:**
```
frontend/src/components/ReasoningDisplay.jsx  # Optional
```

**Modified Files:**
```
backend/council.py       # Include reasoning_details
frontend/src/components/Stage1.jsx  # Show reasoning
frontend/src/components/Stage1.css  # Style reasoning
```

**Packages:** None

**Steps:**
1. Include `reasoning_details` in Stage 1 response
2. Add collapsible reasoning section
3. Style the reasoning display
4. Test with o1, DeepSeek-R1

---

### Feature #8: Cost Tracking

| Attribute | Value |
|-----------|-------|
| **Complexity** | Medium |
| **Effort** | 6-8 hours |
| **Dependencies** | None |
| **Unlocks** | **HARD: #18** (with #10) |

**New Files:**
```
backend/pricing.py
frontend/src/components/CostDisplay.jsx
frontend/src/components/CostDisplay.css
```

**Modified Files:**
```
backend/openrouter.py    # Capture usage data
backend/main.py          # Include cost in response
frontend/src/App.jsx     # Display cost
```

**Packages:** None

**Steps:**
1. Update `query_model()` to return usage
2. Create pricing lookup table
3. Calculate cost per query
4. Build cost display component
5. Show per-model breakdown

---

### Feature #9: Debate Mode

| Attribute | Value |
|-----------|-------|
| **Complexity** | Hard |
| **Effort** | 10-14 hours |
| **Dependencies** | Soft: #4 (Streaming) |
| **Unlocks** | None |

**New Files:**
```
backend/debate.py
frontend/src/components/DebateView.jsx
frontend/src/components/DebateView.css
```

**Modified Files:**
```
backend/main.py          # Add debate endpoint
frontend/src/api.js      # Add debate API
frontend/src/App.jsx     # Add debate toggle
```

**Packages:** None

**Steps:**
1. Design debate flow (position → critique → rebuttal)
2. Implement round orchestration
3. Add debate endpoint
4. Build debate view UI
5. Add mode toggle
6. Test multi-round debates

---

### Feature #10: Confidence Voting ✅ COMPLETED

| Attribute | Value |
|-----------|-------|
| **Complexity** | Medium |
| **Effort** | 4-6 hours |
| **Status** | ✅ **IMPLEMENTED** |
| **Dependencies** | None |
| **Unlocks** | Soft: #14, **HARD: #18** (with #8) |

**New Files:**
```
frontend/src/components/ConfidenceDisplay.jsx
frontend/src/components/ConfidenceDisplay.css
```

**Modified Files:**
```
backend/council.py                           # Add confidence prompts, parsing, aggregation
backend/main.py                              # Emit confidence metadata in streaming
frontend/src/components/Stage1.jsx           # Show confidence badges and summary
frontend/src/components/Stage1.css           # Stage header styling
frontend/src/components/ChatInterface.jsx    # Pass aggregateConfidence prop
frontend/src/App.jsx                         # Handle confidence metadata from streaming
```

**Packages:** None

**What was built:**
- Backend appends `CONFIDENCE_PROMPT_SUFFIX` to Stage 1 queries requesting 1-10 score
- `parse_confidence()` extracts score using multiple regex patterns for robustness
- `extract_response_without_confidence()` removes confidence line for clean display
- `calculate_aggregate_confidence()` computes average, min, max, count, distribution
- `ConfidenceBadge` component shows color-coded individual scores on model tabs
- `AggregateConfidenceSummary` component shows average in Stage 1 header
- Streaming endpoint includes `aggregate_confidence` in `stage1_complete` metadata
- Color scheme: green (9-10), teal (7-8), amber (5-6), orange (3-4), red (1-2)

---

### Feature #11: Live Process Monitor ✅ COMPLETED

| Attribute | Value |
|-----------|-------|
| **Complexity** | Medium-Hard |
| **Effort** | 8-10 hours |
| **Status** | ✅ **IMPLEMENTED** |
| **Dependencies** | Soft: #4 (Streaming) |
| **Unlocks** | None |

**New Files:**
```
backend/process_logger.py
frontend/src/components/ProcessMonitor.jsx
frontend/src/components/ProcessMonitor.css
```

**Modified Files:**
```
backend/main.py          # Accept verbosity, emit process events
frontend/src/api.js      # Pass verbosity parameter
frontend/src/App.jsx     # Integrate ProcessMonitor, handle process events
frontend/src/App.css     # Add process monitor button styling
```

**Packages:** None

**What was built:**
- `process_logger.py`: Process logging module with verbosity levels
  - `Verbosity` enum (0=SILENT, 1=BASIC, 2=STANDARD, 3=VERBOSE)
  - `EventCategory` class for color-coded event types (stage, model, info, success, warning, error, data)
  - `create_process_event()` function to create event dicts with timestamps
  - `should_emit()` function to filter events based on verbosity level
  - Pre-defined event generators for common operations (stage_start, stage_complete, model_query_start, model_query_complete, etc.)
- Streaming endpoint accepts `verbosity` parameter (0-3) and emits `process` events
- ProcessMonitor side panel component with:
  - Verbosity slider/knob (0-3) with labeled levels (Silent, Basic, Standard, Verbose)
  - Auto-scrolling event log with scroll position detection
  - Color-coded events by category
  - Timestamps for each event
  - Collapsible panel with toggle button
  - Event count badge when collapsed
  - "Scroll to latest" button when user scrolls up
- `VerbosityControl` standalone component for inline verbosity selection
- "Process" button in settings bar toggles monitor visibility (shows verbosity level when active)
- Verbosity preference persisted to localStorage
- Dark theme (#1a1a2e) to visually distinguish from main content area

---

### Feature #12: Dynamic Model Routing

| Attribute | Value |
|-----------|-------|
| **Complexity** | Hard |
| **Effort** | 10-14 hours |
| **Dependencies** | **HARD: #3** (Custom System Prompts) |
| **Unlocks** | None |

**New Files:**
```
backend/router.py
```

**Modified Files:**
```
backend/council.py       # Use router
backend/main.py          # Add routing option
```

**Packages:** None

**Steps:**
1. Define model pools per question type
2. Implement classifier prompt
3. Create routing logic
4. Integrate into council
5. Add enable/disable option
6. Test classification accuracy

---

### Feature #13: Weighted Consensus Voting

| Attribute | Value |
|-----------|-------|
| **Complexity** | Easy |
| **Effort** | 3-5 hours |
| **Dependencies** | Soft: #1 (Dashboard) |
| **Unlocks** | None |

**New Files:**
```
backend/weights.py
```

**Modified Files:**
```
backend/council.py       # Use weighted aggregation
backend/main.py          # Update weights after queries
```

**Packages:** None

**Steps:**
1. Create weight storage
2. Implement weighted calculation
3. Add weight updates after queries
4. Consider decay for old data

---

### Feature #14: Early Consensus Exit ✅ COMPLETED

| Attribute | Value |
|-----------|-------|
| **Complexity** | Easy |
| **Effort** | 2-4 hours |
| **Status** | ✅ **IMPLEMENTED** |
| **Dependencies** | Soft: #10 (Confidence) |
| **Unlocks** | None |

**New Files:** None

**Modified Files:**
```
backend/council.py                           # Add consensus detection constants and detect_consensus() function
backend/main.py                              # Add use_early_consensus parameter, consensus_detected event
frontend/src/api.js                          # Add useEarlyConsensus parameter to sendMessageStream
frontend/src/App.jsx                         # Add useEarlyConsensus state, toggle, SSE event handling
frontend/src/App.css                         # Early consensus toggle and indicator styling
frontend/src/components/Stage3.jsx           # Add consensus mode display with metrics
frontend/src/components/Stage3.css           # Consensus mode styling (cyan/teal theme)
frontend/src/components/ChatInterface.jsx    # Pass isConsensus and consensusInfo props to Stage3
```

**Packages:** None

**What was built:**
- Backend consensus detection with three criteria:
  - `CONSENSUS_MAX_AVG_RANK = 1.5`: Top model's average rank threshold
  - `CONSENSUS_MIN_AGREEMENT = 0.8`: 80% first-place vote agreement
  - `CONSENSUS_MIN_CONFIDENCE = 7`: Minimum average confidence
- `detect_consensus()` function returns detailed consensus info with metrics
- Streaming endpoint emits `consensus_detected` event with model, response, metrics
- `stage3_complete` includes `is_consensus` and `consensus_info` when consensus triggered
- API accepts `use_early_consensus` boolean parameter (default false)
- Stage3 consensus view displays:
  - Cyan/teal theme (#ecfeff background, #06b6d4 accents)
  - "Consensus" badge, green notice explaining Stage 3 skipped
  - Metrics grid: winning model, average rank, first-place votes, confidence
  - Winner's response displayed with "Winner:" label
- Early Consensus toggle in settings panel with cyan slider
- "EC" indicator badge in settings bar when enabled
- Setting persisted to localStorage (default false)
- Process monitor events for consensus detection

---

### Feature #15: Iterative Refinement Loop ✅ COMPLETED

| Attribute | Value |
|-----------|-------|
| **Complexity** | Medium |
| **Effort** | 6-8 hours |
| **Status** | ✅ **IMPLEMENTED** |
| **Dependencies** | Soft: #20 (Chain-of-Thought) |
| **Unlocks** | None |

**New Files:**
```
backend/refinement.py
frontend/src/components/RefinementView.jsx
frontend/src/components/RefinementView.css
```

**Modified Files:**
```
backend/main.py                              # Add refinement endpoints, SSE events
frontend/src/api.js                          # Add refinement API calls
frontend/src/App.jsx                         # Add refinement state, toggle, event handling
frontend/src/App.css                         # Refinement toggle styling
frontend/src/components/Stage3.jsx           # Integrate RefinementView component
frontend/src/components/Stage3.css           # Refining indicator styling
frontend/src/components/ChatInterface.jsx    # Pass refinement props, loading indicator
frontend/src/components/ChatInterface.css    # Refinement loading styles
```

**Packages:** None

**What was built:**
- `refinement.py`: Iterative refinement loop module
  - `is_substantive_critique()`: Filters non-actionable feedback
  - `count_substantive_critiques()`: Counts critiques warranting revision
  - `collect_critiques()`: Council models critique synthesis in parallel
  - `generate_revision()`: Chairman revises based on substantive critiques
  - `run_refinement_loop_streaming()`: Full streaming refinement orchestration
  - Convergence detection: Stops when fewer than 2 substantive critiques
- Refinement endpoint: `GET /api/refinement/config`
- Streaming events: `refinement_start`, `iteration_start`, `critiques_start`, `critique_complete`, `critiques_complete`, `revision_start`, `revision_token`, `revision_complete`, `iteration_complete`, `refinement_converged`, `refinement_complete`
- RefinementView.jsx component with collapsible iteration history
  - Critique cards (amber=substantive, gray=non-substantive)
  - Revision sections with purple gradient
  - Progress indicator during active refinement
- RefinementBadge: Shows iteration count and convergence status
- RefinementToggle: Enable/disable with max iterations dropdown (1-5)
- "IR" indicator badge in settings bar when enabled
- Purple/violet color theme (#7c3aed, #8b5cf6, #6d28d9)
- Settings persisted to localStorage (useRefinement, refinementMaxIterations)

---

### Feature #16: Parallel Sub-Question Decomposition

| Attribute | Value |
|-----------|-------|
| **Complexity** | Hard |
| **Effort** | 10-14 hours |
| **Dependencies** | None |
| **Unlocks** | None |

**New Files:**
```
backend/decompose.py
frontend/src/components/DecomposedView.jsx
```

**Modified Files:**
```
backend/main.py          # Add endpoint
frontend/src/api.js      # Add API
```

**Packages:** None

**Steps:**
1. Design decomposer prompt
2. Implement decomposition logic
3. Run parallel councils
4. Implement merge/synthesis
5. Build sub-question UI
6. Test with complex questions

---

### Feature #17: Adversarial Validation Stage ✅ COMPLETED

| Attribute | Value |
|-----------|-------|
| **Complexity** | Medium |
| **Effort** | 6-8 hours |
| **Status** | ✅ **IMPLEMENTED** |
| **Dependencies** | Soft: #20 (Chain-of-Thought) |
| **Unlocks** | None |

**New Files:**
```
backend/adversary.py
frontend/src/components/AdversaryReview.jsx
frontend/src/components/AdversaryReview.css
```

**Modified Files:**
```
backend/main.py                              # Add adversary parameter, endpoint, SSE events
frontend/src/api.js                          # Add useAdversary parameter, getAdversaryConfig
frontend/src/App.jsx                         # Add adversary state, toggle, event handling
frontend/src/App.css                         # Adversary toggle styling (red/rose theme)
frontend/src/components/Stage3.jsx           # Integrate AdversaryReview component
```

**Packages:** None

**What was built:**
- `adversary.py`: Adversarial validation module
  - `has_genuine_issues()`: Determines if critique contains real issues vs praise
  - `parse_severity()`: Extracts severity (critical, major, minor, none)
  - `run_adversarial_validation_streaming()`: Full streaming validation
  - Default adversary model: `google/gemini-2.0-flash-001`
  - Revision threshold: only critical and major trigger revision
- Streaming events: `adversary_start`, `adversary_token`, `adversary_complete`, `adversary_revision_start`, `adversary_revision_token`, `adversary_revision_complete`, `adversary_validation_complete`
- API endpoint: `GET /api/adversary/config`
- AdversaryReview.jsx with AdversaryReview, AdversaryBadge, AdversaryToggle components
- Red/rose color theme (#e11d48, #be123c, #f87171, #fda4af)
- "AV" indicator badge in settings bar when enabled
- Runs after Iterative Refinement (if enabled)
- Disabled when Early Consensus Exit triggers

---

### Feature #18: Confidence-Gated Escalation ✅ COMPLETED

| Attribute | Value |
|-----------|-------|
| **Complexity** | Medium |
| **Effort** | 6-8 hours |
| **Status** | ✅ **IMPLEMENTED** |
| **Dependencies** | **HARD: #8 + #10** |
| **Unlocks** | Soft: #19 |

**New Files:**
```
backend/escalation.py
frontend/src/components/TierIndicator.jsx
frontend/src/components/TierIndicator.css
```

**Modified Files:**
```
backend/config_api.py    # Add tier configurations and getters
backend/main.py          # Add escalation parameter, endpoints, SSE events
frontend/src/api.js      # Add useEscalation parameter, escalation APIs
frontend/src/App.jsx     # Add escalation state, toggle, event handlers
frontend/src/App.css     # Add escalation toggle styling
frontend/src/components/Stage1.jsx           # Show tier indicators
frontend/src/components/ChatInterface.jsx    # Pass escalation props, tier loading
frontend/src/components/ChatInterface.css    # Tier loading styles
```

**Packages:** None

**What was built:**
- `escalation.py`: Tiered model selection module
  - `TIER_INFO`: Display info for Tier 1 (Fast, blue) and Tier 2 (Premium, amber)
  - `should_escalate()`: Evaluates three escalation triggers:
    - Low average confidence (< 6.0)
    - Low minimum confidence (any model < 4)
    - Low first-place agreement (< 50%)
  - `merge_tier_results()`: Combines Tier 1 and Tier 2 results with tier markers
  - `get_tier_info()`: Returns tier configuration for display
- Tier configurations in `config_api.py`:
  - `tier1_models`: Gemini Flash, GPT-4.1-mini, Claude Haiku, DeepSeek
  - `tier2_models`: Claude Sonnet 4.5, GPT-5.1, Gemini 3 Pro, o3
  - Escalation thresholds: confidence (6.0), min_confidence (4), agreement (0.5)
- API endpoints: `GET /api/escalation/tiers`, `GET /api/escalation/thresholds`
- SSE events: `tier1_start`, `tier1_complete`, `escalation_triggered`, `tier2_start`
- TierIndicator components: TierIndicator, TierBadge, EscalationBanner, TierSummary
- Escalation toggle in settings panel with pink theme (#ec4899)
- "CG" indicator badge in settings bar when enabled
- Escalation disabled when Dynamic Routing is active (routing takes precedence)
- Tier badges on model tabs (T1/T2) showing which tier each model belongs to

---

### Feature #19: Response Caching Layer

| Attribute | Value |
|-----------|-------|
| **Complexity** | Hard |
| **Effort** | 12-16 hours |
| **Dependencies** | Soft: #18 (Escalation) |
| **Unlocks** | None |

**New Files:**
```
backend/cache.py
backend/embeddings.py  # Optional
```

**Modified Files:**
```
backend/main.py          # Check cache before council
```

**Packages:**
- Required: `numpy`
- Option A: OpenAI embeddings API
- Option B: `sentence-transformers`
- Option C: `chromadb` or `pinecone-client`

**Steps:**
1. Choose embedding approach
2. Implement embedding generation
3. Implement similarity search
4. Add cache storage
5. Integrate into request flow
6. Add cache invalidation
7. Tune similarity threshold

---

### Feature #20: Chain-of-Thought Orchestration ✅ COMPLETED

| Attribute | Value |
|-----------|-------|
| **Complexity** | Medium |
| **Effort** | 6-8 hours |
| **Status** | ✅ **IMPLEMENTED** |
| **Dependencies** | None |
| **Unlocks** | Soft: #15, #17 |

**New Files:**
```
frontend/src/components/ReasoningView.jsx
frontend/src/components/ReasoningView.css
```

**Modified Files:**
```
backend/council.py       # Add CoT prompts, parsing, stage functions accept use_cot
backend/main.py          # Accept use_cot parameter in endpoints
frontend/src/api.js      # Pass useCot to API
frontend/src/App.jsx     # Add CoT toggle, pass to sendMessageStream
frontend/src/App.css     # CoT toggle styling
frontend/src/components/Stage1.jsx  # Show ReasoningView for CoT responses
frontend/src/components/Stage1.css  # CoT badge styling
```

**Packages:** None

**What was built:**
- `COT_PROMPT_SUFFIX`: Prompts models to structure response as THINKING → ANALYSIS → CONCLUSION
- `parse_cot_response()`: Extracts structured sections with multiple regex pattern support
- `extract_response_without_cot()`: Returns just the conclusion when CoT is present
- Stage 1 and Stage 2 functions accept `use_cot` parameter
- Stage 2 CoT-aware ranking evaluates reasoning quality, analysis depth, conclusion accuracy
- API endpoints accept `use_cot` boolean parameter
- ReasoningView component displays three-step reasoning with step indicators
- Step colors: blue (thinking), purple (analysis), green (conclusion)
- Progress visualization in header showing completed reasoning steps
- Collapsible sections with conclusion expanded by default
- CoT toggle in settings panel with slider switch
- "CoT" indicator badge in settings bar when enabled
- CoT badges on model tabs indicating structured reasoning available
- Setting persisted to localStorage

---

### Feature #21: Multi-Chairman Synthesis ✅ COMPLETED

| Attribute | Value |
|-----------|-------|
| **Complexity** | Medium |
| **Effort** | 6-8 hours |
| **Status** | ✅ **IMPLEMENTED** |
| **Dependencies** | None |
| **Unlocks** | None |

**New Files:**
```
backend/multi_chairman.py
frontend/src/components/MultiSynthesis.jsx
frontend/src/components/MultiSynthesis.css
```

**Modified Files:**
```
backend/config_api.py    # Add multi_chairman_models config, getter, validation
backend/main.py          # Add use_multi_chairman parameter, streaming events
frontend/src/api.js      # Add useMultiChairman parameter
frontend/src/App.jsx     # Add multi-chairman toggle, event handling
frontend/src/App.css     # Multi-chairman toggle styling
frontend/src/components/Stage3.jsx      # Integrate MultiSynthesis component
frontend/src/components/Stage3.css      # Multi-chairman badge styling
frontend/src/components/ChatInterface.jsx  # Pass multi-chairman props
```

**Packages:** None

**What was built:**
- Backend `multi_chairman.py` module implementing two-phase synthesis:
  - `stage3_multi_chairman_synthesis()`: Queries all multi-chairman models in parallel
  - `stage3_supreme_chairman_selection()`: Supreme chairman evaluates and selects best synthesis
  - `stage3_multi_chairman_streaming()`: Streaming version with real-time events
  - `calculate_multi_chairman_costs()`: Cost aggregation across all syntheses
  - `get_supreme_chairman_model()`: Returns configured chairman as supreme chairman
- Configuration in `config_api.py`:
  - `multi_chairman_models` in DEFAULT_CONFIG (Gemini, Claude, GPT-4)
  - `get_multi_chairman_models()` getter function
  - Validation for multi_chairman_models (minimum 2 models)
  - Configuration persisted to `data/council_config.json`
- API accepts `use_multi_chairman` boolean parameter
- Streaming events:
  - `multi_synthesis_start`: Multi-chairman synthesis begins
  - `synthesis_complete`: A chairman finished synthesizing
  - `multi_synthesis_complete`: All chairmen finished
  - `selection_start`: Supreme chairman selection begins
  - `selection_token`: Token from supreme chairman
- MultiSynthesis.jsx component:
  - Tabbed view of all chairman syntheses (labeled A, B, C...)
  - Checkmark icon on selected synthesis tab
  - Active/selected tab highlighting with green accent
  - Supreme chairman selection section (amber/yellow theme)
  - Collapsible "Show Evaluation Details" with reasoning
  - Final response section with green border
- Multi-Chairman toggle in settings panel:
  - Green color scheme (#22c55e) matching Stage 3 theme
  - Slider switch with title and description
  - "MC" indicator badge in settings bar when enabled
- Stage3.jsx branches between single and multi-chairman modes
- Setting persisted to localStorage

---

## Package Requirements

### Backend (Python)

**Current dependencies** (already installed):
```
fastapi, uvicorn, httpx, python-dotenv, pydantic
```

**New packages needed:**

| Feature | Package | Required? |
|---------|---------|-----------|
| #19 Response Caching | `numpy` | Yes |
| #19 Response Caching | `sentence-transformers` | Option B |
| #19 Response Caching | `chromadb` | Option C |

**Installation:**
```bash
# For Feature #19 only
uv add numpy

# Optional (choose one)
uv add sentence-transformers  # Local embeddings
uv add chromadb               # Vector database
```

### Frontend (JavaScript)

**Current dependencies** (already installed):
```
react, react-dom, react-markdown
```

**New packages needed:**

| Feature | Package | Required? |
|---------|---------|-----------|
| #1 Performance Dashboard | `recharts` | Optional |

**Installation:**
```bash
# For Feature #1 only (optional)
cd frontend && npm install recharts
```

### Summary

| Features | Additional Packages |
|----------|-------------------|
| 1-18, 20-21 | **None** |
| #1 Dashboard | `recharts` (optional) |
| #19 Caching | `numpy` + embedding solution |

**20 out of 21 features require NO new packages.**

---

## Implementation Checklists

### Checklist Template

Use this format to track progress on any feature:

```
Feature #X: [Name]
─────────────────
[ ] Read feature description in FEATURE_IDEAS.md
[ ] Check dependencies are complete
[ ] Create new files
[ ] Modify existing files
[ ] Write/update tests
[ ] Manual testing
[ ] Update documentation (if needed)
```

### Quick Start Checklist (First Feature)

If you're starting with #3 Custom System Prompts:

```
Feature #3: Custom System Prompts
─────────────────────────────────
[ ] Backend: Add system_prompt param to stage1_collect_responses()
[ ] Backend: Update MessageRequest model in main.py
[ ] Backend: Pass system_prompt through to query
[ ] Frontend: Add textarea input for system prompt
[ ] Frontend: Save preference to localStorage
[ ] Test: Default behavior unchanged
[ ] Test: Custom prompt affects responses
[ ] Test: Persistence across page reload
```

### Dependency Unlock Checklist

Track which features you've unlocked:

```
Completed Prerequisites          Unlocked Features
─────────────────────           ─────────────────
[x] #3 Custom System Prompts    → [x] #12 Dynamic Routing DONE ✅
[x] #8 Cost Tracking            ┐
                                ├→ [x] #18 Conf-Gated Escalation DONE ✅
[x] #10 Confidence Voting       ┘

[x] #4 Streaming                → [x] #11 Live Process Monitor DONE ✅, [ ] #9 Debate Mode works optimally
[x] #5 Model Config UI          ┐
                                ├→ [x] #1 Performance Dashboard DONE ✅
[x] #1 Performance Dashboard    ┘→ [x] #13 Weighted Consensus DONE ✅
[x] #20 Chain-of-Thought        → [x] #15 Iterative Refinement DONE ✅, [x] #17 Adversarial Validation DONE ✅
[x] #18 Conf-Gated Escalation   → [ ] #19 Response Caching works better
```

---

## Total Effort Summary

| Complexity | Count | Total Effort |
|------------|-------|--------------|
| Easy | 4 | 9-17 hours |
| Medium | 10 | 54-74 hours |
| Medium-Hard | 1 | 8-10 hours |
| Hard | 6 | 56-78 hours |
| **Total** | **21** | **127-179 hours** |

---

## Key Takeaways

1. **90% of features are independent** - Pick what interests you
2. **Only 2 hard dependencies** - Just don't do #12 before #3, or #18 before #8+#10
3. **No new packages for 20/21 features** - Start building immediately
4. **Soft dependencies are optional** - Features work without them
5. **Follow tracks or go freestyle** - Both approaches work

**Recommended first feature:** #3 Custom System Prompts
- Easy (2-4 hours)
- Teaches prompt engineering
- Unlocks #12 later
- Immediate user value
