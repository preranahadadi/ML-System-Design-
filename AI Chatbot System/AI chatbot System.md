# System Design: Company Knowledge Base Chatbot

## 1. Clarify the Problem

Before designing anything, ask the follwoing questions:

- **What type of bot?** Customer support (task-oriented) or general assistant (open-domain)?
- **What domain?** E-commerce, healthcare, banking?
- **What knowledge?** Help articles, product catalog, internal policies?
- **Can it take actions?** Like checking order status, processing returns?
- **Scale?** Thousands or millions of daily users?
- **Voice or text?**

**Assume for this design:** A text-based customer support chatbot for an e-commerce company. It answers from a company knowledge base, handles multi-turn conversations, and can perform actions like checking orders.

---

## 2. Requirements

### Functional Requirements

1. Understand natural language questions and reply conversationally
2. Handle multi-turn conversations (remember what was said earlier)
3. Answer questions grounded in company knowledge base (help articles, product catalog, policies)
4. Perform actions on behalf of the user (check order status, initiate returns) via API calls
5. Escalate to a human agent when it can't resolve the issue
6. Route queries by type — FAQ, knowledge question, action request, chitchat, escalation
7. Block harmful or manipulative inputs (prompt injection, toxic messages)
8. Avoid hallucination — only answer based on retrieved evidence, say "I don't know" otherwise

### Non-Functional Requirements

1. **Latency:** First token within 1 second, full response within 3 seconds
2. **Availability:** 99.9% uptime
3. **Scalability:** 10K+ concurrent conversations, millions of daily messages
4. **Cost Efficiency:** Minimize compute cost per conversation
5. **Security:** PII redaction, GDPR compliance, conversation audit trail
6. **Extensibility:** Easy to add new documents, tools, or intents without retraining everything
7. **Observability:** Full logging, metric dashboards, tracing for debugging

---

## 3. The Core Idea — RAG (Retrieval Augmented Generation)

You don't train an LLM from scratch. You take an existing smart LLM and feed it your company's documents at query time so it can answer based on your data.

**Think of it like this:**

- LLM alone = A very smart new employee who knows nothing about your company
- LLM + RAG = That same smart employee, but you hand them the relevant documents before they answer every question

### How RAG Works (3 Phases)

**Phase 1 — Prepare your documents (one-time, offline):**
Take your docs (PDFs, help articles, product pages) → break them into small chunks (paragraphs, ~200-500 words) → convert each chunk into a numerical vector (embedding) → store all vectors in a vector database.

**Phase 2 — User asks a question (real-time):**
Convert the question into a vector → search the vector database for the most similar chunks → top 3-5 matching chunks come back.

**Phase 3 — Generate the answer (real-time):**
Take the retrieved chunks + the user's question → stuff them into a prompt → send to the LLM → LLM reads the context and generates a grounded answer.

### Why RAG Instead of Fine-Tuning?

- Docs change frequently. With RAG you just re-index. With fine-tuning you retrain the model.
- Fine-tuning is expensive and slow.
- Fine-tuned models still hallucinate. RAG gives the model actual source text.
- RAG lets you cite sources. Fine-tuning can't.
- **Best approach:** RAG for knowledge + good prompting (or light fine-tuning) for tone/behavior.

---

## 4. First Big Decision — Open Source vs Closed Source LLM

### Option A: Closed Source API (GPT-4, Claude)

- **Pros:** No GPU infra to manage, best quality, fast to ship (weeks)
- **Cons:** Data leaves your network (compliance issue), per-token cost at scale, rate limits, third-party dependency

### Option B: Open Source Self-Hosted (Llama, Mistral, Qwen)

- **Pros:** Data stays private, no per-token cost, full control, can fine-tune
- **Cons:** Need GPU infrastructure, quality gap (shrinking), you own maintenance

### Option C: Hybrid (Most Common in Production)

- Simple questions → small open source model (cheap, fast, private)
- Complex questions → closed API (better quality)
- A lightweight classifier routes between them

---

## 5. Model Choices for Each Component

You don't use one model for everything. Use cheap fast models for routing and safety, expensive smart models only for generation.

| Component | Model Choice | Why |
|---|---|---|
| **Embedding Model** (text → vectors) | BGE-large, E5-large, Cohere Embed, OpenAI ada | Converts text to vectors for search |
| **Reranker** (rescores retrieved chunks) | BGE-reranker, Cohere Rerank, fine-tuned BERT | Cross-encoder is more accurate than bi-encoder |
| **LLM** (generates the answer) | Closed: Claude, GPT-4. Open: Llama 70B, Mistral | The brain that reads context and answers |
| **Intent Router** (classifies query type) | Fine-tuned DistilBERT or rule-based | Fast, cheap — don't waste LLM calls on routing |
| **Guardrail Models** (safety checks) | Llama Guard, fine-tuned BERT classifiers | Safety checks must be fast |

---

## 6. Knowledge Base Ingestion Pipeline (Offline)

This is the one-time pipeline that makes your documents searchable.

```
Admin uploads documents (PDF, HTML, Docx, CSV)
    ↓
Document Parser — extract text, preserve structure
    ↓
Chunking — split into 256-512 token segments with overlap
    ↓
Embedding — run each chunk through embedding model → 768/1024-dim vector
    ↓
Store in TWO places:
  • Vector Database (Pinecone / Qdrant / pgvector) — for semantic search
  • Search Index (Elasticsearch) — for keyword/BM25 search
```

### Why Two Stores?

- **Vector search** catches semantic similarity ("how do I send something back" matches "return policy")
- **BM25 lexical search** catches exact keywords (product names, error codes, order IDs)
- Using both (hybrid search) gives the best retrieval

### Chunking Strategies (Critical — Bad Chunking = Bad Answers)

| Strategy | How It Works | Tradeoff |
|---|---|---|
| **Fixed size** | Every 512 tokens, 50 token overlap | Simple but can split mid-thought |
| **Semantic** | Split at paragraph/section boundaries | Respects structure, variable sizes |
| **Recursive** | Try section → paragraph → sentence | Best balance |
| **Parent-child** | Index small chunks for matching, return larger parent for context | Precise matching + complete context |

---

## 7. Retrieval Pipeline (Online, at Query Time)

```
User: "Can I return headphones after 30 days?"
    ↓
Query Rewriter (if multi-turn) — uses conversation history to add context
    ↓
Embed the query using same embedding model
    ↓
Parallel search:
  • Vector DB → top 20 chunks (semantic match)
  • Elasticsearch BM25 → top 20 chunks (keyword match)
    ↓
Merge results using Reciprocal Rank Fusion (RRF)
    ↓
Reranker (cross-encoder) rescores merged results → picks top 3-5
    ↓
These chunks become the "context" for the LLM
```

### Why Reranking?

- **Bi-encoder (retrieval):** Encodes query and chunk separately → fast but less accurate
- **Cross-encoder (reranker):** Encodes query AND chunk together → slow but much more accurate
- This is the classic **two-stage retrieval → ranking** pattern used across all ML system designs

---

## 8. Prompt Assembly & Generation

```
SYSTEM: You are a helpful support assistant for {company}.
Answer ONLY from the provided context.
If the context doesn't contain the answer, say "I don't have information about that."
Always cite which document your answer comes from.

CONTEXT:
[Document: Return Policy, Page 3] {chunk_1}
[Document: Electronics Warranty FAQ] {chunk_2}

CONVERSATION HISTORY:
User: I bought AirPods last month
Assistant: Sure, happy to help. What do you need?
User: Can I return them after 30 days?

Answer the user's latest question.
```

The LLM reads the context and generates a grounded, cited answer.

---

## 9. Tool Use / Function Calling

For action queries ("check my order", "process a return"), the LLM calls external APIs.

**Defined tools:**
- `check_order_status(order_id)` → returns order details
- `initiate_return(order_id, reason)` → starts a return
- `get_product_info(product_id)` → returns product details

**Flow:**
1. LLM decides to call a tool and outputs structured arguments
2. System validates arguments (type check, auth check)
3. For irreversible actions → ask user for confirmation first
4. Execute the API call → feed result back to LLM
5. LLM generates natural language response with the result

**Key concern:** Validate all tool arguments against real data. The LLM might hallucinate an order ID.

---

## 10. Session & Conversation Storage

### The Problem

LLMs are stateless. Every API call is independent. The model doesn't remember previous turns. YOU have to send the full conversation history every time.

### Architecture

**Hot Store — Redis:**
- Stores conversation history keyed by session ID
- Sub-millisecond reads (needed on every user message)
- TTL of 30 minutes for idle sessions
- Scales horizontally with Redis Cluster

**Cold Store — DynamoDB / PostgreSQL / S3:**
- On session expire or close, write full conversation to cold storage
- Used for analytics, training data, compliance audit

### Session Flow on Every Message

1. User sends message
2. Backend reads session from Redis using session_id
3. Appends new user message to the messages array
4. Sends full message history to the LLM as conversation context
5. Gets LLM response
6. Appends assistant response to messages array
7. Writes updated session back to Redis
8. Returns response to user

### Context Window Management (Long Conversations)

When conversations get long, they exceed the LLM's context window.

| Strategy | How It Works | Tradeoff |
|---|---|---|
| **Sliding Window** | Keep only last 10 turns | Simple but loses early context |
| **Summarize + Recent** | Summarize old turns, keep recent verbatim | Best balance — recommended |
| **Hierarchical Memory** | Recent turns verbatim + older turns summarized + key facts extracted | Most sophisticated, complex to build |

### What to Store Per Message

```json
{
  "role": "assistant",
  "content": "AirPods can be returned within 30 days...",
  "timestamp": "2026-03-26T10:30:00Z",
  "sources": ["return_policy.pdf#page3"],
  "model_used": "llama-70b",
  "latency_ms": 1200,
  "confidence": 0.87
}
```

Storing metadata enables debugging, analytics, and evaluation.

---

## 11. Guardrails

### Input Guardrails

| Check | What It Does | Action |
|---|---|---|
| **Toxicity Classifier** | Detects harmful/toxic input | Block + canned response |
| **Prompt Injection Detector** | Detects "ignore previous instructions..." attacks | Strip/block + alert |
| **PII Detector** | Finds personal info (SSN, credit cards) | Redact before logging |

### Output Guardrails

| Check | What It Does | Action |
|---|---|---|
| **Hallucination Detector** | Checks if response follows retrieved context (NLI model) | Regenerate or say "I'm not sure" |
| **Policy Compliance Filter** | Checks against business rules | Block + substitute safe response |
| **PII Filter** | Catches personal info in output | Redact before sending |

### Fallback Strategy

If guardrails flag an issue → don't just block. Provide a graceful fallback: "I'm not sure about that, let me connect you with a human agent." Always pass full conversation context to the human agent.

---

## 12. Full System Architecture

```
User → Load Balancer → API Gateway (auth, rate limiting)
    → Session Manager (Redis)
    → Orchestrator Service
        → Input Guardrails
        → Query Understanding (rewrite, intent routing)
        → [Knowledge Query] → RAG Pipeline (Vector DB + BM25 + Reranker)
        → [Action Query] → Tool Executor (API calls)
        → LLM Generation (GPU cluster or API)
        → Output Guardrails
        → [Escalation] → Human Handoff Service
    → Response to User

OFFLINE: Doc Upload → Chunker → Embedding Model → Vector DB + Elasticsearch
```

---

## 13. Metrics & Evaluation

### Why This Matters

Evaluating conversational AI is hard. Automated metrics are weak. Interviewers specifically test whether you know this and have a multi-layered strategy.

### Retrieval Metrics (Most Important — Bad Retrieval = Bad Answers)

| Metric | What It Measures | Target |
|---|---|---|
| **Recall@k** | Of chunks that SHOULD be retrieved, how many were? | > 0.85 at k=10 |
| **Precision@k** | Of chunks we retrieved, how many are relevant? | > 0.7 at k=5 |
| **MRR** | Where does the first relevant chunk appear? | > 0.7 |
| **NDCG** | Are the most relevant chunks ranked highest? | > 0.75 at k=5 |

### Generation Metrics

| Metric | What It Measures | Usefulness |
|---|---|---|
| **BLEU / ROUGE** | Word overlap with reference answer | LOW — mostly useless for chatbots |
| **BERTScore** | Semantic similarity to reference | MEDIUM — good for regression testing |
| **Groundedness** | Does the answer ONLY use info from context? | CRITICAL — the #1 metric |
| **Relevance** | Does it answer what the user asked? | HIGH |
| **Completeness** | Does it cover all relevant info? | HIGH |

### LLM-as-Judge (Most Important Evaluation Method)

Since automated metrics are weak, use a strong LLM to evaluate responses:

Feed it: the user query + retrieved context + generated response. Ask it to rate on 5 dimensions (groundedness, relevance, completeness, tone, safety) with scores 1-5 and justification.

Run this on a curated eval set of 500+ examples and on a daily random sample of production conversations.

### Online Metrics (Production)

| Metric | What It Measures | Target |
|---|---|---|
| **Task Completion Rate** | Did the user accomplish their goal? | > 80% (NORTH STAR) |
| **Containment Rate** | Resolved without human handoff? | > 70-85% |
| **CSAT** | User satisfaction (thumbs up/down, surveys) | > 4.0/5.0 |
| **Hallucination Rate** | % of responses with fabricated info (sampled) | < 2% |
| **Abandonment Rate** | User left without resolution | < 15% |

### Business Metrics

| Metric | What It Measures |
|---|---|
| **Cost per resolution** | Compute + human cost per resolved conversation |
| **Ticket deflection rate** | Conversations handled by bot that would've been human tickets |
| **Agent productivity** | When escalated, agent gets full context → faster resolution |

### Evaluation Pipeline

**Offline (before deployment):**
Run full pipeline on curated eval set → measure retrieval recall, groundedness, LLM-as-judge scores → compare against baseline → only deploy if all metrics ≥ baseline.

**Online (after deployment):**
A/B test every change → primary metric: task completion rate → guardrail: hallucination rate must not increase → shadow mode for risky changes.

**Continuous:**
Daily 1% conversation sample for LLM-as-judge audit → weekly human review of 200 conversations → drift detection dashboards → feedback loop from thumbs down → alert if hallucination rate spikes.

---

## 14. Reducing Hallucinations (Defense in Depth)

No single technique eliminates hallucination. You layer multiple strategies.

### Why Hallucination Happens

1. **Retrieval failure** — right document wasn't retrieved, LLM gets wrong context
2. **Context gap** — answer doesn't exist in knowledge base, LLM fills the gap
3. **Context confusion** — multiple chunks have conflicting info, LLM blends incorrectly
4. **Instruction drift** — in long conversations, LLM gradually ignores "only answer from context"
5. **Over-inference** — context says "return within 30 days", LLM infers "so after 30 days you can't" (might not be true)
6. **Training knowledge leakage** — LLM uses its pre-training knowledge instead of the provided context

### Layer 1: Better Retrieval (Most Impact)

**Hybrid search** — vector + BM25 together improves recall by 15-25%.

**Better chunking** — overlap between chunks so answers aren't split across boundaries. Parent-child chunking: index small chunks for precise matching, return larger parent for full context.

**Metadata filtering** — filter by product category, region, document recency before search. Prevents outdated or irrelevant docs from reaching the LLM.

**Reranking** — cross-encoder rescores retrieved chunks. Improves precision@3 by 20-30%.

**Query expansion / HyDE** — expand short queries or generate hypothetical answers to improve search.

### Layer 2: Better Prompting

**Explicit grounding instructions** — "Answer ONLY from the provided context. If the answer is not there, say I don't know. Do NOT use your training knowledge."

**Citation forcing** — force the LLM to cite source documents. Makes it trace its answer back to specific chunks, reducing fabrication.

**Chain-of-thought with quotes** — force the LLM to first quote the relevant text from context, then answer based on those quotes.

**Negative examples** — show the model examples of hallucination and the correct "I don't know" response.

### Layer 3: Better Generation

**Model selection** — larger models (70B+, GPT-4, Claude) hallucinate less than smaller ones due to better instruction following.

**Low temperature (0.1-0.3)** — reduces randomness. For a knowledge base chatbot you want accuracy, not creativity.

**Constrained generation** — for factual fields, extract structured data from context and template the response instead of free-form generation.

### Layer 4: Post-Generation Checks

**NLI-based groundedness check** — break the response into claims, run each through a Natural Language Inference model against the retrieved context. If a claim isn't entailed → hallucination detected.

**LLM-as-judge verification** — a second LLM checks whether the first LLM's answer is supported by the context. More expensive but more accurate.

**Self-consistency** — generate the answer 3 times. If they agree → likely correct. If they disagree → low confidence, escalate.

**Confidence scoring** — combine retrieval score + reranker score + NLI score into a composite confidence. Below threshold → add disclaimer or escalate to human.

### Layer 5: Training / Fine-Tuning

**Fine-tune to say "I don't know"** — include 20-30% "unanswerable" examples in training data.

**DPO against hallucination** — collect preference pairs where grounded answers are preferred over hallucinated ones. The model learns that "I don't know" > making something up.

### Layer 6: Monitoring & Feedback Loop

**Track daily:** hallucination rate, "I don't know" rate (too low = overconfident, too high = retrieval broken), top hallucinated topics.

**User feedback loop:** thumbs down → human review → confirmed hallucination → add to regression test set + fix knowledge gap.

**Knowledge gap detection:** cluster queries where retrieval confidence is low or bot says "I don't know" → these are missing topics → add documents → reindex.


## 15. Model Training — SFT, RLHF, DPO

### SFT (Supervised Fine-Tuning)

Take a pre-trained LLM and train it on curated (input, ideal output) pairs so it learns to follow instructions and respond in your company's style.

- A base LLM is just a next-token predictor. It doesn't know how to answer questions properly.
- SFT teaches it to be a helpful assistant by showing it thousands of good examples.
- Data quality > data quantity. 1,000 high-quality examples often beats 100,000 noisy ones.
- **LoRA/QLoRA** makes SFT practical — train small adapter layers instead of all parameters.

### Where SFT Fits in the Pipeline

1. **Pre-training** (trillions of tokens) → base model that predicts next tokens
2. **SFT** (thousands of curated examples) → model that follows instructions
3. **RLHF or DPO** (human preference data) → model that prefers safe, helpful, honest responses

### RLHF vs DPO

- **RLHF:** Collect human preferences (A vs B), train a reward model, optimize LLM with PPO. Complex but proven.
- **DPO:** Directly optimize LLM on preference pairs without a separate reward model. Simpler, increasingly preferred.

### When to Fine-Tune vs Prompt

- Start with prompting (cheaper, faster iteration)
- Fine-tune when: you need consistent style, domain-specific behavior, smaller model for cost, or at scale where per-token cost matters

---

## 16. Scaling & Cost

### Cost Per Conversation (Closed API)

| Component | Cost |
|---|---|
| Embedding query | ~$0.0001 |
| Vector DB search | ~$0.0001 |
| Reranker | ~$0.001 |
| LLM generation (5 turns) | ~$0.05-0.15 |
| **Total per conversation** | **~$0.05-0.15** |
| **At 1M conversations/day** | **~$50K-150K/month** |

### Cost Optimization

1. **Model routing** — simple questions → small model (90% of queries, 10x cheaper)
2. **Caching** — identical questions → cached response (saves ~20-30% of LLM calls)
3. **Shorter prompts** — summarize history instead of full conversation
4. **Self-hosting** — at scale, GPU infra cheaper than per-token API costs
5. **Quantization** — INT8/INT4 models use less memory, serve more per GPU

### When Self-Hosting Becomes Cheaper

At roughly 100K+ conversations/day, self-hosting on GPUs costs less than API pricing. Below that, APIs are more cost-effective when you factor in infrastructure management overhead.

---

## 17. Failure Modes & Mitigations

| Failure | Mitigation |
|---|---|
| Hallucination | RAG + NLI check + "I don't know" training |
| Prompt injection | Input classifier + system prompt isolation + output filter |
| Context overflow | Summarize older turns, sliding window |
| Retrieval miss | Hybrid search, chunk overlap, fallback to web search |
| Tool call errors | Retry logic, graceful error messages, human fallback |
| Stale knowledge base | Scheduled reindexing, webhook triggers on content updates |
| Adversarial users | Rate limiting, toxicity filtering, conversation termination |

---

## 18. Evolution Roadmap

**V1 (MVP):** Hosted LLM API + managed vector DB + basic RAG + simple intent router + human fallback. Focus on top 10 high-traffic intents.

**V2:** Fine-tuned smaller model for cost reduction. Add tool calling. Proper guardrail pipeline. A/B testing framework.

**V3:** Model routing (small/medium/large). Proactive suggestions. Multi-modal (image upload). Personalization. Multi-language support.

---
