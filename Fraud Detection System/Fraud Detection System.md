# ML Fraud Detection System Design
---

## Table of Contents

1. [How to Approach This in an Interview](#1-how-to-approach-this-in-an-interview)
2. [Clarify & Scope](#2-clarify--scope)
3. [Functional Requirements](#3-functional-requirements)
4. [Non-Functional Requirements](#4-non-functional-requirements)
5. [Data Requirements](#5-data-requirements)
6. [High-Level Architecture](#6-high-level-architecture)
7. [ML Model Design](#7-ml-model-design)
8. [Infrastructure Deep Dive](#8-infrastructure-deep-dive)
9. [Scale & Reliability](#9-scale--reliability)
10. [Key Trade-offs](#10-key-trade-offs)

---

## 1. How to Approach This in an Interview

```
Scope → Functional → Non-Functional → Data → Architecture → ML Depth → Scale → Trade-offs
```

---

## 2. Clarify & Scope

Before drawing anything, ask:

| Question | Why It Matters |
|---|---|
| What type of fraud? (CNP, ATO, promo abuse?) | Changes the feature set entirely |
| Transaction volume? (10k/day vs 10M/day?) | Changes infra choices |
| Real-time decision or near-real-time? | Determines latency budget |
| Who consumes the output? | Analyst queue vs auto-block vs API |

"I'll assume a fintech processing ~1M transactions/day, mix of card payments and bank transfers, and we need a real-time decision at checkout under 100ms."
---

## 3. Functional Requirements

### Core Capabilities

- Score every transaction in real time → return **approve / step-up / block**
- Consume events from multiple channels: web checkout, mobile app, bank API, 3rd-party feeds
- Compute velocity features over sliding windows: 1 min, 1 hr, 24 hr, 7 day
- Support a **rule engine** for hard blocks (stolen card list, velocity caps)
- Route high-risk transactions to an **analyst review queue**
- Feed analyst decisions (labels) back to retrain models automatically

### ML-Specific Functional

- Serve multiple model types: tabular (XGBoost), sequential (LSTM), graph (GNN)
- Support **A/B shadow deployment** of new model versions before promotion
- Provide **SHAP-based explanations** for every decision (regulatory requirement)

---

## 4. Non-Functional Requirements

| Requirement | Target | Notes |
|---|---|---|
| **Latency** | p99 < 100ms | End-to-end at checkout |
| **Throughput** | 10k–50k TPS | Peak on Black Friday |
| **Availability** | 99.99% | Scoring outage = lost revenue or unblocked fraud |
| **Consistency** | Eventual for analytics; strong for fraud label writes | |
| **Scalability** | Horizontal — stateless scoring pods | |
| **Data retention** | 7 years | PCI-DSS, GDPR compliance |

### The Core Trade-off

> A 1% false positive rate on 1M transactions/day = **10,000 wrongly blocked customers per day**. The business sets the threshold — not engineering.

---

## 5. Data Requirements

### Input Data (per transaction event)

```
Transaction:  amount, currency, merchant_id, MCC, channel, timestamp
User:         user_id, account_age, historical_fraud_rate, device_id, IP, geo
Behavioral:   session_duration, keystroke_cadence, scroll_pattern
```

### Derived / Computed Data

| Store | Technology | Purpose | Freshness |
|---|---|---|---|
| Online feature store | Redis | Velocity counts, rolling aggregates | < 1 second |
| Offline feature store | S3 / Parquet | Historical aggregates for training | Minutes–hours |
| Graph store | Neo4j / DGL | Entity relationships (shared device, IP, card) | Near real-time |

### Labels for Training

- **Chargebacks** — ground truth, but delayed ~45–60 days
- **Analyst decisions** — faster signal, but has human bias
- **Class imbalance** — only 0.05–0.3% of transactions are fraud

> **Key insight to drop in an interview:** *"The hardest data problem here isn't storage — it's label delay. Chargebacks arrive 45 days later, so your model is always learning from the past."*

---

## 6. High-Level Architecture

```
┌─────────────────────────────────────────────────────┐
│                  DATA INGESTION                      │
│  Payment API │ Mobile SDK │ Bank Feeds │ 3rd-party   │
└──────────────────────┬──────────────────────────────┘
                       │ Kafka (partitioned by user_id)
┌──────────────────────▼──────────────────────────────┐
│              STREAM PROCESSING (Flink)               │
│  Deduplication │ Schema validation │ Feature writes  │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│                  FEATURE STORE                       │
│        Redis (online)  │  S3 Parquet (offline)       │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              ML SCORING ENGINE (<100ms)              │
│  Rule engine → XGBoost → LSTM → GNN → Meta-learner  │
└──────────────────────┬──────────────────────────────┘
                       │ risk score [0.0 – 1.0]
┌──────────────────────▼──────────────────────────────┐
│               DECISION & ACTION                      │
│    Approve  │ Step-up auth │ Review queue │ Block     │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│           FEEDBACK & RETRAINING LOOP                 │
│  Label collection → Data warehouse → MLflow → Deploy │
└─────────────────────────────────────────────────────┘
```

### Tech Choices & Justifications

| Component | Choice | Why |
|---|---|---|
| Message queue | Kafka | Replay, partitioning by user_id, large ecosystem |
| Stream processing | Flink | Stateful windowed aggregates, exactly-once semantics |
| Online feature store | Redis | Sub-ms reads, TTL built-in, clustered mode |
| Offline feature store | S3 + Parquet | Cheap, columnar for ML training, integrates with Spark |
| Primary ML model | XGBoost | Fast inference (<10ms), handles missing values, SHAP explainability |
| Graph processing | DGL / PyG | GNN support, integrates with Neo4j |
| Experiment tracking | MLflow | Model versioning, A/B shadow deployment |

---

## 7. ML Model Design

### The Ensemble Stack

```
Rule Engine  ──┐
XGBoost      ──┤
LSTM         ──┼──► Meta-learner (logistic stacker) ──► Risk score [0,1]
GNN          ──┘
```

| Model | Input | Catches |
|---|---|---|
| **Rule engine** | Hard-coded thresholds | Known stolen cards, obvious velocity abuse |
| **XGBoost / LightGBM** | Tabular features | Most common fraud patterns, fast inference |
| **LSTM / Transformer** | Transaction sequence | Test-charge-then-big-charge patterns |
| **GNN** | Entity relationship graph | Ring fraud, mule networks, device sharing |
| **Meta-learner** | All model scores | Optimal blend per business threshold |

### Feature Engineering

**Transaction features:** amount, currency, merchant_category_code, channel, time_of_day, day_of_week, geo, IP

**Velocity features (sliding windows):**
- `txn_count_1h`, `txn_count_24h`
- `spend_1h`, `spend_7d`
- `unique_merchants_1h`
- `failed_attempts_10min`
- `geo_distance_from_last_txn`

**Behavioral features:**
- Deviation from user's typical spend pattern
- Known vs unknown merchant
- Device fingerprint match
- Session duration, keystroke cadence

**Graph features:**
- `shared_device_count` — how many accounts use this device
- `shared_ip_fraud_rate` — fraud rate of accounts sharing this IP
- `community_risk_score` — GNN embedding of entity neighborhood

### Training Challenges

| Challenge | Solution |
|---|---|
| Class imbalance (0.1% fraud) | SMOTE oversampling + class weights |
| Concept drift (patterns change monthly) | Retrain weekly with recent data window |
| Feature leakage | Time-based train/test splits only — never random |
| Label delay (chargebacks at 45 days) | Proxy labels from analyst queue for faster signal |

### Evaluation Metrics

- **Primary:** Precision-Recall AUC (not accuracy — useless on imbalanced data)
- **Secondary:** KS statistic, F-beta score (beta > 1 to weight recall higher)
- **Business:** Chargeback rate, false positive rate, analyst queue volume, revenue at risk

---

## 8. Infrastructure Deep Dive

### Stateless Pods + Load Balancer

A **pod** is a single running instance of the scoring service.

**Stateless** means the pod stores nothing about users in its own memory. All state (velocity counts, session data) lives in Redis. Any pod can handle any request → you can spin up 50 pods under load and kill 40 when traffic drops.

```
               ┌──► Pod 1 (reads Redis) ──┐
Requests ──► LB├──► Pod 2 (reads Redis) ──┤──► Response
               └──► Pod 3 (reads Redis) ──┘
                      ▲
                    Redis
                (shared state)
```

**Load balancer algorithms:**

| Algorithm | How It Works | Use When |
|---|---|---|
| Round robin | 1→2→3→1→2→3 | Requests take equal time (our case) |
| Least connections | Send to pod with fewest active requests | Request duration varies widely |
| IP hash | Same IP always → same pod | Need session stickiness (not needed here) |
| Weighted | Bigger pods get more traffic | Mixed pod sizes/capacities |

> For fraud scoring: **round robin** is fine because pods are stateless and requests are roughly equal in processing time.

The load balancer also runs **health checks** every ~10 seconds and stops sending traffic to any pod that fails — automatic fault tolerance.

### Kafka Partitioning by `user_id`

Kafka splits a topic into **partitions**. Within one partition, messages are always processed **in order**.

```
Incoming events (mixed):  A:$50  B:$200  A:$30  C:$500  A:$800

         hash(user_id) % num_partitions
                    │
         ┌──────────┼──────────┐
         ▼          ▼          ▼
  Partition 0  Partition 1  Partition 2
  [user A]     [user B]     [user C]
  A:$50        B:$200       C:$500
  A:$30        B:$10        C:$90
  A:$800
  (in order)   (in order)   (in order)
```

**Why ordering matters for fraud:**

Without ordering, you might compute velocity on `$800` before seeing the preceding `$50 + $30` — the pattern is broken. With ordering, you always process `$50 → $30 → $800` in sequence and can correctly flag the sudden spike.

**The formula:** `partition = hash(user_id) % total_partitions`

Same user always hashes to the same partition number → all their events go to the same Flink consumer → velocity features are always computed on a complete, ordered history.

**Parallelism:** Different users land on different partitions, processed simultaneously. You get parallelism across users and ordering within each user.

---

## 9. Scale & Reliability

### Scaling Strategy

| Component | Strategy |
|---|---|
| Scoring pods | Stateless → horizontal scale on CPU, auto-scale group |
| Kafka | Partition by user_id, scale partitions = scale parallelism |
| Redis | Clustered mode with read replicas — writes to primary, reads distributed |
| Flink | Scale consumer group size to match Kafka partition count |
| Model serving | ONNX export for fast cross-platform inference |

### Failure Modes & Mitigations

| Failure | Impact | Mitigation |
|---|---|---|
| Feature store (Redis) down | Can't compute velocity | Fallback to stale features + freshness flag — don't block payment |
| Model service timeout | No ML score | Fall back to rule engine only — degrade gracefully |
| Kafka consumer lag | Delayed scoring | Alert at >30s lag, auto-scale consumer group |
| Model drift | Degraded precision | Weekly retraining + score distribution monitoring |
| Training data poisoning | Corrupted model | Shadow deploy, A/B test before promotion |

### Monitoring

```
Infra:    p50 / p95 / p99 scoring latency, error rates, Kafka consumer lag
Data:     Feature distribution shift (KL divergence or PSI) — weekly
Model:    Score distribution shift, chargeback rate per cohort
Business: FP rate, analyst queue volume, revenue at risk per threshold
```

---

## 10. Key Trade-offs

| Trade-off | Option A | Option B | Decision driver |
|---|---|---|---|
| Precision vs recall | Fewer false positives | Higher fraud detection | Business risk appetite |
| Model complexity vs latency | GNN adds 30–40ms | Catches ring fraud | Only if ring fraud is significant |
| Rule engine vs ML | Auditable, instant deploy | Generalizes, black box | Use both — rules first, ML second |
| Real-time vs batch | Expensive, accurate | Cheaper, delayed | Low-value txns can batch score |
| Label freshness | Chargebacks (45d delay) | Analyst labels (fast, biased) | Use both as complementary signals |






