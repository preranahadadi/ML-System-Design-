# Anomaly / Failure Detection System Design
---

## Table of Contents

1. [How to Scope This in an Interview](#1-how-to-scope-this-in-an-interview)
2. [Functional Requirements](#2-functional-requirements)
3. [Non-Functional Requirements](#3-non-functional-requirements)
4. [Data Requirements](#4-data-requirements)
5. [High-Level Architecture](#5-high-level-architecture)
6. [ML Model Design — The Ensemble](#6-ml-model-design--the-ensemble)
7. [Alert Engine Design](#7-alert-engine-design)
8. [Scale & Reliability](#8-scale--reliability)
9. [Key Trade-offs](#9-key-trade-offs)

---

## 1. How to Scope This in an Interview

| Question | Why It Matters |
|---|---|
| What are we monitoring? (metrics, logs, traces, KPIs?) | Completely changes the feature set |
| Real-time alerts or batch reports? | Latency budget differs by 100x |
| Do we have labeled incidents? | Determines supervised vs unsupervised approach |
| How many metrics? (10 or 10 million?) | Changes storage, compute, and model strategy |
| Who gets alerted? (human, auto-remediation, dashboard?) | Shapes the entire output layer |

> *"I'll assume we're building a production monitoring system for a mid-size tech company — thousands of metrics, mix of infra and app signals, real-time alerts under 60 seconds, and very few labeled incidents. This is primarily an unsupervised problem."*

---

## 2. Functional Requirements

### Detection

- Detect anomalies in real time across metrics, logs, and traces
- Support **point anomalies** (a sudden spike) and **contextual anomalies** (a normal value at the wrong time — e.g. low traffic at 2pm Monday)
- Handle **seasonality** — Monday 9am baseline ≠ Saturday midnight baseline
- Detect **collective anomalies** — three services degrading together is a signal even if each one individually looks fine
- Output anomaly scores [0–1] mapped to severity: `monitor / warn / page`

### Alerting & Operations

- **Deduplicate**: one incident → one alert, not 500 alerts firing simultaneously
- **Correlate** related alerts into a single incident object
- Route alerts to the correct on-call team by service owner
- **Suppression windows**: silence alerts during maintenance, known deploys, or scheduled events
- **Two-way feedback**: engineer can mark alert as false positive or confirm true positive

### ML-Specific Functional

- Dynamic baselines that adapt to metric history (not hardcoded static thresholds)
- Per-metric model instances — each metric's seasonality is different
- Explanation of *why* a point is anomalous (which features drove the score)
- Auto-remediation hooks: trigger runbooks or rollback pipelines for known failure patterns

---

## 3. Non-Functional Requirements

| Requirement | Target | Notes |
|---|---|---|
| **Detection latency** | < 60 seconds end-to-end | From metric emission to alert fired |
| **Ingest throughput** | 1M+ data points / minute | Peak across all services |
| **Availability** | 99.9% | Monitoring going dark is a P0 |
| **False positive rate** | < 5% | Alert fatigue destroys trust in the system |
| **Data retention** | Raw: 15 days · Aggregated: 2 years | Storage cost vs debugging need |
| **Model update frequency** | Daily retrain per metric | Catch concept drift quickly |
| **Alert MTTR impact** | Reduce mean time to detect by > 50% | The actual business metric |

### The Core Trade-off

> A 5% false positive rate on 1,000 alerts/day = **50 false pages per day**. Engineers start ignoring pages → monitoring system becomes worthless. **False positive rate is more important than recall in practice.**

---

## 4. Data Requirements

### Input Data Types

```
Time-series metrics:  (timestamp, service_id, metric_name, value)  @ 10–60s intervals
Logs:                 structured JSON — error_code, latency, status_code, trace_id
Traces:               distributed spans with parent-child service relationships
Business events:      deploys, feature flag changes, traffic spikes — context for anomalies
```

### Derived Features Per Metric

| Feature | Window | Purpose |
|---|---|---|
| Rolling mean / std | 1min, 5min, 1hr, 24hr | Baseline for Z-score |
| Seasonal baseline | Same hour, last 7 days | Handles weekly seasonality |
| Rate of change | 1st derivative | Catches trending degradations |
| Peer deviation | Across correlated services | Collective anomaly detection |
| Deploy flag | Last 15 minutes | Suppresses false positives post-deploy |

### The Label Problem — The Most Important Thing to Understand

This is what makes anomaly detection fundamentally different from fraud or spam:

```
Fraud detection:    transaction → chargeback (label arrives in 45 days, but it arrives)
Content moderation: post → human review (label created on demand)
Anomaly detection:  metric spike → ??? (you don't know it's real until the service is down)
```

**Consequences:**

- You cannot label 1M metric time points — there are too many
- A company may have only 50 confirmed incidents/year across 10,000 metrics
- Supervised learning is **impossible at cold start**
- Labels from past incidents are used only to **tune thresholds post-hoc**, not to train the core model

**Solution:** Train unsupervised (Isolation Forest, Autoencoder) on normal data. Use labeled incidents to calibrate the final threshold mapping: `score > X → page on-call`.

---

## 5. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                              │
│  App metrics │ Infra metrics │ Logs │ Traces │ Business KPIs     │
└──────────────────────────┬──────────────────────────────────────┘
                           │ push / scrape
┌──────────────────────────▼──────────────────────────────────────┐
│               INGESTION & NORMALIZATION                          │
│  Kafka topics per metric type                                    │
│  Schema validation · timestamp alignment · downsampling          │
└──────────────────────────┬──────────────────────────────────────┘
                           │ partitioned by metric_name
┌──────────────────────────▼──────────────────────────────────────┐
│               FEATURE ENGINEERING (Flink)                        │
│  Rolling stats │ Seasonal decomp │ Rate of change │ Correlation  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│               ML DETECTION ENGINE (ensemble)                     │
│                                                                  │
│  ┌─────────────┐  ┌──────────────────┐  ┌──────────────────┐   │
│  │ Statistical  │  │ Isolation Forest │  │ LSTM Autoencoder │   │
│  │ Z-score bands│  │ unsupervised     │  │ recon. error     │   │
│  └─────────────┘  └──────────────────┘  └──────────────────┘   │
│                    ┌─────────────────┐                           │
│                    │ Prophet forecast│                           │
│                    │ seasonality fit │                           │
│                    └─────────────────┘                           │
│                                                                  │
│              Weighted ensemble → anomaly score [0,1]             │
└──────────────────────────┬──────────────────────────────────────┘
                           │ score + explanation
┌──────────────────────────▼──────────────────────────────────────┐
│                     ALERT ENGINE                                 │
│  Deduplication · Correlation · Severity scoring                  │
│  Suppression windows · On-call routing · Cool-down               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│              HUMAN-IN-THE-LOOP FEEDBACK                          │
│  Engineer labels (TP/FP) → threshold tuning → retrain trigger   │
│  Drift monitoring · A/B threshold testing                        │
└──────────────────────────┬──────────────────────────────────────┘
                           │ ↻ feeds back into detection engine
```

### Tech Stack Choices

| Component | Choice | Why |
|---|---|---|
| Message queue | Kafka | Partitioned by `metric_name` preserves ordering per metric |
| Stream processing | Flink | Stateful windowed aggregates, exactly-once semantics |
| Time-series store | InfluxDB / Prometheus | Native time-series queries, efficient compression |
| Model store | S3 + in-memory cache | Serialize trained models, load at worker startup |
| Batch retrain | Spark | Parallelize across 10k metric models nightly |
| Alert routing | PagerDuty / OpsGenie | On-call schedules, escalation policies |

---

## 6. ML Model Design — The Ensemble

Each model catches a different failure mode. No single model wins across all metric types.

### Model 1: Statistical Baselines (always-on, instant)

**How it works:**
```
anomaly_score = |value - rolling_mean| / rolling_std

if same_hour_last_7_days available:
    baseline = mean(same_hour_last_7_days)  # handles weekly seasonality
else:
    baseline = rolling_mean(last_1hr)

alert if score > dynamic_k   # k tuned per metric from historical variance
```

**Catches:** Sudden spikes and drops, immediate detection < 1 second  
**Misses:** Slow trends, multivariate anomalies  
**Training:** None required — computed on the fly  
**Inference latency:** < 1ms

---

### Model 2: Isolation Forest (unsupervised, multivariate)

**How it works:**
- Randomly partition the feature space using decision trees
- Anomalies are isolated in **fewer splits** than normal points
- Anomaly score = average path length across trees (shorter = more anomalous)
- Trained on 30 days of normal operational data (no labels needed)

**Catches:** Multivariate anomalies — where no single metric is alarming but the **combination** is unusual (e.g. CPU normal + latency normal + error rate slightly elevated = real incident)  
**Misses:** Anomalies that look normal in the training feature space  
**Training:** Daily retrain on rolling 30-day window of normal data  
**Inference latency:** ~5ms

**Feature vector per prediction:**
```
[value, rolling_mean_1m, rolling_std_1m, rolling_mean_1h,
 rate_of_change, peer_correlation_score, hour_of_day, day_of_week]
```

---

### Model 3: LSTM Autoencoder (sequence anomaly)

**How it works:**
```
Encoder: input sequence (60 time steps) → compressed latent vector (32-dim)
Decoder: latent vector → reconstructed sequence

anomaly_score = MSE(original_sequence, reconstructed_sequence)
```

- Train **only on normal sequences** (no anomalies in training data)
- At inference: high reconstruction error = sequence is unfamiliar = anomaly
- The model has learned what "normal over the last 60 minutes" looks like

**Catches:** Slow degradations that trend over 1–2 hours — latency creeping up gradually, memory leak, queue depth slowly growing. These are invisible to statistics because the rolling baseline keeps adjusting.  
**Misses:** Single-point spikes (not enough sequence context)  
**Training:** Weekly retrain, sequence window = 60 minutes  
**Inference latency:** ~20ms (GPU recommended for production)

---

### Model 4: Prophet (forecasting-based, seasonality-aware)

**How it works:**
```
y(t) = trend(t) + seasonality_weekly(t) + seasonality_daily(t) + holidays(t) + noise

anomaly if actual_value outside [forecast - 2σ, forecast + 2σ] confidence band
```

- Facebook Prophet decomposes time series into trend + seasonal components
- Trains a forecast model per metric
- If actual value falls outside the 95% confidence interval → flagged

**Catches:** Anything with strong seasonality — business KPIs (traffic, revenue, signups) that behave completely differently on weekday vs weekend, or during a sale event.  
**Misses:** Infrastructure metrics with no seasonality pattern  
**Training:** Daily retrain  
**Inference latency:** ~50ms (Prophet is slower — run async, not inline)

---

### Ensemble Combination

```python
final_score = (
    w1 * statistical_score +
    w2 * isolation_forest_score +
    w3 * lstm_autoencoder_score +
    w4 * prophet_score
)

# Weights tuned per metric category:
# Business KPIs:    w4 (Prophet) weighted highest
# Infra metrics:    w2 (Isolation Forest) + w1 (Statistical) weighted highest
# App latency:      w3 (LSTM) weighted higher for slow trends

if final_score > threshold_page:    → page on-call (P1)
elif final_score > threshold_warn:  → warn in dashboard (P2)
else:                               → log and monitor
```

---

## 7. Alert Engine Design

This is the layer most candidates miss — and the one that determines whether your system is actually usable in production.

### Deduplication

A single outage can trigger 500 simultaneous alerts. Without deduplication, every on-call engineer quits within a month.

```
1. Time window grouping:    alerts on same service within 5 min → same incident
2. Metric correlation:      latency spike + error rate spike on same service = one alert
3. Causal ordering:         if upstream DB is down, silence downstream app alerts
4. Cool-down:               same alert cannot re-fire within 30 min without severity increase
```

### Suppression Windows

```
Deploy just happened  → raise threshold for 10 min (expected noise)
Maintenance window    → silence all non-critical alerts
Traffic ramp event    → adjust baselines for expected load increase
```

### Severity Routing

```
score > 0.95  → P1: page primary on-call immediately
score > 0.80  → P2: page secondary on-call, Slack alert
score > 0.60  → P3: create ticket, no page
score < 0.60  → log and monitor
```

### Two-way Feedback Loop

```
Engineer marks alert as FP  →  lower metric's weight in ensemble
                             →  adjust threshold for this metric
                             →  log for weekly retrain

Engineer confirms alert TP  →  validate model is working
                             →  log incident in incident DB
                             →  potentially adjust routing rules
```

---

## 8. Scale & Reliability

### Scaling Strategy

| Component | Strategy |
|---|---|
| Per-metric models | Each of 10k metrics gets its own model — shard by `metric_id`, embarrassingly parallel |
| Kafka partitions | Partition by `metric_name` so same metric's points always go to same consumer (ordering preserved) |
| Model serving | Serialize models to S3 as pickle/ONNX, load into worker memory at startup |
| Batch retrain | Nightly Spark job retrains all models in parallel, pushes artifacts to S3 |
| Detection workers | Stateless pods, auto-scale on Kafka consumer lag |

### Failure Modes & Mitigations

| Failure | Impact | Mitigation |
|---|---|---|
| **Model goes stale** | New normal looks anomalous after a service change | Weekly PSI (Population Stability Index) check, auto-trigger retrain on drift |
| **Ingestion lag** | Timestamps drift, seasonal baselines break | Alert if Kafka consumer lag > 30s; skip seasonal model if data is stale |
| **Cold start on new metric** | No history to train on | Use statistical baseline only for first 7 days, then activate full ensemble |
| **Alert storm** | 500 alerts fire at once during major outage | Deduplication + incident grouping + severity-based suppression |
| **Detection system outage** | Blind to all failures | Monitoring-of-monitoring: separate health check pipeline, alert if no metrics received for 60s |

### Monitoring the Monitor

```
Infra:     consumer lag per topic, detection worker CPU/memory, model inference p99 latency
Data:      feature distribution drift (PSI per metric, weekly), missing metrics alerts
Model:     FP rate per week, TP rate vs confirmed incidents, score distribution shift
Business:  MTTD (mean time to detect), alert volume, engineer-acknowledged rate
```

---

## 9. Key Trade-offs

| Trade-off | Option A | Option B | Decision |
|---|---|---|---|
| Static vs dynamic thresholds | Simple to implement | Handles seasonality | **Always dynamic** — static breaks on Monday mornings and Black Friday |
| Unsupervised vs supervised | Works cold, no labels needed | Higher precision with labels | **Unsupervised core** — use labels only to tune final threshold |
| Per-metric vs global model | More accurate, costly | Cheaper, less precise | **Per-metric** for critical services, global for the long tail |
| Sensitivity vs specificity | Catch all incidents | Minimize false pages | **Tune toward specificity** — alarm fatigue is a bigger risk than missed alerts |
| Isolation Forest vs Autoencoder | Fast, multivariate | Catches slow trends | **Both** — they catch different failure modes, use ensemble |
| Real-time vs batch scoring | Expensive, immediate | Cheap, delayed | **Real-time** for P1 metrics, batch for dashboards and trend reports |

---

