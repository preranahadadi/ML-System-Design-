# Netflix Watch-Time Prediction – ML System Design

A complete interview-ready guide for designing a Netflix-style watch-time prediction system.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Why Watch Time Matters](#2-why-watch-time-matters)
3. [Recommendation System vs Ranking System](#3-recommendation-system-vs-ranking-system)
4. [Supervised vs Unsupervised vs Bandits](#4-supervised-vs-unsupervised-vs-bandits)
5. [What Exactly Are We Predicting](#5-what-exactly-are-we-predicting)
6. [Functional Requirements](#6-functional-requirements)
7. [Non-Functional Requirements](#7-non-functional-requirements)
8. [High-Level Architecture](#8-high-level-architecture)
9. [End-to-End Request Flow](#9-end-to-end-request-flow)
10. [Data Sources and Features](#10-data-sources-and-features)
11. [Candidate Generation](#11-candidate-generation)
12. [Ranking Model](#12-ranking-model)
13. [Model Choices](#13-model-choices)
14. [Label Engineering](#14-label-engineering)
15. [Loss Functions](#15-loss-functions)
16. [Training Pipeline](#16-training-pipeline)
17. [Offline Evaluation](#17-offline-evaluation)
18. [Online Evaluation](#18-online-evaluation)
19. [Feature Freshness](#19-feature-freshness)
20. [Cold Start](#20-cold-start)
21. [Exploration vs Exploitation](#21-exploration-vs-exploitation)
22. [Re-Ranking and Business Rules](#22-re-ranking-and-business-rules)
23. [Serving Architecture](#23-serving-architecture)
24. [Data Logging](#24-data-logging)
25. [Bias and Feedback Loops](#25-bias-and-feedback-loops)
26. [Failure Modes](#26-failure-modes)
27. [Schema Design](#27-schema-design)
28. [API Design](#28-api-design)
29. [Training / Inference Separation](#29-training--inference-separation)
30. [One-Line Summary](#30-one-line-summary)

---

## 1. Problem Statement

We want to predict:

**How much time will a user spend watching a title if we show it right now?**

This prediction is used to:
- rank titles on the homepage
- rank titles inside rows
- rank titles inside recommendation surfaces
- improve engagement and satisfaction
- optimize long-term retention, not just clicks

The core product objective is not merely:
- "Will the user click?"

It is closer to:
- "Will the user meaningfully watch and enjoy this title?"

---

## 2. Why Watch Time Matters

If we optimize only for click-through rate, the system may show clickbait-style titles.

Example:
- Title A gets many clicks, but users stop after 2 minutes
- Title B gets fewer clicks, but users watch for 45 minutes

In most streaming scenarios, Title B is better.

So watch time is a stronger signal than click alone because it is closer to:
- satisfaction
- intent match
- session quality
- retention value

---

## 3. Recommendation System vs Ranking System

This is where many people get confused.

### Recommendation System
The **full end-to-end system** that decides what to show.

It includes:
- retrieval / candidate generation
- ranking
- re-ranking
- personalization
- business rules
- exploration
- homepage construction

### Ranking System
One stage inside the recommendation system.

After retrieval gives, say, 500 to 2000 candidate titles, the ranking model orders them.

---

## 4. Supervised vs Unsupervised vs Bandits

### Supervised Learning
This is the **main learning setup** for watch-time prediction.

Why?
Because we have:
- input features: user, title, context, past behavior
- label: actual watch time, completion, play, etc.

Example training row:
- user = U1
- title = Stranger Things
- context = Friday 9 PM on TV
- actual watch time = 42 minutes

That is supervised learning.

### Unsupervised Learning
Used for:
- title embeddings
- user embeddings
- clustering similar content
- clustering similar viewers
- latent taste discovery

This helps retrieval and feature engineering.

### Bandits / RL-Like Methods
Used for:
- exploration vs exploitation
- trying new titles
- reducing popularity lock-in
- learning from online feedback


---

## 5. What Exactly Are We Predicting

There are multiple possible targets.

### Option A: Raw Watch Minutes
Predict:

```text
expected_watch_minutes(user, title, context)
```

### Option B: Normalized Watch Ratio
Predict:

```text
watch_minutes / runtime
```

Useful because:
- a 20-minute watch on a 22-minute episode is great
- a 20-minute watch on a 3-hour movie is weak

### Option C: Satisfaction-Weighted Watch Time
Use a custom target that combines:
- watch minutes
- completion ratio
- return-to-series behavior
- abandonment penalty
- long-term satisfaction proxy

---

## 6. Functional Requirements

The system should:

- predict watch time for a `(user, title, context)` pair
- rank titles on homepage and other recommendation surfaces
- personalize recommendations per user
- adapt quickly to changing user behavior
- support real-time or near-real-time updates
- support A/B testing of new ranking models
- handle new users and new titles
- support business constraints like region and maturity filtering

---

## 7. Non-Functional Requirements

The system should provide:

- low latency, ideally under 100 ms for ranking stage
- high throughput for millions of users
- high availability
- fault tolerance and graceful degradation
- scalable storage and feature computation
- feature freshness
- monitoring and observability
- reproducibility of training pipelines

---

## 8. High-Level Architecture

```text
+------------------+
| User opens app   |
+------------------+
         |
         v
+------------------------------+
| Request Context Builder      |
| user, device, time, surface  |
+------------------------------+
         |
         v
+------------------------------+
| Candidate Generation Layer   |
| CF / content / ANN / trends  |
+------------------------------+
         |
         v
+------------------------------+
| Candidate Set                |
| 500 to 2000 titles           |
+------------------------------+
         |
         v
+------------------------------+
| Feature Store Lookup         |
| user + title + context       |
+------------------------------+
         |
         v
+------------------------------+
| Ranking Model                |
| predict expected watch time  |
+------------------------------+
         |
         v
+------------------------------+
| Re-Ranking Layer             |
| diversity / freshness /      |
| filters / business rules     |
+------------------------------+
         |
         v
+------------------------------+
| Top N Recommendations        |
+------------------------------+
         |
         v
+------------------------------+
| Logging Pipeline             |
| impressions / plays / watch  |
+------------------------------+
         |
         v
+------------------------------+
| Offline Training Pipeline    |
+------------------------------+
```

---

## 9. End-to-End Request Flow

When a user opens Netflix:

1. The app sends a recommendation request
2. The request context is built
   - user ID
   - device type
   - country
   - time of day
   - session state
3. Candidate generation retrieves a few hundred or thousand possible titles
4. Features are fetched from online feature store or cache
5. The ranking model scores each candidate
6. The re-ranking layer applies business and product constraints
7. Top titles are returned
8. Impressions and user interactions are logged
9. These logs later become training data

---

## 10. Data Sources and Features

### User Features
- viewing history
- genres preferred
- actors/directors preferred
- session time patterns
- weekday/weekend preference
- language preference
- binge behavior
- abandonment rate
- recency of last watch
- search history
- device type

### Title Features
- genre
- subgenre
- language
- cast
- director
- maturity rating
- runtime
- release year
- popularity
- novelty
- embeddings from synopsis, trailer, artwork, and metadata

### Context Features
- time of day
- day of week
- country/region
- device type
- network type
- homepage row position
- whether user is in a short session vs long session

### Interaction Features
- user-title past impressions
- previous clicks
- previous partial watches
- completion ratio
- dwell time before play
- whether title was shown recently and ignored

---

## 11. Candidate Generation

We do **not** rank the entire catalog online. That would be too expensive.

So we first retrieve a smaller set.

### Retrieval Sources
- collaborative filtering candidates
- "users like you watched"
- content-based similar titles
- continue watching
- trending in region
- new releases
- editorial or business-promoted titles
- embeddings-based nearest neighbors using ANN
- two-tower retrieval model

### Why Candidate Generation Matters
If the correct title never reaches the ranking stage, the ranker cannot recover it.

So retrieval should optimize for:
- high recall
- low latency
- source diversity

---

## 12. Ranking Model

Now for each candidate title, we predict expected watch value.

### Input
```text
(user_features, title_features, context_features, user-title interaction features)
```

### Output
```text
expected_watch_minutes
```

Or a combined score such as:
```text
final_score =
w1 * P(play)
+ w2 * expected_watch_minutes
+ w3 * P(completion)
+ w4 * long_term_value
```

This makes the system more robust than optimizing for one metric only.

---

## 13. Model Choices

### Baseline
- linear regression
- logistic regression on thresholded watch outcomes

Pros:
- easy to debug
- fast
- interpretable

### Strong Practical Choice
- Gradient Boosted Trees such as XGBoost or LightGBM

Pros:
- handles tabular features well
- strong performance on mixed structured data
- efficient inference
- good first production model

### Advanced Choice
- deep ranking model
- user tower + item tower + cross features
- MLP over concatenated features
- sequence model over recent viewing history
- transformer-based session encoder

Use a two-stage approach:
1. **Two-tower model** for retrieval
2. **GBDT or deep ranker** for final watch-time prediction

That is realistic and production-friendly.

---

## 14. Label Engineering

This is one of the most important parts.

Raw watch time is noisy.

### Simple Label
```text
watch_label = min(actual_watch_minutes, max_cap)
```

### Better Label
```text
weighted_watch =
a * watch_minutes
+ b * completion_ratio
+ c * return_to_series
- d * immediate_abandonment
```

### Why Better Labels Matter
Because:
- 3 minutes watched on a 2-hour movie is weak
- 20 minutes watched on a 22-minute episode is excellent
- long watch with return behavior is stronger than accidental autoplay
- completion carries extra meaning beyond raw minutes

---

## 15. Loss Functions

Depending on the formulation:

### Regression Framing
Predict watch minutes:
- MSE
- MAE
- Huber loss
- Tweedie or Gamma loss for skewed positive labels

### Ranking Framing
Predict better ordering:
- pairwise ranking loss
- listwise ranking loss
- LambdaMART

### Multi-Task Framing
Jointly predict:
- click probability
- play probability
- expected watch minutes
- completion probability

Then combine them into a final score.
---

## 16. Training Pipeline

### Data Flow
1. collect impression logs
2. join them with play and watch logs
3. build training examples
4. generate features
5. compute labels
6. train model
7. validate offline
8. register model
9. deploy to online serving

### Training Example
For each shown title:
- user at time T
- title shown in slot S
- context C
- label = watch minutes after impression within the defined attribution window

### Attribution Window
Need to define:
- immediate session only
- next session window
- 24-hour / 48-hour delayed watch
- autoplay handling
- repeat watch handling

This matters because some watches happen later, not immediately after the impression.

---

## 17. Offline Evaluation

Do **not** say only accuracy.

### For Regression
- RMSE
- MAE

### For Ranking
- NDCG
- MAP
- Precision@K
- Recall@K

### Product-Oriented Offline Metrics
- average watch minutes per session
- completion rate
- plays per homepage visit
- user return rate
- long-term satisfaction proxy

---

## 18. Online Evaluation

Use A/B testing.

### Success Metrics
- total watch time per member
- plays per session
- completion rate
- abandonment rate
- session length
- retention after 7 / 30 days
- diversity of consumed content
- novelty consumption

### Guardrail Metrics
- latency
- error rate
- recommendation freshness
- catalog coverage
- region or profile safety violations

---

## 19. Feature Freshness

Some features can be batch and some should be near real time.

### Batch Features
- long-term genre affinity
- favorite actors
- average session length
- popularity trends over days

### Real-Time Features
- last watched title
- current session actions
- recent skips
- recent binge signal
- current device and time context

Why this matters:
A user at 8 PM on a TV may want a movie.
The same user at 8 AM on a phone may want something short.

---

## 20. Cold Start

### New User Cold Start
Use:
- onboarding preferences
- country and language defaults
- trending titles
- broad priors
- session context
- controlled exploration

### New Title Cold Start
Use:
- metadata embeddings
- synopsis embeddings
- trailer embeddings
- artwork embeddings
- similarity to existing titles
- editorial boosts
- exploration traffic buckets

---

## 21. Exploration vs Exploitation

If the system always recommends what is already known to work, it becomes stale.

### Exploitation
Show titles likely to maximize watch time based on known signals.

### Exploration
Try:
- new titles
- new genres
- lesser-known content
- different artwork
- different row placements

### Common Strategies
- epsilon-greedy
- contextual bandits
- exploration buckets

This helps:
- reduce feedback loops
- learn user preferences faster
- give new titles a chance

---

## 22. Re-Ranking and Business Rules

Pure model score is not enough.

After ranking, apply re-ranking to enforce:

- diversity
- freshness
- maturity filters
- kids profile rules
- region licensing constraints
- suppress already completed titles
- continue-watching priority
- deduplication
- row-specific constraints
- fairness for new content exposure

### Example
If the top 8 titles are all dark crime thrillers, that may hurt homepage quality.
The re-ranker should create a healthier mix.

---

## 23. Serving Architecture

### Online Path
1. request hits recommendation service
2. fetch request context
3. retrieve candidates
4. fetch features from online feature store / cache
5. score candidates
6. apply re-ranking
7. return top results

### Latency Optimizations
- precompute user embeddings
- precompute title embeddings
- ANN index for fast retrieval
- cache popular item features
- batch feature fetches
- lightweight final ranker
- asynchronous logging

### Fallback Path
If ranker fails:
- use cached recommendations
- fall back to popularity + continue watching
- still enforce profile safety filters

---

## 24. Data Logging

You must log both exposure and outcome.

### Impression Logs
- impression_id
- user_id
- title_id
- timestamp
- rank_position
- row_id
- artwork_id
- device_type
- country
- experiment_id

### Engagement Logs
- play_id
- impression_id
- play_start
- watch_duration
- completion_ratio
- pause / resume
- stop event
- exit point
- replay
- thumbs up / down if available

### Why Logging Matters
If you do not log impressions correctly, training becomes biased because you only know what got watched, not what was shown and ignored.

---

## 25. Bias and Feedback Loops

This is a great interview discussion point.

### Problem
If the system always shows popular titles:
- popular titles get more impressions
- more impressions generate more watch data
- model becomes more confident those are the best
- long-tail content gets starved

### Solutions
- exploration traffic
- randomized buckets
- inverse propensity weighting
- position-bias correction
- re-ranking diversity constraints
- new-title exposure rules

---

## 26. Failure Modes

### 1. Clickbait Recommendations
High clicks, low watch time

**Fix:** optimize for watch time or satisfaction-weighted watch value

### 2. Filter Bubble
Recommendations become too narrow

**Fix:** enforce diversity and exploration

### 3. New Title Starvation
No historical interactions

**Fix:** use metadata embeddings + exploration

### 4. Session Mismatch
Short-session user gets long-form content

**Fix:** use session-aware and context-aware features

### 5. Position Bias
Higher-ranked items get more interaction just because of placement

**Fix:** debias logged data and use randomized experiments

### 6. Delayed Feedback
User watches later, outside the attribution window

**Fix:** design the label attribution window carefully

---

## 27. Schema Design

### impressions
```sql
CREATE TABLE impressions (
    impression_id      BIGINT PRIMARY KEY,
    user_id            BIGINT NOT NULL,
    title_id           BIGINT NOT NULL,
    ts                 TIMESTAMP NOT NULL,
    row_id             VARCHAR(100),
    rank_position      INT,
    artwork_id         VARCHAR(100),
    device_type        VARCHAR(50),
    country_code       VARCHAR(10),
    experiment_id      VARCHAR(100)
);
```

### plays
```sql
CREATE TABLE plays (
    play_id            BIGINT PRIMARY KEY,
    impression_id      BIGINT,
    user_id            BIGINT NOT NULL,
    title_id           BIGINT NOT NULL,
    play_start_ts      TIMESTAMP NOT NULL,
    play_end_ts        TIMESTAMP,
    watch_minutes      DOUBLE,
    completion_ratio   DOUBLE,
    replay_count       INT DEFAULT 0
);
```

### user_features
```sql
CREATE TABLE user_features (
    user_id                    BIGINT PRIMARY KEY,
    language_pref              VARCHAR(50),
    avg_session_length         DOUBLE,
    binge_score                DOUBLE,
    abandonment_rate           DOUBLE,
    recency_last_watch_hours   DOUBLE,
    preferred_genres_json      TEXT
);
```

### title_features
```sql
CREATE TABLE title_features (
    title_id             BIGINT PRIMARY KEY,
    genre                VARCHAR(100),
    subgenre             VARCHAR(100),
    language             VARCHAR(50),
    runtime_minutes      INT,
    maturity_rating      VARCHAR(20),
    release_year         INT,
    popularity_score     DOUBLE,
    novelty_score        DOUBLE
);
```

---

## 28. API Design

### Recommendation API
```http
GET /recommendations?user_id=123&surface=homepage
```

### Sample Response
```json
{
  "user_id": 123,
  "surface": "homepage",
  "titles": [
    {
      "title_id": "T101",
      "score": 0.91,
      "reason": "Because you watched thrillers"
    },
    {
      "title_id": "T202",
      "score": 0.84,
      "reason": "Trending in your region"
    }
  ]
}
```

### Impression Logging API
```http
POST /impressions
```

### Play Event Logging API
```http
POST /play-events
```

### Optional Feature Debug API
```http
GET /debug/features?user_id=123&title_id=T101
```

Useful for:
- debugging feature values
- troubleshooting poor recommendations
- validating online feature freshness

---

## 29. Training / Inference Separation

### Offline Layer
Handles:
- heavy joins
- feature generation
- embeddings
- model training
- backfills
- evaluation

### Online Layer
Handles:
- fast retrieval
- low-latency feature fetch
- scoring
- re-ranking
- serving

### Why Separation Matters
Training can be expensive and slow.
Inference must be fast and reliable.

---

## 30. One-Line Summary

**Netflix watch-time prediction is a supervised ranking problem inside a larger recommendation system, where candidates are retrieved first, then scored for expected watch value, and finally re-ranked using business and product constraints.**

---