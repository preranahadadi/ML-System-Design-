# Instagram Feed ranking model

A complete, interview-ready and revision-friendly guide to designing an Instagram Feed Ranking system.

This README is written so that:
- a student can revise it quickly,
- an interview candidate can explain it confidently,
- and the design stays structured from requirements to ML training and serving.

---

## 1. Functional Requirements

### Core Product Requirements
- Show a personalized feed when a user opens Instagram.
- Rank posts based on the likelihood of engagement and overall user satisfaction.
- Include posts from:
  - followed accounts,
  - recommended creators,
  - trending or fresh content.

### Filtering Requirements
The system should remove or suppress:
- blocked users,
- muted users,
- already seen posts,
- unsafe or policy-violating content,
- duplicate or near-duplicate posts.

### Interaction Logging
The system should track user actions such as:
- impressions,
- likes,
- comments,
- shares,
- saves,
- skips,
- dwell time,
- follows after viewing content,
- hides and reports.

### Additional Product Goals
- Feed diversity.
- Exploration of new creators/content.
- Freshness boost for recent posts.
- Good balance between relevance and discovery.

---

## 2. Non-Functional Requirements

- **Latency:** roughly 100 to 300 ms for feed generation.
- **Scalability:** should support billions of users and massive post volume.
- **High availability:** feed should still work even if some ranking components fail.
- **Freshness:** new posts and recent engagement signals should appear quickly.
- **Personalization:** feed should adapt to each user's behavior.
- **Reliability:** ranking and logging pipelines must be consistent and fault-tolerant.

---

## 3. High-Level Flow

The overall flow is:

**Candidate Generation -> Filtering -> Feature Generation -> Ranking -> Re-ranking -> Serving -> Logging -> Training**

This is a standard multi-stage ranking pipeline because scoring every post in the system is too expensive.

---

## 4. Architecture Diagram

### Simple Flow Diagram

```text
User
 |
 v
API Gateway
 |
 v
Feed Service
 |
 +-------------------------+------------------------+
 |                         |                        |
 v                         v                        v
Follow Graph        Recommendation System     Trending Service
 |
 +----------- Candidate Aggregator ------------+
                     |
                     v
               Filtering Layer
                     |
                     v
            Feature Enrichment
                     |
                     v
                Pre-Ranking
                     |
                     v
                Main Ranking
                     |
                     v
                Re-Ranking
                     |
                     v
               Feed Response
                     |
                     v
                   Client
                     |
                     v
               Logging System
                     |
        +-----------------------------+
        |                             |
        v                             v
 Real-time Features         Offline Training Pipeline
```

### How to Explain This in an Interview

A good way to explain this is:

1. The **Feed Service** receives the feed request.
2. It gathers candidates from multiple sources such as the follow graph, recommendation system, and trending service.
3. The system filters out invalid or unwanted posts.
4. Features are generated for each remaining candidate.
5. A **pre-ranking model** reduces the list quickly.
6. A heavier **main ranking model** scores the shortlisted posts more accurately.
7. A **re-ranking layer** applies diversity, freshness, and business rules.
8. The ranked feed is returned to the client.
9. User interactions are logged and later used for feature updates and training.

---

## 5. Data and Features

This is one of the most important interview sections.

For feed ranking, features usually come from five major buckets.

### A. User Features
These describe the user.

Examples:
- user ID embedding,
- language,
- region,
- account age,
- device type,
- active hours of day,
- average session length,
- preferred content type such as reel, image, or carousel,
- recent topic interests,
- past engagement rates,
- follow count and follower count,
- recent watch history,
- recently liked or saved topic clusters.

**Why these matter:**
They tell us what kind of content the user usually prefers.

### B. Creator Features
These describe the author of the post.

Examples:
- creator ID embedding,
- creator popularity,
- creator category,
- average engagement rate,
- posting frequency,
- historical quality score,
- trust and safety score,
- whether the creator is followed by the user,
- past interaction strength between user and creator.

**Why these matter:**
Users often engage more with creators they already know, trust, or repeatedly interact with.

### C. Post Features
These describe the post itself.

Examples:
- post age or freshness,
- media type,
- caption length,
- hashtags,
- topic embedding,
- audio embedding for reels,
- image or video embedding,
- early engagement velocity,
- number of likes, comments, and shares so far,
- language of caption,
- whether the post is from a close friend or frequent contact,
- quality or spam score,
- content safety classification.

**Why these matter:**
These features help the system understand what the content is and how promising it looks.

### D. User-Post Interaction Features
These are usually the most powerful features.

Examples:
- has the user liked this creator before,
- has the user commented on this creator before,
- number of interactions with the creator in the last 7 or 30 days,
- cosine similarity between user embedding and post embedding,
- how often the user skips similar posts,
- time since the last interaction with this creator,
- user engagement with the same topic recently,
- whether the user watches similar reels to completion,
- whether the user hides this type of content often.

**Why these matter:**
These answer the key question:

**For this specific user and this specific post, how likely is engagement?**

### E. Context Features
These describe the current session.

Examples:
- current time of day,
- day of week,
- current network quality,
- scroll depth in session,
- recent actions in the same session,
- whether the user is in exploration mode or normal browsing mode,
- time since app open,
- whether the user just consumed many reels from the same category.

**Why these matter:**
User intent changes during a session, so the same user may behave differently at different times.

---

## 6. ML Models

Feed ranking is usually not one single model. It is typically a **multi-stage ML system**.

### A. Candidate Generation Models
This stage is about **fast retrieval**.

**Goal:**
From millions of posts, retrieve maybe **500 to 2000 candidates**.

Possible approaches:
- collaborative filtering,
- graph-based retrieval,
- two-tower retrieval model,
- approximate nearest neighbor search over embeddings,
- heuristic retrieval from follow graph and recent posts.

#### Two-Tower Retrieval Model Explained Properly
This is a very strong concept to explain in interviews.

A **two-tower model** has two separate neural networks:

- one tower creates a **user embedding**,
- the other tower creates a **post embedding**.

Then we compute similarity between the two embeddings, usually using:
- dot product,
- cosine similarity,
- or another learned similarity function.

If similarity is high, the post is likely relevant to the user.

#### Why Two-Tower Works Well
- It is fast at large scale.
- Post embeddings can be precomputed offline.
- User embeddings can be computed online or refreshed frequently.
- ANN search can quickly retrieve the nearest posts to the user embedding.

#### How It Works End to End
1. Train the model on historical interactions.
2. User tower learns a vector representation of the user.
3. Post tower learns a vector representation of each post.
4. During serving, compute the user vector.
5. Search for top nearest post vectors using ANN.
6. Return the top candidates to downstream ranking stages.

#### What Inputs Go Into the Two Towers
**User tower inputs:**
- user embedding ID,
- recent engagement history,
- followed creators,
- favorite topics,
- session context.

**Post tower inputs:**
- creator embedding,
- post topic,
- media embeddings,
- freshness,
- post metadata.

#### What It Learns
It learns a latent space where:
- users are close to content they like,
- and far from content they tend to skip.

This makes retrieval efficient because we do not score every post individually with a heavy model.

### B. Pre-Ranking Model
**Goal:**
Reduce candidates further, for example from **1000 to 200**.

Possible models:
- logistic regression,
- gradient boosted decision trees,
- lightweight neural networks.

**Why this stage exists:**
It gives a quick quality estimate before spending more compute on the expensive ranking model.

### C. Main Ranking Model
**Goal:**
Score the shortlisted posts accurately.

Possible models:
- Gradient Boosted Trees such as XGBoost or LightGBM,
- Wide and Deep models,
- deep neural networks,
- multi-task neural networks,
- DeepFM or similar ranking DNN variants.

A strong interview answer is:

> I would likely use a multi-task ranking model because feed quality depends on multiple engagement objectives like likes, comments, saves, shares, and dwell time.

### D. Multi-Task Learning
Instead of predicting just one label, the model predicts multiple outcomes:
- P(like)
- P(comment)
- P(save)
- P(share)
- P(long dwell)
- P(follow after impression)

A final score can be computed as:

```text
FinalScore = w1 * P(like)
           + w2 * P(comment)
           + w3 * P(save)
           + w4 * P(share)
           + w5 * P(dwell)
```

This is useful because Instagram does not only care about likes.
Sometimes a **save** or **share** is a much stronger quality signal.

### E. Re-Ranking Model or Rules Layer
After the main ranking stage, a final pass adjusts the feed order.

This may not always be a pure ML model. It can include:
- heuristic rules,
- constrained optimization,
- diversity-aware rerankers,
- lightweight bandit or exploration logic.

Common goals:
- diversify creators,
- diversify content types,
- add freshness boost,
- avoid too many similar posts,
- ensure policy constraints,
- give exposure to new creators.

---

## 7. Training Data

The system logs impressions and downstream user actions.
For every shown post, labels are created.

### Positive Labels
- liked,
- commented,
- shared,
- saved,
- watched for long duration,
- profile click,
- follow after viewing content.

### Negative Labels
- skipped quickly,
- hidden,
- reported,
- scrolled past immediately,
- no engagement after impression.

### Important Interview Point
A non-click is **not always a true negative**.
Sometimes the user never properly noticed the content.

So ranking data requires careful label design and possibly position-bias correction.

### Example Training Row
```text
(user123, post456, features...) -> like=1, save=0, comment=0, dwell=1
```

Each row includes:
- user features,
- creator features,
- post features,
- interaction features,
- context features,
- labels.

---

## 8. Training Pipeline

### Step 1: Event Logging
Collect:
- impression logs,
- engagement logs,
- skip logs,
- hide and report logs.

### Step 2: Data Processing
Use batch or streaming pipelines to:
- join impressions with downstream actions,
- build labels,
- compute aggregates,
- generate embeddings,
- clean noisy or invalid data.

### Step 3: Feature Computation
- Offline features are computed in data warehouses or Spark jobs.
- Near-real-time features are updated using streaming systems.

### Step 4: Model Training
Train:
- candidate generation models,
- pre-ranking models,
- main ranking models,
- reranking or exploration models if needed.

### Step 5: Validation and Offline Evaluation
Test on holdout datasets and compare against baseline models.

### Step 6: Model Deployment
Deploy the model to online inference services.

### Step 7: Online Experimentation
Compare the new model against the production model using A/B testing.

---

## 9. Offline Training vs Online Serving

Interviewers like this distinction a lot.

### Offline
Heavy computations happen here:
- model training,
- feature aggregation,
- embedding generation,
- historical statistics,
- dataset building.

### Online
Fast computations happen here:
- retrieve candidates,
- fetch online features,
- infer model scores,
- rerank,
- return the feed response.

A strong interview line is:

> Offline computes expensive intelligence, while online uses it under tight latency constraints.

---

## 10. Testing and Evaluation

This section should be split into **offline evaluation** and **online evaluation**.

### A. Offline Evaluation
This happens before production rollout.

#### Ranking Metrics
- AUC,
- log loss,
- precision@k,
- recall@k,
- NDCG@k,
- MAP@k.

For feed ranking, **NDCG@k** is especially strong to mention because ranking order matters.

#### Multi-Objective Evaluation
Evaluate performance on:
- like prediction,
- save prediction,
- comment prediction,
- dwell prediction.

#### Calibration
Check whether predicted probabilities are meaningful and well-calibrated.

#### Slice-Based Analysis
Evaluate across slices such as:
- new users,
- power users,
- new creators,
- language and geography,
- different content types.

This is important because average metrics can hide poor performance on important segments.

### B. Online Evaluation
Real proof comes from live traffic.

#### A/B Testing Metrics
- likes per impression,
- saves per impression,
- shares per impression,
- comments per impression,
- dwell time,
- session duration,
- retention,
- number of posts consumed,
- hide or report rate.

#### Guardrail Metrics
Improvements should not damage platform health.
Monitor:
- latency,
- crash rate,
- policy violation exposure,
- diversity score,
- new creator exposure,
- spam exposure,
- negative feedback rate.

A very strong line to say in an interview is:

> I would not ship a ranking model purely on better CTR if it worsens hide rate, policy risk, or latency.

---

## 11. Cold Start Handling

This is a classic ranking problem, so mention it explicitly.

### New User Cold Start
No history exists yet.

Possible solutions:
- onboarding preference selection,
- region and language priors,
- device and signup signals,
- default popular content,
- follow graph bootstrap,
- stronger exploration initially.

### New Post or New Creator Cold Start
No engagement history exists.

Possible solutions:
- content embeddings,
- creator metadata,
- similarity to known posts,
- early engagement velocity,
- exploration budget,
- freshness boosting.

---

## 12. Exploration vs Exploitation

This is another strong ML system design topic.

### Exploitation
Show content that is already known to work well for the user.

### Exploration
Show some new types of content to:
- discover new interests,
- help new creators grow,
- prevent the feed from becoming repetitive.

Possible techniques:
- epsilon-greedy exploration,
- contextual bandits,
- boosted sampling for new creators,
- freshness-based exploration.

Without exploration:
- the feed becomes repetitive,
- new creators never grow,
- user interests do not expand over time.

---

## 13. Data Infrastructure and Feature Store

This is good to mention briefly in interviews.

### Offline Feature Store
Stores:
- historical aggregates,
- embeddings,
- creator metrics,
- user topic profiles.

### Online Feature Store
Stores:
- recent interaction counts,
- current session features,
- last active time,
- fresh engagement counters.

### Important Concern: Training-Serving Skew
The same feature definitions must be used in both training and serving.
Otherwise, the model performs well offline but poorly online.

A very interview-friendly phrase is:

> I would avoid training-serving skew by using shared feature definitions between offline training and online inference.

---

## 14. Model Retraining Strategy

Possible strategies:
- daily full retraining,
- hourly refresh for lightweight models,
- streaming updates for fast-moving features,
- periodic embedding refresh.

### Why Retraining Matters
User interests change quickly.
Content freshness matters a lot.
Creator trends also shift rapidly.

So the system should combine:
- strong historical learning,
- fast online updates,
- periodic model refresh.

---

## 15. Failure Handling and Fallbacks

If the ranking service or feature systems fail, the feed should still work.

Fallback options:
- show chronological feed from followed accounts,
- use cached feed slices,
- use a simpler backup ranking heuristic,
- degrade gracefully by skipping heavy ranking stages.

This improves availability and user experience during failures.

---

## 16. Best Way to Explain This End to End in an Interview

You can explain the design in this order:

1. Start with the goal: build a personalized, low-latency feed.
2. Say you cannot rank millions of posts directly, so you use a **multi-stage pipeline**.
3. Explain candidate generation from follow graph, recommendations, and trending.
4. Explain filtering and feature enrichment.
5. Explain pre-ranking and main ranking.
6. Explain reranking for diversity, freshness, and exploration.
7. Explain logging and training loops.
8. Mention offline metrics, A/B testing, cold start, and fallback strategies.

A concise summary line:

> The feed ranking system is a multi-stage retrieval and ranking architecture where cheap models narrow the search space, expensive models optimize relevance, and reranking balances personalization with diversity, freshness, and business constraints.

---

## 17. Quick Revision Summary

If you want to revise this in 30 seconds, remember this flow:

```text
Retrieve candidates -> Filter -> Build features -> Pre-rank -> Rank -> Re-rank -> Serve -> Log -> Train
```

And remember these major talking points:
- Functional requirements,
- Non-functional requirements,
- Candidate generation,
- Two-tower retrieval,
- Main ranking model,
- Multi-task learning,
- Re-ranking,
- Offline vs online,
- Evaluation,
- Cold start,
- Exploration vs exploitation,
- Feature store,
- Retraining,
- Failure fallback.

---

## 18. Final Interview Closing Line

If the interviewer asks for a final summary, you can say:

> I would design Instagram feed ranking as a multi-stage ML ranking system. First, I would retrieve a manageable candidate set using follow graph signals, recommendation systems, and embedding-based retrieval such as a two-tower model. Then I would filter and enrich candidates with user, creator, post, interaction, and context features. After that, I would use pre-ranking and multi-task ranking models to score engagement likelihood, followed by a reranking layer for diversity, freshness, and exploration. Finally, I would log all user interactions and continuously retrain the models using both offline and online feedback loops.
