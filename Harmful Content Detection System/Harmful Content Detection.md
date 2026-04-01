#  ML System Design: Harmful Content / Hate Speech Detection

---

##  Table of Contents

1. [What Are We Building?](#what-are-we-building)
2. [Functional Requirements](#functional-requirements)
3. [Non-Functional Requirements](#non-functional-requirements)
4. [ML Problem Framing](#ml-problem-framing)
5. [Data](#data)
6. [Feature Engineering](#feature-engineering)
7. [Model Design](#model-design)
8. [System Architecture](#system-architecture)
9. [Offline Metrics](#offline-metrics)
10. [Online Metrics](#online-metrics)
11. [Feedback Loop](#feedback-loop)
12. [Extensibility](#extensibility)
13. [Key Concepts Glossary](#key-concepts-glossary)

---

## What Are We Building?

A system that **automatically detects harmful, hateful, or abusive content** posted on a social media platform (think Twitter/Meta/Instagram scale) and decides what action to take — allow it, send it for human review, or remove it automatically.

**Why is this hard?**
- Hate speech is subjective and contextual
- Bad actors deliberately obfuscate their language (h@te, coded slang)
- Platform has 100M+ posts/day — can't have humans review everything
- Content comes in 100+ languages
- New harmful patterns emerge every week

---

## Functional Requirements

These are the **core things the system must do.**

| Requirement | Details |
|---|---|
| **Detect harmful text** | Classify posts/comments as harmful or not harmful |
| **Real-time detection** | Check content before it goes live on the platform |
| **Three-level action** | Allow / Send to human review / Auto-remove |
| **Multilingual support** | Handle content in 100+ languages |
| **Severity scoring** | Output a probability score, not just a binary label |
| **Multi-category detection** | Identify type: hate speech, violence, self-harm, sexual content, misinformation |

---

## Non-Functional Requirements

These are the **qualities the system must have** — performance, reliability, scale.

| Requirement | Target |
|---|---|
| **Latency** | < 100ms end-to-end (real-time before post goes live) |
| **Throughput** | 100 million posts/day (~1,200 posts/second) |
| **Availability** | 99.99% uptime — content moderation can't go down |
| **Scalability** | Horizontally scalable — handle traffic spikes (viral events, elections) |
| **Accuracy** | High recall — missing harmful content is worse than over-flagging |
| **Fairness** | Must not disproportionately flag content from specific dialects or communities |
| **Explainability** | Moderators need to understand why content was flagged |

---

## ML Problem Framing

**Task type:** Binary Classification  
*(Can be extended to multi-label classification for categories)*

```
Input  → Text of a post or comment
Output → P(harmful) — a probability score between 0 and 1
```

**Based on the score, a threshold decision engine decides:**

```
Score < 0.3   →  Allow (post goes live)
Score 0.3–0.7 →  Send to human review queue
Score > 0.7   →  Auto-remove
```

> These thresholds are tunable based on business policy — not fixed ML decisions.

---

## Data

### Where Does Training Data Come From?

| Source | What it is | Quality |
|---|---|---|
| **Human-labeled data** | Content moderation teams manually label posts as harmful/not harmful |  High quality |
| **User reports** | When users report a post, that's a label signal | ⚠️ Noisy — users misreport |
| **Appealed removals** | When a user appeals a wrong removal, that's a confirmed false positive |  Gold label |
| **Public datasets** | Jigsaw/Google Civil Comments, HatEval, OffComms | ✅ Good for bootstrapping |
| **Synthetic data** | Paraphrasing or augmenting harmful examples to generate more training samples |  Use carefully |

---

### Data Challenges

#### 1. Class Imbalance
Only ~1–2% of posts are actually harmful. If you train naively, the model just learns to predict "not harmful" for everything and gets 98% accuracy — but it's useless.

**Fix:**
- Oversample harmful examples (SMOTE or simply duplicate minority class)
- Use **weighted loss** — tell the model that missing a harmful post costs more than falsely flagging a clean one

---

#### 2. Label Subjectivity (Why Soft Labels Exist)
Is this post harmful? Ask 5 people and you might get 3 "yes" and 2 "no." If you force a hard 0 or 1, you throw away that disagreement signal.

**Hard Label:** 4/5 annotators say harmful → label = 1  
**Soft Label:** 4/5 annotators say harmful → label = 0.8

The model now learns this example is *probably* harmful, but not certain. This leads to better-calibrated scores and correctly routes these cases to human review instead of auto-removal.

---

#### 3. Adversarial Users
Bad actors know the rules. They'll write:
- `h@te` instead of `hate`
- Use coded language and dog whistles
- Embed meaning in memes (image + text)

**Fix:** Use contextual models (transformers) that understand meaning, not just exact word matches.

---

#### 4. Label Drift
New slang, new coded language, new memes emerge every week. A model trained 6 months ago won't know what a new harmful phrase means.

**Fix:**
- Weekly/monthly retraining with fresh data
- Active learning on new uncertain examples
- Monitor for distribution shift using PSI (explained in glossary)

---

## Feature Engineering

Since we use a transformer model, raw text is the primary input — transformers do their own internal feature learning. But we add **contextual signals** on top:

| Feature | Why It Helps |
|---|---|
| **Raw text** | Core content signal |
| **User history** | Has this user posted harmful content before? Raises prior probability |
| **Account age** | New accounts are more likely to be trolls or bots |
| **Follower/following ratio** | Abnormal ratios indicate bot accounts |
| **Report rate on past posts** | High past reports → higher suspicion |
| **Hashtags / @mentions** | Targeting a specific person with slurs is more serious |

> Example: "I'll kill you" from a friend in a gaming context vs. a stranger targeting a public figure — same words, very different harm level. User signals help disambiguate.

---

## Model Design

### Option Comparison

| Model | Pros | Cons |
|---|---|---|
| Logistic Regression + TF-IDF | Extremely fast, interpretable | No context understanding, broken by obfuscation |
| fastText | Fast, handles some word variation | Still bag-of-words, no deep semantics |
| BiLSTM | Understands word order | Weaker than transformers, slower to train |
| BERT / RoBERTa | Strong context understanding | ~50ms/post, too expensive at 100M posts/day |
| **XLM-R** | Same as BERT but multilingual (100 languages) | Even heavier than BERT |

---

### Recommended: Two-Stage Pipeline

The key insight: **95% of posts are clean.** Don't run your expensive model on all of them.

```
Every Post Comes In
        │
        ▼
┌─────────────────────────────┐
│  Stage 1: DistilBERT        │  ← Fast, ~5ms, runs on 100% of posts
│  (Lightweight Classifier)   │
└─────────────────────────────┘
        │
   Score < 0.2?
        │
        ├── YES → Allow immediately (95% of posts end here)
        │
        └── NO  →
                 ▼
        ┌─────────────────────────────┐
        │  Stage 2: XLM-R             │  ← Accurate, ~50ms, runs on ~5% of posts
        │  (Full Transformer)         │
        └─────────────────────────────┘
                 │
            ┌────┴─────────┐
         Score           Score           Score
         < 0.3           0.3–0.7         > 0.7
           │               │               │
           ▼               ▼               ▼
         Allow      Human Review     Auto-Remove
```

**Why two-stage?**  
Running XLM-R on 100M posts/day would cost millions in GPU compute. Two-stage gives you accuracy where it matters and speed everywhere else.

---

### What is XLM-R?

XLM-R (Cross-Lingual Model RoBERTa) is a transformer model trained on **2.5 terabytes of text across 100 languages**.

- **BERT** = great at understanding context in English
- **XLM** = BERT but trained on many languages at once
- **XLM-R** = XLM + better training settings from RoBERTa

**The key power:** It learns a shared understanding across languages. If it sees hate speech patterns in English, it can recognise similar patterns in Hindi or Arabic — even with little labeled data in those languages. This is called **zero-shot cross-lingual transfer**.

**For us:** One model handles all 100+ languages instead of maintaining 100 separate models.

---

### Training Details

| Setting | Choice |
|---|---|
| **Base model** | Pretrained XLM-R from HuggingFace |
| **Fine-tuning** | On our labeled harmful content dataset |
| **Loss function** | Binary cross-entropy with class weights (for imbalance) |
| **Optimizer** | AdamW with warmup scheduler |
| **Hard negative mining** | Focus extra training on borderline examples the model gets wrong |

---

## System Architecture

```
User submits post
       │
       ▼
API Gateway
       │
       ▼
Content Moderation Service
       │
       ├── Text Preprocessing (clean, tokenize, normalize)
       │
       ▼
Stage 1 Model Server (TorchServe / Triton)
       │
  if borderline
       ▼
Stage 2 Model Server
       │
       ▼
Decision Engine (applies thresholds)
       │
   ┌───┼────────────┐
   ▼   ▼            ▼
Allow  Human     Auto-Remove
      Review
      Queue
       │
       ▼
Human Reviewer Dashboard
       │
  decision fed back
       ▼
Training Data Pipeline
```

**Key infrastructure points:**
- Model servers are **horizontally scalable** — add more instances during traffic spikes
- **Caching:** Near-duplicate posts (same text reposted) can be cached using MinHash/SimHash — don't run inference twice
- **Async processing:** For non-real-time checks (e.g. post-publish scanning), use a message queue (Kafka) to decouple ingestion from processing

---

## Offline Metrics

These are metrics you compute **before deployment**, on a held-out test set.

---

### Precision
**"Of all posts my model flagged as harmful, how many were actually harmful?"**

```
Precision = True Positives / (True Positives + False Positives)
```

- Low precision = wrongly removing good content = users angry, trust broken
- High precision = when we flag something, we're usually right

---

### Recall
**"Of all actually harmful posts, how many did my model catch?"**

```
Recall = True Positives / (True Positives + False Negatives)
```

- Low recall = harmful posts slip through and stay on the platform
- High recall = we catch most of the bad stuff

---

### The Precision-Recall Tradeoff
You can't maximize both at the same time. Raising your threshold increases precision but lowers recall. Lowering your threshold increases recall but lowers precision.

**For harmful content: we prioritize recall over precision.**  
Reason: Letting harmful content stay up causes real-world harm. Wrongly flagging a borderline post just sends it to human review — it doesn't auto-remove it.

---

### F-beta Score
F1 is the harmonic mean of precision and recall — equal weight.  
F-beta lets you weight recall more by setting β > 1.

```
F-beta = (1 + β²) × (Precision × Recall) / (β² × Precision + Recall)
```

For hate speech detection: use **β = 2** (recall twice as important as precision).

---

### AUC-ROC
**"How well does the model separate harmful from non-harmful posts, regardless of threshold?"**

- Score ranges from 0.5 (random) to 1.0 (perfect)
- Threshold-independent — useful for comparing two models overall
- If Model A has AUC 0.95 and Model B has 0.88, Model A is better overall even before you pick a threshold

---

### Why Not Just Use Accuracy?
With 1% harmful content:
- A model that always predicts "not harmful" gets **99% accuracy**
- But it catches **zero** harmful posts — completely useless
- Accuracy is meaningless with class imbalance

---

### Calibration
**"Does a score of 0.8 actually mean 80% likely to be harmful?"**

If yes, the model is well-calibrated. This matters because our threshold decisions (0.3, 0.7) only make sense if we trust the scores. A poorly calibrated model might output 0.9 for things it's actually 50/50 on.

---

## Online Metrics

These are metrics you measure **after deployment**, in production, on real traffic.

---

### Appeal Rate
When we remove a post, users can appeal. High appeal rate = our precision is low = we're removing too much legitimate content.

**What to do:** If appeal rate spikes, raise the auto-remove threshold.

---

### Harmful Content Slip Rate
Posts that were allowed but later confirmed harmful (via user reports + human review).  
This is your real-world recall signal — how much are we missing?

---

### Human Reviewer Agreement Rate
Of posts we sent to the human review queue, what % did reviewers agree were harmful?

- High agreement = Stage 1 threshold is well-calibrated
- Low agreement = Stage 1 is too aggressive, flooding reviewers with false positives (wastes their time)

---

### Latency (P99)
99th percentile response time. Even if average latency is 30ms, if 1% of posts take 5 seconds, users notice — the post is held up.

Target: P99 < 100ms.

---

### Prevalence of Harmful Content Over Time
What % of all posts are being flagged week over week?

- Sudden spike → new adversarial pattern or troll campaign emerged
- Sudden drop → model became too aggressive (or too lenient)
- Gradual rise → model is drifting, needs retraining

---

## Feedback Loop

The model should continuously improve from production data.

```
Production
    │
    ├── User appeals confirmed false positives → add to training data
    │
    ├── Human reviewer decisions → weekly batch added to training set
    │
    ├── Active learning → prioritize labeling posts with score near 0.5
    │   (model is most uncertain → most valuable to label)
    │
    ├── Scheduled retraining → weekly/monthly with fresh data
    │
    └── Drift monitoring → if PSI spikes, trigger emergency retraining
```

**Active Learning explained simply:**  
Instead of randomly labeling new posts, you ask human labelers to label the posts the model is most confused about (score near 0.5). These are the most informative examples — labeling them improves the model faster than labeling easy ones.

---

## Extensibility

Things you can add after the core system is built:

| Extension | Why |
|---|---|
| **Multimodal (images + text)** | Memes are a major vector for hate speech — need to understand image + caption together. Use CLIP (vision-language model). |
| **Multi-label classification** | Instead of just harmful/not harmful, classify into: hate speech / violence / self-harm / sexual content / misinformation |
| **On-device moderation** | For private messages (end-to-end encrypted) — run a small model on the user's device so you never see message content on server |
| **Context-aware detection** | Same slur can be harmful from a stranger, reclaimed language from within a community — account for who is saying it to whom |
| **Shadow mode deployment** | Run new model in parallel with old one, compare decisions before switching over |
| **Fairness monitoring** | Track false positive rates by language, dialect, region — ensure the model isn't disproportionately silencing specific communities |

---

## Key Concepts Glossary

| Term | Simple Explanation |
|---|---|
| **Binary Classification** | Model outputs one of two answers: harmful or not harmful |
| **Multi-label Classification** | Model can output multiple categories at once: hate speech AND violence |
| **Class Imbalance** | When one category (harmful) is much rarer than the other (clean). Makes training tricky. |
| **Weighted Loss** | Telling the model during training that getting harmful posts wrong costs more than getting clean posts wrong |
| **Soft Labels** | Instead of 0 or 1, use 0.8 to reflect that 4/5 annotators agreed. Captures uncertainty. |
| **Hard Labels** | Traditional 0 or 1 — either harmful or not, no in-between |
| **Active Learning** | Prioritize labeling the examples the model is most uncertain about — most efficient use of human labeling effort |
| **Label Drift** | When the meaning of "harmful" shifts over time as new slang/patterns emerge |
| **PSI (Population Stability Index)** | Measures how much the distribution of incoming data has changed. High PSI = model needs retraining. |
| **Calibration** | Whether the model's confidence scores are trustworthy. Score of 0.7 should mean ~70% likely harmful. |
| **AUC-ROC** | How well the model separates classes across all thresholds. 0.5 = random, 1.0 = perfect. |
| **F-beta Score** | Weighted combination of precision and recall. β > 1 weights recall higher. |
| **Hard Negative Mining** | During training, focus extra attention on difficult borderline examples the model keeps getting wrong |
| **Two-Stage Pipeline** | Use a fast cheap model first to filter obvious cases, then run the slow expensive model only on uncertain ones |
| **XLM-R** | Transformer model trained on 100 languages. One model handles all languages via shared multilingual understanding. |
| **Zero-shot Cross-lingual Transfer** | Model trained on English hate speech can detect hate speech in Hindi without Hindi training data |
| **DistilBERT** | A smaller, faster version of BERT — 60% smaller, 97% of BERT's accuracy. Good for Stage 1. |
| **TorchServe / Triton** | Model serving frameworks — they host your model and handle incoming inference requests at scale |
| **MinHash / SimHash** | Algorithms that detect near-duplicate text — used for caching results of similar posts |
| **Shadow Mode** | Running a new model in parallel with the old one to compare decisions before making it live |
| **Appeal Rate** | % of removed posts that users appeal — proxy for false positive rate in production |
| **Slip Rate** | % of allowed posts later confirmed as harmful — proxy for false negative rate in production |
| **AdamW** | Optimizer used during model training — an improved version of Adam with weight decay |
| **Warmup Scheduler** | Gradually increases learning rate at the start of training to stabilize early learning |

---

## Summary: Full System at a Glance

```
┌─────────────────────────────────────────────────────────────────┐
│                    HARMFUL CONTENT DETECTION                    │
├─────────────────────────────────────────────────────────────────┤
│  INPUT        │ Text post/comment + user context signals        │
│  OUTPUT       │ P(harmful) score → Allow / Review / Remove      │
│  SCALE        │ 100M posts/day, <100ms latency, 100+ languages  │
├─────────────────────────────────────────────────────────────────┤
│  DATA         │ Human labels + user reports + appeals           │
│  CHALLENGES   │ Imbalance, subjectivity, adversarial, drift     │
├─────────────────────────────────────────────────────────────────┤
│  MODEL        │ Two-stage: DistilBERT → XLM-R                   │
│  TRAINING     │ Weighted loss + hard negative mining            │
├─────────────────────────────────────────────────────────────────┤
│  OFFLINE      │ Precision, Recall, F-beta (β=2), AUC, Calib.   │
│  ONLINE       │ Appeal rate, slip rate, reviewer agreement, P99 │
├─────────────────────────────────────────────────────────────────┤
│  FEEDBACK     │ Appeals + reviewer decisions → retraining       │
│  MONITORING   │ PSI drift detection, prevalence tracking        │
└─────────────────────────────────────────────────────────────────┘
```

---
