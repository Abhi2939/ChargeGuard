# ChargeGuard

A calibrated dispute agent that predicts whether a merchant will win a
chargeback dispute, with a trustworthy, calibrated confidence score — and
decides whether to auto-recommend fighting it, dropping it, or escalating to
human review.

Built for the Razorpay AI Buildathon 2026, Track 2 (AI Risk Manager).

## The problem

When a customer disputes a payment, the issuing bank reverses the money
immediately — before the merchant gets to respond. The merchant then has a
short window to submit evidence ("representment") and fight the chargeback.
Deciding whether a given dispute is worth fighting is currently a manual,
inconsistent judgment call, and getting it wrong is costly in both directions:
fighting a losing case wastes staff effort, and dropping a winnable one loses
real revenue.

ChargeGuard automates that decision — but the core design problem isn't
"predict who wins." It's "produce a confidence score that can actually be
trusted enough to act on automatically." A model that says "90% confident"
should be right about 90% of the time, or auto-acting on that number is
dangerous. That's what calibration solves, and it's the centerpiece of this
project.

## What it does

Given a dispute case (reason code, delivery status, evidence signals, customer
history), ChargeGuard:

1. Predicts the probability the merchant would win the dispute
2. Calibrates that probability so it reflects real-world frequency, not just
   raw model confidence
3. Explains which factors drove the prediction
4. Recommends fight / drop / human review based on fixed, auditable thresholds
5. Generates a plain-language explanation of the recommendation
6. Logs the full decision and reasoning to an audit trail

## Architecture

```
Case input
    |
    v
Ingestion Agent        validates and encodes the case
    |
    v
Prediction Agent        XGBoost classifier -> Platt-scaled calibration
    |                    -> SHAP explainability layer
    v
Decision Agent           deterministic threshold branching (fight / drop / review)
    |                    -> LLM call for plain-language narration only
    v
Audit Log                every decision logged with full reasoning
```

The pipeline is orchestrated with LangGraph and exposed two ways: a scripted
console demo (`demo/run_demo.py`) and a FastAPI backend with a Streamlit
frontend on top.

An important design decision: the fight/drop/review threshold logic is
**deterministic code**, not an LLM decision. An LLM's own confidence in
generated text is not a calibrated probability, so letting it make or override
the threshold call would undo the entire point of the calibration layer. The
LLM here is used only to turn an already-made decision into readable prose —
never to decide.

## Key results

The model was trained on a synthetic dataset of 6,000 dispute cases, where
each reason code (fraud, item not received, not as described, duplicate
charge, subscription not cancelled) is driven by a different, realistic
subset of evidence features, with noise layered on top of the label so the
task isn't trivially easy.

**Calibration method:** two approaches were tested and compared, not just one
applied blindly:

- Isotonic regression was tried first and rejected — with a calibration set
  of only 1,200 rows, it produced unstable exact-1.0 predictions and actually
  worsened log loss.
- Platt scaling (sigmoid) was adopted instead, giving a real, stable
  improvement on both metrics:
  - Brier score: 0.2322 -> 0.2309
  - Log loss: 0.6564 -> 0.6536

The raw model showed a mild S-shaped miscalibration — underconfident on weak
cases, overconfident on strong ones. Platt scaling compressed the probability
range and pulled predictions measurably closer to the diagonal on the
reliability diagram (see `reports/reliability_before_after.png`).

**False-positive cost analysis:** precision/recall and a threshold sweep were
run against an explicit cost model (flat cost per wrongly-fought case vs. the
real order value of missed winnable cases). This surfaced a genuine tension —
under this cost model, a lower threshold minimizes total cost, while a higher
threshold maximizes precision. ChargeGuard's thresholds (fight above 75%,
drop below 40%) are set to control precision and avoid recommending fights
the merchant will likely lose, not to minimize raw cost — a deliberate,
explainable tradeoff rather than an unexamined default.

## Project structure

```
data/
  generate_dataset.py       synthetic dataset generator
  raw/dispute_dataset.csv

notebooks/
  calibration_analysis.ipynb   exploratory training/calibration/explainability work

src/
  model/
    train.py                 trains XGBoost, calibrates with Platt scaling, saves artifacts
    explain.py                SHAP-based explainability layer
    evaluate.py                reliability diagrams, precision/recall, cost analysis
    artifacts/                 saved model files
  agents/
    ingestion.py               case validation and feature encoding
    prediction.py               calibrated probability + explanation
    decision.py                  threshold logic + LLM narration
    orchestrator.py               LangGraph pipeline wiring
  utils/
    audit_log.py                 JSON Lines audit trail

demo/
  sample_cases.json             three curated cases: strong, weak, borderline
  run_demo.py                    console walkthrough of the full pipeline

reports/
  audit_log.jsonl                generated at runtime
  reliability_before_after.png   generated by evaluate.py

main.py                          FastAPI backend
streamlit_app.py                 Streamlit frontend
requirements.txt
.env.example
```

## Running it

**Install dependencies:**

```
pip install -r requirements.txt
```

**Train the model** (or skip this — trained artifacts are already included):

```
python src/model/train.py
```

**Run the console demo** (fastest way to see the full pipeline work):

```
python demo/run_demo.py
```

**Run the full API + UI stack:**

```
uvicorn main:app --reload
```

In a second terminal:

```
streamlit run streamlit_app.py
```

Then open the Streamlit URL shown in the terminal and submit a case through
the form. Check "Show recent audit trail" to see logged decisions live.

**API endpoints** (FastAPI, `http://localhost:8000`):

- `GET /` — health check
- `POST /evaluate-case` — runs a case through the full pipeline, returns the
  calibrated probability, explanation, decision, and narration
- `GET /audit-log?limit=N` — returns the N most recent logged decisions

Interactive API docs are available at `http://localhost:8000/docs`.

**Environment variables** (`.env`):

```
GROQ_API_KEY=your_key_here
```

Without a Groq key, the narration step falls back to a template-based
explanation rather than failing — the fight/drop/review decision itself
never depends on the LLM call succeeding.

## Honest scope and limitations

- The dataset is synthetic, generated with realistic but deliberately
  imperfect rules — it is not validated against real merchant data.
- ChargeGuard is scoped to a limited set of merchant/product categories'
  worth of behavior patterns baked into the generator, not a general-purpose
  model across all dispute types.
- The recommendation is a first-response suggestion, not a final word —
  real disputes can go through pre-arbitration and arbitration rounds beyond
  the first response, which this system does not model.
- Calibration was validated on synthetic data with a 1,200-row calibration
  set; a production system would need meaningfully more data before the
  calibrated probabilities could be trusted at the same level.
