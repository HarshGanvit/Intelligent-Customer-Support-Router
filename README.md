# Customer Support Router - Streamlit app

A small web app that reads a customer message and decides who should handle it. One fine-tuned DistilBERT model predicts the **department**, **sentiment** and **urgency** of the message. A rule layer then sends it to one of four tiers, and routine questions are answered automatically using sentence-embedding retrieval.

| Tier | Meaning | Action |
|---|---|---|
| 1 | Critical (fraud, repeated charges, request for a human) | Escalate to a human |
| 2 | Routine request, frustrated customer (or possibly critical) | Priority ticket for a human |
| 3 | Calm, in-domain question | Bot answers with a retrieved reply |
| 4 | Greeting, off-topic text, gibberish | Canned reply |

## Project structure

```
support_router_app/
├── app.py              Streamlit UI (router, model performance, tickets)
├── inference.py        model loading + prediction + retrieval
├── routing.py          routing rules (same file the notebook uses)
├── requirements.txt
├── README.md
└── router_artifacts/   created by the notebook (see below)
```

## Step 1: Get the model files (`router_artifacts`)

The trained model is **not stored in this repository** because the weights and embeddings are large. You create it by running the notebook once.

1. Open `intelligent_customer_support_router_v5.ipynb` in Jupyter, Google Colab or Kaggle. A GPU is strongly recommended (the zero-shot labelling step is slow on CPU).
2. Run all cells from top to bottom. The notebook downloads the Bitext dataset from Hugging Face, so it needs internet access.
3. The last cells write everything to a folder called `router_artifacts/` and also pack it into `router_artifacts.zip`.
4. Download the zip and unzip it **next to `app.py`**:
   * **Colab:** run `from google.colab import files; files.download("router_artifacts.zip")`, or use the file browser on the left.
   * **Kaggle:** the zip appears under *Output* (`/kaggle/working/`).
   * **Local Jupyter:** it is saved in the notebook's working directory.

If you keep the folder somewhere else, you can change the path in the app's sidebar.

## Step 2: Install and run

Python 3.9 or newer is recommended. Using a virtual environment keeps things clean:

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501. If your artifacts live elsewhere, change the path in the sidebar.

## How it works

1. **Classify.** The message goes through the shared DistilBERT encoder. Three heads output probabilities for department (12 classes), sentiment (casual, neutral, frustrated) and urgency (critical, greeting, nonsense, routine).
2. **Route.** Rules are applied in this order:
   1. P(critical) ≥ 0.50 → Tier 1, escalate.
   2. 0.25 ≤ P(critical) < 0.50 → Tier 2, priority ticket.
   3. Greeting or nonsense predicted → Tier 4, canned reply.
   4. P(frustrated) ≥ 0.60 → Tier 2, human follow-up.
   5. Otherwise look up the closest known question. If it is not similar enough → Tier 4. If it is → Tier 3, answer with the stored reply.
3. **Retrieve.** For Tier 3, the message is embedded with `all-MiniLM-L6-v2` and compared with the stored questions (restricted to the predicted department when the classifier is confident). The similarity threshold is calibrated on data (about 0.55 in the reported run).
4. **Ticket.** Tier 1 and Tier 2 cases create a ticket that appears on the tickets page.

The thresholds live in `routing_config.json`, so you can adjust behaviour without retraining.

## Reported performance

From the notebook's evaluation on its own validation split and gold set:

| Measure | Result |
|---|---|
| Department accuracy (validation) | 1.000 |
| Urgency macro-F1 (validation) | 0.998 |
| Sentiment macro-F1 (validation) | 0.979 |
| Tier accuracy (3,000 validation messages) | 0.985 |
| Tier accuracy (34-message gold set) | 1.000 |
| Missed emergencies | 0 |
| Retrieval top-1 intent accuracy | 0.994 |
| Median latency on GPU | about 16 ms |

Your numbers may differ slightly because training is not bit-for-bit repeatable across hardware.
