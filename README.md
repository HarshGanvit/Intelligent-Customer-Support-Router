# Customer Support Router - Streamlit app

```
support_router_app/
├── app.py              Streamlit UI (router, model performance, tickets)
├── inference.py        model loading + prediction + retrieval
├── routing.py          routing rules (same file the notebook uses)
├── requirements.txt
└── router_artifacts/   weights, labels, thresholds, embeddings, replies, metrics.json, figures
```

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
Open http://localhost:8501. If the artifacts live elsewhere, change the path in the sidebar.

## Notes
* Everything runs offline if `router_artifacts/retrieval_model/` is present (the notebook saves it).
* CPU is fine: a message takes well under a second.
* On Colab/Kaggle: `pip install streamlit pyngrok`, then expose port 8501 with ngrok or localtunnel.
