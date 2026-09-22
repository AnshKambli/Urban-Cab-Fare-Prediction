# Urban Cab Fare Predictor — Streamlit App

A live fare estimator and surge-risk predictor built on the models from the EDA notebook.

## What it does

- **Fare Predictor** — enter trip details (city, distance, traffic, demand, etc.) and get an
  estimated `Final_Fare` from the Random Forest regressor (plus a Linear Regression estimate for comparison).
- **Surge Predictor** — enter current conditions and get a probability that the ride will hit
  high surge pricing (≥ 1.5×), from the Random Forest classifier.
- **Data Insights** — key charts from the training data (fare distribution, city/time patterns, correlations).
- **Model Performance** — held-out test metrics and feature importance for both models.

Models are trained **on app startup** from `UrbanCabFare.csv` and cached (`st.cache_resource`), so
the first load takes a couple of seconds and every prediction after that is instant. Upload a new
CSV from the sidebar at any time to retrain on fresh data.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Place `UrbanCabFare.csv` in the same folder as `app.py`, or upload it from the sidebar when the
app opens — either works.

## Deploy for real-world use

Any of these work with zero code changes:

- **Streamlit Community Cloud** (free, easiest): push this folder to a GitHub repo, then deploy
  at [share.streamlit.io](https://share.streamlit.io) pointing at `app.py`. Upload the CSV via the
  sidebar after it's live, or commit it to the repo so it loads automatically.
- **Docker / your own server**: build an image with `requirements.txt` installed, `COPY` this
  folder in, and run `streamlit run app.py --server.port 8501 --server.address 0.0.0.0`.
- **Hugging Face Spaces**: create a Streamlit Space, push these files — it builds automatically.

## Keeping it accurate in production

- **Retrain regularly.** The app retrains automatically on whatever CSV is loaded — schedule a
  job to refresh `UrbanCabFare.csv` with recent rides (e.g. weekly) so the models don't go stale.
- **Watch the surge model's real-world accuracy.** The near-perfect scores on the sample dataset
  come from `Surge_Multiplier` following a near-fixed rule based on `Demand_Supply_Ratio` there.
  Live data will likely be noisier — monitor precision/recall on real predictions and expect
  (and plan for) a drop from the numbers shown in this app.
- **Log predictions vs. actual outcomes** if you connect this to real traffic, so you can retrain
  on ground truth rather than the original synthetic sample over time.

## Files

| File | Purpose |
|---|---|
| `app.py` | The Streamlit app (all pages, training, and prediction logic) |
| `requirements.txt` | Python dependencies |
| `UrbanCabFare.csv` | Training data (add your own copy here, or upload via the sidebar) |
