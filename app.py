import os, time, threading, csv
import pandas as pd
import numpy as np
from flask import Flask
import plotly.express as px
import plotly.io as pio
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# ---------- config ----------
FEATURES = ["Income","CCAvg","Mortgage","SecuritiesAccount"]
TARGET   = "PersonalLoan"
STREAM_INTERVAL = float(os.environ.get("STREAM_INTERVAL", "0.5"))  # seconds
CSV_PATH = "scored_events.csv"

# ---------- data & model ----------
def load_data():
    # Prefer CSV in repo: data/training-testing-data.csv or root CSV
    for path in ["data/training-testing-data.csv", "training-testing-data.csv"]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            break
    else:
        # fallback: synthetic data that mimics the IBM columns
        n = 5000
        rng = np.random.default_rng(7)
        df = pd.DataFrame({
            "Income": rng.integers(10, 200, size=n).astype(float),
            "CCAvg":  rng.uniform(0.0, 8.0, size=n),
            "Mortgage": rng.integers(0, 300, size=n).astype(float),
            "SecuritiesAccount": rng.integers(0, 2, size=n).astype(float),
        })
        # simple rule to create a target with ~imbalanced classes
        z = 0.008*df["Income"] + 0.35*df["CCAvg"] + 0.002*df["Mortgage"] + 0.6*df["SecuritiesAccount"] - 3.8
        p = 1/(1+np.exp(-z))
        df["PersonalLoan"] = (rng.uniform(0,1,size=n) < p).astype(int)

    # column cleanups if needed
    rename_map = {
        "Personal Loan": "PersonalLoan",
        "Securities Account": "SecuritiesAccount",
        "ZIP Code": "ZIPCode",
    }
    df = df.rename(columns=rename_map)

    # keep only what we need
    need = FEATURES + [TARGET]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    for c in need:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=need).reset_index(drop=True)
    return df

df = load_data()
X = df[FEATURES].copy()
y = df[TARGET].astype(int).copy()
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
MODEL = GaussianNB().fit(Xtr, ytr)
ACCURACY = float(accuracy_score(yte, MODEL.predict(Xte)))

# prepare scored CSV
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        csv.writer(f).writerow(["timestamp"] + FEATURES + ["probability","prediction"])

# ---------- background streamer ----------
_streamer_started = False
def stream_loop():
    while True:
        row = df.sample(1).iloc[0]
        x = [[float(row[f]) for f in FEATURES]]
        proba = float(MODEL.predict_proba(x)[0][1])
        pred  = int(MODEL.predict(x)[0])
        ts = int(time.time())
        with open(CSV_PATH, "a", newline="") as f:
            csv.writer(f).writerow([ts] + x[0] + [proba, pred])
        time.sleep(STREAM_INTERVAL)

# ---------- dashboard ----------
def build_dashboard_html():
    if not os.path.exists(CSV_PATH):
        return "<h2>No scored_events.csv yet.</h2>"
    s = pd.read_csv(CSV_PATH)
    if s.empty:
        return "<h2>Waiting for events… refresh in a few seconds.</h2>"
    s["datetime"] = pd.to_datetime(s["timestamp"], unit="s")

    # Table (last 20)
    table_html = s.tail(20).to_html(index=False)

    figs = []

    # 1) predictor "importance" proxy (mean-by-class)
    pos, neg = s[s["prediction"]==1], s[s["prediction"]==0]
    if not pos.empty and not neg.empty:
        comp = pd.DataFrame({
            "feature": FEATURES*2,
            "mean_value": list(pos[FEATURES].mean().values) + list(neg[FEATURES].mean().values),
            "class": ["Positive"]*len(FEATURES) + ["Negative"]*len(FEATURES)
        })
        fig = px.bar(comp, x="feature", y="mean_value", color="class",
                     barmode="group", title="Predictor Importance (proxy)")
        figs.append(pio.to_html(fig, include_plotlyjs="cdn", full_html=False))

    # 2) positives per 5s
    by_5s = s.set_index("datetime")["prediction"].resample("5S").sum().reset_index(name="positives")
    fig = px.line(by_5s, x="datetime", y="positives",
                  title="People who will actually take loan (per 5s)")
    figs.append(pio.to_html(fig, include_plotlyjs="cdn", full_html=False))

    # 3) probability trend
    fig = px.line(s, x="datetime", y="probability",
                  title="Prediction Probability Trend (streaming over time)")
    figs.append(pio.to_html(fig, include_plotlyjs="cdn", full_html=False))

    # 4) class counts
    cc = (s["prediction"].map({0:"No",1:"Yes"})
            .value_counts().rename_axis("class").reset_index(name="count"))
    fig = px.bar(cc, x="class", y="count", title="Prediction count (Yes / No)")
    figs.append(pio.to_html(fig, include_plotlyjs="cdn", full_html=False))

    yes = int((s["prediction"]==1).sum())
    no  = int((s["prediction"]==0).sum())
    total = len(s)

    return f"""
    <html><head>
      <meta charset="utf-8"/>
      <meta http-equiv="refresh" content="10">
      <title>Live Streaming Dashboard</title>
      <style>
        body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 24px; }}
        .grid {{ display: grid; grid-template-columns: 1fr; gap: 24px; }}
        .card {{ padding: 16px; border: 1px solid #eee; border-radius: 12px; box-shadow: 0 1px 4px rgba(0,0,0,.06); }}
        h1 {{ margin: 0 0 8px; }}
      </style>
    </head><body>
      <h1>Live Streaming Dashboard</h1>
      <div style="margin:10px 0; padding:10px; border:1px solid #eee; border-radius:8px;">
        <b>Accuracy:</b> {ACCURACY:.3f}
        &nbsp;•&nbsp; <b>Yes:</b> {yes} / {total}
        &nbsp;•&nbsp; <b>No:</b> {no} / {total}
      </div>
      <p>Auto-refreshes every 10 seconds. Source: <code>scored_events.csv</code></p>
      <div class="grid">
        <div class="card"><h2>Incoming / Processed (last 20 rows)</h2>{table_html}</div>
        <div class="card">{figs[0] if len(figs)>0 else ""}</div>
        <div class="card">{figs[1] if len(figs)>1 else ""}</div>
        <div class="card">{figs[2] if len(figs)>2 else ""}</div>
        <div class="card">{figs[3] if len(figs)>3 else ""}</div>
      </div>
    </body></html>
    """

app = Flask(__name__)

@app.get("/")
def root():
    return '<meta http-equiv="refresh" content="0; url=/dashboard">'

@app.get("/dashboard")
def dashboard():
    global _streamer_started
    # start background streamer once (single worker only)
    if not _streamer_started:
        t = threading.Thread(target=stream_loop, daemon=True)
        t.start()
        _streamer_started = True
    return build_dashboard_html()

@app.get("/healthz")
def healthz():
    return "ok", 200

# Dev run (Render will use gunicorn)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
