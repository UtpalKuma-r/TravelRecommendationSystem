# app.py
# Streamlit Travel Destination Recommender (Traveler Trip Dataset)
# Dataset: https://www.kaggle.com/datasets/rkiattisak/traveler-trip-data
# UI inputs: preferred duration (days), traveler age, gender (+ optional month)
# Recommends destinations; users can like/dislike; likes update a multinomial model online; dislikes apply a soft penalty.

import os
import json
import pickle
from datetime import datetime, date
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Travel Recommender (Traveler Trip Dataset)", layout="wide")

import tempfile
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(tempfile.gettempdir(), "trip_model"))
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "clf.pkl")
ENC_PATH = os.path.join(MODEL_DIR, "enc.pkl")
LBL_PATH = os.path.join(MODEL_DIR, "lbl.pkl")
PENALTY_PATH = os.path.join(MODEL_DIR, "penalties.json")
FEEDBACK_PATH = os.path.join(MODEL_DIR, "feedback.csv")

# -------------------------
# Helpers
# -------------------------

def infer_columns(df: pd.DataFrame):
    cols = {c.lower(): c for c in df.columns}
    def find(*cands):
        for c in cands:
            if c in cols:
                return cols[c]
        # fuzzy contains
        for k in cols:
            for c in cands:
                if c.replace("_"," ") in k:
                    return cols[k]
        return None

    return {
        "dest": find("destination", "place", "city", "country", "location"),
        "start": find("start_date", "start", "depart", "travel_date", "date"),
        "end": find("end_date", "return", "end"),
        "duration": find("duration", "days", "trip_duration"),
        "age": find("age"),
        "gender": find("gender", "sex"),
        "transport": find("transport", "transportation", "mode"),
        "accom": find("accommodation", "accomodation", "stay"),
        "cost": find("cost", "price", "total_cost", "expense"),
        "month": find("month"),
        "nationality": find("nationality", "country_of_origin")
    }


def ensure_duration(df: pd.DataFrame, c):
    if c["duration"] and c["duration"] in df:
        d = pd.to_numeric(df[c["duration"]], errors="coerce")
    else:
        d = None
    if d is None or d.isna().all():
        # try compute from dates
        if c["start"] and c["end"] and c["start"] in df and c["end"] in df:
            sd = pd.to_datetime(df[c["start"]], errors="coerce")
            ed = pd.to_datetime(df[c["end"]], errors="coerce")
            d = (ed - sd).dt.days
        else:
            d = pd.Series(np.nan, index=df.index)
    df["_duration"] = d.clip(lower=1)


def ensure_month(df: pd.DataFrame, c):
    if c["month"] and c["month"] in df:
        m = pd.to_numeric(df[c["month"]], errors="coerce")
    elif c["start"] and c["start"] in df:
        sd = pd.to_datetime(df[c["start"]], errors="coerce")
        m = sd.dt.month
    else:
        m = pd.Series(np.nan, index=df.index)
    df["_month"] = m


@st.cache_data(show_spinner=False)
def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    cols = infer_columns(df)
    # basic cleaning
    for key in ["dest", "gender", "transport", "accom", "nationality"]:
        if cols.get(key) and cols[key] in df:
            df[cols[key]] = df[cols[key]].astype(str).str.strip()
    ensure_duration(df, cols)
    ensure_month(df, cols)
    # drop rows without destination
    if cols.get("dest") is None or cols["dest"] not in df:
        raise ValueError("Could not detect a destination column. Rename a header to include 'Destination' or 'City'.")
    df = df[pd.notna(df[cols["dest"]])]
    return df.reset_index(drop=True), cols


def build_pipeline(df: pd.DataFrame, cols: dict) -> Tuple[Pipeline, LabelEncoder]:
    # features
    num_features = ["_duration"]
    if cols.get("age") and cols["age"] in df:
        df["_age"] = pd.to_numeric(df[cols["age"]], errors="coerce")
        num_features.append("_age")
    if "_month" in df:
        num_features.append("_month")

    cat_maps = {
        "gender": cols.get("gender"),
        "transport": cols.get("transport"),
        "accom": cols.get("accom"),
        "nationality": cols.get("nationality")
    }
    cat_features = [v for v in cat_maps.values() if v and v in df]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_features),
            ("cat", cat_pipe, cat_features),
        ], remainder="drop"
    )

    clf = SGDClassifier(loss="log_loss", max_iter=10, random_state=42)
    pipe = Pipeline([("pre", pre), ("clf", clf)])

    # labels
    le = LabelEncoder()
    # ensure labels have no NaN
    y_raw = df[cols["dest"]].astype(str)
    mask = y_raw.notna()
    y = le.fit_transform(y_raw[mask])

    X = df.loc[mask, num_features + cat_features]

    # warm start with partial_fit on batches to keep snappy
    # fit partial to establish classes using imputed/transformed features
    Xt = pre.fit_transform(X)
    pipe.named_steps["clf"].partial_fit(
        Xt, y, classes=np.unique(y)
    )
    return pipe, le


def save_model(pipe: Pipeline, le: LabelEncoder):
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipe, f)
    with open(LBL_PATH, "wb") as f:
        pickle.dump(le, f)


def load_model() -> Tuple[Optional[Pipeline], Optional[LabelEncoder]]:
    pipe = le = None
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                pipe = pickle.load(f)
        except Exception:
            pipe = None
    if os.path.exists(LBL_PATH):
        try:
            with open(LBL_PATH, "rb") as f:
                le = pickle.load(f)
        except Exception:
            le = None
    return pipe, le


def read_penalties():
    if os.path.exists(PENALTY_PATH):
        try:
            return json.load(open(PENALTY_PATH, "r"))
        except Exception:
            return {}
    return {}


def write_penalties(p):
    json.dump(p, open(PENALTY_PATH, "w"))


# -------------------------
# Sidebar & inputs
# -------------------------

st.sidebar.header("Data & Inputs")
# Allow multiple ways to provide data: env var, common paths, or upload
csv_env = os.environ.get("CSV_PATH", "")
common_candidates = [
    csv_env,
    csv_env.strip() if csv_env else "",
    "Traveler_trip_dataset.csv",
    "/app/Traveler_trip_dataset.csv",
    "/app/data/Traveler_trip_dataset.csv",
    "data/Traveler_trip_dataset.csv",
]

uploaded = st.sidebar.file_uploader("Or upload the traveler trip CSV", type=["csv"])

def resolve_csv():
    # 1) uploaded file takes precedence
    if uploaded is not None:
        return uploaded, True
    # 2) sidebar path if provided
    p = st.sidebar.text_input(
        "Path to Traveler Trip CSV",
        value=common_candidates[2],
        help="Path inside the container/Space. You can also upload above.",
    )
    if os.path.exists(p):
        return p, False
    # 3) try common candidates
    for cand in common_candidates:
        if cand and os.path.exists(cand):
            return cand, False
    return None, False

csv_source, is_buffer = resolve_csv()
if csv_source is None:
    st.info("Provide the traveler trip CSV via **upload** or a valid **path** in the sidebar to begin. You can also set the CSV_PATH environment variable.")
    st.stop()

with st.spinner("Loading data…"):
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        cols = infer_columns(df)
        ensure_duration(df, cols); ensure_month(df, cols)
        df = df[pd.notna(df[cols["dest"]])].reset_index(drop=True)
    else:
        df, cols = load_data(csv_source)

st.sidebar.caption(f"Detected columns → {cols}")(f"Detected columns → {cols}")

# User inputs
age = st.sidebar.number_input("Traveler age", min_value=5, max_value=100, value=30)
gender = st.sidebar.selectbox("Gender", options=["Not specified", "Male", "Female", "Other"])
trip_days = st.sidebar.slider("Preferred duration (days)", 1, 60, 7)
month = st.sidebar.selectbox("Planned month", list(range(1,13)), index=(date.today().month-1))

k = st.sidebar.slider("How many recommendations?", 3, 20, 6)

# -------------------------
# Model
# -------------------------
pipe, le = load_model()
if pipe is None or le is None:
    pipe, le = build_pipeline(df, cols)
    save_model(pipe, le)

# Build a single-row feature frame from the user input
row = {
    "_duration": trip_days,
    "_age": age,
    "_month": month,
    cols.get("gender", "gender"): gender if gender != "Not specified" else np.nan,
}
# include columns that exist
feat_cols = []
for c in ["_duration", "_age", "_month", cols.get("gender"), cols.get("transport"), cols.get("accom"), cols.get("nationality")]:
    if c and (c in df.columns or c.startswith("_")):
        feat_cols.append(c)
# ensure all columns expected by preprocessor exist, even if missing in this single-row input
X_user = pd.DataFrame([{c: row.get(c, np.nan) for c in feat_cols}])

# Predict destination probabilities
proba = pipe.predict_proba(X_user)[0]
labels = le.inverse_transform(np.arange(len(proba)))

# Apply dislike penalties
penalties = read_penalties()
penalty_vec = np.array([penalties.get(lbl, 0.0) for lbl in labels])
adj_score = proba - 0.2 * penalty_vec  # soft penalty
order = np.argsort(-adj_score)

# -------------------------
# UI
# -------------------------

st.title("✈️ Traveler Trip Recommender")
left, right = st.columns([3,1])

with right:
    st.metric("Destinations", len(np.unique(df[cols["dest"]])))
    if os.path.exists(FEEDBACK_PATH):
        fb_df = pd.read_csv(FEEDBACK_PATH)
        st.metric("Feedback count", len(fb_df))
    else:
        st.metric("Feedback count", 0)

# Aggregate helper for display
# prepare numeric cost if present
if cols.get("cost") and cols["cost"] in df:
    df["_cost_num"] = pd.to_numeric(df[cols["cost"]], errors="coerce")

agg = df.groupby(cols["dest"]).agg(
    avg_days=("_duration", "mean"),
    median_cost=("_cost_num", "median") if ("_cost_num" in df.columns) else ("_duration", "count"),
    common_transport=(cols["transport"], lambda s: s.dropna().value_counts().index[0] if (cols.get("transport") and s.dropna().size>0) else None),
    trips=(cols["dest"], "count")
).reset_index()

shown = 0
for idx in order:
    dest = labels[idx]
    score = adj_score[idx]
    if score <= 0:
        continue
    row_agg = agg[agg[cols["dest"]] == dest]
    with st.container(border=True):
        c1, c2 = st.columns([4,1])
        with c1:
            st.markdown(f"### {dest}")
            if not row_agg.empty:
                days = row_agg.iloc[0].get("avg_days", np.nan)
                trans = row_agg.iloc[0].get("common_transport", None)
                trips = int(row_agg.iloc[0].get("trips", 0))
                cost = row_agg.iloc[0].get("median_cost", np.nan)
                line = []
                if not np.isnan(days):
                    line.append(f"avg {days:.1f} days")
                if trans and str(trans) != "nan":
                    line.append(f"usually via {trans}")
                if cols.get("cost") and not pd.isna(cost):
                    line.append(f"median cost ~ {cost}")
                line.append(f"based on {trips} trips")
                st.caption(" • ".join(line))
        with c2:
            like = st.button("👍 Like", key=f"like_{dest}")
            dislike = st.button("👎 Dislike", key=f"dislike_{dest}")
            if like or dislike:
                y_lbl = dest
                # record feedback
                row_fb = {
                    "dest": y_lbl,
                    "label": int(1 if like else 0),
                    "age": age,
                    "gender": gender,
                    "days": trip_days,
                    "month": month,
                    "ts": datetime.utcnow().isoformat()
                }
                mode = "a" if os.path.exists(FEEDBACK_PATH) else "w"
                pd.DataFrame([row_fb]).to_csv(FEEDBACK_PATH, index=False, mode=mode, header=not os.path.exists(FEEDBACK_PATH))
                if like:
                    # positive online update
                    X_like = X_user
                    y_like = le.transform([y_lbl])
                    pipe.named_steps["clf"].partial_fit(
                        pipe.named_steps["pre"].transform(X_like), y_like, classes=np.arange(len(le.classes_))
                    )
                    save_model(pipe, le)
                    st.toast("Thanks! Model nudged toward this destination.")
                else:
                    # apply penalty
                    penalties[y_lbl] = min(5.0, penalties.get(y_lbl, 0.0) + 1.0)
                    write_penalties(penalties)
                    st.toast("Got it. We'll show this less often.")
    shown += 1
    if shown >= k:
        break

st.divider()
st.markdown(
    """
**How this model recommends**  
- Trains a multinomial logistic regression on the historical trips to predict destination from **age, gender, duration, month** and any available **transport/accommodation/nationality**.  
- Your **Likes** perform online learning (partial_fit) to boost that destination for similar profiles.  
- **Dislikes** apply a soft penalty so the item ranks lower; penalties are stored per destination.  
- All assets are saved under `.trip_model/`.
"""
)
