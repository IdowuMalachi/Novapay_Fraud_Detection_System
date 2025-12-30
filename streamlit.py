\
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import joblib

# Optional: SHAP (loaded only when needed)
try:
    import shap  # noqa: F401
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

# -----------------------------
# Paths (match your repo layout)
# -----------------------------
ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "Models"
DATA_DIR = ROOT / "Data"

st.set_page_config(page_title="NovaPay Fraud Detection", page_icon="🛡️", layout="wide")

# -----------------------------
# Utils
# -----------------------------
def first_existing(paths):
    for p in paths:
        p = Path(p)
        if p.exists():
            return p
    return None

def human_mb(n_bytes: int) -> str:
    return f"{n_bytes / (1024 * 1024):.1f} MB"

@st.cache_resource(show_spinner=False)
def load_core_artifacts():
    """
    FAST load: only the minimum needed for scoring.
    (No SHAP files loaded here.)
    """
    model_path = first_existing([
        MODELS_DIR / "rf_fraud_model.joblib",
        ROOT / "rf_fraud_model.joblib",
    ])
    scaler_path = first_existing([
        MODELS_DIR / "standard_scaler.joblib",
        ROOT / "standard_scaler.joblib",
    ])
    encoders_path = first_existing([
        MODELS_DIR / "label_encoders.joblib",
        ROOT / "label_encoders.joblib",
    ])
    feature_list_path = first_existing([
        MODELS_DIR / "rf_feature_list.json",
        ROOT / "rf_feature_list.json",
    ])

    missing = []
    if model_path is None: missing.append("Models/rf_fraud_model.joblib")
    if scaler_path is None: missing.append("Models/standard_scaler.joblib")
    if encoders_path is None: missing.append("Models/label_encoders.joblib")
    if feature_list_path is None: missing.append("Models/rf_feature_list.json")
    if missing:
        raise FileNotFoundError("Missing required files:\n- " + "\n- ".join(missing))

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    label_encoders = joblib.load(encoders_path)
    with open(feature_list_path, "r", encoding="utf-8") as f:
        feature_list = json.load(f)

    return model, scaler, label_encoders, feature_list

@st.cache_data(show_spinner=False)
def read_csv_cached(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """
    Cache CSV reading by file content (fast re-runs).
    """
    from io import BytesIO
    return pd.read_csv(BytesIO(file_bytes))

@st.cache_data(show_spinner=False)
def read_repo_csv_cached(path_str: str) -> pd.DataFrame:
    return pd.read_csv(Path(path_str))

def safe_label_encode(series: pd.Series, le) -> pd.Series:
    classes = getattr(le, "classes_", None)
    if classes is None:
        return series.astype(str)

    mapping = {cls: i for i, cls in enumerate(classes)}
    s = series.astype(str).fillna("")
    unseen = ~s.isin(mapping)
    if unseen.any():
        s = s.where(~unseen, other=str(classes[0]))
    return s.map(mapping).astype(int)

def prepare_features(df: pd.DataFrame, feature_list, label_encoders, scaler):
    """
    Produce model-ready matrix aligned to feature_list.
    - adds missing cols
    - applies saved label encoders where possible
    - numeric conversion + median fill
    - scales with saved scaler
    """
    X = df.copy()

    # Add missing columns
    for col in feature_list:
        if col not in X.columns:
            X[col] = np.nan

    # Align to expected order
    X = X[feature_list]

    # Apply encoders (only for columns present in encoders dict)
    # NOTE: we do encoding before numeric coercion
    for col, le in label_encoders.items():
        if col in X.columns:
            X[col] = safe_label_encode(X[col], le)

    # Convert to numeric & fill NaNs
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        med = X[col].median()
        if pd.isna(med):
            med = 0
        X[col] = X[col].fillna(med)

    X_scaled = scaler.transform(X.values)
    return X, X_scaled

def score_prob(model, X_scaled: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_scaled)[:, 1]
    if hasattr(model, "decision_function"):
        raw = model.decision_function(X_scaled)
        return 1 / (1 + np.exp(-raw))
    return model.predict(X_scaled).astype(float)

# Lazy SHAP loaders (ONLY when user asks)
@st.cache_resource(show_spinner=False)
def load_precomputed_shap(feature_list):
    """
    Loads precomputed SHAP assets if present.
    Done lazily to keep startup fast.
    """
    shap_values_path = first_existing([
        MODELS_DIR / "shap_values_rf (2).npy",
        MODELS_DIR / "shap_values_rf.npy",
        ROOT / "shap_values_rf (2).npy",
        ROOT / "shap_values_rf.npy",
    ])
    x_test_path = first_existing([
        MODELS_DIR / "X_test_for_shap (2).csv",
        MODELS_DIR / "X_test_for_shap.csv",
        ROOT / "X_test_for_shap (2).csv",
        ROOT / "X_test_for_shap.csv",
    ])

    shap_values = None
    X_test = None

    if shap_values_path is not None:
        shap_values = np.load(shap_values_path)
    if x_test_path is not None:
        X_test = pd.read_csv(x_test_path)

    # Validate shapes if both exist
    if shap_values is not None and X_test is not None:
        if shap_values.shape[1] != len(feature_list):
            # still allow, but warn later
            pass

    return shap_values, X_test, shap_values_path, x_test_path

# -----------------------------
# UI
# -----------------------------
st.title("🛡️ NovaPay Fraud Detection System (Fast)")
st.caption("Optimized for quick startup on Streamlit Cloud. SHAP loads only when requested.")

# Load core artifacts quickly
try:
    with st.spinner("Loading model (fast)..."):
        model, scaler, label_encoders, feature_list = load_core_artifacts()
except Exception as e:
    st.error(str(e))
    st.stop()

# Sidebar
with st.sidebar:
    st.header("Input")
    use_repo_data = st.toggle("Use CSV from /Data", value=False)
    repo_csv = None
    if use_repo_data:
        if DATA_DIR.exists():
            csvs = sorted([p.name for p in DATA_DIR.glob("*.csv")])
            if csvs:
                repo_csv = st.selectbox("Select Data CSV", csvs)
            else:
                st.info("No CSV found in /Data.")
        else:
            st.info("/Data folder not found.")

    uploaded = st.file_uploader("Or upload a CSV", type=["csv"])

    st.divider()
    st.header("Scoring")
    threshold = st.slider("Fraud threshold", 0.05, 0.95, 0.50, 0.01)

    st.divider()
    st.header("Explainability")
    show_explain = st.toggle("Show explanations", value=False)
    show_local_shap = st.toggle("Local SHAP (single row)", value=False, disabled=not HAS_SHAP)

    st.divider()
    st.caption("Expected repo layout:")
    st.code("streamlit.py\nrequirements.txt\nruntime.txt\nModels/\nData/\nNotebooks/", language="text")

# Load input data
df_in = None
source_label = None

if repo_csv:
    try:
        df_in = read_repo_csv_cached(str(DATA_DIR / repo_csv))
        source_label = f"Data/{repo_csv}"
    except Exception as e:
        st.error(f"Failed to read {repo_csv}: {e}")

if df_in is None and uploaded is not None:
    try:
        file_bytes = uploaded.getvalue()
        df_in = read_csv_cached(file_bytes, uploaded.name)
        source_label = f"Uploaded: {uploaded.name} ({human_mb(len(file_bytes))})"
    except Exception as e:
        st.error(f"Failed to read uploaded CSV: {e}")

if df_in is None:
    st.info("Pick a CSV from **/Data** or upload one to start.")
    st.stop()

# Main preview
c1, c2, c3 = st.columns([1.4, 1, 1])
with c1:
    st.subheader("Preview")
    st.write(f"Source: **{source_label}**")
with c2:
    st.subheader("Model")
    st.write("Features:", len(feature_list))
    st.write("Encoders:", len(label_encoders))
with c3:
    st.subheader("Performance tips")
    st.write("Explanations are OFF by default ✅")
    st.write("Turn ON only when needed")

st.dataframe(df_in.head(30), use_container_width=True)

# Score with progress for large datasets
st.subheader("Score transactions")
N = len(df_in)
chunk_size = 20000  # tune for performance
progress = st.progress(0)

probas = []
preds = []

for start in range(0, N, chunk_size):
    end = min(N, start + chunk_size)
    batch = df_in.iloc[start:end].copy()
    X_aligned, X_scaled = prepare_features(batch, feature_list, label_encoders, scaler)
    p = score_prob(model, X_scaled)
    probas.append(p)
    preds.append((p >= threshold).astype(int))
    progress.progress(end / N)

progress.empty()

proba = np.concatenate(probas) if probas else np.array([])
pred = np.concatenate(preds) if preds else np.array([])

out = df_in.copy()
out["fraud_probability"] = proba
out["fraud_flag"] = pred

m1, m2, m3 = st.columns(3)
m1.metric("Rows scored", f"{len(out):,}")
m2.metric("Flagged fraud", f"{int(out['fraud_flag'].sum()):,}")
m3.metric("Flag rate", f"{(out['fraud_flag'].mean() * 100):.2f}%")

st.dataframe(out.sort_values("fraud_probability", ascending=False).head(150), use_container_width=True)

st.download_button(
    "⬇️ Download scored CSV",
    data=out.to_csv(index=False).encode("utf-8"),
    file_name="novapay_scored.csv",
    mime="text/csv",
)

# Explanations (lazy loaded)
if show_explain:
    st.divider()
    st.subheader("Explanations (loaded on demand)")

    left, right = st.columns(2)

    with left:
        st.markdown("### Feature importance (model-based)")
        importances = getattr(model, "feature_importances_", None)
        if importances is None or len(importances) != len(feature_list):
            st.info("Model-based feature importance is not available.")
        else:
            imp_df = (
                pd.DataFrame({"feature": feature_list, "importance": importances})
                .sort_values("importance", ascending=False)
                .reset_index(drop=True)
            )
            st.dataframe(imp_df.head(25), use_container_width=True, height=420)
            st.bar_chart(imp_df.set_index("feature").head(25))

    with right:
        st.markdown("### SHAP summary (precomputed, optional)")
        try:
            shap_values, X_test_for_shap, shap_path, x_path = load_precomputed_shap(feature_list)
            if shap_values is None or X_test_for_shap is None:
                st.info("No precomputed SHAP files found in /Models (optional).")
            else:
                mean_abs = np.mean(np.abs(shap_values), axis=0)
                shap_df = (
                    pd.DataFrame({"feature": feature_list, "mean_abs_shap": mean_abs})
                    .sort_values("mean_abs_shap", ascending=False)
                    .reset_index(drop=True)
                )
                st.caption(f"Loaded: {shap_path.name} and {x_path.name}")
                st.dataframe(shap_df.head(25), use_container_width=True, height=420)
                st.bar_chart(shap_df.set_index("feature").head(25))
        except Exception as e:
            st.warning(f"Could not load precomputed SHAP: {e}")

    # Local SHAP (compute per-row only)
    if show_local_shap:
        if not HAS_SHAP:
            st.warning("SHAP not installed; disable Local SHAP.")
        else:
            st.divider()
            st.subheader("Local SHAP (single row) — compute only when you select")
            idx = st.number_input("Row index", min_value=0, max_value=max(0, len(out) - 1), value=0, step=1)
            if st.button("Compute Local SHAP", type="primary"):
                try:
                    # explain one row only (fast-ish)
                    row_df = out.iloc[[int(idx)]].copy()
                    X_aligned, _ = prepare_features(row_df, feature_list, label_encoders, scaler)

                    explainer = shap.TreeExplainer(model)
                    sv = explainer.shap_values(X_aligned.values)

                    if isinstance(sv, list) and len(sv) == 2:
                        sv_row = sv[1][0]
                    else:
                        sv_row = np.array(sv)[0]

                    local = (
                        pd.DataFrame({"feature": feature_list, "shap_value": sv_row})
                        .assign(abs_shap=lambda d: d["shap_value"].abs())
                        .sort_values("abs_shap", ascending=False)
                        .reset_index(drop=True)
                    )

                    st.write(f"Fraud probability for row {idx}: **{out.loc[int(idx), 'fraud_probability']:.4f}**")
                    st.dataframe(local.head(25), use_container_width=True)
                    st.bar_chart(local.set_index("feature").head(25)[["abs_shap"]])
                except Exception as e:
                    st.warning(f"Local SHAP failed: {e}")

st.caption("Fast mode: startup loads only model/scaler/encoders/features. SHAP loads only when enabled.")
