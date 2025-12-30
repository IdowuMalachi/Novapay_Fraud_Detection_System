\
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import joblib

# Optional: SHAP
try:
    import shap  # noqa: F401
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False


# -----------------------------
# Repo-aware paths
# -----------------------------
ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "Models"
DATA_DIR = ROOT / "Data"

# If user runs streamlit from a different working directory, use __file__ paths above.
# All artifact loading uses these absolute paths.


def _first_existing(paths):
    for p in paths:
        if p and Path(p).exists():
            return Path(p)
    return None


@st.cache_resource(show_spinner=False)
def load_artifacts():
    """
    Loads artifacts from ./Models (your repo structure).
    Falls back to root for compatibility.
    """
    # Model
    model_path = _first_existing([
        MODELS_DIR / "rf_fraud_model.joblib",
        MODELS_DIR / "rf_fraud_model.joblib".replace("rf_", "rf_"),  # harmless
        ROOT / "rf_fraud_model.joblib",
    ])

    scaler_path = _first_existing([
        MODELS_DIR / "standard_scaler.joblib",
        ROOT / "standard_scaler.joblib",
    ])

    encoders_path = _first_existing([
        MODELS_DIR / "label_encoders.joblib",
        ROOT / "label_encoders.joblib",
    ])

    feature_list_path = _first_existing([
        MODELS_DIR / "rf_feature_list.json",
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        ROOT / "rf_feature_list.json",
    ])

    # Some repos use slightly different names. Try common alternatives too.
    if feature_list_path is None:
        feature_list_path = _first_existing([
            MODELS_DIR / "rf_feature_list.json",
            MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
            MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
            MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
            MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
            MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
            MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
            MODELS_DIR / "rf_feature_list.json".replace("rf_feature_list", "rf_feature_list"),
        ])

    # SHAP assets (optional)
    shap_values_path = _first_existing([
        MODELS_DIR / "shap_values_rf (2).npy",
        MODELS_DIR / "shap_values_rf.npy",
        ROOT / "shap_values_rf (2).npy",
        ROOT / "shap_values_rf.npy",
    ])

    x_test_shap_path = _first_existing([
        MODELS_DIR / "X_test_for_shap (2).csv",
        MODELS_DIR / "X_test_for_shap.csv",
        ROOT / "X_test_for_shap (2).csv",
        ROOT / "X_test_for_shap.csv",
    ])

    missing = []
    if model_path is None:
        missing.append("Models/rf_fraud_model.joblib")
    if scaler_path is None:
        missing.append("Models/standard_scaler.joblib")
    if encoders_path is None:
        missing.append("Models/label_encoders.joblib")
    if feature_list_path is None:
        missing.append("Models/rf_feature_list.json")

    if missing:
        raise FileNotFoundError(
            "Missing required files:\n- " + "\n- ".join(missing) +
            "\n\nMake sure your repo keeps artifacts inside the 'Models' folder."
        )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    label_encoders = joblib.load(encoders_path)
    with open(feature_list_path, "r", encoding="utf-8") as f:
        feature_list = json.load(f)

    shap_values = None
    X_test_for_shap = None
    if shap_values_path is not None:
        try:
            shap_values = np.load(shap_values_path)
        except Exception:
            shap_values = None
    if x_test_shap_path is not None:
        try:
            X_test_for_shap = pd.read_csv(x_test_shap_path)
        except Exception:
            X_test_for_shap = None

    return model, scaler, label_encoders, feature_list, shap_values, X_test_for_shap


def safe_label_encode(series: pd.Series, le) -> pd.Series:
    classes = getattr(le, "classes_", None)
    if classes is None:
        return series.astype(str)

    mapping = {cls: i for i, cls in enumerate(classes)}
    s = series.astype(str).fillna("")

    unseen = ~s.isin(mapping)
    if unseen.any():
        # map unseen to first class
        s = s.where(~unseen, other=str(classes[0]))

    return s.map(mapping).astype(int)


def prepare_features(df: pd.DataFrame, feature_list, label_encoders, scaler):
    X = df.copy()

    # Ensure all expected columns exist
    for col in feature_list:
        if col not in X.columns:
            X[col] = np.nan

    # align and order
    X = X[feature_list]

    # apply encoders (if any)
    for col, le in label_encoders.items():
        if col in X.columns:
            X[col] = safe_label_encode(X[col], le)

    # numeric coercion + missing fill
    for col in X.columns:
        # force numeric if possible
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Fill remaining NaNs with median (or 0)
    for col in X.columns:
        med = X[col].median()
        if pd.isna(med):
            med = 0
        X[col] = X[col].fillna(med)

    X_scaled = scaler.transform(X.values)
    return X, X_scaled


def score_prob(model, X_scaled: np.ndarray):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_scaled)[:, 1]
    if hasattr(model, "decision_function"):
        raw = model.decision_function(X_scaled)
        return 1 / (1 + np.exp(-raw))
    return model.predict(X_scaled).astype(float)


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="NovaPay Fraud Detection", page_icon="🛡️", layout="wide")
st.title("🛡️ NovaPay Fraud Detection System")
st.caption("Upload a CSV → score fraud risk → download results. Includes feature importance + optional SHAP summaries.")

# Sidebar: data sources
with st.sidebar:
    st.header("Inputs")
    use_repo_data = st.toggle("Use a CSV already in /Data", value=False)

    repo_csv = None
    if use_repo_data:
        if DATA_DIR.exists():
            csvs = sorted([p.name for p in DATA_DIR.glob("*.csv")])
            if csvs:
                repo_csv = st.selectbox("Choose a CSV from /Data", csvs)
            else:
                st.info("No CSV files found in the Data folder.")
        else:
            st.info("Data folder not found in this repo.")

    uploaded = st.file_uploader("Or upload a CSV to score", type=["csv"])
    st.divider()

    st.header("Scoring")
    threshold = st.slider("Fraud threshold", 0.05, 0.95, 0.50, 0.01)
    show_explain = st.toggle("Show explanations", value=True)
    local_shap = st.toggle("Local SHAP (single row)", value=False, disabled=not HAS_SHAP)

    st.divider()
    st.caption("Repo structure expected:")
    st.code("Data/  Models/  Notebooks/\nstreamlit.py  requirements.txt", language="text")

# Load artifacts
try:
    with st.spinner("Loading model artifacts from /Models..."):
        model, scaler, label_encoders, feature_list, shap_values, X_test_for_shap = load_artifacts()
except Exception as e:
    st.error(str(e))
    st.stop()

# Choose input dataframe
df_in = None
source_label = None

if repo_csv:
    try:
        df_in = pd.read_csv(DATA_DIR / repo_csv)
        source_label = f"Data/{repo_csv}"
    except Exception as e:
        st.error(f"Failed to read {repo_csv}: {e}")

if df_in is None and uploaded is not None:
    df_in = pd.read_csv(uploaded)
    source_label = "Uploaded CSV"

if df_in is None:
    st.info("Select a CSV from **/Data** or upload one to begin.")
    st.stop()

# Main layout
top1, top2, top3 = st.columns([1.2, 1, 1])
with top1:
    st.subheader("Preview")
    st.write(f"Source: **{source_label}**")
with top2:
    st.subheader("Model")
    st.write("Expected features:", len(feature_list))
    st.write("Encoders:", len(label_encoders))
with top3:
    st.subheader("Explainability")
    st.write("SHAP installed:", "Yes" if HAS_SHAP else "No")
    st.write("Precomputed SHAP:", "Yes" if (shap_values is not None and X_test_for_shap is not None) else "No")

st.dataframe(df_in.head(25), use_container_width=True)

# Prepare + score
with st.spinner("Preparing features and scoring..."):
    X_aligned, X_scaled = prepare_features(df_in, feature_list, label_encoders, scaler)
    proba = score_prob(model, X_scaled)
    pred = (proba >= threshold).astype(int)

out = df_in.copy()
out["fraud_probability"] = proba
out["fraud_flag"] = pred

st.subheader("Results")
m1, m2, m3 = st.columns(3)
m1.metric("Rows scored", f"{len(out):,}")
m2.metric("Flagged fraud", f"{int(out['fraud_flag'].sum()):,}")
m3.metric("Flag rate", f"{(out['fraud_flag'].mean() * 100):.2f}%")

st.dataframe(out.sort_values("fraud_probability", ascending=False).head(100), use_container_width=True)

st.download_button(
    "⬇️ Download scored CSV",
    data=out.to_csv(index=False).encode("utf-8"),
    file_name="novapay_scored.csv",
    mime="text/csv",
)

# Explanations
if show_explain:
    st.divider()
    st.subheader("Explanations")

    c1, c2 = st.columns(2)

    with c1:
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

    with c2:
        st.markdown("### SHAP summary (precomputed)")
        if shap_values is None or X_test_for_shap is None:
            st.info("Precomputed SHAP files not found in /Models (optional).")
        else:
            # mean absolute shap
            mean_abs = np.mean(np.abs(shap_values), axis=0)
            shap_df = (
                pd.DataFrame({"feature": feature_list, "mean_abs_shap": mean_abs})
                .sort_values("mean_abs_shap", ascending=False)
                .reset_index(drop=True)
            )
            st.dataframe(shap_df.head(25), use_container_width=True, height=420)
            st.bar_chart(shap_df.set_index("feature").head(25))

    if local_shap:
        if not HAS_SHAP:
            st.warning("Install SHAP to enable local explanations.")
        else:
            st.divider()
            st.subheader("Local SHAP (single row)")
            idx = st.number_input("Row index", min_value=0, max_value=max(0, len(out) - 1), value=0, step=1)
            try:
                explainer = shap.TreeExplainer(model)
                row = X_aligned.iloc[[int(idx)]].values
                sv = explainer.shap_values(row)

                # Binary RF often returns list [class0, class1]
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

st.caption("If the model fails to load on a new machine, match your training versions (Python / sklearn / numpy).")
