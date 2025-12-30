\
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# Optional: SHAP (used only when user requests explanations)
try:
    import shap
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False

st.set_page_config(page_title="NovaPay Fraud Risk Scorer", page_icon="🛡️", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = joblib.load("rf_fraud_model.joblib")
    scaler = joblib.load("standard_scaler.joblib")
    label_encoders = joblib.load("label_encoders.joblib")
    with open("rf_feature_list.json", "r", encoding="utf-8") as f:
        feature_list = json.load(f)

    # Precomputed SHAP assets (optional)
    shap_values = None
    X_test_for_shap = None
    try:
        shap_values = np.load("shap_values_rf (2).npy")
    except Exception:
        try:
            shap_values = np.load("shap_values_rf.npy")
        except Exception:
            shap_values = None

    try:
        X_test_for_shap = pd.read_csv("X_test_for_shap (2).csv")
    except Exception:
        try:
            X_test_for_shap = pd.read_csv("X_test_for_shap.csv")
        except Exception:
            X_test_for_shap = None

    return model, scaler, label_encoders, feature_list, shap_values, X_test_for_shap


def safe_label_encode(series: pd.Series, le) -> pd.Series:
    """
    LabelEncoder can't transform unseen labels.
    Strategy:
      - map known classes to their codes
      - unseen -> 0 (and warn)
    """
    classes = getattr(le, "classes_", None)
    if classes is None:
        # Fallback (shouldn't happen with sklearn LabelEncoder)
        return series.astype(str)

    mapping = {cls: i for i, cls in enumerate(classes)}
    s = series.astype(str).fillna("")

    unseen_mask = ~s.isin(mapping.keys())
    if unseen_mask.any():
        # assign unseen to 0 (first class)
        s = s.where(~unseen_mask, other=str(classes[0]))

    return s.map(mapping).astype(int)


def prepare_features(df: pd.DataFrame, feature_list, label_encoders, scaler):
    """
    Produce model-ready matrix aligned to feature_list.
    - ensures all expected features exist
    - encodes categoricals using saved label encoders
    - fills missing values
    - scales numeric features using saved scaler
    """
    X = df.copy()

    # Ensure all expected columns exist
    for col in feature_list:
        if col not in X.columns:
            X[col] = np.nan

    # Keep only expected columns, correct order
    X = X[feature_list]

    # Apply label encoders where available
    for col, le in label_encoders.items():
        if col in X.columns:
            X[col] = safe_label_encode(X[col], le)

    # Coerce to numeric where possible
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="ignore")

    # Fill missing values: numeric -> median, non-numeric -> 0
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            med = X[col].median()
            if pd.isna(med):
                med = 0
            X[col] = X[col].fillna(med)
        else:
            X[col] = X[col].astype(str).fillna("")
            # If still non-numeric at this point, try coerce again, else set 0
            coerced = pd.to_numeric(X[col], errors="coerce")
            if coerced.notna().any():
                X[col] = coerced.fillna(0)
            else:
                X[col] = 0

    # Scale
    X_scaled = scaler.transform(X.values)
    return X, X_scaled


@st.cache_resource(show_spinner=False)
def get_shap_explainer(model):
    if not _HAS_SHAP:
        return None
    try:
        return shap.TreeExplainer(model)
    except Exception:
        return None


def score(model, X_scaled: np.ndarray):
    # binary classifier expected
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled)[:, 1]
    else:
        # fallback: use decision_function if present
        if hasattr(model, "decision_function"):
            raw = model.decision_function(X_scaled)
            proba = 1 / (1 + np.exp(-raw))
        else:
            preds = model.predict(X_scaled)
            proba = preds.astype(float)
    return proba


# -----------------------------
# UI
# -----------------------------
st.title("🛡️ NovaPay Fraud Risk Scorer")
st.caption("Fast scoring + smart explanations (feature importance + optional SHAP).")

with st.spinner("Loading model artifacts..."):
    model, scaler, label_encoders, feature_list, shap_values, X_test_for_shap = load_artifacts()

colA, colB, colC = st.columns([1.2, 1, 1])
with colA:
    st.subheader("1) Upload transactions")
    uploaded = st.file_uploader("Upload a CSV to score (one or many rows).", type=["csv"])
    st.markdown("**Tip:** Your file should include the same engineered columns used in training. Missing columns will be created automatically.")

with colB:
    st.subheader("2) Settings")
    threshold = st.slider("Fraud threshold", 0.05, 0.95, 0.50, 0.01)
    show_explain = st.toggle("Show explanations (feature importance / SHAP)", value=True)
    show_shap_local = st.toggle("Local SHAP for a selected row", value=False, disabled=not _HAS_SHAP)

with colC:
    st.subheader("3) Quick sanity check")
    st.write("Expected feature count:", len(feature_list))
    st.write("Encoders:", len(label_encoders))
    st.write("SHAP available:", "Yes" if (_HAS_SHAP and show_explain) else ("Install shap" if show_explain else "Off"))

st.divider()

# Feature importance (global)
if show_explain:
    st.subheader("Global model signals")
    left, right = st.columns([1, 1])

    with left:
        st.markdown("### Feature importance (model-based)")
        try:
            importances = getattr(model, "feature_importances_", None)
            if importances is not None and len(importances) == len(feature_list):
                imp_df = pd.DataFrame({"feature": feature_list, "importance": importances}).sort_values("importance", ascending=False)
                st.dataframe(imp_df.head(20), use_container_width=True, height=420)
                st.bar_chart(imp_df.set_index("feature").head(20))
            else:
                st.info("Model-based importances not available for this model.")
        except Exception as e:
            st.warning(f"Could not compute model importances: {e}")

    with right:
        st.markdown("### SHAP summary (precomputed)")
        if shap_values is not None and X_test_for_shap is not None:
            # mean absolute shap
            mean_abs = np.mean(np.abs(shap_values), axis=0)
            shap_df = pd.DataFrame({"feature": feature_list, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False)
            st.dataframe(shap_df.head(20), use_container_width=True, height=420)
            st.bar_chart(shap_df.set_index("feature").head(20))
        else:
            st.info("Precomputed SHAP files not found in the app folder. (Optional)")

st.divider()

# Scoring
st.subheader("Score transactions")
if uploaded is None:
    st.info("Upload a CSV to begin scoring.")
else:
    df_in = pd.read_csv(uploaded)
    st.markdown("### Preview (uploaded)")
    st.dataframe(df_in.head(20), use_container_width=True)

    with st.spinner("Preparing features and scoring..."):
        X_aligned, X_scaled = prepare_features(df_in, feature_list, label_encoders, scaler)
        proba = score(model, X_scaled)
        pred = (proba >= threshold).astype(int)

    out = df_in.copy()
    out["fraud_probability"] = proba
    out["fraud_flag"] = pred

    st.markdown("### Results")
    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("Rows scored", f"{len(out):,}")
    with k2:
        st.metric("Flagged as fraud", f"{int(out['fraud_flag'].sum()):,}")
    with k3:
        st.metric("Flag rate", f"{(out['fraud_flag'].mean()*100):.2f}%")

    st.dataframe(out.sort_values("fraud_probability", ascending=False).head(50), use_container_width=True)

    # Download
    csv_bytes = out.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download scored CSV", data=csv_bytes, file_name="novapay_scored.csv", mime="text/csv")

    # Local SHAP explanation
    if show_explain and show_shap_local:
        if not _HAS_SHAP:
            st.warning("SHAP is not installed in this environment.")
        else:
            st.divider()
            st.subheader("Local SHAP explanation (single row)")
            idx = st.number_input("Row index to explain", min_value=0, max_value=max(0, len(out)-1), value=0, step=1)
            explainer = get_shap_explainer(model)
            if explainer is None:
                st.warning("Could not create SHAP explainer for this model.")
            else:
                # Compute SHAP for the selected row
                row_X = X_aligned.iloc[[int(idx)]]
                row_scaled = scaler.transform(row_X.values)
                try:
                    sv = explainer.shap_values(row_X.values)
                    # binary: often list [class0, class1] or array
                    if isinstance(sv, list) and len(sv) == 2:
                        sv_row = sv[1][0]
                    elif isinstance(sv, np.ndarray):
                        sv_row = sv[0]
                    else:
                        sv_row = np.array(sv)[0]

                    local_df = pd.DataFrame({"feature": feature_list, "shap_value": sv_row, "abs_shap": np.abs(sv_row)})
                    local_df = local_df.sort_values("abs_shap", ascending=False)

                    st.write(f"Fraud probability for row {idx}: **{out.loc[int(idx), 'fraud_probability']:.4f}**")
                    st.dataframe(local_df.head(25), use_container_width=True)

                    st.bar_chart(local_df.set_index("feature").head(25)[["abs_shap"]])
                except Exception as e:
                    st.warning(f"Local SHAP failed: {e}")

st.caption("Note: If you see model loading errors, ensure your Streamlit environment uses the same sklearn/numpy versions used during training.")
