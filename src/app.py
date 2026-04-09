"""
EquiLend AI — Streamlit Dashboard
Fixes applied (Task 00):
  Bug 1 — Division-by-zero: utility_bill clamped to min 1 before any division
  Bug 2 — Age guard: hard block on applicants < 18
  Bug 3 — Linear formula replaced with trained XGBoost model
  Bug 4 — State persistence: decisions saved to MongoDB (session log fallback)

New feature: Threshold Optimizer (Lender Rules Engine)
"""

import os
import sys
import time

import matplotlib
matplotlib.use("Agg")          # must come before any other matplotlib import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
_SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SRC_DIR)
for _p in (_SRC_DIR, _ROOT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve

# ── Constants ─────────────────────────────────────────────────────────────────
PRIMARY  = "#2E7D32"   # Forest Green
ACCENT   = "#5D4037"   # Soil Brown
ORANGE   = "#F57C00"

MODELS_DIR = os.path.join(_ROOT_DIR, "models")

# Support both data locations (generate_data.py vs upstream mock_data_generator.py)
_DATA_CANDIDATES = [
    os.path.join(_ROOT_DIR, "data",          "equilend_mock_data.csv"),
    os.path.join(_ROOT_DIR, "scripts", "data", "equilend_mock_data.csv"),
]
DATA_PATH = next((p for p in _DATA_CANDIDATES if os.path.exists(p)), _DATA_CANDIDATES[0])

# ── Cached loaders ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _load_artifacts():
    """Load saved model + preprocessor + test predictions. Returns None if absent."""
    try:
        from models.train_xgb import load_artifacts
        return load_artifacts(MODELS_DIR)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _sweep_thresholds(y_test_tup, y_prob_tup):
    """Threshold sweep — result cached so plots reuse it without recomputing."""
    from evaluation.thresholds import sweep_thresholds
    return sweep_thresholds(np.array(y_test_tup), np.array(y_prob_tup))


# ── Shared training helper ────────────────────────────────────────────────────

def _do_train():
    """Train XGBoost, clear cache, rerun. Runs inside a Streamlit spinner."""
    from models.train_xgb import train_and_save
    model, pre, y_test, y_prob, auc = train_and_save(DATA_PATH, MODELS_DIR)
    _load_artifacts.clear()
    return auc


# ── Page: New Application ─────────────────────────────────────────────────────

def page_new_application():
    st.subheader("Manual Loan Application")

    col1, col2 = st.columns(2)

    with col1:
        name              = st.text_input("Full Name")
        # Bug 2 fix: min_value kept at 0 so the field accepts any input,
        # but we validate >= 18 before scoring.
        age               = st.number_input("Age", min_value=0, max_value=120, step=1)
        income            = st.number_input("Monthly Income (₹)", min_value=0, step=500)
        gender            = st.selectbox("Gender", ["Male", "Female", "Non-Binary"])

    with col2:
        # Bug 1 fix: min_value=0 is fine for display; we clamp to 1 before use.
        utility_bill      = st.number_input("Average Utility Bill (₹)", min_value=0, step=100)
        repayment_history = st.slider("Past Repayment Consistency (%)", 0, 100, 50)
        employment_length = st.selectbox(
            "Employment Length",
            ["< 1 year", "1-3 years", "4-7 years", "8+ years"],
        )

    if st.button("Analyze Risk", type="primary"):

        # ── Bug 2: Age guard ──────────────────────────────────────────────────
        if age < 18:
            st.error("❌ Applicant must be at least 18 years old to apply.")
            return

        if not name.strip():
            st.warning("Please enter the applicant's full name.")
            return

        with st.spinner("AI Model Calculating…"):
            time.sleep(0.4)

            artifacts = _load_artifacts()

            if artifacts is None:
                # ── Bug 1 & 3 fallback (no model trained yet) ────────────────
                safe_bill  = max(utility_bill, 1)          # Bug 1 fix
                base_score = (income / safe_bill) * (repayment_history / 100)
                risk_level = "High" if base_score < 5 else "Low"
                st.warning("⚠️ ML model not trained yet. Using formula fallback.")
                st.metric("Formula Score", round(base_score, 2))
                st.write(f"Recommended Decision: **{risk_level} Risk**")
                return

            model, preprocessor, _, _ = artifacts

            # Bug 3 fix: use the trained ML model
            input_df = pd.DataFrame([{
                "monthly_income":       income,
                "utility_bill_average": max(utility_bill, 1),   # Bug 1 fix
                "repayment_history_pct": repayment_history,
                "employment_length":    employment_length,
                "gender":               gender,
            }])
            X_proc       = preprocessor.transform(input_df)
            prob_default = float(model.predict_proba(X_proc)[0, 1])

            # Read the lender's active threshold (set in Threshold Optimizer)
            threshold = st.session_state.get("thresh_slider", 0.50)
            decision  = "Deny — Default Risk" if prob_default >= threshold else "Approve"
            is_deny   = prob_default >= threshold

        st.success(f"Analysis complete for **{name}**")

        m1, m2, m3 = st.columns(3)
        m1.metric("Default Probability", f"{prob_default * 100:.1f}%")
        m2.metric("Active Threshold",    f"{threshold:.2f}")
        m3.metric("Decision",            decision)

        if is_deny:
            st.error(f"🚫 {decision}")
        else:
            st.success(f"✅ {decision}")

        # ── Bug 4 fix: persist decision to session log + MongoDB ──────────────
        record = {
            "name":                  name,
            "age":                   int(age),
            "gender":                gender,
            "monthly_income":        income,
            "utility_bill":          utility_bill,
            "repayment_history_pct": repayment_history,
            "employment_length":     employment_length,
            "prob_default":          round(prob_default, 4),
            "threshold":             threshold,
            "decision":              decision,
            "timestamp":             pd.Timestamp.now().isoformat(),
        }

        if "audit_log" not in st.session_state:
            st.session_state.audit_log = []
        st.session_state.audit_log.append(record)

        try:
            from data_ingestion.mongo_loader import save_decision
            save_decision(record)
            st.caption("✅ Decision saved to MongoDB.")
        except Exception:
            st.caption("ℹ️ Decision saved to session log (MongoDB not configured).")


# ── Page: Dashboard ───────────────────────────────────────────────────────────

def page_dashboard():
    st.subheader("📊 Model Performance Overview")

    artifacts = _load_artifacts()
    if artifacts is None:
        st.info(
            "No trained model found. Open **Threshold Optimizer** in the sidebar "
            "to train the model first."
        )
        return

    _, _, y_test, y_prob = artifacts
    auc = roc_auc_score(y_test, y_prob)

    c1, c2, c3 = st.columns(3)
    c1.metric("ROC-AUC",          f"{auc:.4f}")
    c2.metric("Test Samples",     f"{len(y_test):,}")
    c3.metric("Default Rate (test)", f"{y_test.mean():.1%}")

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(fpr, tpr, color=PRIMARY, lw=2, label=f"XGBoost  AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)


# ── Page: Threshold Optimizer ─────────────────────────────────────────────────

def page_threshold_optimizer():
    st.subheader("🎯 Threshold Optimizer — Lender Rules Engine")

    st.markdown("""
The model outputs a **probability of default** for every applicant (0 = safe payer → 1 = likely default).
A **decision threshold** converts that probability into a binary verdict — *Approve* or *Deny*.

| Threshold direction | Effect |
|---|---|
| **Lower** (e.g. 0.30) | Approve more applicants → higher revenue, more defaults slip through |
| **Higher** (e.g. 0.70) | Stricter approval → fewer defaults, but more good borrowers rejected |

Use this tool to explore the trade-off and lock in the threshold that matches your lending policy.
    """)

    # ── Step 1: model status ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Step 1 — Model Status")

    artifacts = _load_artifacts()

    if artifacts is None:
        data_ok = os.path.exists(DATA_PATH)
        if not data_ok:
            st.error(
                "❌ Dataset not found. Run `python scripts/generate_data.py` "
                "(or `python scripts/mock_data_generator.py`) from the project root first."
            )
            return

        col_msg, col_btn = st.columns([3, 1])
        col_msg.warning("⚠️ No trained model found. Click **Train Model** to begin.")
        if col_btn.button("🚀 Train Model"):
            try:
                with st.spinner("Training XGBoost on the synthetic dataset (this takes ~10 s)…"):
                    auc = _do_train()
                st.success(f"✅ Training complete!  ROC-AUC = **{auc:.4f}**")
                st.rerun()
            except Exception as exc:
                st.error(f"Training failed: {exc}")
        return

    _, _, y_test, y_prob = artifacts
    auc = roc_auc_score(y_test, y_prob)

    col_a, col_b, col_c, col_retrain = st.columns([2, 1, 1, 1])
    col_a.success(f"✅ Model loaded — ROC-AUC = **{auc:.4f}**")
    col_b.metric("Test samples", f"{len(y_test):,}")
    col_c.metric("Default rate", f"{y_test.mean():.1%}")
    if col_retrain.button("🔄 Retrain"):
        try:
            with st.spinner("Retraining…"):
                auc = _do_train()
            st.success(f"Retrained — AUC = {auc:.4f}")
            st.rerun()
        except Exception as exc:
            st.error(f"Retrain failed: {exc}")

    # ── Sweep (cached) ────────────────────────────────────────────────────────
    sweep_df = _sweep_thresholds(tuple(y_test.tolist()), tuple(y_prob.tolist()))

    # ── Step 2: threshold slider ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Step 2 — Set Decision Threshold")

    threshold = st.slider(
        "Decision Threshold  ←  lower = approve more  |  higher = approve fewer  →",
        min_value=0.01,
        max_value=0.99,
        value=float(st.session_state.get("thresh_slider", 0.50)),
        step=0.01,
        format="%.2f",
        key="thresh_slider",
    )

    # ── Step 3: live metrics ──────────────────────────────────────────────────
    from evaluation.thresholds import get_metrics_at_threshold, find_optimal_threshold

    m = get_metrics_at_threshold(y_test, y_prob, threshold)

    st.markdown("---")
    st.markdown(f"#### Step 3 — Live Metrics at Threshold = **{threshold:.2f}**")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Precision",     f"{m['precision']:.3f}",
              help="Of all *denied* applicants, what fraction were true defaulters?")
    c2.metric("Recall",        f"{m['recall']:.3f}",
              help="Of all true defaulters, what fraction did we correctly identify?")
    c3.metric("F1 Score",      f"{m['f1']:.3f}",
              help="Harmonic mean of Precision and Recall.")
    c4.metric("Accuracy",      f"{m['accuracy']:.3f}",
              help="Overall fraction of correctly classified applicants.")
    c5.metric("Approval Rate", f"{m['approval_rate']:.1%}",
              help="Fraction of applicants that would be *approved* at this threshold.")

    # ── Step 4: confusion matrix ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Confusion Matrix")

    cm_col, explain_col = st.columns([1, 1])

    with cm_col:
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        fig, ax = plt.subplots(figsize=(4, 3))
        im = ax.imshow(cm, cmap="Greens")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Predicted\nApproved", "Predicted\nDenied"])
        ax.set_yticklabels(["Actual\nPaid", "Actual\nDefault"])
        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i, f"{cm[i, j]:,}",
                    ha="center", va="center", fontsize=13, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() * 0.55 else "black",
                )
        fig.colorbar(im, ax=ax)
        ax.set_title(f"Threshold = {threshold:.2f}", pad=10)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with explain_col:
        st.markdown(f"""
| Quadrant | Count | Meaning |
|---|---|---|
| **True Negative (TN)** | {m['tn']:,} | Correctly **approved** good borrowers ✅ |
| **False Positive (FP)** | {m['fp']:,} | Incorrectly **denied** good borrowers ❌ |
| **False Negative (FN)** | {m['fn']:,} | Defaulters **approved** (risk leaked) ⚠️ |
| **True Positive (TP)** | {m['tp']:,} | Correctly **denied** defaulters ✅ |

---
**Approval Rate:** {m['approval_rate']:.1%} of test applicants approved.
**Default leakage:** {m['fn']:,} defaulters would slip through as approved.
**Good borrowers blocked:** {m['fp']:,} creditworthy applicants incorrectly denied.
        """)

    # ── Step 5: metrics vs threshold chart ───────────────────────────────────
    st.markdown("---")
    st.markdown("#### How Metrics Shift with Threshold")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(sweep_df["threshold"], sweep_df["precision"],     label="Precision",     color="#1976D2", lw=1.8)
    ax.plot(sweep_df["threshold"], sweep_df["recall"],        label="Recall",        color="#E53935", lw=1.8)
    ax.plot(sweep_df["threshold"], sweep_df["f1"],            label="F1 Score",      color=PRIMARY,   lw=2.5)
    ax.plot(sweep_df["threshold"], sweep_df["approval_rate"], label="Approval Rate", color=ACCENT,    lw=1.8, linestyle="--")
    ax.axvline(threshold, color=ORANGE, lw=2, linestyle=":", label=f"Current  ({threshold:.2f})")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score / Rate")
    ax.set_title("Metric Trade-offs Across Thresholds")
    ax.legend(loc="center right", framealpha=0.9)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── Step 6: ROC and PR curves ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### ROC Curve & Precision-Recall Curve")

    fpr_arr, tpr_arr, roc_thresh = roc_curve(y_test, y_prob)
    pr_prec, pr_rec, pr_thresh   = precision_recall_curve(y_test, y_prob)

    roc_col, pr_col = st.columns(2)

    with roc_col:
        # Mark the point on the ROC curve closest to the chosen threshold
        if len(roc_thresh) > 1:
            roc_idx = int(np.argmin(np.abs(roc_thresh - threshold)))
            pt_fpr, pt_tpr = fpr_arr[roc_idx], tpr_arr[roc_idx]
        else:
            pt_fpr, pt_tpr = None, None

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr_arr, tpr_arr, color=PRIMARY, lw=2, label=f"AUC = {auc:.4f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random baseline")
        if pt_fpr is not None:
            ax.scatter([pt_fpr], [pt_tpr], color=ORANGE, s=120, zorder=5,
                       label=f"Threshold {threshold:.2f}")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate (Recall)")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with pr_col:
        if len(pr_thresh) > 1:
            pr_idx = int(np.argmin(np.abs(pr_thresh - threshold)))
            pt_rec_v, pt_prec_v = pr_rec[pr_idx], pr_prec[pr_idx]
        else:
            pt_rec_v, pt_prec_v = None, None

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(pr_rec, pr_prec, color=ACCENT, lw=2, label="PR Curve")
        ax.axhline(float(y_test.mean()), color="k", linestyle="--", lw=1,
                   label=f"Baseline ({y_test.mean():.2f})")
        if pt_rec_v is not None:
            ax.scatter([pt_rec_v], [pt_prec_v], color=ORANGE, s=120, zorder=5,
                       label=f"Threshold {threshold:.2f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.legend(loc="upper right")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Step 7: optimal threshold recommendations ─────────────────────────────
    st.markdown("---")
    st.markdown("#### Optimal Threshold Recommendations")
    st.markdown(
        "The table below shows the best threshold for each business objective. "
        "Click a button to instantly apply that threshold."
    )

    OBJECTIVES = {
        "Maximize F1":        "f1",
        "Maximize Precision": "precision",
        "Maximize Recall":    "recall",
        "Balanced (G-Mean)":  "balanced",
        "Maximize Profit":    "profit",
    }
    OBJECTIVE_TIPS = {
        "f1":        "Best balanced trade-off between catching defaulters and approving good borrowers.",
        "precision": "Risk-averse lender: minimise bad loans approved (may reject good borrowers).",
        "recall":    "Catch-all policy: identify every defaulter (accepts more false rejections).",
        "balanced":  "Equal weight to sensitivity and specificity — suitable for imbalanced datasets.",
        "profit":    "Weighted by business value: each approved good borrower = +1, bad loan = −5.",
    }

    rows = []
    opt_results = {}
    for label, obj in OBJECTIVES.items():
        best = find_optimal_threshold(y_test, y_prob, objective=obj)
        opt_results[obj] = best
        rows.append({
            "Objective":          label,
            "Optimal Threshold":  f"{best['threshold']:.2f}",
            "Precision":          f"{best['precision']:.3f}",
            "Recall":             f"{best['recall']:.3f}",
            "F1":                 f"{best['f1']:.3f}",
            "Approval Rate":      f"{best['approval_rate']:.1%}",
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Quick-apply buttons
    st.markdown("**Quick-apply an objective:**")
    btn_cols = st.columns(len(OBJECTIVES))
    for idx, (label, obj) in enumerate(OBJECTIVES.items()):
        with btn_cols[idx]:
            short = label.replace("Maximize ", "").replace("Balanced (G-Mean)", "Balanced")
            if st.button(short, key=f"apply_{obj}", help=OBJECTIVE_TIPS[obj]):
                st.session_state["thresh_slider"] = float(opt_results[obj]["threshold"])
                st.rerun()

    st.caption(
        "💡 Tip: start with **Maximize F1** for balanced performance, "
        "switch to **Maximize Recall** if reducing default leakage is the priority, "
        "or **Maximize Profit** for a dollar-weighted optimum."
    )


# ── Page: Audit Logs ──────────────────────────────────────────────────────────

def page_audit_logs():
    st.subheader("📋 Audit Logs")

    logs = list(st.session_state.get("audit_log", []))

    # Also try to pull from MongoDB if available
    try:
        from data_ingestion.mongo_loader import load_decisions
        mongo_records = load_decisions()
        if mongo_records:
            st.info(f"Showing {len(mongo_records)} record(s) from MongoDB.")
            logs = mongo_records
    except Exception:
        pass

    if not logs:
        st.info(
            "No decisions recorded yet. "
            "Use **New Application** in the sidebar to score applicants."
        )
        return

    df = pd.DataFrame(logs)
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download as CSV",
        data=csv,
        file_name="equilend_audit_log.csv",
        mime="text/csv",
    )


# ── Page: Fairness Analyzer ───────────────────────────────────────────────────

def page_fairness_analyzer():
    """Fairness analysis page — detect and report bias in loan approvals"""
    st.subheader("⚖️ Fairness Analyzer — Bias Detection & Equity Audit")
    
    st.markdown("""
    Evaluate the fairness of lending decisions across demographic groups.
    Detects potential bias using industry-standard metrics.
    """)
    
    artifacts = _load_artifacts()
    if not artifacts:
        st.error("❌ No trained model found. Train a model in **Threshold Optimizer** first.")
        return
    
    model, preprocessor, y_test, y_prob = artifacts
    
    st.info(
        "📊 Using test set predictions for fairness analysis. "
        f"Test set size: **{len(y_test)}** samples"
    )
    
    # ── Protected Attribute Selection ─────────────────────────────────────────
    st.subheader("Step 1: Select Protected Attribute")
    
    protected_attr_choice = st.selectbox(
        "Which demographic group to analyze?",
        ["Gender", "Age Group", "Synthetic (Demo)"]
    )
    
    # Generate synthetic protected attributes for demo
    if protected_attr_choice == "Synthetic (Demo)":
        np.random.seed(42)
        protected_attr = pd.Series(
            np.random.choice(["Group_A", "Group_B"], len(y_test)),
            index=range(len(y_test))
        )
        st.caption("Using randomly generated groups for demonstration")
    else:
        st.warning(f"⚠️ {protected_attr_choice} attribute not in test data. Using synthetic demo groups.")
        np.random.seed(42)
        protected_attr = pd.Series(
            np.random.choice(["Group_A", "Group_B"], len(y_test)),
            index=range(len(y_test))
        )
    
    # ── Threshold Selection ───────────────────────────────────────────────────
    st.subheader("Step 2: Set Decision Threshold")
    
    threshold = st.slider(
        "Decision Threshold",
        min_value=0.01,
        max_value=0.99,
        value=0.50,
        step=0.01,
        help="Predictions >= threshold → Denied (1), < threshold → Approved (0)"
    )
    
    # Convert probabilities to binary predictions
    y_pred = (y_prob >= threshold).astype(int)
    
    # ── Generate Fairness Report ──────────────────────────────────────────────
    st.subheader("Step 3: Fairness Report")
    
    try:
        from evaluation.fairness import FairnessReportGenerator
        
        report_gen = FairnessReportGenerator(
            y_true=y_test,
            y_pred=y_pred,
            protected_attr=protected_attr
        )
        report = report_gen.fairness_summary()
        
        # ── Data Sufficiency Check ────────────────────────────────────────────
        data_suff = report.get("data_sufficiency", {})
        
        if not data_suff.get("is_sufficient", True):
            st.error("❌ **Data Insufficient**: Analysis cannot proceed with missing data.")
            for error in data_suff.get("errors", []):
                st.error(f"  • {error}")
            return
        
        if data_suff.get("warnings"):
            for warning in data_suff.get("warnings", []):
                st.warning(f"⚠️ {warning}")
        
        # Overall Metrics
        col1, col2, col3, col4 = st.columns(4)
        metrics = report["overall_metrics"]
        
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        with col2:
            st.metric("Precision", f"{metrics['precision']:.4f}")
        with col3:
            st.metric("Recall", f"{metrics['recall']:.4f}")
        with col4:
            st.metric("Specificity", f"{metrics['specificity']:.4f}")
        
        # ── Demographic Parity ────────────────────────────────────────────────
        st.subheader("📊 Demographic Parity (Selection Rates)")
        
        dp = report["demographic_parity"]
        
        if "by_group" in dp:
            dp_data = []
            for group, metrics_dict in dp["by_group"].items():
                dp_data.append({
                    "Group": group,
                    "Approval Rate": f"{metrics_dict['selection_rate']:.2%}",
                    "Sample Size": metrics_dict['count']
                })
            
            st.dataframe(pd.DataFrame(dp_data), use_container_width=True)
            
            if "disparate_impact_ratio" in dp:
                di_ratio = dp["disparate_impact_ratio"]
                passes = "✅ PASS" if dp["passes_80_percent_rule"] else "❌ FAIL"
                
                col_di1, col_di2 = st.columns(2)
                with col_di1:
                    st.metric(
                        "Disparate Impact Ratio",
                        f"{di_ratio:.4f}",
                        help="Min approval rate ÷ Max approval rate. ≥ 0.80 = Fair"
                    )
                with col_di2:
                    st.metric(
                        "80% Rule",
                        passes.split()[-1],
                        delta="Fair" if dp["passes_80_percent_rule"] else "Unfair"
                    )
                
                if di_ratio < 0.80:
                    st.warning(
                        f"⚠️ **Disparate Impact Alert**: Ratio is {di_ratio:.4f} (< 0.80). "
                        "This suggests potential discrimination."
                    )
        
        # ── Equalized Odds ────────────────────────────────────────────────────
        st.subheader("📈 Equalized Odds (Error Rates)")
        
        eo = report["equalized_odds"]
        
        if "by_group" in eo:
            eo_data = []
            for group, metrics_dict in eo["by_group"].items():
                eo_data.append({
                    "Group": group,
                    "FPR": f"{metrics_dict['fpr']:.4f}",
                    "TPR": f"{metrics_dict['tpr']:.4f}"
                })
            
            st.dataframe(pd.DataFrame(eo_data), use_container_width=True)
            
            if "max_fpr_diff" in eo:
                col_eo1, col_eo2 = st.columns(2)
                with col_eo1:
                    st.metric("Max FPR Difference", f"{eo['max_fpr_diff']:.4f}")
                with col_eo2:
                    st.metric("Max TPR Difference", f"{eo['max_tpr_diff']:.4f}")
        
        # ── Equal Opportunity ─────────────────────────────────────────────────
        st.subheader("🎯 Equal Opportunity (True Positive Rate)")
        
        eop = report["equal_opportunity"]
        
        if "by_group" in eop:
            eop_data = []
            for group, metrics_dict in eop["by_group"].items():
                eop_data.append({
                    "Group": group,
                    "TPR": f"{metrics_dict['tpr']:.4f}",
                    "True Positives": metrics_dict['true_positives'],
                    "False Negatives": metrics_dict['false_negatives']
                })
            
            st.dataframe(pd.DataFrame(eop_data), use_container_width=True)
            
            if "max_tpr_diff" in eop:
                st.metric("Max TPR Difference", f"{eop['max_tpr_diff']:.4f}")
                
                if eop["max_tpr_diff"] > 0.10:
                    st.warning(
                        f"⚠️ **Equal Opportunity Alert**: TPR differs by {eop['max_tpr_diff']:.4f} "
                        "across groups. Lower rates may indicate bias."
                    )
        
        # ── Predictive Parity ─────────────────────────────────────────────────
        st.subheader("🔍 Predictive Parity (Precision)")
        
        pp = report["predictive_parity"]
        
        if "by_group" in pp:
            pp_data = []
            for group, metrics_dict in pp["by_group"].items():
                pp_data.append({
                    "Group": group,
                    "PPV (Precision)": f"{metrics_dict['ppv']:.4f}",
                    "True Positives": metrics_dict['true_positives'],
                    "False Positives": metrics_dict['false_positives']
                })
            
            st.dataframe(pd.DataFrame(pp_data), use_container_width=True)
        
        # ── Recommendations ───────────────────────────────────────────────────
        st.subheader("💡 Recommendations & Alerts")
        
        recommendations = report["recommendations"]
        
        for i, rec in enumerate(recommendations, 1):
            if "✓" in rec:
                st.success(rec)
            elif "⚠️" in rec or "Alert" in rec:
                st.warning(rec)
            else:
                st.info(rec)
        
        # ── Export Report ─────────────────────────────────────────────────────
        st.subheader("📥 Export Reports")
        
        try:
            from evaluation.fairness import generate_fairness_html_report, generate_fairness_markdown_report
            
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            
            # HTML Report
            html_report, html_path = generate_fairness_html_report(
                y_true=y_test,
                y_pred=y_pred,
                protected_attr=protected_attr,
                title="EquiLend AI — Fairness Audit Report"
            )
            
            col_html, col_md = st.columns(2)
            
            with col_html:
                st.download_button(
                    "📊 Download HTML Report",
                    data=html_report,
                    file_name=f"fairness_report_{timestamp}.html",
                    mime="text/html"
                )
                if html_path:
                    st.caption(f"✅ Also saved to: {html_path}")
                else:
                    st.caption("ℹ️ Could not save to disk (permission denied)")
            
            # Markdown Report
            with col_md:
                try:
                    md_report, md_path = generate_fairness_markdown_report(
                        y_true=y_test,
                        y_pred=y_prob,  # Use probabilities for AUC calculation
                        protected_attr=protected_attr
                    )
                    
                    st.download_button(
                        "📝 Download Markdown Report",
                        data=md_report,
                        file_name=f"fairness_report_{timestamp}.md",
                        mime="text/markdown"
                    )
                    if md_path:
                        st.caption(f"✅ Also saved to: {md_path}")
                    else:
                        st.caption("ℹ️ Could not save to disk (permission denied)")
                except Exception as md_error:
                    st.error(f"Could not generate markdown: {md_error}")
        
        except ImportError:
            st.error("❌ Fairness export functions not found.")
        except PermissionError as perm_err:
            st.warning(f"⚠️ Permission denied writing to disk: {perm_err}. Reports available for download only.")
        except Exception as e:
            st.error(f"❌ Error generating reports: {e}")
    
    except ImportError:
        st.error("❌ Fairness module not found. Please ensure fairness.py is installed.")
    except Exception as e:
        st.error(f"❌ Error generating fairness report: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="EquiLend AI — Credit Scoring",
        layout="wide",
        page_icon="⚖️",
    )

    st.title("⚖️ EquiLend AI: Transparent Credit Scoring")
    st.markdown("*Bridging the credit gap with fair, explainable, alternative-data ML.*")

    menu   = ["New Application", "Dashboard", "Threshold Optimizer", "Fairness Analyzer", "Audit Logs"]
    choice = st.sidebar.selectbox("Navigation", menu)

    st.sidebar.markdown("---")
    active_thresh = st.session_state.get("thresh_slider", 0.50)
    st.sidebar.metric("Active Threshold", f"{active_thresh:.2f}")
    st.sidebar.caption("Set in **Threshold Optimizer**")

    if choice == "New Application":
        page_new_application()
    elif choice == "Dashboard":
        page_dashboard()
    elif choice == "Threshold Optimizer":
        page_threshold_optimizer()
    elif choice == "Fairness Analyzer":
        page_fairness_analyzer()
    elif choice == "Audit Logs":
        page_audit_logs()


if __name__ == "__main__":
    main()
