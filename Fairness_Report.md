# Fairness Report Generator — Task 14

## Overview

The Fairness Report Generator is a comprehensive module for evaluating and identifying bias in the EquiLend AI loan approval system. It implements industry-standard fairness metrics to ensure that lending decisions are equitable across demographic groups.

## Fairness Metrics Implemented

### 1. **Demographic Parity**
- **Definition**: Selection (approval) rates should be similar across protected groups
- **Metric**: Disparate Impact Ratio
- **Rule**: The 80% rule — minority group approval rate should be at least 80% of majority group rate
- **Use Case**: Ensures no group is systematically denied more often

### 2. **Equalized Odds**
- **Definition**: Both False Positive Rate (FPR) and True Positive Rate (TPR) should be equal across groups
- **Metrics**: 
  - FPR: Rate of false positives among actual negatives
  - TPR: Rate of true positives among actual positives
- **Use Case**: Ensures balanced error rates across demographics

### 3. **Equal Opportunity**
- **Definition**: True Positive Rate should be equal across groups
- **Focus**: Ensures disadvantaged groups have equal "opportunity" to be correctly identified
- **Metric**: Maximum TPR difference across groups
- **Use Case**: Prevents systematic undercounting of qualified borrowers in protected groups

### 4. **Predictive Parity**
- **Definition**: Positive Predictive Value (precision) should be equal across groups
- **Metric**: The probability that a positive prediction is correct
- **Use Case**: Ensures loan decisions have equal confidence across groups

## API Usage

### Basic Usage

```python
from src.evaluation.fairness import FairnessReportGenerator
import numpy as np
import pandas as pd

# Prepare data
y_true = np.array([...])  # Ground truth labels (0=approved, 1=denied)
y_pred = np.array([...])  # Model predictions
gender = pd.Series([...])  # Protected attribute (e.g., gender)

# Generate report
report_gen = FairnessReportGenerator(y_true, y_pred, protected_attr=gender)
summary = report_gen.fairness_summary()

print(summary["recommendations"])
```

### Using Utility Functions

```python
from src.evaluation.fairness import calculate_fairness_metrics, generate_fairness_html_report

# Calculate all fairness metrics
metrics = calculate_fairness_metrics(y_true, y_pred, gender)

# Generate HTML report for export/documentation
html_report = generate_fairness_html_report(y_true, y_pred, gender, title="Monthly Fairness Report")

# Save to file
with open("fairness_report.html", "w") as f:
    f.write(html_report)
```

## Integration with Streamlit Dashboard

The fairness report can be integrated into the EquiLend AI Streamlit app:

```python
from src.evaluation.fairness import FairnessReportGenerator
import streamlit as st

# In your app.py
if st.sidebar.checkbox("Show Fairness Analysis"):
    st.subheader("Fairness Report")
    
    # Get predictions and protected attributes
    report_gen = FairnessReportGenerator(y_test, y_prob, protected_attr=demographics)
    summary = report_gen.fairness_summary()
    
    # Display overall metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", summary["overall_metrics"]["accuracy"])
    with col2:
        st.metric("Sample Size", summary["overall_metrics"]["total_samples"])
    
    # Display recommendations
    st.warning("⚠️ Fairness Alerts:")
    for rec in summary["recommendations"]:
        st.write(rec)
    
    # Display demographic parity
    st.subheader("Demographic Parity Analysis")
    for group, metrics in summary["demographic_parity"]["by_group"].items():
        st.write(f"**{group}**: {metrics['selection_rate']} approval rate")
```

## Model Output Interpretation

### Disparate Impact Ratio
- **≥ 0.80**: ✓ Passes the 80% rule (Fair)
- **< 0.80**: ✗ Fails the 80% rule (Potential bias detected)

### TPR/FPR Differences
- **< 0.10**: ✓ Small difference (Good)
- **0.10 - 0.15**: ⚠️ Moderate difference (Review recommended)
- **> 0.15**: ✗ Large difference (Action needed)

## Recommendations for Addressing Bias

When fairness metrics reveal issues:

1. **Demographic Parity Failures**:
   - Review threshold settings
   - Consider threshold-based fairness constraints
   - Examine feature importance for demographic information

2. **Equalized Odds Failures**:
   - Investigate false positive vs false negative trade-offs
   - Consider separate thresholds for different groups (when legal)
   - Use fairness-aware learning algorithms

3. **Equal Opportunity Failures**:
   - Ensure training data representation
   - Address data quality issues in specific groups
   - Consider re-weighting samples for underrepresented groups

## Testing

Comprehensive test suite included in `tests/test_fairness.py`:

```bash
pytest tests/test_fairness.py -v
```

Tests cover:
- Metric calculations on fair and biased data
- Edge cases (perfect predictions, random models)
- HTML report generation
- Utility function behavior

## References

- **Fairness Definitions**: Buolamwini & Gebru (2018) "Gender Shades"
- **Legal Basis**: 4/5 Rule in Equal Employment Opportunity Commission (EEOC) guidelines
- **Implementation**: Adapted from IBM AI Fairness 360 toolkit
- **Standards**: NIST AI Risk Management Framework (NIST AI 600-1)

## Future Enhancements

- [ ] Causal fairness metrics
- [ ] Intersectional fairness (multiple protected attributes)
- [ ] Fairness constraints in model training
- [ ] Sensitivity analysis for threshold changes
- [ ] Monitoring dashboard for fairness drift over time

