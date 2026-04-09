"""
Task 14 — Fairness Report Generator
Comprehensive fairness metrics and bias detection for loan approval models.
Implements demographic parity, equalized odds, and equal opportunity metrics.
Includes markdown generation, NaN validation, and permission error handling.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import confusion_matrix
from datetime import datetime
import os
import tempfile


# ── Fairness Metrics ──────────────────────────────────────────────────────────

class FairnessReportGenerator:
    """
    Generate comprehensive fairness reports for binary classification models.
    Identifies potential bias across protected attributes (e.g., gender, age groups).
    """

    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, 
                 protected_attr: Optional[pd.Series] = None,
                 positive_label: int = 0):
        """
        Initialize fairness report generator.
        
        Args:
            y_true: Ground truth binary labels (0 = approved/non-default, 1 = denied/default)
            y_pred: Model predictions (binary or probabilities)
            protected_attr: Series with protected attribute groups (e.g., gender, age)
            positive_label: Which class represents positive outcome (default: 0 = approval)
        """
        self.y_true = np.asarray(y_true)
        self.y_pred = np.asarray(y_pred)
        self.protected_attr = protected_attr
        self.positive_label = positive_label
        self.negative_label = 1 - positive_label
        self.data_sufficiency = self._validate_data_sufficiency()
        
        # Ensure binary predictions
        if self.y_pred.min() >= 0 and self.y_pred.max() <= 1:
            if self.y_pred.max() > 0.5:  # Likely probabilities
                self.y_pred_binary = (self.y_pred >= 0.5).astype(int)
            else:
                self.y_pred_binary = self.y_pred.astype(int)
        else:
            self.y_pred_binary = self.y_pred.astype(int)
    
    def _validate_data_sufficiency(self) -> Dict[str, any]:
        """Validate data sufficiency and identify potential issues."""
        sufficiency = {
            "is_sufficient": True,
            "warnings": [],
            "errors": []
        }
        
        # Check for empty arrays
        if len(self.y_true) == 0 or len(self.y_pred) == 0:
            sufficiency["is_sufficient"] = False
            sufficiency["errors"].append("Data insufficient: empty arrays")
            return sufficiency
        
        # Check for NaN in y_true and y_pred
        nan_count_true = np.isnan(self.y_true).sum()
        nan_count_pred = np.isnan(self.y_pred).sum()
        
        if nan_count_true > 0:
            nan_pct = (nan_count_true / len(self.y_true)) * 100
            sufficiency["warnings"].append(
                f"Data insufficient: {nan_count_true} NaN values in y_true ({nan_pct:.1f}%)"
            )
            if nan_pct > 10:
                sufficiency["is_sufficient"] = False
        
        if nan_count_pred > 0:
            nan_pct = (nan_count_pred / len(self.y_pred)) * 100
            sufficiency["warnings"].append(
                f"Data insufficient: {nan_count_pred} NaN values in y_pred ({nan_pct:.1f}%)"
            )
            if nan_pct > 10:
                sufficiency["is_sufficient"] = False
        
        # Check for protected attribute NaN
        if self.protected_attr is not None:
            nan_count_attr = self.protected_attr.isna().sum()
            if nan_count_attr > 0:
                nan_pct = (nan_count_attr / len(self.protected_attr)) * 100
                sufficiency["warnings"].append(
                    f"Data insufficient: {nan_count_attr} NaN values in protected_attr ({nan_pct:.1f}%)"
                )
                if nan_pct > 10:
                    sufficiency["is_sufficient"] = False
        
        # Check for minimum sample size
        if len(self.y_true) < 30:
            sufficiency["warnings"].append(
                f"Data insufficient: only {len(self.y_true)} samples (minimum recommended: 30)"
            )
        
        return sufficiency

    def demographic_parity(self) -> Dict[str, float]:
        """
        Demographic Parity: Selection rate should be similar across groups.
        Measures the ratio of approval rates between groups.
        
        Returns:
            dict with group statistics and disparate impact ratio
        """
        if self.protected_attr is None:
            return {"error": "Protected attribute required"}
        
        groups = self.protected_attr.unique()
        results = {"by_group": {}}
        
        for group in groups:
            mask = self.protected_attr == group
            selection_rate = (self.y_pred_binary[mask] == self.positive_label).mean()
            results["by_group"][str(group)] = {
                "selection_rate": round(float(selection_rate), 4),
                "count": int(mask.sum())
            }
        
        # Disparate Impact Ratio
        if len(groups) == 2:
            rates = [results["by_group"][str(g)]["selection_rate"] for g in groups]
            disparate_impact = min(rates) / max(rates) if max(rates) > 0 else 0.0
            results["disparate_impact_ratio"] = round(disparate_impact, 4)
            results["passes_80_percent_rule"] = disparate_impact >= 0.80
        
        return results

    def equalized_odds(self) -> Dict[str, Dict]:
        """
        Equalized Odds: FPR and TPR should be equal across groups.
        Ensures false positive and true positive rates don't differ by group.
        
        Returns:
            dict with FPR and TPR by group
        """
        if self.protected_attr is None:
            return {"error": "Protected attribute required"}
        
        groups = self.protected_attr.unique()
        results = {"by_group": {}}
        
        for group in groups:
            mask = self.protected_attr == group
            y_true_group = self.y_true[mask]
            y_pred_group = self.y_pred_binary[mask]
            
            # False Positive Rate (FPR): rate of false positive among actual negatives
            tn, fp, fn, tp = confusion_matrix(
                y_true_group, y_pred_group, 
                labels=[self.positive_label, self.negative_label]
            ).ravel()
            
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            results["by_group"][str(group)] = {
                "fpr": round(float(fpr), 4),
                "tpr": round(float(tpr), 4),
                "count": int(mask.sum())
            }
        
        # Maximum difference
        if len(groups) == 2:
            fprs = [results["by_group"][str(g)]["fpr"] for g in groups]
            tprs = [results["by_group"][str(g)]["tpr"] for g in groups]
            results["max_fpr_diff"] = round(max(fprs) - min(fprs), 4)
            results["max_tpr_diff"] = round(max(tprs) - min(tprs), 4)
        
        return results

    def equal_opportunity(self) -> Dict[str, Dict]:
        """
        Equal Opportunity: TPR should be equal across groups.
        Focuses on equal true positive rates for disadvantaged groups.
        
        Returns:
            dict with TPR by group and max difference
        """
        if self.protected_attr is None:
            return {"error": "Protected attribute required"}
        
        groups = self.protected_attr.unique()
        results = {"by_group": {}}
        
        for group in groups:
            mask = self.protected_attr == group
            y_true_group = self.y_true[mask]
            y_pred_group = self.y_pred_binary[mask]
            
            tn, fp, fn, tp = confusion_matrix(
                y_true_group, y_pred_group,
                labels=[self.positive_label, self.negative_label]
            ).ravel()
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            results["by_group"][str(group)] = {
                "tpr": round(float(tpr), 4),
                "true_positives": int(tp),
                "false_negatives": int(fn),
                "count": int(mask.sum())
            }
        
        if len(groups) == 2:
            tprs = [results["by_group"][str(g)]["tpr"] for g in groups]
            results["max_tpr_diff"] = round(max(tprs) - min(tprs), 4)
            results["alert"] = (results["max_tpr_diff"] > 0.1)
        
        return results

    def predictive_parity(self) -> Dict[str, Dict]:
        """
        Predictive Parity: Positive Predictive Value (precision) should be equal.
        
        Returns:
            dict with PPV (precision) by group
        """
        if self.protected_attr is None:
            return {"error": "Protected attribute required"}
        
        groups = self.protected_attr.unique()
        results = {"by_group": {}}
        
        for group in groups:
            mask = self.protected_attr == group
            y_true_group = self.y_true[mask]
            y_pred_group = self.y_pred_binary[mask]
            
            tn, fp, fn, tp = confusion_matrix(
                y_true_group, y_pred_group,
                labels=[self.positive_label, self.negative_label]
            ).ravel()
            
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            
            results["by_group"][str(group)] = {
                "ppv": round(float(ppv), 4),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "count": int(mask.sum())
            }
        
        if len(groups) == 2:
            ppvs = [results["by_group"][str(g)]["ppv"] for g in groups]
            results["max_ppv_diff"] = round(max(ppvs) - min(ppvs), 4)
        
        return results

    def fairness_summary(self) -> Dict:
        """
        Generate complete fairness summary with all metrics.
        Returns data sufficiency warnings if applicable.
        
        Returns:
            dict with all fairness metrics and recommendations
        """
        summary = {
            "data_sufficiency": self.data_sufficiency,
            "overall_metrics": self._get_overall_metrics(),
            "demographic_parity": self.demographic_parity(),
            "equalized_odds": self.equalized_odds(),
            "equal_opportunity": self.equal_opportunity(),
            "predictive_parity": self.predictive_parity(),
            "recommendations": self._generate_recommendations()
        }
        return summary

    def _get_overall_metrics(self) -> Dict[str, float]:
        """Calculate overall model performance metrics."""
        tn, fp, fn, tp = confusion_matrix(
            self.y_true, self.y_pred_binary,
            labels=[self.positive_label, self.negative_label]
        ).ravel()
        
        accuracy = (tp + tn) / len(self.y_true)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        return {
            "accuracy": round(float(accuracy), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "specificity": round(float(specificity), 4),
            "total_samples": len(self.y_true),
            "positive_rate": round(float((self.y_true == self.positive_label).mean()), 4)
        }

    def _generate_recommendations(self) -> List[str]:
        """Generate fairness improvement recommendations."""
        recommendations = []
        
        if self.protected_attr is None:
            return ["Add protected attributes to enable demographic bias detection"]
        
        dp = self.demographic_parity()
        if "passes_80_percent_rule" in dp and not dp["passes_80_percent_rule"]:
            recommendations.append(
                "⚠️ Demographic Parity Issue: Disparate impact detected. "
                f"Disparate Impact Ratio = {dp['disparate_impact_ratio']:.4f} (< 0.80)"
            )
        
        eo = self.equalized_odds()
        if "max_fpr_diff" in eo and eo["max_fpr_diff"] > 0.15:
            recommendations.append(
                f"⚠️ Equalized Odds Alert: False Positive Rate differs by {eo['max_fpr_diff']:.4f} across groups"
            )
        
        eop = self.equal_opportunity()
        if "max_tpr_diff" in eop and eop["max_tpr_diff"] > 0.10:
            recommendations.append(
                f"⚠️ Equal Opportunity Alert: True Positive Rate differs by {eop['max_tpr_diff']:.4f} across groups"
            )
        
        if not recommendations:
            recommendations.append("✓ Model appears fair across all measured dimensions")
        
        return recommendations

    def to_dataframe(self) -> pd.DataFrame:
        """Convert fairness report to readable DataFrame format."""
        report = self.fairness_summary()
        
        # Flatten for better readability
        rows = []
        
        # Overall metrics
        for key, val in report["overall_metrics"].items():
            rows.append({"Category": "Overall", "Metric": key, "Value": val})
        
        # Append recommendations
        for rec in report["recommendations"]:
            rows.append({"Category": "Recommendation", "Metric": "-", "Value": rec})
        
        return pd.DataFrame(rows)


# ── Utility Functions ─────────────────────────────────────────────────────────

def calculate_fairness_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                              protected_attr: Optional[pd.Series] = None) -> Dict:
    """
    Convenience function to calculate all fairness metrics at once.
    
    Args:
        y_true: Ground truth labels
        y_pred: Model predictions
        protected_attr: Protected attribute groups
    
    Returns:
        Complete fairness report dictionary
    """
    generator = FairnessReportGenerator(y_true, y_pred, protected_attr)
    return generator.fairness_summary()


def generate_fairness_html_report(y_true: np.ndarray, y_pred: np.ndarray,
                                 protected_attr: Optional[pd.Series] = None,
                                 title: str = "Fairness Report",
                                 output_path: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """
    Generate HTML fairness report with permission error handling.
    
    Args:
        y_true: Ground truth labels
        y_pred: Model predictions
        protected_attr: Protected attribute groups
        title: Report title
        output_path: Path to save HTML file (optional)
    
    Returns:
        Tuple of (HTML string, file_path or None if couldn't write)
    """
    generator = FairnessReportGenerator(y_true, y_pred, protected_attr)
    report = generator.fairness_summary()
    
    html = f"""
    <html>
    <head>
        <title>{title}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #2E7D32; }}
            h2 {{ color: #5D4037; margin-top: 20px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .warning {{ color: #ff9800; font-weight: bold; }}
            .success {{ color: #4caf50; font-weight: bold; }}
            .recommendation {{ background-color: #fff3cd; padding: 10px; margin: 10px 0; border-left: 4px solid #ffc107; }}
        </style>
    </head>
    <body>
        <h1>{title}</h1>
        
        <h2>Overall Metrics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
    """
    
    for key, val in report["overall_metrics"].items():
        html += f"<tr><td>{key}</td><td>{val}</td></tr>"
    
    html += """
        </table>
        
        <h2>Demographic Parity</h2>
        <table>
            <tr><th>Group</th><th>Selection Rate</th><th>Sample Count</th></tr>
    """
    
    if "by_group" in report["demographic_parity"]:
        for group, metrics in report["demographic_parity"]["by_group"].items():
            html += f"<tr><td>{group}</td><td>{metrics['selection_rate']}</td><td>{metrics['count']}</td></tr>"
    
    if "disparate_impact_ratio" in report["demographic_parity"]:
        di_ratio = report["demographic_parity"]["disparate_impact_ratio"]
        passes = "✓ PASS" if report["demographic_parity"]["passes_80_percent_rule"] else "✗ FAIL"
        html += f"<tr><td colspan='3'><strong>Disparate Impact Ratio: {di_ratio} {passes}</strong></td></tr>"
    
    html += """
        </table>
        
        <h2>Recommendations</h2>
    """
    
    for rec in report["recommendations"]:
        html += f'<div class="recommendation">{rec}</div>'
    
    html += """
    </body>
    </html>
    """
    
    # Attempt to save file with permission error handling
    saved_path = None
    if output_path:
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(html)
            saved_path = output_path
        except PermissionError:
            # Fallback to temp directory
            try:
                temp_dir = tempfile.gettempdir()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                saved_path = os.path.join(temp_dir, f"fairness_report_{timestamp}.html")
                with open(saved_path, 'w') as f:
                    f.write(html)
            except Exception:
                # Could not save anywhere
                pass
        except Exception:
            # Other I/O errors - try temp fallback
            try:
                temp_dir = tempfile.gettempdir()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                saved_path = os.path.join(temp_dir, f"fairness_report_{timestamp}.html")
                with open(saved_path, 'w') as f:
                    f.write(html)
            except Exception:
                pass
    
    return html, saved_path


def generate_fairness_markdown_report(y_true: np.ndarray, y_pred: np.ndarray,
                                     protected_attr: Optional[pd.Series] = None,
                                     output_path: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """
    Generate markdown fairness report with AUC/SHAP/bias metrics summary.
    Includes permission error handling and fallback to temp directory.
    
    Args:
        y_true: Ground truth labels
        y_pred: Model predictions (probabilities for AUC calculation)
        protected_attr: Protected attribute groups
        output_path: Path to save markdown file (optional)
    
    Returns:
        Tuple of (markdown string, file_path or None if couldn't write)
    """
    from sklearn.metrics import roc_auc_score
    
    generator = FairnessReportGenerator(y_true, y_pred, protected_attr)
    report = generator.fairness_summary()
    
    # Calculate AUC if predictions appear to be probabilities
    auc_score = None
    if y_pred.min() >= 0 and y_pred.max() <= 1 and y_pred.max() > 0.5:
        try:
            auc_score = roc_auc_score(y_true, y_pred)
        except Exception:
            pass
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Build warnings section
    warnings_text = ''
    if report['data_sufficiency']['warnings']:
        warnings_lines = [f'- {w}' for w in report['data_sufficiency']['warnings']]
        warnings_text = '\n'.join(warnings_lines) + '\n'
    
    markdown = f"""# Fairness Analysis Report

**Generated:** {timestamp}

## Executive Summary

Data Sufficiency Status: **{'✓ Data Sufficient' if report['data_sufficiency']['is_sufficient'] else '⚠️ Data Insufficient'}**

{warnings_text}
## Model Performance Metrics

| Metric | Value |
|--------|-------|
| Accuracy | {report['overall_metrics']['accuracy']:.4f} |
| Precision | {report['overall_metrics']['precision']:.4f} |
| Recall | {report['overall_metrics']['recall']:.4f} |
| Specificity | {report['overall_metrics']['specificity']:.4f} |
| Total Samples | {report['overall_metrics']['total_samples']} |
| Default/Positive Rate | {report['overall_metrics']['positive_rate']:.4f} |
"""
    
    if auc_score is not None:
        markdown += f"""| AUC-ROC | {auc_score:.4f} |
"""
    
    markdown += """
## Fairness Metrics

### Demographic Parity (EEOC 4/5 Rule)

Measures whether approval rates are equal across demographic groups.

"""
    
    if "by_group" in report["demographic_parity"]:
        markdown += "| Group | Approval Rate | Sample Size |\n|-------|---------------|-------------|\n"
        for group, metrics in report["demographic_parity"]["by_group"].items():
            markdown += f"| {group} | {metrics['selection_rate']:.2%} | {metrics['count']} |\n"
        markdown += "\n"
    
    if "disparate_impact_ratio" in report["demographic_parity"]:
        di_ratio = report["demographic_parity"]["disparate_impact_ratio"]
        passes = "✓ PASS" if report["demographic_parity"]["passes_80_percent_rule"] else "✗ FAIL"
        markdown += f"""**Disparate Impact Ratio:** {di_ratio:.4f}  
**80% Rule Compliance:** {passes}  
*(Ratio ≥ 0.80 indicates fair treatment)*

"""
    
    markdown += """### Equalized Odds

Ensures False Positive Rate and True Positive Rate are balanced across groups.

"""
    
    if "by_group" in report["equalized_odds"]:
        markdown += "| Group | FPR | TPR | Sample Size |\n|-------|-----|-----|-------------|\n"
        for group, metrics in report["equalized_odds"]["by_group"].items():
            markdown += f"| {group} | {metrics['fpr']:.4f} | {metrics['tpr']:.4f} | {metrics['count']} |\n"
        markdown += "\n"
    
    if "max_fpr_diff" in report["equalized_odds"]:
        markdown += f"""**Max FPR Difference:** {report['equalized_odds']['max_fpr_diff']:.4f}  
**Max TPR Difference:** {report['equalized_odds']['max_tpr_diff']:.4f}  
"""
    
    markdown += """
### Equal Opportunity

Measures True Positive Rate equality across groups.

"""
    
    if "by_group" in report["equal_opportunity"]:
        markdown += "| Group | TPR | True Positives | False Negatives |\n|-------|-----|---|---|\n"
        for group, metrics in report["equal_opportunity"]["by_group"].items():
            markdown += f"| {group} | {metrics['tpr']:.4f} | {metrics['true_positives']} | {metrics['false_negatives']} |\n"
        markdown += "\n"
    
    if "max_tpr_diff" in report["equal_opportunity"]:
        markdown += f"""**Max TPR Difference:** {report['equal_opportunity']['max_tpr_diff']:.4f}  
**Alert Status:** {'⚠️ Alert' if report['equal_opportunity']['alert'] else '✓ OK'}  
"""
    
    markdown += """
### Predictive Parity

Measures Positive Predictive Value (precision) equality across groups.

"""
    
    if "by_group" in report["predictive_parity"]:
        markdown += "| Group | Precision (PPV) | True Positives | False Positives |\n|-------|---|---|---|\n"
        for group, metrics in report["predictive_parity"]["by_group"].items():
            markdown += f"| {group} | {metrics['ppv']:.4f} | {metrics['true_positives']} | {metrics['false_positives']} |\n"
        markdown += "\n"
    
    if "max_ppv_diff" in report["predictive_parity"]:
        markdown += f"**Max Precision Difference:** {report['predictive_parity']['max_ppv_diff']:.4f}\n\n"
    
    markdown += """## Recommendations & Action Items

"""
    
    for i, rec in enumerate(report["recommendations"], 1):
        markdown += f"{i}. {rec}\n"
    
    markdown += """
## Compliance Notes

- **EEOC Standard:** Selection rate ratio must be ≥ 0.80 (4/5 rule)
- **AI Fairness Framework:** Aligns with NIST AI Risk Management Framework
- **Interpretation:** Lower disparate impact ratio indicates greater disparity

---
*This report is automatically generated for compliance and audit purposes.*
"""
    
    # Attempt to save file with permission error handling
    saved_path = None
    if output_path:
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(markdown)
            saved_path = output_path
        except PermissionError:
            # Fallback to temp directory
            try:
                temp_dir = tempfile.gettempdir()
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                saved_path = os.path.join(temp_dir, f"fairness_report_{timestamp_str}.md")
                with open(saved_path, 'w') as f:
                    f.write(markdown)
            except Exception:
                pass
        except Exception:
            # Other I/O errors - try temp fallback
            try:
                temp_dir = tempfile.gettempdir()
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                saved_path = os.path.join(temp_dir, f"fairness_report_{timestamp_str}.md")
                with open(saved_path, 'w') as f:
                    f.write(markdown)
            except Exception:
                pass
    
    return markdown, saved_path

