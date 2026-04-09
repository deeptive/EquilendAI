"""
Task 14 — Fairness Report Generator
Comprehensive fairness metrics and bias detection for loan approval models.
Implements demographic parity, equalized odds, and equal opportunity metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import confusion_matrix


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
        
        # Ensure binary predictions
        if self.y_pred.min() >= 0 and self.y_pred.max() <= 1:
            if self.y_pred.max() > 0.5:  # Likely probabilities
                self.y_pred_binary = (self.y_pred >= 0.5).astype(int)
            else:
                self.y_pred_binary = self.y_pred.astype(int)
        else:
            self.y_pred_binary = self.y_pred.astype(int)

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
        
        Returns:
            dict with all fairness metrics and recommendations
        """
        summary = {
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
                                 title: str = "Fairness Report") -> str:
    """
    Generate HTML fairness report for documentation/export.
    
    Args:
        y_true: Ground truth labels
        y_pred: Model predictions
        protected_attr: Protected attribute groups
        title: Report title
    
    Returns:
        HTML string of the report
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
    
    return html

