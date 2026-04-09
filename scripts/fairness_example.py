"""
Example: Using the Fairness Report Generator
Demonstrates how to analyze fairness metrics for loan approval predictions
"""

import numpy as np
import pandas as pd
from src.evaluation.fairness import (
    FairnessReportGenerator,
    calculate_fairness_metrics,
    generate_fairness_html_report
)


def generate_example_data():
    """Generate synthetic loan approval data with demographic information"""
    np.random.seed(42)
    n_samples = 500
    
    # Create features
    income = np.random.exponential(50000, n_samples)
    credit_score = np.random.normal(650, 100, n_samples)
    debt_ratio = np.random.uniform(0, 1, n_samples)
    
    # Create ground truth labels (1 = defaulted, 0 = paid)
    y_true = (income < 40000).astype(int)
    
    # Model predictions (probabilities of default)
    y_prob = 0.3 * (income < 40000) + 0.2 * (credit_score < 620) + 0.1 * (debt_ratio > 0.5)
    y_prob = np.clip(y_prob, 0, 1)
    
    # Model predictions (binary: 0 = approved, 1 = denied)
    y_pred = (y_prob > 0.5).astype(int)
    
    # Create demographic information
    gender = np.random.choice(['Male', 'Female'], n_samples)
    age_group = np.random.choice(['18-25', '26-35', '36-50', '50+'], n_samples)
    
    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'gender': pd.Series(gender),
        'age_group': pd.Series(age_group),
        'income': income,
        'credit_score': credit_score,
        'debt_ratio': debt_ratio
    }


def example_basic_usage():
    """Example 1: Basic usage — generate fairness report for gender"""
    print("=" * 80)
    print("EXAMPLE 1: Basic Fairness Report Generation")
    print("=" * 80)
    
    data = generate_example_data()
    
    # Create fairness report generator
    report_gen = FairnessReportGenerator(
        y_true=data['y_true'],
        y_pred=data['y_pred'],
        protected_attr=data['gender']
    )
    
    # Generate summary
    summary = report_gen.fairness_summary()
    
    # Display overall metrics
    print("\n📊 OVERALL PERFORMANCE METRICS")
    print("-" * 40)
    for metric, value in summary['overall_metrics'].items():
        print(f"  {metric:.<30} {value}")
    
    # Display demographic parity
    print("\n⚖️  DEMOGRAPHIC PARITY (Selection Rates by Gender)")
    print("-" * 40)
    for group, metrics in summary['demographic_parity']['by_group'].items():
        print(f"  {group:.<30} {metrics['selection_rate']} (n={metrics['count']})")
    
    if 'disparate_impact_ratio' in summary['demographic_parity']:
        di_ratio = summary['demographic_parity']['disparate_impact_ratio']
        passes = "✓ PASS" if summary['demographic_parity']['passes_80_percent_rule'] else "✗ FAIL"
        print(f"\n  Disparate Impact Ratio: {di_ratio} {passes}")
    
    # Display recommendations
    print("\n💡 RECOMMENDATIONS")
    print("-" * 40)
    for i, rec in enumerate(summary['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    return summary


def example_comparative_analysis():
    """Example 2: Comparative analysis across multiple protected attributes"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Comparative Analysis (Gender vs Age Group)")
    print("=" * 80)
    
    data = generate_example_data()
    
    print("\n▶ Analysis by GENDER:")
    print("-" * 40)
    report_gender = FairnessReportGenerator(
        y_true=data['y_true'],
        y_pred=data['y_pred'],
        protected_attr=data['gender']
    )
    summary_gender = report_gender.fairness_summary()
    dp_gender = summary_gender['demographic_parity']
    print(f"  Disparate Impact Ratio: {dp_gender.get('disparate_impact_ratio', 'N/A')}")
    
    print("\n▶ Analysis by AGE GROUP:")
    print("-" * 40)
    report_age = FairnessReportGenerator(
        y_true=data['y_true'],
        y_pred=data['y_pred'],
        protected_attr=data['age_group']
    )
    summary_age = report_age.fairness_summary()
    
    print("  Equalized Odds (TPR by group):")
    for group, metrics in summary_age['equalized_odds']['by_group'].items():
        print(f"    {group:.<20} TPR={metrics['tpr']}, FPR={metrics['fpr']}")


def example_utility_functions():
    """Example 3: Using utility functions for quick analysis"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Utility Functions")
    print("=" * 80)
    
    data = generate_example_data()
    
    # Quick metrics calculation
    print("\n✓ Calculate all fairness metrics with one function:")
    metrics = calculate_fairness_metrics(
        y_true=data['y_true'],
        y_pred=data['y_pred'],
        protected_attr=data['gender']
    )
    
    print(f"  Total metrics collected: {len(metrics)}")
    print(f"  Keys: {', '.join(metrics.keys())}")
    
    # Generate HTML report
    print("\n✓ Generate HTML report for export:")
    html_report = generate_fairness_html_report(
        y_true=data['y_true'],
        y_pred=data['y_pred'],
        protected_attr=data['gender'],
        title="EquiLend AI — Monthly Fairness Audit"
    )
    
    print(f"  HTML report generated: {len(html_report)} characters")
    print("  Sample HTML snippet:")
    print(f"    {html_report[:200]}...")


def example_dataframe_export():
    """Example 4: Export fairness report to DataFrame"""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Export to DataFrame")
    print("=" * 80)
    
    data = generate_example_data()
    
    report_gen = FairnessReportGenerator(
        y_true=data['y_true'],
        y_pred=data['y_pred'],
        protected_attr=data['gender']
    )
    
    df = report_gen.to_dataframe()
    
    print("\n📋 Fairness Report as DataFrame:")
    print("-" * 60)
    print(df.to_string())
    
    return df


def example_interpretation_guidelines():
    """Example 5: How to interpret fairness metrics"""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Interpretation Guidelines")
    print("=" * 80)
    
    print("""
DISPARATE IMPACT RATIO (0-1 scale):
  ≥ 0.80  ✓ PASS  — Model passes the "4/5 rule" (legal threshold)
  < 0.80  ✗ FAIL  — Potential illegal discrimination
  
EQUALIZED ODDS:
  FPR Difference < 0.10  ✓ Good   — False positive rates are similar
  TPR Difference < 0.10  ✓ Good   — True positive rates are similar
  Differences > 0.15     ✗ Alert  — Significant bias detected

EQUAL OPPORTUNITY (TPR):
  Max Difference < 0.10  ✓ Good   — Equal opportunity across groups
  Max Difference > 0.15  ✗ Alert  — Some groups face higher false negative rates

PREDICTIVE PARITY (Precision):
  Low variance           ✓ Good   — Predictions equally reliable for all groups
  High variance          ✗ Alert  — Some groups get less reliable predictions

ACTION ITEMS:
  1. If Disparate Impact Ratio < 0.80:
     → Review thresholds, reweight samples, or retrain model
  
  2. If Equalized Odds fails:
     → Investigate cost-sensitive learning
     → Consider group-specific thresholds (where legal)
  
  3. If Equal Opportunity fails:
     → Ensure diverse training data
     → Check for missing features for underrepresented groups
    """)


def main():
    """Run all examples"""
    print("\n" + "=" * 80)
    print("FAIRNESS REPORT GENERATOR — USAGE EXAMPLES")
    print("=" * 80)
    
    # Run examples
    example_basic_usage()
    example_comparative_analysis()
    example_utility_functions()
    example_dataframe_export()
    example_interpretation_guidelines()
    
    print("\n" + "=" * 80)
    print("Examples completed! See Fairness_Report.md for full documentation.")
    print("=" * 80)


if __name__ == "__main__":
    main()
