"""
Tests for the Fairness Report Generator module
"""

import numpy as np
import pandas as pd
import pytest
from src.evaluation.fairness import (
    FairnessReportGenerator,
    calculate_fairness_metrics,
    generate_fairness_html_report
)


class TestFairnessReportGenerator:
    """Test suite for FairnessReportGenerator class"""

    @pytest.fixture
    def synthetic_fair_data(self):
        """Create synthetic data with fair predictions"""
        np.random.seed(42)
        n = 200
        
        y_true = np.concatenate([
            np.random.binomial(1, 0.3, n // 2),
            np.random.binomial(1, 0.3, n // 2)
        ])
        
        y_pred = np.concatenate([
            np.random.binomial(1, 0.35, n // 2),
            np.random.binomial(1, 0.35, n // 2)
        ])
        
        protected_attr = pd.Series(
            ['Group_A'] * (n // 2) + ['Group_B'] * (n // 2)
        )
        
        return y_true, y_pred, protected_attr

    @pytest.fixture
    def synthetic_biased_data(self):
        """Create synthetic data with biased predictions"""
        np.random.seed(42)
        n = 200
        
        y_true = np.concatenate([
            np.random.binomial(1, 0.3, n // 2),
            np.random.binomial(1, 0.3, n // 2)
        ])
        
        # Group A has higher approval rate
        y_pred_a = np.random.binomial(1, 0.2, n // 2)  # 20% approval
        y_pred_b = np.random.binomial(1, 0.6, n // 2)  # 60% approval
        y_pred = np.concatenate([y_pred_a, y_pred_b])
        
        protected_attr = pd.Series(
            ['Group_A'] * (n // 2) + ['Group_B'] * (n // 2)
        )
        
        return y_true, y_pred, protected_attr

    def test_initialization(self, synthetic_fair_data):
        """Test FairnessReportGenerator initialization"""
        y_true, y_pred, protected_attr = synthetic_fair_data
        
        generator = FairnessReportGenerator(y_true, y_pred, protected_attr)
        
        assert len(generator.y_true) == len(y_true)
        assert len(generator.y_pred_binary) == len(y_pred)
        assert generator.positive_label == 0

    def test_demographic_parity_fair(self, synthetic_fair_data):
        """Test demographic parity with fair data"""
        y_true, y_pred, protected_attr = synthetic_fair_data
        
        generator = FairnessReportGenerator(y_true, y_pred, protected_attr)
        dp_report = generator.demographic_parity()
        
        assert "by_group" in dp_report
        assert "disparate_impact_ratio" in dp_report
        assert "passes_80_percent_rule" in dp_report
        assert dp_report["passes_80_percent_rule"] is True

    def test_demographic_parity_biased(self, synthetic_biased_data):
        """Test demographic parity with biased data"""
        y_true, y_pred, protected_attr = synthetic_biased_data
        
        generator = FairnessReportGenerator(y_true, y_pred, protected_attr)
        dp_report = generator.demographic_parity()
        
        assert "disparate_impact_ratio" in dp_report
        # With biased data, we expect to fail the 80% rule
        assert dp_report["disparate_impact_ratio"] < 0.80

    def test_equalized_odds(self, synthetic_fair_data):
        """Test equalized odds computation"""
        y_true, y_pred, protected_attr = synthetic_fair_data
        
        generator = FairnessReportGenerator(y_true, y_pred, protected_attr)
        eo_report = generator.equalized_odds()
        
        assert "by_group" in eo_report
        for group in eo_report["by_group"]:
            assert "fpr" in eo_report["by_group"][group]
            assert "tpr" in eo_report["by_group"][group]

    def test_equal_opportunity(self, synthetic_fair_data):
        """Test equal opportunity metrics"""
        y_true, y_pred, protected_attr = synthetic_fair_data
        
        generator = FairnessReportGenerator(y_true, y_pred, protected_attr)
        eop_report = generator.equal_opportunity()
        
        assert "by_group" in eop_report
        assert "max_tpr_diff" in eop_report
        assert eop_report["max_tpr_diff"] >= 0

    def test_predictive_parity(self, synthetic_fair_data):
        """Test predictive parity computation"""
        y_true, y_pred, protected_attr = synthetic_fair_data
        
        generator = FairnessReportGenerator(y_true, y_pred, protected_attr)
        pp_report = generator.predictive_parity()
        
        assert "by_group" in pp_report
        for group in pp_report["by_group"]:
            assert "ppv" in pp_report["by_group"][group]

    def test_fairness_summary(self, synthetic_fair_data):
        """Test complete fairness summary generation"""
        y_true, y_pred, protected_attr = synthetic_fair_data
        
        generator = FairnessReportGenerator(y_true, y_pred, protected_attr)
        summary = generator.fairness_summary()
        
        assert "overall_metrics" in summary
        assert "demographic_parity" in summary
        assert "equalized_odds" in summary
        assert "equal_opportunity" in summary
        assert "predictive_parity" in summary
        assert "recommendations" in summary

    def test_overall_metrics(self, synthetic_fair_data):
        """Test overall performance metrics"""
        y_true, y_pred, protected_attr = synthetic_fair_data
        
        generator = FairnessReportGenerator(y_true, y_pred, protected_attr)
        metrics = generator._get_overall_metrics()
        
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "specificity" in metrics
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["precision"] <= 1

    def test_recommendations_generation(self, synthetic_fair_data):
        """Test recommendation generation"""
        y_true, y_pred, protected_attr = synthetic_fair_data
        
        generator = FairnessReportGenerator(y_true, y_pred, protected_attr)
        recommendations = generator._generate_recommendations()
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

    def test_to_dataframe(self, synthetic_fair_data):
        """Test DataFrame conversion"""
        y_true, y_pred, protected_attr = synthetic_fair_data
        
        generator = FairnessReportGenerator(y_true, y_pred, protected_attr)
        df = generator.to_dataframe()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert all(col in df.columns for col in ["Category", "Metric", "Value"])

    def test_without_protected_attr(self, synthetic_fair_data):
        """Test behavior when protected_attr is None"""
        y_true, y_pred, _ = synthetic_fair_data
        
        generator = FairnessReportGenerator(y_true, y_pred, protected_attr=None)
        dp_report = generator.demographic_parity()
        
        assert "error" in dp_report


class TestUtilityFunctions:
    """Test utility functions"""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        np.random.seed(42)
        y_true = np.random.binomial(1, 0.3, 100)
        y_pred = np.random.binomial(1, 0.35, 100)
        protected_attr = pd.Series(['A'] * 50 + ['B'] * 50)
        
        return y_true, y_pred, protected_attr

    def test_calculate_fairness_metrics(self, sample_data):
        """Test calculate_fairness_metrics function"""
        y_true, y_pred, protected_attr = sample_data
        
        result = calculate_fairness_metrics(y_true, y_pred, protected_attr)
        
        assert isinstance(result, dict)
        assert "overall_metrics" in result
        assert "recommendations" in result

    def test_generate_fairness_html_report(self, sample_data):
        """Test HTML report generation"""
        y_true, y_pred, protected_attr = sample_data
        
        html_report = generate_fairness_html_report(y_true, y_pred, protected_attr)
        
        assert isinstance(html_report, str)
        assert "<html>" in html_report
        assert "Fairness Report" in html_report


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_perfect_predictions(self):
        """Test with perfect model predictions"""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1])
        protected_attr = pd.Series(['A', 'A', 'A', 'B', 'B', 'B'])
        
        generator = FairnessReportGenerator(y_true, y_pred, protected_attr)
        summary = generator.fairness_summary()
        
        assert summary["overall_metrics"]["accuracy"] == 1.0

    def test_random_predictions(self):
        """Test with random predictions"""
        np.random.seed(42)
        y_true = np.random.binomial(1, 0.5, 100)
        y_pred = np.random.rand(100)
        protected_attr = pd.Series(['A'] * 50 + ['B'] * 50)
        
        generator = FairnessReportGenerator(y_true, y_pred, protected_attr)
        summary = generator.fairness_summary()
        
        assert "overall_metrics" in summary

    def test_very_small_dataset(self):
        """Test with very small dataset"""
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])
        protected_attr = pd.Series(['A', 'B'])
        
        generator = FairnessReportGenerator(y_true, y_pred, protected_attr)
        summary = generator.fairness_summary()
        
        assert summary is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
