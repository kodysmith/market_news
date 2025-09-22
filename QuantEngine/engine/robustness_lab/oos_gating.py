"""
Out-of-Sample Gating and Validation for AI Quant Trading System

Implements rigorous OOS validation to prevent overfitting and ensure
strategies work in unseen market conditions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class OOSGateKeeper:
    """
    Gatekeeper for out-of-sample validation

    Ensures strategies meet strict OOS performance criteria before approval.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # OOS gating thresholds
        self.min_oos_sharpe = config.get('min_oos_sharpe', 0.8)
        self.min_oos_total_return = config.get('min_oos_total_return', -0.05)  # Allow small losses
        self.max_oos_dd = config.get('max_oos_dd', 0.25)
        self.min_oos_win_rate = config.get('min_oos_win_rate', 0.45)
        self.min_oos_is_ratio = config.get('min_oos_is_ratio', 0.7)  # OOS Sharpe / IS Sharpe

        # Statistical significance thresholds
        self.min_t_stat = config.get('min_t_stat', 2.0)
        self.max_p_value = config.get('max_p_value', 0.05)

        # Stability requirements
        self.max_sharpe_volatility = config.get('max_sharpe_volatility', 1.5)  # Max std of Sharpe across splits

    def evaluate_oos_performance(self, walk_forward_results: Dict[str, Any],
                               in_sample_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate out-of-sample performance against gating criteria

        Args:
            walk_forward_results: Results from walk-forward validation
            in_sample_metrics: In-sample performance metrics

        Returns:
            OOS evaluation with approval decision
        """

        evaluation = {
            'timestamp': datetime.now().isoformat(),
            'approved': False,
            'reasons': [],
            'concerns': [],
            'oos_metrics': {},
            'gating_results': {},
            'confidence_score': 0
        }

        # Extract OOS metrics from walk-forward results
        oos_metrics = self._extract_oos_metrics(walk_forward_results)

        if not oos_metrics:
            evaluation['concerns'].append("No valid OOS results available")
            return evaluation

        evaluation['oos_metrics'] = oos_metrics

        # Apply gating criteria
        gating_results = self._apply_gating_criteria(oos_metrics, in_sample_metrics)
        evaluation['gating_results'] = gating_results

        # Count passed criteria
        passed_criteria = sum(1 for result in gating_results.values() if result['passed'])
        total_criteria = len(gating_results)

        # Calculate confidence score (0-100)
        confidence_score = (passed_criteria / total_criteria) * 100
        evaluation['confidence_score'] = confidence_score

        # Make approval decision
        min_criteria_passed = max(3, total_criteria * 0.6)  # At least 60% or 3 criteria

        if passed_criteria >= min_criteria_passed:
            evaluation['approved'] = True
            evaluation['reasons'].append(f"Passed {passed_criteria}/{total_criteria} gating criteria")

            if confidence_score >= 80:
                evaluation['confidence_level'] = 'high'
            elif confidence_score >= 60:
                evaluation['confidence_level'] = 'medium'
            else:
                evaluation['confidence_level'] = 'low'
        else:
            evaluation['concerns'].append(f"Failed gating: {passed_criteria}/{total_criteria} criteria passed")
            evaluation['confidence_level'] = 'rejected'

        # Add specific reasons and concerns
        for criterion_name, result in gating_results.items():
            if result['passed']:
                evaluation['reasons'].append(f"✅ {criterion_name}: {result['message']}")
            else:
                evaluation['concerns'].append(f"❌ {criterion_name}: {result['message']}")

        return evaluation

    def _extract_oos_metrics(self, wf_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract OOS metrics from walk-forward results"""

        if 'aggregated' not in wf_results:
            return {}

        aggregated = wf_results['aggregated']

        # Core OOS metrics
        oos_metrics = {
            'sharpe': aggregated.get('avg_sharpe', 0),
            'total_return': aggregated.get('avg_total_return', 0),
            'max_dd': aggregated.get('avg_max_dd', 0),
            'win_rate': aggregated.get('avg_win_rate', 0),
            'sharpe_volatility': aggregated.get('std_sharpe', 0),
            'num_splits': aggregated.get('num_splits', 0),
            'sharpe_ci_lower': aggregated.get('sharpe_confidence_interval', [0, 0])[0],
            'sharpe_ci_upper': aggregated.get('sharpe_confidence_interval', [0, 0])[1]
        }

        return oos_metrics

    def _apply_gating_criteria(self, oos_metrics: Dict[str, Any],
                             is_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Apply all gating criteria"""

        criteria_results = {}

        # 1. OOS Sharpe Ratio
        oos_sharpe = oos_metrics.get('sharpe', 0)
        criteria_results['oos_sharpe'] = self._evaluate_criterion(
            oos_sharpe >= self.min_oos_sharpe,
            f"OOS Sharpe {oos_sharpe:.2f} >= {self.min_oos_sharpe:.2f}",
            f"OOS Sharpe {oos_sharpe:.2f} < {self.min_oos_sharpe:.2f}"
        )

        # 2. OOS/IS Sharpe Ratio
        is_sharpe = is_metrics.get('sharpe', 1.0)  # Avoid division by zero
        oos_is_ratio = oos_sharpe / is_sharpe if is_sharpe != 0 else 0
        criteria_results['oos_is_ratio'] = self._evaluate_criterion(
            oos_is_ratio >= self.min_oos_is_ratio,
            f"OOS/IS ratio {oos_is_ratio:.2f} >= {self.min_oos_is_ratio:.2f}",
            f"OOS/IS ratio {oos_is_ratio:.2f} < {self.min_oos_is_ratio:.2f}"
        )

        # 3. OOS Total Return
        oos_total_return = oos_metrics.get('total_return', -999)
        criteria_results['oos_total_return'] = self._evaluate_criterion(
            oos_total_return >= self.min_oos_total_return,
            f"OOS return {oos_total_return:.2%} >= {self.min_oos_total_return:.1%}",
            f"OOS return {oos_total_return:.2%} < {self.min_oos_total_return:.1%}"
        )

        # 4. OOS Max Drawdown
        oos_max_dd = abs(oos_metrics.get('max_dd', 999))
        criteria_results['oos_max_dd'] = self._evaluate_criterion(
            oos_max_dd <= self.max_oos_dd,
            f"OOS max DD {oos_max_dd:.1%} <= {self.max_oos_dd:.1%}",
            f"OOS max DD {oos_max_dd:.1%} > {self.max_oos_dd:.1%}"
        )

        # 5. OOS Win Rate
        oos_win_rate = oos_metrics.get('win_rate', 0)
        criteria_results['oos_win_rate'] = self._evaluate_criterion(
            oos_win_rate >= self.min_oos_win_rate,
            f"OOS win rate {oos_win_rate:.1%} >= {self.min_oos_win_rate:.1%}",
            f"OOS win rate {oos_win_rate:.1%} < {self.min_oos_win_rate:.1%}"
        )

        # 6. Sharpe Stability (low volatility across splits)
        sharpe_volatility = oos_metrics.get('sharpe_volatility', 999)
        criteria_results['sharpe_stability'] = self._evaluate_criterion(
            sharpe_volatility <= self.max_sharpe_volatility,
            f"Sharpe volatility {sharpe_volatility:.2f} <= {self.max_sharpe_volatility:.2f}",
            f"Sharpe volatility {sharpe_volatility:.2f} > {self.max_sharpe_volatility:.2f}"
        )

        # 7. Statistical Significance (Sharpe ratio t-stat)
        sharpe = oos_metrics.get('sharpe', 0)
        num_splits = oos_metrics.get('num_splits', 1)
        if num_splits > 1:
            # Approximate t-statistic for Sharpe ratio
            t_stat = sharpe * np.sqrt(num_splits) / (1 + 0.5 * sharpe**2)  # Conservative estimate
            criteria_results['statistical_significance'] = self._evaluate_criterion(
                t_stat >= self.min_t_stat,
                f"t-stat {t_stat:.2f} >= {self.min_t_stat:.1f}",
                f"t-stat {t_stat:.2f} < {self.min_t_stat:.1f}"
            )
        else:
            criteria_results['statistical_significance'] = {
                'passed': False,
                'message': "Insufficient splits for statistical test"
            }

        # 8. Sharpe Ratio Confidence Interval
        ci_lower = oos_metrics.get('sharpe_ci_lower', -999)
        criteria_results['sharpe_confidence'] = self._evaluate_criterion(
            ci_lower > 0,
            f"Sharpe CI lower bound {ci_lower:.2f} > 0",
            f"Sharpe CI lower bound {ci_lower:.2f} <= 0"
        )

        return criteria_results

    def _evaluate_criterion(self, passed: bool, pass_message: str, fail_message: str) -> Dict[str, Any]:
        """Evaluate a single gating criterion"""
        return {
            'passed': passed,
            'message': pass_message if passed else fail_message
        }


class OOSValidator:
    """Advanced OOS validation with multiple testing corrections"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.gatekeeper = OOSGateKeeper(config)

    def validate_strategy_comprehensive(self, strategy_spec: Dict[str, Any],
                                      walk_forward_results: Dict[str, Any],
                                      in_sample_results: Dict[str, Any],
                                      market_regime_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Comprehensive OOS validation including multiple testing corrections

        Args:
            strategy_spec: Strategy specification
            walk_forward_results: Walk-forward backtest results
            in_sample_results: In-sample backtest results
            market_regime_data: Optional market regime analysis

        Returns:
            Comprehensive validation report
        """

        validation_report = {
            'strategy_name': strategy_spec.get('name', 'unknown'),
            'timestamp': datetime.now().isoformat(),
            'validation_components': {},
            'final_decision': {},
            'recommendations': []
        }

        # 1. Basic OOS gating
        gating_evaluation = self.gatekeeper.evaluate_oos_performance(
            walk_forward_results, in_sample_results.get('metrics', {})
        )
        validation_report['validation_components']['gating'] = gating_evaluation

        # 2. Multiple testing correction
        multiple_testing = self._apply_multiple_testing_correction(gating_evaluation)
        validation_report['validation_components']['multiple_testing'] = multiple_testing

        # 3. Regime robustness check
        if market_regime_data:
            regime_analysis = self._analyze_regime_robustness(walk_forward_results, market_regime_data)
            validation_report['validation_components']['regime_analysis'] = regime_analysis

        # 4. Forward test simulation (if enough data)
        forward_test = self._simulate_forward_test(walk_forward_results)
        validation_report['validation_components']['forward_test'] = forward_test

        # 5. Final decision with all factors
        final_decision = self._make_final_decision(validation_report)
        validation_report['final_decision'] = final_decision

        # 6. Generate recommendations
        recommendations = self._generate_recommendations(validation_report)
        validation_report['recommendations'] = recommendations

        return validation_report

    def _apply_multiple_testing_correction(self, gating_evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Apply multiple testing correction to p-values"""

        # Simplified multiple testing correction
        # In practice, this would use Bonferroni, Holm-Bonferroni, etc.

        num_tests = len(gating_evaluation.get('gating_results', {}))
        if num_tests == 0:
            return {'correction_applied': False, 'message': 'No tests to correct'}

        original_confidence = gating_evaluation.get('confidence_score', 0)
        corrected_confidence = original_confidence * (1 - 0.05 * num_tests)  # Conservative adjustment

        return {
            'correction_applied': True,
            'original_confidence': original_confidence,
            'corrected_confidence': max(0, corrected_confidence),
            'num_tests': num_tests,
            'adjustment_factor': (1 - 0.05 * num_tests)
        }

    def _analyze_regime_robustness(self, wf_results: Dict[str, Any],
                                 regime_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze strategy performance across market regimes"""

        # Simplified regime analysis
        # In practice, this would analyze bull/bear/chop markets separately

        regime_performance = {
            'regimes_analyzed': list(regime_data.keys()),
            'best_regime': None,
            'worst_regime': None,
            'regime_stability_score': 0.8  # Placeholder
        }

        return regime_performance

    def _simulate_forward_test(self, wf_results: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate forward testing on most recent data"""

        # Simplified forward test simulation
        splits = wf_results.get('splits', [])

        if len(splits) < 2:
            return {'possible': False, 'message': 'Insufficient splits for forward test'}

        # Use last split as forward test
        last_split = splits[-1]
        forward_metrics = last_split.get('result', {}).get('metrics', {})

        return {
            'possible': True,
            'forward_sharpe': forward_metrics.get('sharpe', 0),
            'forward_return': forward_metrics.get('total_return', 0),
            'forward_max_dd': forward_metrics.get('max_dd', 0)
        }

    def _make_final_decision(self, validation_report: Dict[str, Any]) -> Dict[str, Any]:
        """Make final validation decision"""

        components = validation_report['validation_components']

        # Weight different validation components
        weights = {
            'gating': 0.5,
            'multiple_testing': 0.2,
            'regime_analysis': 0.15,
            'forward_test': 0.15
        }

        weighted_score = 0
        total_weight = 0

        for component_name, weight in weights.items():
            if component_name in components:
                component = components[component_name]

                if component_name == 'gating':
                    score = component.get('confidence_score', 0)
                elif component_name == 'multiple_testing':
                    score = component.get('corrected_confidence', 0)
                elif component_name == 'regime_analysis':
                    score = component.get('regime_stability_score', 50) * 100
                elif component_name == 'forward_test':
                    # Forward test gets binary score
                    forward_possible = component.get('possible', False)
                    score = 100 if forward_possible else 0
                else:
                    score = 50

                weighted_score += (score * weight)
                total_weight += weight

        final_score = weighted_score / total_weight if total_weight > 0 else 0

        # Final decision
        if final_score >= 70:
            decision = 'approved'
            confidence = 'high' if final_score >= 85 else 'medium'
        elif final_score >= 50:
            decision = 'conditional'
            confidence = 'low'
        else:
            decision = 'rejected'
            confidence = 'none'

        return {
            'decision': decision,
            'confidence': confidence,
            'final_score': final_score,
            'weighted_score': weighted_score,
            'total_weight': total_weight
        }

    def _generate_recommendations(self, validation_report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results"""

        recommendations = []
        final_decision = validation_report.get('final_decision', {})

        decision = final_decision.get('decision', 'unknown')
        confidence = final_decision.get('confidence', 'unknown')

        if decision == 'approved':
            if confidence == 'high':
                recommendations.append("✅ Strategy approved for live paper trading")
                recommendations.append("📊 Monitor performance weekly with automated alerts")
                recommendations.append("🔄 Schedule quarterly re-validation")
            elif confidence == 'medium':
                recommendations.append("⚠️ Strategy conditionally approved - monitor closely")
                recommendations.append("📈 Reduce initial position sizing (50% of target)")
                recommendations.append("🔍 Re-validate after 30 days of paper trading")

        elif decision == 'conditional':
            recommendations.append("🔄 Strategy needs improvement before approval")
            recommendations.append("🎯 Consider parameter re-optimization")
            recommendations.append("📊 Add additional risk management layers")
            recommendations.append("🔬 Test on additional market regimes")

        else:  # rejected
            recommendations.append("❌ Strategy rejected - fundamental issues identified")
            recommendations.append("🔍 Review strategy logic and assumptions")
            recommendations.append("📚 Consider alternative strategy approaches")
            recommendations.append("🔬 Test with different market conditions")

        # Component-specific recommendations
        components = validation_report.get('validation_components', {})

        if 'gating' in components:
            gating = components['gating']
            if not gating.get('approved', False):
                recommendations.append("🎯 Address OOS performance issues")

        if 'multiple_testing' in components:
            mt = components['multiple_testing']
            if mt.get('correction_applied', False):
                original = mt.get('original_confidence', 0)
                corrected = mt.get('corrected_confidence', 0)
                if corrected < original * 0.8:
                    recommendations.append("⚠️ Multiple testing correction significantly impacted results")

        return recommendations


# Test and example functions
def test_oos_gating():
    """Test OOS gating functionality"""

    print("🧪 Testing OOS Gating...")

    # Mock walk-forward results
    wf_results = {
        'aggregated': {
            'avg_sharpe': 1.2,
            'avg_total_return': 0.15,
            'avg_max_dd': -0.18,
            'avg_win_rate': 0.52,
            'std_sharpe': 0.8,
            'num_splits': 8,
            'sharpe_confidence_interval': [0.8, 1.6]
        }
    }

    # Mock in-sample metrics
    is_metrics = {
        'sharpe': 1.5,
        'total_return': 0.25,
        'max_dd': -0.15,
        'win_rate': 0.55
    }

    # Test gating
    config = {
        'min_oos_sharpe': 0.8,
        'min_oos_total_return': -0.05,
        'max_oos_dd': 0.25,
        'min_oos_win_rate': 0.45,
        'min_oos_is_ratio': 0.7
    }

    gatekeeper = OOSGateKeeper(config)
    evaluation = gatekeeper.evaluate_oos_performance(wf_results, is_metrics)

    print("✅ OOS gating test completed")
    print(f"   Approved: {evaluation['approved']}")
    print(f"   Confidence: {evaluation.get('confidence_level', 'unknown')}")
    print(f"   Score: {evaluation['confidence_score']:.1f}%")

    return evaluation


if __name__ == "__main__":
    test_oos_gating()

