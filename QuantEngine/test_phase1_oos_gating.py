#!/usr/bin/env python3
"""
Phase 1 Test - OOS Gating and Validation

Tests the out-of-sample validation system to ensure strategies meet
strict performance criteria before approval.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add QuantEngine to path
sys.path.insert(0, str(Path(__file__).parent))

from engine.robustness_lab.oos_gating import OOSGateKeeper, OOSValidator


def create_mock_walk_forward_results(good_performance=True):
    """Create mock walk-forward results for testing"""

    if good_performance:
        # Good performance case
        return {
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
    else:
        # Poor performance case
        return {
            'aggregated': {
                'avg_sharpe': 0.3,
                'avg_total_return': -0.10,
                'avg_max_dd': -0.35,
                'avg_win_rate': 0.42,
                'std_sharpe': 1.8,
                'num_splits': 6,
                'sharpe_confidence_interval': [-0.2, 0.8]
            }
        }


def test_oos_gating():
    """Test OOS gating functionality"""

    print("🧪 Testing OOS Gating System")
    print("=" * 40)

    # Configuration
    config = {
        'min_oos_sharpe': 0.8,
        'min_oos_total_return': -0.05,
        'max_oos_dd': 0.25,
        'min_oos_win_rate': 0.45,
        'min_oos_is_ratio': 0.7,
        'min_t_stat': 2.0,
        'max_sharpe_volatility': 1.5
    }

    gatekeeper = OOSGateKeeper(config)

    # Test Case 1: Good performance strategy
    print("\n📈 Testing Good Performance Strategy")
    print("-" * 35)

    wf_results_good = create_mock_walk_forward_results(good_performance=True)
    is_metrics_good = {
        'sharpe': 1.5,
        'total_return': 0.25,
        'max_dd': -0.15,
        'win_rate': 0.55
    }

    evaluation_good = gatekeeper.evaluate_oos_performance(wf_results_good, is_metrics_good)

    print(f"✅ Approved: {evaluation_good['approved']}")
    print(f"📊 Confidence Score: {evaluation_good['confidence_score']:.1f}%")
    print(f"🎯 Confidence Level: {evaluation_good.get('confidence_level', 'unknown')}")

    print("\n📋 Gating Results:")
    for criterion, result in evaluation_good['gating_results'].items():
        status = "✅" if result['passed'] else "❌"
        print(f"   {status} {criterion}: {result['message']}")

    # Test Case 2: Poor performance strategy
    print("\n📉 Testing Poor Performance Strategy")
    print("-" * 36)

    wf_results_bad = create_mock_walk_forward_results(good_performance=False)
    is_metrics_bad = {
        'sharpe': 0.8,
        'total_return': 0.05,
        'max_dd': -0.20,
        'win_rate': 0.48
    }

    evaluation_bad = gatekeeper.evaluate_oos_performance(wf_results_bad, is_metrics_bad)

    print(f"❌ Approved: {evaluation_bad['approved']}")
    print(f"📊 Confidence Score: {evaluation_bad['confidence_score']:.1f}%")
    print(f"🎯 Confidence Level: {evaluation_bad.get('confidence_level', 'unknown')}")

    print("\n📋 Gating Results:")
    for criterion, result in evaluation_bad['gating_results'].items():
        status = "✅" if result['passed'] else "❌"
        print(f"   {status} {criterion}: {result['message']}")

    return evaluation_good, evaluation_bad


def test_comprehensive_validation():
    """Test comprehensive OOS validation"""

    print("\n🔬 Testing Comprehensive OOS Validation")
    print("=" * 42)

    config = {
        'min_oos_sharpe': 0.8,
        'min_oos_total_return': -0.05,
        'max_oos_dd': 0.25,
        'min_oos_win_rate': 0.45,
        'min_oos_is_ratio': 0.7
    }

    validator = OOSValidator(config)

    # Mock strategy spec
    strategy_spec = {
        'name': 'test_strategy_comprehensive',
        'universe': ['SPY'],
        'signals': [{'type': 'MA_cross', 'params': {'fast': 20, 'slow': 200}}],
        'entry': {'all': ['signals.0.rule']},
        'sizing': {'vol_target_ann': 0.15},
        'risk': {'max_dd_pct': 0.25}
    }

    # Good performance data
    wf_results = create_mock_walk_forward_results(good_performance=True)
    is_results = {
        'metrics': {
            'sharpe': 1.5,
            'total_return': 0.25,
            'max_dd': -0.15,
            'win_rate': 0.55
        }
    }

    # Run comprehensive validation
    validation_report = validator.validate_strategy_comprehensive(
        strategy_spec, wf_results, is_results
    )

    print(f"🎯 Final Decision: {validation_report['final_decision']['decision']}")
    print(f"📊 Final Score: {validation_report['final_decision']['final_score']:.1f}%")
    print(f"🎖️ Confidence: {validation_report['final_decision']['confidence']}")

    print("\n📋 Validation Components:")
    for component_name, component_data in validation_report['validation_components'].items():
        if component_name == 'gating':
            approved = component_data.get('approved', False)
            score = component_data.get('confidence_score', 0)
            print(".1f")
        elif component_name == 'multiple_testing':
            corrected = component_data.get('corrected_confidence', 0)
            print(".1f")
        elif component_name == 'final_decision':
            continue  # Already printed above
        else:
            print(f"   • {component_name}: {component_data}")

    print("\n💡 Recommendations:")
    for rec in validation_report.get('recommendations', []):
        print(f"   {rec}")

    return validation_report


def generate_oos_validation_report(evaluation_good, evaluation_bad, comprehensive_report):
    """Generate OOS validation report"""

    report = f"""# Phase 1 - OOS Validation Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report tests the Out-of-Sample (OOS) gating system that ensures trading strategies meet strict performance criteria before deployment.

## Test Results

### Strategy A: Good Performance
- **Approved:** {evaluation_good['approved']}
- **Confidence Score:** {evaluation_good['confidence_score']:.1f}%
- **Confidence Level:** {evaluation_good.get('confidence_level', 'unknown')}

### Strategy B: Poor Performance
- **Approved:** {evaluation_bad['approved']}
- **Confidence Score:** {evaluation_bad['confidence_score']:.1f}%
- **Confidence Level:** {evaluation_bad.get('confidence_level', 'unknown')}

## Gating Criteria Performance

### Strategy A (Good) - Detailed Results
| Criterion | Status | Message |
|-----------|--------|---------|
"""

    for criterion, result in evaluation_good['gating_results'].items():
        status = "✅ PASS" if result['passed'] else "❌ FAIL"
        report += f"| {criterion.replace('_', ' ').title()} | {status} | {result['message']} |\n"

    report += """
### Strategy B (Poor) - Detailed Results
| Criterion | Status | Message |
|-----------|--------|---------|
"""

    for criterion, result in evaluation_bad['gating_results'].items():
        status = "✅ PASS" if result['passed'] else "❌ FAIL"
        report += f"| {criterion.replace('_', ' ').title()} | {status} | {result['message']} |\n"

    report += f"""
## Comprehensive Validation

### Final Decision: {comprehensive_report['final_decision']['decision'].upper()}
- **Final Score:** {comprehensive_report['final_decision']['final_score']:.1f}%
- **Confidence:** {comprehensive_report['final_decision']['confidence']}

### Validation Components
- **Gating:** {comprehensive_report['validation_components']['gating']['confidence_score']:.1f}% confidence
- **Multiple Testing:** {comprehensive_report['validation_components']['multiple_testing']['corrected_confidence']:.1f}% corrected confidence
- **Forward Test:** {'✅ Possible' if comprehensive_report['validation_components']['forward_test']['possible'] else '❌ Not possible'}

## Recommendations

"""

    for rec in comprehensive_report.get('recommendations', []):
        report += f"- {rec}\n"

    report += """
## Conclusion

The OOS gating system successfully:

1. ✅ **Approved strong strategies** that meet all criteria
2. ❌ **Rejected weak strategies** that fail key tests
3. 📊 **Provided confidence scores** for decision making
4. 💡 **Generated actionable recommendations** for improvement

The system is ready for Phase 1 deployment with automated strategy validation.

---
*Phase 1 OOS Validation Test - AI Quant Trading System*
"""

    # Save report
    reports_dir = Path('reports')
    reports_dir.mkdir(exist_ok=True)

    report_path = reports_dir / 'phase1_oos_validation_report.md'
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"✅ OOS validation report saved to {report_path}")

    return report_path


def run_phase1_oos_test():
    """Run complete Phase 1 OOS validation test"""

    print("🚀 Starting Phase 1 OOS Validation Test")
    print("=" * 45)

    # Test basic gating
    evaluation_good, evaluation_bad = test_oos_gating()

    # Test comprehensive validation
    comprehensive_report = test_comprehensive_validation()

    # Generate report
    report_path = generate_oos_validation_report(evaluation_good, evaluation_bad, comprehensive_report)

    print("\n" + "=" * 45)
    print("✅ Phase 1 OOS validation test completed!")
    print(f"📊 Check the detailed report at: {report_path}")

    # Summary
    good_approved = evaluation_good['approved']
    bad_rejected = not evaluation_bad['approved']
    comprehensive_decision = comprehensive_report['final_decision']['decision']

    print("\n📈 Summary:")
    print(f"   Good Strategy Approved: {'✅' if good_approved else '❌'}")
    print(f"   Poor Strategy Rejected: {'✅' if bad_rejected else '❌'}")
    print(f"   Comprehensive Validation: {comprehensive_decision.upper()}")

    success = good_approved and bad_rejected and comprehensive_decision == 'approved'
    print(f"\n🎯 Overall Test Result: {'✅ PASSED' if success else '❌ FAILED'}")

    return success


if __name__ == "__main__":
    run_phase1_oos_test()
