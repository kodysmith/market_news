"""
Test suite for options scanner
"""

import unittest
import os
import sys
from unittest.mock import Mock, patch
import json

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from iv_solver import IVSolver
from probability import ProbabilityCalculator
from spreads import SpreadAnalyzer, BullPutSpread
from database import DatabaseManager
from alerts import AlertManager

class TestIVSolver(unittest.TestCase):
    """Test implied volatility solver"""
    
    def setUp(self):
        self.solver = IVSolver()
    
    def test_black_scholes_put(self):
        """Test Black-Scholes put price calculation"""
        # Test case: S=100, K=95, T=0.25, r=0.02, sigma=0.2
        price = self.solver.black_scholes_put(100, 95, 0.25, 0.02, 0.2)
        self.assertGreater(price, 0)
        self.assertLess(price, 5)  # Should be reasonable put price
    
    def test_implied_vol_put(self):
        """Test implied volatility solving"""
        # Test case: known price should give back known volatility
        S, K, T, r, sigma = 100, 95, 0.25, 0.02, 0.2
        known_price = self.solver.black_scholes_put(S, K, T, r, sigma)
        
        solved_iv = self.solver.implied_vol_put(known_price, S, K, T, r)
        
        # Should be close to original volatility
        self.assertAlmostEqual(solved_iv, sigma, places=2)
    
    def test_edge_cases(self):
        """Test edge cases for IV solver"""
        # Zero time to expiration
        iv = self.solver.implied_vol_put(1.0, 100, 99, 0, 0.02)
        self.assertEqual(iv, 0.2)  # Should return default
        
        # Zero price
        iv = self.solver.implied_vol_put(0, 100, 99, 0.25, 0.02)
        self.assertEqual(iv, 0.2)  # Should return default (not minimum)

class TestProbabilityCalculator(unittest.TestCase):
    """Test probability calculations"""
    
    def setUp(self):
        self.calc = ProbabilityCalculator()
    
    def test_pop_short_put(self):
        """Test probability of profit for short put"""
        # Test case: S=100, K=95, T=0.25, r=0.02, sigma=0.2
        pop = self.calc.pop_short_put(100, 95, 0.25, 0.02, 0.2)
        
        # Should be between 0 and 1
        self.assertGreaterEqual(pop, 0)
        self.assertLessEqual(pop, 1)
        
        # Should be reasonable (around 60-70% for this case)
        self.assertGreater(pop, 0.5)
        self.assertLess(pop, 0.8)
    
    def test_pop_bull_put_spread(self):
        """Test probability of profit for bull put spread"""
        # Test case: S=100, K_short=95, K_long=90, T=0.25, r=0.02, sigma=0.2
        pop = self.calc.pop_bull_put_spread(100, 95, 90, 0.25, 0.02, 0.2)
        
        # Should be between 0 and 1
        self.assertGreaterEqual(pop, 0)
        self.assertLessEqual(pop, 1)
        
        # Should be same as short put POP
        short_put_pop = self.calc.pop_short_put(100, 95, 0.25, 0.02, 0.2)
        self.assertAlmostEqual(pop, short_put_pop, places=6)

class TestSpreadAnalyzer(unittest.TestCase):
    """Test spread analysis"""
    
    def setUp(self):
        self.analyzer = SpreadAnalyzer()
    
    def test_calculate_fill_score(self):
        """Test fill score calculation"""
        # Mock option quotes
        short_put = Mock()
        short_put.open_interest = 1000
        short_put.volume = 50
        short_put.bid = 1.0
        short_put.ask = 1.1
        
        long_put = Mock()
        long_put.open_interest = 800
        long_put.volume = 30
        long_put.bid = 0.5
        long_put.ask = 0.6
        
        spot_price = 100.0
        bid_ask_width = 0.2  # 0.1 + 0.1
        
        score = self.analyzer._calculate_fill_score(short_put, long_put, bid_ask_width, spot_price)
        
        # Should be between 0 and 1
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)
    
    def test_apply_filters(self):
        """Test spread filtering"""
        # Create mock spreads with proper attributes
        spreads = []
        
        # Good spread (EV per $100 = 0.15 / (5.0 * 100) = 0.0003, which is < 0.10)
        good_spread = Mock()
        good_spread.pop = 0.6
        good_spread.ev = 50.0  # High EV to pass the per-100 test
        good_spread.width = 5.0
        good_spread.bid_ask_width = 0.05
        good_spread.fill_score = 0.8
        spreads.append(good_spread)
        
        # Low POP
        low_pop = Mock()
        low_pop.pop = 0.4
        low_pop.ev = 0.15
        low_pop.width = 5.0
        low_pop.bid_ask_width = 0.05
        low_pop.fill_score = 0.8
        spreads.append(low_pop)
        
        # Low EV (EV per $100 = 0.05 / (5.0 * 100) = 0.0001, which is < 0.10)
        low_ev = Mock()
        low_ev.pop = 0.6
        low_ev.ev = 0.05
        low_ev.width = 5.0
        low_ev.bid_ask_width = 0.05
        low_ev.fill_score = 0.8
        spreads.append(low_ev)
        
        # Wide spread
        wide_spread = Mock()
        wide_spread.pop = 0.6
        wide_spread.ev = 0.15
        wide_spread.width = 5.0
        wide_spread.bid_ask_width = 0.15
        wide_spread.fill_score = 0.8
        spreads.append(wide_spread)
        
        # Low fill score
        low_fill = Mock()
        low_fill.pop = 0.6
        low_fill.ev = 0.15
        low_fill.width = 5.0
        low_fill.bid_ask_width = 0.05
        low_fill.fill_score = 0.2
        spreads.append(low_fill)
        
        filtered = self.analyzer.apply_filters(spreads)
        
        # Should keep the first spread (good spread)
        # The wide spread should be filtered out due to bid_ask_width > 0.10
        self.assertEqual(len(filtered), 1)

class TestDatabaseManager(unittest.TestCase):
    """Test database operations"""
    
    def setUp(self):
        # Use in-memory database for testing
        self.db_manager = DatabaseManager(":memory:")
        # Ensure database is initialized
        self.db_manager.init_database()
    
    def test_generate_spread_id(self):
        """Test spread ID generation"""
        spread_id = self.db_manager.generate_spread_id("SPY", "2025-10-18", 500, 495)
        
        # Should be consistent
        self.assertEqual(len(spread_id), 32)  # MD5 hash length
        self.assertEqual(spread_id, self.db_manager.generate_spread_id("SPY", "2025-10-18", 500, 495))
    
    def test_alert_deduplication(self):
        """Test alert deduplication logic"""
        # Create a fresh database manager for this test
        db_manager = DatabaseManager(":memory:")
        
        spread_id = "test_spread_123"
        
        # First time - should send alert
        self.assertTrue(db_manager.should_send_alert(spread_id, 0.15))
        
        # Mark as sent
        db_manager.upsert_pushed_alert(spread_id, 0.15)
        
        # Same EV - should not send
        self.assertFalse(db_manager.should_send_alert(spread_id, 0.15))
        
        # Higher EV - should send
        self.assertTrue(db_manager.should_send_alert(spread_id, 0.20))

class TestAlertManager(unittest.TestCase):
    """Test alert management"""
    
    def setUp(self):
        self.db_manager = DatabaseManager(":memory:")
        self.alert_manager = AlertManager(self.db_manager)
    
    def test_format_alert_message(self):
        """Test alert message formatting"""
        spread = Mock()
        spread.ticker = "SPY"
        spread.pop = 0.65
        spread.ev = 0.15
        spread.credit = 0.95
        spread.dte = 33
        spread.short_strike = 510
        spread.long_strike = 505
        spread.spread_id = "test_123"
        spread.expiry = "2025-10-18"
        
        alert = self.alert_manager.format_alert_message(spread)
        
        self.assertIn("SPY", alert['title'])
        self.assertIn("65%", alert['body'])
        self.assertEqual(alert['data']['ticker'], "SPY")
        self.assertEqual(alert['data']['strategy'], "BULL_PUT")

def run_tests():
    """Run all tests"""
    print("🧪 Running options scanner tests...")
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestIVSolver,
        TestProbabilityCalculator,
        TestSpreadAnalyzer,
        TestDatabaseManager,
        TestAlertManager
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
    else:
        print(f"\n❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
