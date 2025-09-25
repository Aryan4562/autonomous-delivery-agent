# run_tests.py
import unittest
import sys
import os

# Ensure the parent directory is in sys.path so 'tests' and 'src' can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from tests.test_environment import TestEnvironment
    from tests.test_planners import TestPlanners
    from tests.test_agent import TestAgent
    from tests.test_utils import TestUtils
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

if __name__ == '__main__':
    loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    test_suite.addTests(loader.loadTestsFromTestCase(TestEnvironment))
    test_suite.addTests(loader.loadTestsFromTestCase(TestPlanners))
    test_suite.addTests(loader.loadTestsFromTestCase(TestAgent))
    test_suite.addTests(loader.loadTestsFromTestCase(TestUtils))

    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    sys.exit(not result.wasSuccessful())

    