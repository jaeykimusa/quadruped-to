# test_runner.py

import unittest

class TestResult(unittest.TextTestResult):
    def print_summary(self):
        total_tests = self.testsRun
        total_failures = len(self.failures)
        total_errors = len(self.errors)
        total_passed = total_tests - total_failures - total_errors
        success_rate = (total_passed / total_tests) * 100 if total_tests > 0 else 0

        print("\n===== Test Summary =====")
        print(f"Total Tests   : {total_tests}")
        print(f"Passed        : {total_passed}")
        print(f"Failures      : {total_failures}")
        print(f"Errors        : {total_errors}")
        print(f"Success Rate  : {success_rate:.2f}%")
        print("========================\n")

class TestRunner(unittest.TextTestRunner):
    def _makeResult(self):
        return TestResult(self.stream, self.descriptions, self.verbosity)

    def run(self, test):
        result = super().run(test)
        result.print_summary()
        return result