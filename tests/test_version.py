import unittest

import svf_gymnasium


class VersionTestCase(unittest.TestCase):
    """Version tests"""

    def test_version(self):
        """check svf_gymnasium exposes a version attribute"""
        self.assertTrue(hasattr(svf_gymnasium, "__version__"))
        self.assertIsInstance(svf_gymnasium.__version__, str)


if __name__ == "__main__":
    unittest.main()
