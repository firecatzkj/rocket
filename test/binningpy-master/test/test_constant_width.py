import unittest
import numpy as np
try:
    from binningpy.unsupervised.constant_width import ConstantWidthBinning
except:
    import sys
    from pathlib import Path
    path = str(
        Path(__file__).absolute().parent.parent
    )
    if path not in sys.path:
        sys.path.append(path)
    from binningpy.unsupervised.constant_width import ConstantWidthBinning


def setUpModule():
    print("setUpModule")


def tearDownModule():
    print("tearUpModule")


class TestConstantWidthBinning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("setUpClass")

    @classmethod
    def tearDownClass(cls):
        print("tearDownClass")

    def setUp(self):
        print("instance setUp")

    def tearDown(self):
        print("instance tearDown")

    def test_confined_transform(self):
        bb = ConstantWidthBinning(4)
        bb.fit(np.array([1, 1, 3, 3, 2, 1, 3, 5, 6, 7, 7, 2]).reshape(-1, 1))
        print(bb._bins)
        result = bb.transform(np.array([1, 1, 3, 3, 2, 1, 3, 5, 6, 7, 7, 2]).reshape(-1, 1))
        target_result = np.array([[0], [0], [1], [1], [0], [0], [1], [2], [3], [3], [3], [0]])
        print(result)
        assert all(map(lambda x: x[0][0] == x[1][0], zip(result, target_result)))


def BinningBase_suite():
    suite = unittest.TestSuite()
    suite.addTest(TestConstantWidthBinning("test_confined_transform"))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    test_suite = BinningBase_suite()
    runner.run(test_suite)
