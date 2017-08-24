import numpy
import unittest
from mta.dataset import Dataset
from mta.profile import Profile

class TestProfile(unittest.TestCase):
    
    def setUp(self):
        ratings = numpy.loadtxt('cust_ratings',delimiter=',')
        touchs = numpy.loadtxt('cust_touchs',delimiter=',')
        self.dataset = Dataset(ratings,touchs)
    
    def test_rating_dist(self):
        dist, labels = Profile.rating_dist(self.dataset, stacks=6, stack_range=1)
        self.assertEqual(len(labels), 6)
        self.assertEqual(len(dist), 6)
        self.assertEqual(dist.tolist(), [0, 6, 0, 1, 3, 3])

    def test_rating_std_dist_on_items(self):
        dist, labels = Profile.rating_std_dist_on_items(self.dataset, stacks=6, stack_range=0.5)
        self.assertEqual(len(labels), 6)
        self.assertEqual(len(dist), 6)
        self.assertEqual(dist.tolist(), [0, 0, 1, 1, 3, 0])

    def test_factor_dist_on_users(self):
        dist, labels = Profile.factor_dist_on_users(self.dataset, stacks=6, stack_range=1)
        self.assertEqual(len(labels),6)
        self.assertEqual(len(dist),6)
        self.assertEqual(dist.tolist(),[0, 1, 4, 0, 0, 0])

    def test_factor_dist_on_items(self):
        dist, labels = Profile.factor_dist_on_items(self.dataset, stacks=6, stack_range=1)
        self.assertEqual(len(labels),6)
        self.assertEqual(len(dist),6)
        self.assertEqual(dist.tolist(),[0, 0, 2, 3, 0, 0])

    def test_item_dist(self):
        dist, labels = Profile.item_dist(self.dataset, stacks=6, stack_range=1)
        self.assertEqual(len(labels),6)
        self.assertEqual(len(dist),6)
        self.assertEqual(dist.tolist(),[0, 2, 0, 1, 2, 0])

    def test_item_dist_on_factors(self):
        dist, labels = Profile.item_dist_on_factors(self.dataset, stacks=6, stack_range=1)
        self.assertEqual(len(labels),6)
        self.assertEqual(len(dist),6)
        self.assertEqual(dist.tolist(),[0, 0, 0, 1, 0, 2])

if __name__ == '__main__' :
    unittest.main()