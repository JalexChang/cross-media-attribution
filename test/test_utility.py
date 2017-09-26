import numpy
import unittest
from mta.dataset import Dataset
import mta.utility as utility

class TestUtility(unittest.TestCase):

    def setUp(self):
        self.rating_list = utility.load_rating_file('cust_ratings_with_datetime')
        self.touch_list = utility.load_touch_file('cust_touchs')
        self.normal = utility.Normalization()

    def test_log(self):
        original_min = min(list(rating[2] for rating in self.rating_list))
        original_max = max(list(rating[2] for rating in self.rating_list))
        
        R_list = self.normal.log(self.rating_list)
        normal_min = min(list(rating[2] for rating in R_list))
        normal_max = max(list(rating[2] for rating in R_list))
        self.assertEqual(len(self.rating_list), len(R_list))
        self.assertEqual(normal_min, 0.)
        self.assertEqual(normal_max, 1.)

        R_list = self.normal.log_revert(R_list)
        revert_min = min(list(rating[2] for rating in R_list))
        revert_max = max(list(rating[2] for rating in R_list))
        self.assertEqual(len(self.rating_list), len(R_list))
        self.assertTrue( (revert_min - original_min) <0.00001)
        self.assertTrue( (revert_max - original_max) <0.00001) 
    
    def test_minmax(self):
        original_min = min(list(rating[2] for rating in self.rating_list))
        original_max = max(list(rating[2] for rating in self.rating_list))
        
        R_list = self.normal.minmax(self.rating_list)
        normal_min = min(list(rating[2] for rating in R_list))
        normal_max = max(list(rating[2] for rating in R_list))
        self.assertEqual(len(self.rating_list), len(R_list))
        self.assertEqual(normal_min, 0.)
        self.assertEqual(normal_max, 1.)

        R_list = self.normal.minmax_revert(R_list)
        revert_min = min(list(rating[2] for rating in R_list))
        revert_max = max(list(rating[2] for rating in R_list))
        self.assertEqual(len(self.rating_list), len(R_list))
        self.assertTrue( (revert_min - original_min) <0.00001)
        self.assertTrue( (revert_max - original_max) <0.00001) 

if __name__ == '__main__' :
    unittest.main()
    
    
