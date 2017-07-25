import numpy
import unittest
import mta.metric as metric 
from mta.ds.rating_row import RatingRow
class TestMetric(unittest.TestCase):
    
    def setUp(self):
        ratings = numpy.loadtxt('cust_ratings',delimiter=',')
        ratings = RatingRow(ratings,[max(ratings[:,0])+1,max(ratings[:,1])+1])
        predicted_ratings = numpy.loadtxt('cust_predicted_ratings',delimiter=',')
        predicted_ratings = RatingRow(predicted_ratings,[max(predicted_ratings[:,0])+1,max(predicted_ratings[:,1])+1])

        self.rating_list = ratings.to_list()
        self.predicted_matrix = predicted_ratings.to_matrix()

    def test_rmse(self):
        rmse = metric.rmse(self.rating_list, self.predicted_matrix)
        self.assertIsInstance(rmse,float)
        self.assertTrue(rmse <= 0.1)

    def test_mae(self):
        mae = metric.mae(self.rating_list, self.predicted_matrix)
        self.assertIsInstance(mae,float)
        self.assertTrue(mae <= 1.)

    def test_mape(self):
        mape = metric.mape(self.rating_list, self.predicted_matrix)
        self.assertIsInstance(mape,float)
        self.assertTrue(mape <= 40.)

    def test_hit_rate(self):
        hit_rate_1 = metric.hit_rate_1(self.rating_list, self.predicted_matrix)
        hit_rate_2 = metric.hit_rate_2(self.rating_list, self.predicted_matrix)
        hit_rate_3 = metric.hit_rate_3(self.rating_list, self.predicted_matrix)
        self.assertIsInstance(hit_rate_1,float)
        self.assertIsInstance(hit_rate_2,float)
        self.assertIsInstance(hit_rate_3,float)
        self.assertTrue(hit_rate_1 <= hit_rate_2)
        self.assertTrue(hit_rate_2 <= hit_rate_3)

if __name__ == '__main__' :
    unittest.main()