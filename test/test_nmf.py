import numpy
import unittest
from mta.dataset import Dataset
from mta.model.nmf import NMF
import mta.metric as metric 

class TestNMF(unittest.TestCase):
    
    def setUp(self):
        rating_rows = numpy.loadtxt('cust_ratings',delimiter=',')
        touch_rows = numpy.loadtxt('cust_touchs',delimiter=',')
        self.dataset = Dataset(rating_rows,touch_rows)
        self.model = NMF(max_iters=10,alpha=0.001,beta=0.001,user_biased =True, item_biased =True)
        self.model.load_dataset(self.dataset)
    
    def test_init(self):
        self.assertIsInstance(self.model,NMF)

    def test_init_with_agrv(self):
        model = NMF(max_iters=500,alpha=0.003,beta=0.1,delta=0.0001,verbose=True)
        self.assertEqual(model.max_iters, 500)
        self.assertEqual(model.alpha, 0.003)
        self.assertEqual(model.beta, 0.1)
        self.assertEqual(model.delta, 0.0001)
        self.assertTrue(model.verbose)
    
    def test_load_dataset(self):
        self.assertTrue(self.model.dataset_loaded)
        self.assertEqual(self.model.ratings.to_list(),self.dataset.ratings.to_list())
        self.assertEqual(self.model.touchs.to_list(),self.dataset.touchs.to_list())
        self.assertEqual(self.model.matrix_shape(),self.dataset.matrix_shape())

    def test_fit(self):
        self.model.fit()
        predict_rating = self.model.predict()
        print(self.dataset.ratings.to_matrix())
        print(predict_rating)
        rmse = metric.rmse(self.dataset.ratings.to_list(),predict_rating)
        print("rmse", rmse)
        self.assertTrue(rmse <= 1.5)

    def test_factor_item_attribution(self):
        self.model.fit()
        rating_matrix = numpy.array(self.dataset.ratings.to_matrix())
        attribution_matrix = self.model.factor_item_attribution(self.dataset.ratings.to_list())
        error = rating_matrix.sum() - attribution_matrix.sum() 
        print("error on factor_item_attribution",error)
        self.assertTrue(error <= 0.0001)

    def test_factor_attribution(self):
        self.model.fit()
        rating_matrix = numpy.array(self.dataset.ratings.to_matrix())
        attribution = self.model.factor_attribution(self.dataset.ratings.to_list())
        error = rating_matrix.sum() - attribution.sum() 
        print("error on factor_attribution",error)
        self.assertTrue(error <= 0.0001)

if __name__ == '__main__' :
    unittest.main()