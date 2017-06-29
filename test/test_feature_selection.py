import numpy
import unittest
from mta.feature_selection.cross_validation import CrossValidation
from mta.dataset import Dataset
from mta.model.svd import SVD
import mta.metric as metric 

class TestFeatureSelection(unittest.TestCase):
    
    def setUp(self):
        rating_rows = numpy.loadtxt('cust_ratings',delimiter=',')
        touch_rows = numpy.loadtxt('cust_touchs',delimiter=',')
        self.dataset = Dataset(rating_rows,touch_rows)
        self.model = SVD(max_iters=20)
        
    def test_cross_validation(self):
        n_fold =4
        cv = CrossValidation(verbose=True)
        cv.load(self.dataset,self.model)
        cv.split(n_fold).run()
        scores = cv.score(metric.rmse)
        self.assertEqual(len(scores),n_fold)
        selected_model = cv.select(metric.rmse)
        self.assertIsInstance(selected_model,SVD)

if __name__ == '__main__' :
    unittest.main()