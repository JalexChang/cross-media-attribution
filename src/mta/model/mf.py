import numpy 
from copy import copy
from mta.dataset import Dataset
from mta.ds.rating_row import RatingRow
from mta.ds.touch_row import TouchRow
import time

class MF:
    trained = False
    dataset_loaded = False

    def __init__ (self,max_iters=100, user_biased =False, item_biased =False, alpha=0.001, beta=0.01, delta=0.001,verbose=False):
        self.max_iters = max_iters
        self.user_biased = user_biased
        self.item_biased = item_biased
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.verbose = verbose

    def load_dataset(self,dataset):
        matrix_shape = dataset.matrix_shape()
        self._size_user = matrix_shape[0]
        self._size_factor = matrix_shape[1]
        self._size_item = matrix_shape[2]
        self.ratings = copy(dataset.ratings)
        self.touchs = copy(dataset.touchs)
        self.dataset_loaded = True
        if not self.trained :
            self.trained_transaction_on_item = []
            for i_id in range(self._size_item):
                self.trained_transaction_on_item.append([])
        if self.verbose:
            print('dataset:', matrix_shape, ' is loaded')

    def _init_latent_factors(self):
        if not self.trained:
            init_mean = self.ratings.mean() - self.mean
            init_std = self.ratings.std()
            self.W = numpy.zeros([self._size_user,self._size_factor])
            for u_id,f_id in self.touchs.to_list():
                self.W[u_id][f_id] = numpy.random.normal(init_mean, init_std)
            self.H = numpy.random.normal(init_mean, init_std,(self._size_factor,self._size_item))
            if self.verbose:
                print('latent factors has been initialized')
               
    def _init_biases(self):
        if not self.trained :
            R = self.ratings.to_matrix()
            self.mean = 0
            self.bias_user = numpy.zeros(self._size_user)
            self.bias_item = numpy.zeros(self._size_item)
            if self.user_biased or self.item_biased:
                self.mean = self.ratings.mean()
                if self.verbose:
                    print('biases has been initialized')
    
    def _mark_matrix(self):
        for u_id, i_id, rating in self.ratings.to_list():
            self.trained_transaction_on_item[i_id].append(u_id)

    def fit(self):
        self._init_biases()
        self._init_latent_factors()
        
        self._mark_matrix()      
        best_W = numpy.copy(self.W)
        best_H = numpy.copy(self.H)
        R_list = self.ratings.to_list()
        #R_predicted = self.predict(R_list)
        for iters in range(self.max_iters):
            begin_time = time.time()
            #update features by stochastic gradient descent
            self._update_sgd(R_list)
            #calculate overall error (with regularization)
            R_predicted = self.predict(R_list)
            total_cost = self._calculate_cost(R_list,R_predicted)
            end_time= time.time()
            if self.verbose :
                    print ('iters-',iters+1,' cost:',total_cost,' time:', end_time - begin_time)
            if total_cost < self.delta:
                break
        self.trained = True
        
    def _update_sgd(self,R_list):
        for u_id, i_id, rating in R_list:
            predicted_rating = self._predict_one_element(u_id, i_id, p_type = "normal")
            error = rating - predicted_rating
            #updated factors
            for f_id in range(self._size_factor):
                if self.W[u_id][f_id] > 0:
                    w_uf = self.W[u_id][f_id] 
                    h_fi = self.H[f_id][i_id]
                    self.W[u_id][f_id] += self.alpha *(error * h_fi + self.beta * w_uf)
                    self.H[f_id][i_id] += self.alpha *(error * w_uf + self.beta * h_fi)
            #update biases
            if self.item_biased:
                self.bias_item[i_id] += self.alpha *( error - self.beta * self.bias_item[i_id])
            if self.user_biased:
                self.bias_user[u_id] += self.alpha *( error - self.beta * self.bias_user[u_id])

    def _calculate_cost(self,R_list,R_predicted): 
        total_cost =0.
        # prediction errors
        for u_id, i_id, rating in R_list:
            error = rating - R_predicted[u_id][i_id]
            total_cost += pow(error,2)
        # regularization errors
        for u_id in range(self._size_user):
            total_cost += self.beta*(numpy.dot(self.W[u_id,:],self.W[u_id,:]) + pow(self.bias_user[u_id],2))
        for i_id in range(self._size_item):
            total_cost += self.beta*(numpy.dot(self.H[:,i_id],self.H[:,i_id]) + pow(self.bias_item[i_id],2))
        return total_cost

    def predict(self, R_list = None):
        return self._predict(R_list, "normal")

    def predict_average(self, R_list = None):
        return self._predict(R_list, "avg")

    def _predict(self, R_list = None, p_type ="normal"):
        R_predicted = numpy.zeros((self._size_user,self._size_item))
        if R_list is None:
            for u_id in range(self._size_user):
                for i_id in range(self._size_item): 
                    R_predicted[u_id][i_id] = self._predict_one_element(u_id, i_id, p_type)    
        else:
            for u_id, i_id, rating in R_list:
                R_predicted[u_id][i_id] = self._predict_one_element(u_id, i_id, p_type)
        return R_predicted


    def _predict_one_element(self, u_id, i_id, p_type ="normal"):
        predicted_element = 0.
        if p_type == "normal":
            predicted_element = numpy.dot(self.W[u_id,:], self.H[:,i_id])
        elif  p_type == "avg" :
            predicted_w = self.average_w(u_id, i_id)
            predicted_element = numpy.dot(predicted_w, self.H[:,i_id])
        predicted_element += self.mean + self.bias_user[u_id] + self.bias_item[i_id]
        return predicted_element

    def average_w(self, u_id, i_id):
        users = self.trained_transaction_on_item[i_id]
        len_user = len(users)
        w = numpy.zeros(self._size_factor)
        if len_user >0:
            for user_id in users:
                w += self.W[user_id]
            for f_id in range(self._size_factor):
                w[f_id] = w[f_id]/ len_user if self.W[u_id][f_id] >0 else 0
        return w

    def matrix_shape(self):
        return (self._size_user,self._size_factor,self._size_item)

    def factor_attribution(self, R_list = None):
        attribution = numpy.zeros(self._size_factor)
        attribution_matrix = self.factor_item_attribution(R_list)
        for f_id in range(self._size_factor):
            attribution[f_id] = attribution_matrix[f_id].sum()
        return attribution

    def factor_item_attribution(self, R_list = None):
        attribution_matrix = numpy.zeros([self._size_factor,self._size_item])
        if R_list is None:
            R_list = self.ratings.to_list() 
        for u_id, i_id, rating in R_list:
            total_weight = numpy.inner(self.W[u_id,:], self.H[:,i_id])
            for f_id in range(self._size_factor):
                attributed_weight = self.W[u_id][f_id] * self.H[f_id][i_id] 
                if total_weight != 0.:
                    attribution_matrix[f_id][i_id] += rating * (attributed_weight / total_weight)
        return attribution_matrix
