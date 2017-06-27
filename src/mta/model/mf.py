import numpy 
from copy import copy
from mta.dataset import Dataset
from mta.ds.rating_row import RatingRow
from mta.ds.touch_row import TouchRow

class MF:
    trained = False
    dataset_loaded = False

    def __init__ (self,max_iters=100, biased =False, alpha=0.001, beta=0.01, delta=0.001,verbose=False):
        self.max_iters = max_iters
        self.biased = biased
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
        if self.verbose:
            print('dataset:', matrix_shape, ' is loaded')

    def _init_latent_factors(self):
        if not self.trained:
            self.W = numpy.zeros([self._size_user,self._size_factor])
            for u_id,f_id in self.touchs.to_list():
                self.W[u_id][f_id] = numpy.random.random()
            self.H = numpy.random.random([self._size_factor,self._size_item])
            if self.verbose:
                print('latent factors has been initialized')
               
    def _init_biases(self):
        if not self.trained :
            R = self.ratings.to_matrix()
            self.mean = 0
            self.bias_user = numpy.zeros(self._size_user)
            self.bias_item = numpy.zeros(self._size_item)
            if self.biased:
                self.mean = self.ratings.mean()
                for u_id in range(len(self.bias_user)):
                    self.bias_user[u_id] = numpy.random.random() if max(R[u_id,:]) > 0 else 0 
                for i_id in range(len(self.bias_item)):
                    self.bias_item[i_id] = numpy.random.random() if max(R[:,i_id]) > 0 else 0
                if self.verbose:
                    print('biases has been initialized')
 
    def fit(self):
        self._init_latent_factors()
        self._init_biases()      
        best_W = numpy.copy(self.W)
        best_H = numpy.copy(self.H)
        R_list = self.ratings.to_list()
        R_predicted = self.predict()
        min_cost = self._calculate_cost(R_list,R_predicted)

        for iters in range(self.max_iters):
            R_predicted = self.predict()
            #update features by stochastic gradient descent
            self._update_sgd(R_list,R_predicted)          
            #calculate overall error (with regularization)
            total_cost = self._calculate_cost(R_list,R_predicted)
            #compare with best factors
            if total_cost < min_cost:
                best_W = numpy.copy(self.W)
                best_H = numpy.copy(self.H)
                min_cost = total_cost
            if self.verbose :
                    print ('iters-',iters+1,':',total_cost)
            if total_cost < self.delta:
                break
        #after training, chosse best factors
        self.W = best_W
        self.H = best_H
        self.trained = True
                    
    def _update_sgd(self,R_list,R_predicted):
        #updated factors
        updated_W = self._calculate_updated_W(R_list,R_predicted)
        updated_H = self._calculate_updated_H(R_list,R_predicted)
        self.W = updated_W
        self.H = updated_H
        #update biases
        if self.biased:
            updated_bias_user = self._calculate_updated_bias_user(R_list,R_predicted)
            updated_bias_item = self._calculate_updated_bias_item(R_list,R_predicted)
            self.bias_user = updated_bias_user
            self.bias_item = updated_bias_item

    def _calculate_updated_W(self,R_list,R_predicted):
        updated_W = numpy.copy(self.W)
        for u_id, i_id, rating in R_list:
            error = rating - R_predicted[u_id][i_id]
            for f_id in range(self._size_factor):
                regu_term = self.beta * self.W[u_id][f_id]
                updated_W[u_id][f_id] += self.alpha * (error* self.H[f_id][i_id] + regu_term)
        return updated_W

    def _calculate_updated_H(self,R_list,R_predicted):
        updated_H = numpy.copy(self.H)
        for u_id, i_id, rating in R_list:
            error = rating - R_predicted[u_id][i_id]
            for f_id in range(self._size_factor):
                regu_term = self.beta * self.H[f_id][u_id]
                updated_H[f_id][i_id] += self.alpha * (error* self.W[u_id][f_id] + regu_term)
        return updated_H

    def _calculate_updated_bias_user(self,R_list,R_predicted):
        updated_bias_user = numpy.copy(self.bias_user)
        for u_id, i_id, rating in R_list:
            error = rating - R_predicted[u_id][i_id]
            updated_bias_user[u_id] += self.alpha*( error - self.beta * self.bias_user[u_id])
        return updated_bias_user

    def _calculate_updated_bias_item(self,R_list,R_predicted):
        updated_bias_item = numpy.copy(self.bias_item)
        for u_id, i_id, rating in R_list:
            error = rating - R_predicted[u_id][i_id]
            updated_bias_item[i_id] += self.alpha*( error - self.beta * self.bias_item[i_id])
        return updated_bias_item
    
    def _calculate_cost(self,R_list,R_predicted): 
        total_cost =0.
        # prediction errors
        for u_id, i_id, rating in R_list:
            error = rating - R_predicted[u_id][i_id]
            total_cost += pow(error,2)
        # regularization errors
        for u_id in range(self._size_user):
            total_cost += self.beta*(numpy.dot(self.W[u_id,:].flat,self.W[u_id,:].flat) + pow(self.bias_user[u_id],2))
        for i_id in range(self._size_item):
            total_cost += self.beta*(numpy.dot(self.H[:,i_id].flat,self.H[:,i_id].flat) + pow(self.bias_item[i_id],2))
        return total_cost

        
    def predict(self):
        R_predicted = numpy.dot(self.W,self.H)
        for u_id in range(self._size_user):
            for i_id in range(self._size_item):
                bias = self.mean + self.bias_user[u_id] + self.bias_item[i_id]
                R_predicted[u_id][i_id]+=bias
        return R_predicted

    def matrix_shape(self):
        return (self._size_user,self._size_factor,self._size_item)

    def factor_attribution(self):
        attribution = numpy.zeros(self._size_factor)
        attribution_matrix = self.factor_item_attribution()
        for f_id in range(self._size_factor):
            attribution[f_id] = attribution_matrix[f_id].sum()
        return attribution

    def factor_item_attribution(self):
        attribution_matrix = numpy.zeros([self._size_factor,self._size_item])
        #factor_value =numpy.zeros(self._size_factor)
        R_list = self.ratings.to_list()
        R_predicted = self.predict()
        for u_id, i_id, rating in R_list:
            for f_id in range(self._size_factor):
                weight_percent = (self.W[u_id][f_id] * self.H[f_id][i_id]) / R_predicted[u_id][i_id]
                attribution_matrix[f_id][i_id] += rating * weight_percent
        return attribution_matrix


