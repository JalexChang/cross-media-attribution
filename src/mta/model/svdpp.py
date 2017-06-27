try:
    import numpy 
except:
    raise Exception('svd module needs numpy module')

class svd:
    trained = False
   
    def __init__ (self,matrix_size=[0,0,0],max_iters=500, biased =False, alpha=0.001, beta=0.01, delta=0.001,verbose=False):
        self.size_n = matrix_size[0]
        self.size_k = matrix_size[1]
        self.size_m = matrix_size[2]
        self.max_iters = max_iters
        self.biased = biased
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.W = numpy.random.random([self.size_n,self.size_k])
        self.H = numpy.random.random([self.size_k,self.size_m])

        self.verbose = verbose

        
    def init_latent_factors(self,cust_W=None,cust_H=None):
        if cust_W is not None:
            self.W = cust_W if cust_W.shape == (self.size_n,self.size_k) else self.W
        if cust_H is not None:
            self.H = cust_H if cust_H.shape == (self.size_k,self.size_m) else self.H

               
    def init_biases(self,R=None):
        if not self.trained and R is not None:
            self.mean = 0
            self.bias_i = numpy.zeros(len(R))
            self.bias_j = numpy.zeros(len(R[0]))
            
            if self.biased:
                self.mean = R.mean()
                for i in range(len(self.bias_i)):
                    self.bias_i[i] = numpy.random.random() if max(R[i,:]) > 0 else 0 
                for j in range(len(self.bias_j)):
                    self.bias_j[j] = numpy.random.random() if max(R[:,j]) > 0 else 0 
    
    def fit(self,R):
        self.init_biases(R)       
        best_W = numpy.copy(self.W)
        best_H = numpy.copy(self.H)
        R_predicted = self.prediction()
        min_cost = self._calculate_cost_value(R,R_predicted)
       
        for iters in range(self.max_iters):
            R_predicted = self.prediction()
            
            #update features by stochastic gradient descent
            self._update_sgd(R,R_predicted)          
            #calculate overall error (with regularization)
            total_cost = self._calculate_cost_value(R,R_predicted)

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
        
                
    def _update_sgd(self,R,R_predicted):
        #calculate updated factors and biases
        updated_W = self._calculate_updated_W(R,R_predicted)
        updated_H = self._calculate_updated_H(R,R_predicted)
        updated_bias_i = self._calculate_updated_bias_i(R,R_predicted)
        updated_bias_j = self._calculate_updated_bias_j(R,R_predicted)
        #update factors and biases
        self.W = updated_W
        self.H = updated_H
        self.bias_i = updated_bias_i if self.biased else self.bias_i
        self.bias_j = updated_bias_j if self.biased else self.bias_j
        #print('W',updated_W)
        #print('H',updated_H)
        #print('bias_i',updated_bias_i)
        #print('bias_j',updated_bias_j)

    def _calculate_updated_W(self,R,R_predicted):
        updated_W = numpy.copy(self.W)
        for i in range(self.size_n):
            for j in range(self.size_m):
                if R[i][j] >0:
                    error_ij = R[i][j] - R_predicted[i][j]
                    for k in range(self.size_k):
                        regu_term = self.beta * self.W[i][k]
                        updated_W[i][k] += self.alpha * (error_ij* self.H[k][j] + regu_term)
        return updated_W

    def _calculate_updated_H(self,R,R_predicted):
        updated_H = numpy.copy(self.H)
        for i in range(self.size_n):
            for j in range(self.size_m):
                if R[i][j] >0:
                    error_ij = R[i][j] - R_predicted[i][j]
                    for k in range(self.size_k):
                        regu_term = self.beta * self.H[k][j]
                        updated_H[k][j] += self.alpha * (error_ij* self.W[i][k] + regu_term)
        return updated_H

  
    def _calculate_updated_bias_i(self,R,R_predicted):
        updated_bias_i = numpy.copy(self.bias_i)
        for i in range(self.size_n):
            for j in range(self.size_m):
                if R[i][j] >0:
                    error_ij = R[i][j] - R_predicted[i][j]
                    updated_bias_i[i] += self.alpha * ( error_ij - self.beta * self.bias_i[i])
        return updated_bias_i


    def _calculate_updated_bias_j(self,R,R_predicted):
        updated_bias_j = numpy.copy(self.bias_j)
        for i in range(self.size_n):
            for j in range(self.size_m):
                if R[i][j] >0:
                    error_ij = R[i][j] - R_predicted[i][j]
                    updated_bias_j[j] += self.alpha*( error_ij - self.beta * self.bias_j[j])
        return updated_bias_j
        
    
    def _calculate_cost_value(self,R,R_predicted): 
        total_cost =0.
        # prediction errors
        for i in range(self.size_n):
            for j in range(self.size_m):
                if R[i][j] >0 : 
                    error_ij = R[i][j] - R_predicted[i][j]
                    total_cost += pow(error_ij,2)
        
        # regularization errors
        for i in range(self.size_n):
            total_cost += self.beta*(numpy.dot(self.W[i,:].flat,self.W[i,:].flat) + pow(self.bias_i[i],2))
        for j in range(self.size_m):
            total_cost += self.beta*(numpy.dot(self.H[:,j].flat,self.H[:,j].flat) + pow(self.bias_j[j],2))
        
        return total_cost

        
    def prediction(self,R=None):
        R_predicted = numpy.dot(self.W,self.H)
        for i in range(self.size_n):
            for j in range(self.size_m):
                bias = self.mean + self.bias_i[i] + self.bias_j[j]
                R_predicted[i][j]+=bias
        return R_predicted