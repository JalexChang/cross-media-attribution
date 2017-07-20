'''
@input: python3 generate_data.py <user_size> <item_size> <factor_size> <user_item_density> <user_factor_density> <folder_name>

@output: 
   rating_table.csv 
   effect_talbe.csv
   conversion_table.csv
   rating_list.csv, formats: user_id, itme_id, rating_value
   conversion_list.csv, formats: user_id, factor_id 

@process:
   1. generate effect matrix  with random value from N(u,sigma^2)
   2. generate conversion matrix with random value from N(u,sigma^2)
   3. filter conversion matrix with  user_factor_density 
   4. generate rating matrix based on effect and conversion matrix 
   5. filter rating matrix based on user_item_density
'''
import sys
import numpy

user_size = int(sys.argv[1]) if len(sys.argv)>1 else 20 
item_size = int(sys.argv[2]) if len(sys.argv)>2 else 4
factor_size = int(sys.argv[3]) if len(sys.argv)>3 else 5
user_item_density = float(sys.argv[4]) if len(sys.argv)>4 else 0.1   
user_factor_density = float(sys.argv[5]) if len(sys.argv)>5 else 0.1   
folder_name = sys.argv[6] if len(sys.argv)>6 else "./sample/"
mu = 5
sigma = 1

rating = numpy.zeros([user_size,item_size])
effect = sigma * numpy.random.randn(factor_size * item_size) + mu
conversion =  sigma * numpy.random.randn(user_size * factor_size) + mu

effect = effect.reshape(factor_size,item_size)

for i in range(len(conversion)):
    if numpy.random.random() >= user_factor_density:
        conversion[i] = 0.
conversion = conversion.reshape(user_size,factor_size)

for i in range(len(rating)):
	for j in range(len(rating[0])):
		rating[i][j] = numpy.dot(conversion[i,:],effect[:,j])

rating = rating.reshape(user_size * item_size)
for i in range(len(rating)):
    if numpy.random.random() >= user_item_density:
        rating[i] = 0.
rating = rating.reshape(user_size, item_size)

print ("rating table: ", rating.shape)
print ("conversion table: ",conversion.shape)
print ("effect table: ",effect.shape)

numpy.savetxt(folder_name+"rating_table.csv",rating,delimiter=',', fmt ='%.4f')
numpy.savetxt(folder_name+"effect_table.csv",effect,delimiter=',', fmt ='%.4f')
numpy.savetxt(folder_name+"conversion_table.csv",conversion,delimiter=',', fmt ='%.4f')

f = open(folder_name+"rating_list.csv","w")
for i in range(len(rating)):
    for j in range(len(rating[0])):
        if rating[i][j] > 0.:
            f.write(str(i)+","+str(j)+","+str(rating[i][j])+"\n")
f.close()

f = open(folder_name+"conversion_list.csv","w")
for i in range(len(conversion)):
    for j in range(len(conversion[0])):
        if conversion[i][j] > 0. :
            f.write(str(i)+","+str(j)+"\n")
f.close()



