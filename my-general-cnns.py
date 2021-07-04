pip install tensorflow==1.14
pip install gast==0.2.2
import tensorflow as tf

import warnings
from PIL import Image, ImageOps
import numpy as np
from mlxtend.data import loadlocal_mnist
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
import tensorflow as tf



class CNNS():
    

    warnings.filterwarnings(action = "ignore", category = DeprecationWarning)
    def __init__(self):
       return
    def Deploy_X_Y(self,architecture):
      m, w, h, d = architecture[0][1:]

      X = tf.placeholder(tf.float32, shape = (None, w, h, d))
      Y = tf.placeholder(tf.float32, shape = (None, architecture[-1][-1]))

      return X, Y

    def toOneHot(self, indices, K):
      return tf.one_hot(indices, K)

    
    def start(self, architecture): #[(input,m,32,32,1), (conv,3,3,16),(conv,3,3,32),(maxpool,2,2,32),......(ord,120),(fc,120),(softmax,10)]
    
      m, w, h, d = architecture[0][1:]
      W, b = [], []
      L = 0
      ordCount = 0
    
      architecture = architecture[1:]

      for layer in architecture:
        if layer[0] == 'conv':
          L+=1
          f, f, _, stride, p = layer[1:] 

          globals()['W' + str(L)] = tf.get_variable('W' + str(L), layer[1:3] + tuple([d, layer[3] ]), dtype= tf.float32, initializer = tf.contrib.layers.xavier_initializer(seed = 6))
          globals()['b' + str(L)] = tf.get_variable('b' + str(L), (1,1,1) + tuple([layer[3]]), dtype= tf.float32, initializer = tf.contrib.layers.xavier_initializer(seed = 6))
          
          W.append(globals()['W' + str(L)])
          b.append(globals()['b' + str(L)])
          
          w = int ((w-f)/stride + 1)*(p == "VALID" or (p == "SAME" and stride >1)) + w*(p == "SAME")
          h = int ((h-f)/stride + 1)*(p == "VALID" or (p == "SAME" and stride >1)) + h*(p == "SAME")
          d = layer[3]
        elif layer[0] == 'ordlayer':
          L+=1
          if ordCount == 0 :
            globals()['W' + str(L)] = tf.get_variable('W' + str(L), layer[1:3] + tuple([d*w*h]), dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer(seed=6))
            globals()['b' + str(L)] = tf.get_variable('b' + str(L), tuple([layer[-1],1]), dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer(seed=6))
          else :
            globals()['W' + str(L)] = tf.get_variable('W' + str(L), layer[1:3] + tuple([prevFeatures[0]]), dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer(seed=6))
            globals()['b' + str(L)] = tf.get_variable('b' + str(L), tuple([layer[-1],1]), dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer(seed=6))
          
          W.append(globals()['W' + str(L)])
          b.append(globals()['b' + str(L)])
          d = layer[-1]
          prevFeatures = layer[1:3]
          ordCount +=1
        elif layer[0] == 'maxpool':
          L+=1
          f, f, stride, p = layer[1:] 

          globals()['W' + str(L)] = tf.get_variable('W' + str(L), layer[1:3] + tuple([d, layer[3] ]), dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
          globals()['b' + str(L)] = tf.get_variable('b' + str(L), (1,1,1) + tuple([layer[3] ]), dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
          
          W.append(globals()['W' + str(L)])
          b.append(globals()['b' + str(L)])

          
          w = int ((w-f)/stride + 1)*(p == "VALID" or (p == "SAME" and stride >1)) + w*(p == "SAME")
          h = int ((h-f)/stride + 1)*(p == "VALID" or (p == "SAME" and stride >1)) + h*(p == "SAME")
          
        else:
          continue
          
      parameters = {}
      parameters['W'] = W;
      parameters['b'] = b;
      return parameters

    

    def forward_prop(self, X, parameters, architecture):
        Z = []
        A =[X]
        architecture = architecture[1:]
        W, b = parameters['W'], parameters['b']
        ordl1 = 0
        
        for i in range(len(architecture)):
          if architecture[i][0] == 'conv':
             
             Zl = tf.nn.conv2d(A[i], W[i], strides = [1,architecture[i][4],architecture[i][4],1], padding = architecture[i][-1])
             Al = tf.nn.relu(Zl)
             Z.append(Zl)
             A.append(Al)
             
            
          elif architecture[i][0] == 'maxpool':
             
             Al = tf.nn.max_pool(A[i], ksize = [1,architecture[i][1],architecture[i][2],1], strides = [1,architecture[i][3],architecture[i][3],1], padding = architecture[i][-1])
             A.append(Al)
             
          elif architecture[i][0] == 'Flatten' :
             Al = tf.contrib.layers.flatten(A[i])
          #  Al = tf.transpose(tf.contrib.layers.flatten(A[i]))
            
          #  Zl = tf.matmul(W[i],Al) + b[i] 
          #  Al = tf.nn.relu(Zl)
          #  ordl1 = 1
             mark = i
             A.append(Al)
             
          #  Z.append(Zl)
            
          elif architecture[i][0] == 'ordlayer' :
             

             Zl = tf.matmul(tf.cast(W[mark],dtype = tf.float32),A[i]) + b[mark] 
             Al = tf.nn.relu(Zl)
             A.append(Al)
             Z.append(Zl)
             mark = i
          
          elif architecture[i][0] == 'fc':
             
            
             Zl = tf.contrib.layers.fully_connected(A[i], architecture[i][-1], activation_fn= None)
             Al = tf.nn.relu(Zl)
             Z.append(Zl)
             A.append(Al)
          elif architecture[i][0] == 'softmax' :
             Al = tf.nn.softmax(Z[-1], axis = -1)
             A.append(Al)
          
        return Z[-1], A[-1]

    def cost(self, Zlast, Y):
      cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Zlast,labels = Y))
      return cost
    
    def model(self, xtrain, ytrain, xtest, ytest, architecture, learning_rate = 0.3, epochs = 1000, printCost = True):
      
      X, Y = self.Deploy_X_Y(architecture)
      
      parameters = self.start(architecture)
      
      Zlast, Alast = self.forward_prop(X,parameters, architecture)

      cost = self.cost(Zlast, Y)

      trainer = tf.train.AdamOptimizer(learning_rate).minimize(loss=cost)

      with tf.Session() as sess : 
        init = tf.global_variables_initializer()
        sess.run(init)
        
        epochdata = []
        epoch_costdata = []

        for epoch in range(epochs):
            
            _, epoch_cost = sess.run(fetches = [trainer, cost], feed_dict = {X:xtrain, Y: ytrain})

            if printCost == True and epoch%1 == 0:
              print("Epoch",epoch," cost : ", epoch_cost)
            epochdata.append(epoch)
            epoch_costdata.append(epoch_cost)

        parameters = sess.run(parameters)
        
        predict_op = tf.argmax(Zlast, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
        
        plt.plot(epochdata, epoch_costdata)
        plt.show()
        predict_op = sess.run(predict_op, feed_dict={X : xtest})
       
      # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
       
        train_accuracy = accuracy.eval({X: Xtrain, Y: Ytrain})
        test_accuracy = accuracy.eval({X: Xtest, Y: Ytest})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)

      return parameters, predict_op

    def predict(self,arch, param, dataIn):
      
      X,_ = self.Deploy_X_Y(arch)
      Zlast, Alast = self.forward_prop(X, param,arch)
      sess1 = tf.Session()
      init = tf.global_variables_initializer()
      sess1.run(init)
      Aout1 = tf.argmax(Zlast, 1)
      Ypredi = sess1.run(Aout1, feed_dict = {X : dataIn})
      Ypredi = Ypredi.T
      
      
      
      sess1.close()
      return Ypredi
    
tf.reset_default_graph()
pathXtrain = '/content/drive/My Drive/train-images.idx3-ubyte'
pathYtrain = '/content/drive/My Drive/train-labels.idx1-ubyte'

pathXtest = '/content/drive/My Drive/t10k-images.idx3-ubyte'
pathYtest = '/content/drive/My Drive/t10k-labels.idx1-ubyte'

Xtrain_orig, Ytrain_orig = loadlocal_mnist(pathXtrain,pathYtrain) 
Xtest_orig, Ytest_orig = loadlocal_mnist(pathXtest,pathYtest)



Xtrain_orig = Xtrain_orig[:]
Ytrain_orig = Ytrain_orig[:]


Xtrain_orig = Xtrain_orig.reshape(-1,28,28)
Xtest_orig = Xtest_orig.reshape(-1,28,28)
Xtrain = Xtrain_orig.reshape(-1,28,28,1)/255
Xtest = Xtest_orig.reshape(-1,28,28,1)/255

sess = tf.Session()
MyModel = CNNS()

Ytrain = sess.run(MyModel.toOneHot(Ytrain_orig,10))

Ytest = sess.run(MyModel.toOneHot(Ytest_orig,10))


sess.close()

#Architechture Blue print : 
#arch := list of tuples 
#Input -types : 
# 1. Input-Image-Array := ('input', m, w, h, d)
# 2. Conv-filters      := ('conv', f, f, c, stride, "SAME"/"VALID")   #NOTE : FOR "SAME" stride >1 HAS TO BE PICKED ACCORDINGLY 
# 3. Maxpool-filters   := ('maxpool', f, f, stride, "SAME"/"VALID")   #NOTE : FOR "SAME" stride >1 HAS TO BE PICKED ACCORDINGLY
# 4. Flatten           := ('Flatten',) 
# 5. Feed-Forward      := ('ordlayers', nodes) #DEPRECATED - LATELY
# 6. Fully-Connected   := ('fc', output_nodes)
# 7. Softmax           := ('softmax', output_nodes)
Xtrain_orig = Xtrain_orig[:]
Ytrain_orig = Ytrain_orig[:]


Xtrain_orig = Xtrain_orig.reshape(-1,28,28)
Xtest_orig = Xtest_orig.reshape(-1,28,28)
Xtrain = Xtrain_orig.reshape(-1,28,28,1)/255
Xtest = Xtest_orig.reshape(-1,28,28,1)/255

sess = tf.Session()
MyModel = CNNS()

Ytrain = sess.run(MyModel.toOneHot(Ytrain_orig,10))

Ytest = sess.run(MyModel.toOneHot(Ytest_orig,10))


sess.close()

tf.reset_default_graph()
                                                              
arch = [('input',Xtrain.shape[0],Xtrain.shape[1],Xtrain.shape[2],1), ('conv',3,3,32,1,"SAME"),('maxpool',2,2,2,"VALID"),('Flatten',),('fc',100),('fc',10),('softmax',10)]

parameters, Ypredi = MyModel.model(Xtrain, Ytrain, Xtest, Ytest, arch, learning_rate = 0.007, epochs= 30, printCost=True)
_ = MyModel.predict(arch, parameters, Xtest)

print(Ypredi[:30])

#print(parameters['W'])


  
          





        
        
