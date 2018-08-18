import tflearn as tl
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
#load data
import tflearn.datasets.mnist as mnist
X, Y, testX, testY = mnist.load_data(one_hot=True)

X = X.reshape([-1,28,28,1])
testX = testX.reshape([-1,28,28,1])



#Convolutional neural network


input = tl.input_data(shape=[None,28,28,1],name="input")

network = conv_2d(input,32,3, activation="relu")
network = max_pool_2d(network,2)

network = conv_2d(network,64,3, activation="relu")
network = max_pool_2d(network,2)

network = local_response_normalization(network)

network = fully_connected(network,256,activation="relu")
network = fully_connected(network,10,activation="softmax")

regression = regression(network, optimizer="adam", learning_rate=0.001, loss="categorical_crossentropy", name="target")


#Training

model = tl.DNN(regression,tensorboard_verbose=0)
model.fit(X,Y, n_epoch=20, validation_set=(testX,testY),show_metric=True, batch_size=100, run_id="convNet_model")
