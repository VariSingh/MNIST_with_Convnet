import tflearn as tl
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected

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

fully_connected = fully_connected(network,100,activation="relu")
fully_connected = fully_connected(fully_connected,10,activation="softmax")

regression = regression(fully_connected, optimizer="adam", learning_rate=0.001, loss="categorical_crossentropy", name="target")


#Training

model = tflearn.DNN(regression,tensorboard_verbose=0)
model.fit(X,Y, n_epoch=20, validation_set=(testX,testY),show_metric=True, batch_size=100, run_id="convNet_model")
