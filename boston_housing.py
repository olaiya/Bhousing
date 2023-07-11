import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPool1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.sparsity import keras as sparsity
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks
import numpy as np
import pandas as pd
import time


data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]


x, y = data, target
print(x.shape)

x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape)

#Generate train and test samples
xtrain, xtest, ytrain, ytest=train_test_split(x, y, test_size=0.2)

#Split test sample into 2, use one sample for validation
xtest, xval, ytest, yval=train_test_split(xtest, ytest, test_size=0.5)

##Build a simple CNN to model Boston housing

base_model = Sequential()
base_model.add(Conv1D(32, 2, activation="relu", input_shape=(13,1), name="inputlayer"))
base_model.add(MaxPool1D(2, name="maxpoolinglayer1"))
base_model.add(Conv1D(32, 2, activation="relu", name="convlayer1"))
#base_model.add(MaxPool1D(2, name="maxpoolinglayer2"))
base_model.add(Flatten())
base_model.add(Dense(24, activation="relu", name="denselayer1"))
base_model.add(Dense(1,name='output_dense'))
base_model.compile(loss="mse", optimizer="adam")
base_model.summary()

modelName = 'bhousing.h5'

#Check the size of the layers in your neural net
print("Checking size of layers in model")
for layer in base_model.layers:
    if layer.__class__.__name__ in ['Conv1D', 'Dense']:
        w = layer.get_weights()[0]
        layersize = np.prod(w.shape)
        print("{}: {}".format(layer.name,layersize)) # 0 = weights, 1 = biases
        if (layersize > 4096): # assuming that shape[0] is batch, i.e., 'None'
            print("Layer {} is too large ({}), are you sure you want to train?".format(layer.name,layersize))


batch_size = 12
n_epochs = 100

##Prune model
print('Shape is: {}'.format(xtrain.shape))
NSTEPS = int(xtrain.shape[0])  // batch_size #90% train, 10% validation in 10-fold cross validation
print('Number of training steps per epoch is {}'.format(NSTEPS))

#
train_model = True # True if you want to retrain, false if you want to load a previsously trained model


modelName = 'base_pruned_cnn_model.h5'

LOSS        = "mse"
OPTIMIZER   = "adam"

if train_model:

    base_model.compile(loss=LOSS, optimizer=OPTIMIZER)

    start = time.time()
    history = base_model.fit(xtrain, ytrain, validation_data=(xval, yval), batch_size=batch_size,epochs=n_epochs)
    end = time.time()
    print('It took {} minutes to train pruned Keras model'.format( (end - start)/60.))

    base_model.save(modelName)

else:
    print('Loading model')
    base_model = tf.keras.models.load_model(modelName)
    base_model.compile(loss=LOSS, optimizer=OPTIMIZER)

print('Running predictions\n')
ypred_prune = base_model.predict(xtest)
print(base_model.evaluate(xtrain, ytrain))
print("MSE: %.4f" % mean_squared_error(ytest, ypred_prune))

x_ax_prune = range(len(ypred_prune))
plt.figure(2)
plt.scatter(x_ax_prune, ytest, s=5, color="blue", label="original")
plt.plot(x_ax_prune, ypred_prune, lw=0.8, color="red", label="predicted")
plt.legend()
plt.savefig('BHousing_pruned.png')
plt.show()

import hls4ml
import plotting

#First, the baseline model
hls_config = hls4ml.utils.config_from_keras_model(base_model, granularity='name')

for layer in hls_config['LayerName'].keys():
    hls_config['LayerName'][layer]['Trace'] = True


hls_model = hls4ml.converters.convert_from_keras_model(base_model,
                                                       hls_config=hls_config,
                                                       output_dir='model_1/hls4ml_prj_2',
                                                       part='xcu250-figd2104-2L-e')

print('\nProfiling model\n')
plots = hls4ml.model.profiling.numerical(model=base_model, hls_model=hls_model, X=xtest)

for i, plot in enumerate(plots):
    plot.savefig(f'hls4mlPlots{i}.png')

# Set the precision and reuse factor for the full model
hls_config['Model']['Precision'] = 'ap_fixed<16,10>'
hls_config['Model']['ReuseFactor'] = 1

hls_config['LayerName']['inputlayer_input']['Precision']['result'] = 'ap_fixed<16,10>'
hls_config['LayerName']['inputlayer']['Precision']['result'] = 'ap_fixed<16,10>'
hls_config['LayerName']['inputlayer']['Precision']['bias'] = 'ap_fixed<16,10>'
hls_config['LayerName']['inputlayer']['Precision']['weight'] = 'ap_fixed<16,10>'
#hls_config['LayerName']['maxpoolinglayer1']['Precision']['result'] = 'ap_fixed<16,10>'
hls_config['LayerName']['inputlayer_relu']['Precision'] = 'ap_fixed<16,10>'
hls_config['LayerName']['inputlayer_relu']['table_t'] = 'ap_fixed<18,10>'

hls_config['LayerName']['maxpoolinglayer1']['Precision'] = 'ap_fixed<16,10>'
hls_config['LayerName']['convlayer1']['Precision']['result'] = 'ap_fixed<16,10>'
hls_config['LayerName']['convlayer1']['Precision']['bias'] = 'ap_fixed<16,10>'
hls_config['LayerName']['convlayer1']['Precision']['weight'] = 'ap_fixed<16,10>'
hls_config['LayerName']['convlayer1_relu']['Precision'] = 'ap_fixed<16,10>'
hls_config['LayerName']['convlayer1_relu']['table_t'] = 'ap_fixed<18,10>'

#hls_config['LayerName']['flatten']['Precision']['result'] = 'ap_fixed<16,10>'
hls_config['LayerName']['denselayer1']['Precision']['result'] = 'ap_fixed<16,10>'
hls_config['LayerName']['denselayer1']['Precision']['bias'] = 'ap_fixed<16,10>'
hls_config['LayerName']['denselayer1']['Precision']['weight'] = 'ap_fixed<16,10>'
hls_config['LayerName']['denselayer1_relu']['Precision'] = 'ap_fixed<16,10>'
hls_config['LayerName']['denselayer1_relu']['table_t'] = 'ap_fixed<18,10>'

hls_config['LayerName']['output_dense']['Precision']['result'] = 'ap_fixed<16,10>'
hls_config['LayerName']['output_dense']['Precision']['bias'] = 'ap_fixed<16,10>'
hls_config['LayerName']['output_dense']['Precision']['weight'] = 'ap_fixed<16,10>'
hls_config['LayerName']['output_dense_linear']['Precision'] = 'ap_fixed<16,10>'
hls_config['LayerName']['output_dense_linear']['table_t'] = 'ap_fixed<18,10>'
#Try model
hls_model = hls4ml.converters.convert_from_keras_model(base_model,
                                                       hls_config=hls_config,
                                                       output_dir='model_2/hls4ml_prj_2',
                                                       part='xcu250-figd2104-2L-e')

print('\nProfiling model\n')
plots2 = hls4ml.model.profiling.numerical(model=base_model, hls_model=hls_model, X=xtest)

for i, plot in enumerate(plots2):
    plot.savefig(f'hls4mlPlots{i}_mod.png')

plotting.print_dict(hls_config)

start = time.time()
hls_model.compile()
end = time.time()
print('It took {} minutes to run HLS compilation\n'.format( (end - start)/60.))

start = time.time()
print('Running predictions\n')
#ypred_hls = hls_model.predict(np.ascontiguousarray(xtest))
ypred_hls = hls_model.predict(xtest)
print("MSE hls: %.4f" % mean_squared_error(ytest, ypred_hls))
end = time.time()
print('It took {} minutes to run predictions\n'.format( (end - start)/60.))

x_ax_hls = range(len(ypred_hls))
plt.figure(6)
plt.scatter(x_ax_hls, ytest, s=5, color="blue", label="original")
plt.plot(x_ax_hls, ypred_hls, lw=0.8, color="red", label="predicted")
plt.legend()
plt.savefig('BHousing_hls.png')
plt.show()
