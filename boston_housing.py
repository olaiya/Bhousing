from sklearn.datasets import load_boston
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
import time


boston = load_boston()
x, y = boston.data, boston.target
print(x.shape)

x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape)

#Generate train and test samples
xtrain, xtest, ytrain, ytest=train_test_split(x, y, test_size=0.2)

#Split test sample into 2, use one sample for validation
xtest, xval, ytest, yval=train_test_split(xtest, ytest, test_size=0.5)

##Build a simple CNN to model Boston housing

model = Sequential()
model.add(Conv1D(32, 2, activation="relu", input_shape=(13,1), name="inputlayer"))
model.add(MaxPool1D(2, name="maxpoolinglayer1"))
model.add(Conv1D(32, 2, activation="relu", name="convlayer1"))
#model.add(MaxPool1D(2, name="maxpoolinglayer2"))
model.add(Flatten())
model.add(Dense(24, activation="relu", name="denselayer1"))
model.add(Dense(1,name='output_dense'))
model.compile(loss="mse", optimizer="adam")
model.summary()

modelName = 'bhousing.h5'

#Check the size of the layers in your neural net
print("Checking size of layers in model")
for layer in model.layers:
    if layer.__class__.__name__ in ['Conv1D', 'Dense']:
        w = layer.get_weights()[0]
        layersize = np.prod(w.shape)
        print("{}: {}".format(layer.name,layersize)) # 0 = weights, 1 = biases
        if (layersize > 4096): # assuming that shape[0] is batch, i.e., 'None'
            print("Layer {} is too large ({}), are you sure you want to train?".format(layer.name,layersize))


cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=modelName,
        save_best_only=True,
        verbose=1)

batch_size = 12
n_epochs = 50


start = time.time()
history = model.fit(xtrain, ytrain, validation_data=(xval, yval), callbacks=[cp_callback], batch_size=batch_size,epochs=n_epochs)
end = time.time()
print('It took {} minutes to train Keras model'.format( (end - start)/60.))

print('\nSaving model\n')
model.save(modelName)
print('Running predictions\n')
ypred = model.predict(xtest)
print(model.evaluate(xtrain, ytrain))
print("MSE: %.4f" % mean_squared_error(ytest, ypred))

x_ax = range(len(ypred))
plt.scatter(x_ax, ytest, s=5, color="blue", label="original")
plt.plot(x_ax, ypred, lw=0.8, color="red", label="predicted")
plt.legend()
plt.savefig('BHousing.png')
plt.show()

##Prune model
print('Shape is: {}'.format(xtrain.shape))
NSTEPS = int(xtrain.shape[0])  // batch_size #90% train, 10% validation in 10-fold cross validation
print('Number of training steps per epoch is {}'.format(NSTEPS))

# Prune all convolutional and dense layers gradually from 0 to 50% sparsity every 2 epochs,
# ending by the 10th epoch
def pruneFunction(layer):
    pruning_params = {'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity = 0.0,
                                                                   final_sparsity = 0.50,
                                                                   begin_step = NSTEPS*2,
                                                                   end_step = NSTEPS*10,
                                                                   frequency = NSTEPS)
                     }
    if isinstance(layer, tf.keras.layers.Conv1D):
        return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
    if isinstance(layer, tf.keras.layers.Dense) and layer.name!='output_dense':
        return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
    return layer

model_pruned = tf.keras.models.clone_model( model, clone_function=pruneFunction)
train_pruned = True # True if you want to retrain, false if you want to load a previsously trained model


modelName_pruned = 'pruned_cnn_model.h5'

LOSS        = "mse"
OPTIMIZER   = "adam"

if train_pruned:

    model_pruned.compile(loss=LOSS, optimizer=OPTIMIZER)

    callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
            pruning_callbacks.UpdatePruningStep()
            ]

    start = time.time()
    history = model_pruned.fit(xtrain, ytrain, validation_data=(xval, yval), callbacks=[pruning_callbacks.UpdatePruningStep()], batch_size=batch_size,epochs=n_epochs)
    end = time.time()
    print('It took {} minutes to train pruned Keras model'.format( (end - start)/60.))

    model_pruned.save(modelName_pruned)

else:
    from tensorflow_model_optimization.sparsity.keras import strip_pruning
    from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper

    from qkeras.utils import _add_supported_quantized_objects

    co = {}
    _add_supported_quantized_objects(co)
    co['PruneLowMagnitude'] = pruning_wrapper.PruneLowMagnitude
    model_pruned = tf.keras.models.load_model(modelName_pruned, custom_objects=co)
    model_pruned  = strip_pruning(model_pruned)
    model_pruned.compile(loss=LOSS, optimizer=OPTIMIZER)

print('Running predictions\n')
ypred_prune = model_pruned.predict(xtest)
print(model_pruned.evaluate(xtrain, ytrain))
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

from tensorflow_model_optimization.sparsity.keras import strip_pruning
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper

from qkeras.utils import _add_supported_quantized_objects

co = {}
_add_supported_quantized_objects(co)
co['PruneLowMagnitude'] = pruning_wrapper.PruneLowMagnitude
model = tf.keras.models.load_model(modelName_pruned, custom_objects=co)
model  = strip_pruning(model_pruned)
model.compile(loss=LOSS, optimizer=OPTIMIZER)

#Check model loaded correctly
print('Running predictions\n')
ypred_prune = model.predict(xtest)
print(model.evaluate(xtrain, ytrain))
print("MSE: %.4f" % mean_squared_error(ytest, ypred_prune))

#First, the baseline model
hls_config = hls4ml.utils.config_from_keras_model(model, granularity='name')

for layer in hls_config['LayerName'].keys():
    hls_config['LayerName'][layer]['Trace'] = True

hls_model = hls4ml.converters.convert_from_keras_model(model,
                                                       hls_config=hls_config,
                                                       output_dir='model_1/hls4ml_prj_2',
                                                       part='xcu250-figd2104-2L-e')

print('\nProfiling model\n')
#plots = hls4ml.model.profiling.numerical(model=model, hls_model=hls_model, X=xtest)
#for i, plot in enumerate(plots):
#    plot.savefig(f'hls4mlPlots{i}.png')

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
hls_model = hls4ml.converters.convert_from_keras_model(model,
                                                       hls_config=hls_config,
                                                       output_dir='model_2/hls4ml_prj_2',
                                                       part='xcu250-figd2104-2L-e')

print('\nProfiling model\n')
#plots2 = hls4ml.model.profiling.numerical(model=model, hls_model=hls_model, X=xtest)
#for i, plot in enumerate(plots2):
#    plot.savefig(f'hls4mlPlots{i}_mod.png')

plotting.print_dict(hls_config)

start = time.time()
hls_model.compile()
end = time.time()
print('It took {} minutes to run HLS compilation\n'.format( (end - start)/60.))

'''
hls4ml_pred, hls4ml_trace = hls_model.trace(xtest)
keras_trace = hls4ml.model.profiling.get_ymodel_keras(model, xtest)


print("Keras layer 'inputlayer', first sample:")
print(keras_trace['inputlayer'][0])
print("hls4ml layer 'inputlayer', first sample:")
print(hls4ml_trace['inputlayer'][0])
print('Length of object is:')
print(len(keras_trace['inputlayer']))
print(keras_trace['inputlayer'][0]-hls4ml_trace['inputlayer'][0])
'''

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
