# use mlp for prediction on multi-output regression
from numpy import asarray
from sklearn.datasets import make_regression
from keras.models import Sequential
from keras.layers import Dense
 
# get the dataset
def get_dataset():
	X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, n_targets=3, random_state=2)
	return X, y
 
# get the model
def get_model(n_inputs, n_outputs):
	model = Sequential()
	model.add(Dense(20, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
	model.add(Dense(n_outputs, kernel_initializer='he_uniform'))
	model.compile(loss='mae', optimizer='adam')
	return model
 

if __name__ == "__main__":
    roni_bot_path = "Dataset/roni_bot"
    # load dataset
    X, y = get_dataset()
    n_inputs, n_outputs = X.shape[1], y.shape[1]
    # get model
    model = get_model(n_inputs, n_outputs)
    # fit the model on all data
    model.fit(X, y, verbose=0, epochs=100)
    # make a prediction for new data
    row = [-0.99859353,2.19284309,-0.42632569,-0.21043258,-1.13655612,-0.55671602,-0.63169045,-0.87625098,-0.99445578,-0.3677487]
    newX = asarray([row])
    yhat = model.predict(newX)
    print('Predicted: %s' % yhat[0])
    