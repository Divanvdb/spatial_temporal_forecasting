from UNet import *
from data_utilities import *
from tensorflow import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


# Hyperparameters
file_name_ = 'ERA5_Data/2023_SouthAfrica.nc'
run_name = 'RRnet_2023'
batch_size = 64
epochs = 5
window_size = 3
variable = 'ws'
seed = 10
steps = 10

# Data Load and Prep

ds, _ = load_data(file_name=file_name_)

lon_shape = ds.longitude.size
lat_shape = ds.latitude.size

X_, y_ = window(ds, window_size, variable)

X_n, y_n, max_ws = normalize(X_, y_)

f_train, t_train, f_test, t_test = train_test_split(X_n, y_n)


X_training = X_n.reshape(-1, lat_shape, lon_shape, window_size)
y_training = y_n.reshape(-1, lat_shape, lon_shape,  1)

f_training, t_training, f_testing, t_testing = train_test_split(X_training, y_training)

print('Train Shape: ', f_train.shape, t_train.shape, '\n\nTest Shape: ', f_test.shape, t_test.shape)
print('\n\nTraining Shape: ', f_training.shape, t_training.shape, '\n\nTesting Shape: ', f_testing.shape, t_testing.shape)

# Model Setup

input_shape = tuple((1, lat_shape, lon_shape, window_size))
plot_shape = t_training[0].shape

model = build_RRnet(input_shape[1:])

model.compile(optimizer='adam', loss='mse')

model.load_weights(f'models/{run_name}.h5')

# Calculate Metrics

targets = t_test[seed:seed + steps]

rollout_pred = rollout(model, f_test[seed:seed + 1], steps, input_shape, plot_shape)

rollout_pred = rollout_pred * max_ws
targets = targets * max_ws

mse = mean_squared_error(targets.flatten(), rollout_pred.flatten())
mae = mean_absolute_error(targets.flatten(), rollout_pred.flatten())
mape = mean_absolute_percentage_error(targets.flatten(), rollout_pred.flatten())
rmse = np.sqrt(mse)

print(f'Mean Squared Error: {mse}, Mean Absolute Error: {mae}, Mean Absolute Percentage Error: {mape}, Root Mean Squared Error: {rmse}')    