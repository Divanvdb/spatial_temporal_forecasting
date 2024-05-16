import xarray as xr
import numpy as np

def coarsen_data(ds):
    lat_factor = ds.sizes['latitude'] // 16
    lon_factor = ds.sizes['longitude'] // 32

    ds_coarsened = ds.coarsen(latitude=lat_factor, longitude=lon_factor).mean()

    return ds_coarsened


def load_data(file_name='ERA5_Data/2023_SouthAfrica.nc', subset = True, coarsen=True):
    ds = xr.open_dataset(file_name)
    
    ds.load()

    ds['ws'] = (ds.u10**2 + ds.v10**2)**0.5

    ds_og = ds.copy()

    if subset:
        lat_slice = slice(1, 33)  
        lon_slice = slice(3, 67)  

        ds = ds.isel(latitude=lat_slice, longitude=lon_slice)

    if coarsen:
        ds = coarsen_data(ds)

    return ds, ds_og


def window(ds, window_size, variable):
    X, y = [], []

    for i in range(ds.time.size - window_size):
        X.append(ds[variable].isel(time=slice(i, i + window_size)))
        y.append(ds[variable].isel(time=i + window_size))

    return np.array(X), np.array(y)


def normalize(X, y):
    max_ws = X.max()
    X_n = X / max_ws
    y_n = y / max_ws

    return X_n, y_n, max_ws


def train_test_split(X, y, split_percent=0.9):
    split = int(X.shape[0] * split_percent)

    f_train, t_train = X[:split].astype('float32'), y[:split].astype('float32')
    f_test, t_test = X[split:].astype('float32'), y[split:].astype('float32')

    return f_train, t_train, f_test, t_test


def rollout(model, current_state, steps, input_shape, plot_shape):
    output = []
    for i in range(steps):
        pred = model.predict(current_state.reshape(input_shape)).reshape(plot_shape)
        current_state = current_state[:,1:]
        current_state = np.append(current_state, pred.reshape(1, 1, pred.shape[0], pred.shape[1]), axis=1)
        output.append(pred)

    return np.array(output).squeeze()
