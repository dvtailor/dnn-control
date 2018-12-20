from .shared_imports import *

from sklearn.preprocessing import StandardScaler, MinMaxScaler


'''
    we split on whole trajectories
    train, valid, test split proportions: (0.8, 0.1, 0.1)
    take sequentially since already random order
'''
def partition_dataset(dataset):
    n_trajs = dataset.shape[0] # 200000

    lower_idx = 0
    upper_idx = int(n_trajs*0.8)
    dataset_train = dataset[lower_idx:upper_idx]

    lower_idx = upper_idx
    upper_idx = lower_idx + int(n_trajs*0.1)
    dataset_valid = dataset[lower_idx:upper_idx]

    lower_idx = upper_idx
    upper_idx = lower_idx + int(n_trajs*0.1)
    dataset_test = dataset[lower_idx:upper_idx]

    return dataset_train, dataset_valid, dataset_test


'''
    normalise input data
'''
def normalise_inputs(inputs):
    input_scaler = StandardScaler().fit(inputs)
    inputs_norm = input_scaler.transform(inputs)
    return inputs_norm, input_scaler


'''
    normalise and scale target data
'''
def normalise_and_scale_targets(targets):
    target_normaliser = StandardScaler().fit(targets)
    targets_norm = target_normaliser.transform(targets)

    target_scaler = MinMaxScaler(feature_range=(-1, 1),).fit(targets_norm)
    targets_scaled = target_scaler.transform(targets_norm)

    return targets_scaled, target_normaliser, target_scaler


'''
    get normalised train, valid, test datasets for machine learning
    also returns dataset transformation functions
'''
def get_train_val_datasets(path_to_data, output_indices=[6,7]):
    dataset = np.load(path_to_data)
    dataset_train, dataset_valid, dataset_test = partition_dataset(dataset)

    input_indices = [1,2,3,4,5]

    X_train, input_scaler = \
        normalise_inputs(dataset_train.reshape(-1, 8)[:,input_indices])

    y_train, target_normaliser, target_scaler = \
        normalise_and_scale_targets(dataset_train.reshape(-1,8)[:,output_indices])

    X_val = input_scaler.transform(dataset_valid.reshape(-1,8)[:, input_indices])
    y_val = target_scaler.transform(
                target_normaliser.transform(
                    dataset_valid.reshape(-1, 8)[:,output_indices]))

    X_test = input_scaler.transform(dataset_test.reshape(-1,8)[:,input_indices])
    y_test = target_scaler.transform(
                target_normaliser.transform(
                    dataset_test.reshape(-1, 8)[:,output_indices]))

    data = (X_train, y_train, X_val, y_val, X_test, y_test)
    scalers = (input_scaler, target_normaliser, target_scaler)

    return data, scalers


'''
functions to process loss history as a result of learning rate trick
'''
# returned array contains 'mov_avg_len' fewer elements
def moving_average(arr, mov_avg_len):
    ret = np.cumsum(arr, dtype=float)
    ret[mov_avg_len:] = ret[mov_avg_len:] - ret[:-mov_avg_len]
    return ret[(mov_avg_len - 1):] / mov_avg_len

def smooth_arr(arr, learn_rates, mov_avg_len):
    new_arr = moving_average(arr, mov_avg_len)
    return new_arr, learn_rates[mov_avg_len-1:]

def compute_loss_grad(loss_hist, learn_rates, skip_len=25):
    loss_grad = (loss_hist[skip_len:] - loss_hist[:-skip_len]) / skip_len
    return loss_grad, learn_rates[skip_len:]
