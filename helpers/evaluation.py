from .shared_imports import *

from keras.models import load_model
from .util import get_identifier
from .nn_training import create_neural_net


def get_model_metrics(config, models_path):
    mdl_id = get_identifier(config)
    loss_metrics = np.load(os.path.join(models_path, mdl_id, 'metrics_loss.npz'))
    mae_metrics = np.load(os.path.join(models_path, mdl_id, 'metrics_mean_absolute_error.npz'))
    metrics = {**loss_metrics, **mae_metrics}
    return metrics


def get_min_error(config, models_path, metric='val_mean_absolute_error'):
    metrics = get_model_metrics(config, models_path)
    return np.min(metrics[metric])


def get_min_errors(configurations, models_path, metric='val_mean_absolute_error'):
    errors = [0 for _ in range(len(configurations))]
    for idx, config in enumerate(configurations):
        errors[idx] = get_min_error(config, models_path, metric)
    return errors


def get_keras_model(config, models_path):
    mdl_id = get_identifier(config)
    return load_model(os.path.join(models_path, mdl_id, 'best_model.h5'))


def get_parameter_count(hid_layer_dims):
    layer_dims = [5] + hid_layer_dims + [2]
    return create_neural_net(layer_dims).count_params()


def get_best_model(configurations, models_path):
    min_errors = get_min_errors(configurations, models_path)
    best_mdl_idx = min_errors.index(min(min_errors))
    return configurations[best_mdl_idx], min_errors[best_mdl_idx]


# gets time taken to train model in hours
def get_training_time(config, models_path):
    path_to_model = os.path.join(models_path, get_identifier(config))

    path_config_export = os.path.join(path_to_model, 'config.txt')
    path_log = os.path.join(path_to_model, 'log.out')

    time_1 = os.path.getmtime(path_config_export)
    time_2 = os.path.getmtime(path_log)

    time_taken_hours = ((time_2 - time_1) / 60) / 60

    return time_taken_hours


# gets time taken for one epoch
# extracted from log file (set to 2nd epoch)
# ---
# future improvement: extract all times from log and average
def get_epoch_train_time(config, models_path):
    path_to_model = os.path.join(models_path, get_identifier(config))
    path_log = os.path.join(path_to_model, 'log.out')

    epoch_time = 0
    with open(path_log, 'r') as fp:
        for i, line in enumerate(fp):
            if i == 11:
                match_regex = re.search('\-([^s]+)', line)
                if match_regex != None:
                    epoch_time = int(match_regex.group(1))
                    break
            # could be on another line due to debug statements
            if i == 16:
                epoch_time = int(re.search('\-([^s]+)', line).group(1))

    return epoch_time


'''
dataset shape: (_,59,8)
'''
class Evaluator():
    def __init__(self, model, scalers, dataset, output_indices=[6,7]):
        self.model = model
        self.scalers = scalers
        self.dataset = dataset
        self.output_indices = output_indices

        self._process_data()

    def _process_data(self):
        input_scaler, target_normaliser, target_scaler = self.scalers
        input_indices = [1,2,3,4,5]

        X = self.dataset.reshape(-1,8)[:,input_indices]
        y = self.dataset.reshape(-1,8)[:,self.output_indices]

        X_norm = input_scaler.transform(X)
        self.model_preds_norm = self.model.predict(X_norm)
        model_preds = target_normaliser.inverse_transform(
                            target_scaler.inverse_transform(
                                self.model_preds_norm))
        self.model_preds = model_preds.reshape(-1, 59, len(self.output_indices))

        self.targets = self.dataset[:,:,self.output_indices]
        self.targets_norm = target_scaler.transform(
                                target_normaliser.transform(y))

        self.traj_times = self.dataset[:,:,0]
        self.n_trajs = self.targets.shape[0]


    def compute_mae(self):
        return np.mean(np.abs(self.targets_norm - self.model_preds_norm), axis=0)


    def compute_mse(self):
        return np.mean(np.square(self.targets_norm - self.model_preds_norm), axis=0)

    '''
    only necessary for comparison to carlos paper
    '''
    def compute_mae_unnormalised(self):
        y_pred = self.model_preds.reshape(-1, len(self.output_indices))
        y_true = self.targets.reshape(-1, len(self.output_indices))
        return np.mean(np.abs(y_pred - y_true), axis=0)

    '''
    range of control variables:
        u1: [0, 20]
        u2: [-2, 2]
    '''
    def plot_trajectory(self, traj_num):
        fig = plt.figure(figsize=(7, 3))
        if self.output_indices == [6,7]:
            plt.subplot(121)
            plt.plot(self.traj_times[traj_num], self.targets[traj_num,:,0], label='targets')
            plt.plot(self.traj_times[traj_num], self.model_preds[traj_num,:,0], label='preds')
            plt.ylim(-1, 21)
            plt.xlabel('time (s)')
            plt.ylabel('$u_1$ (N)')
            plt.legend()

            plt.subplot(122)
            plt.plot(self.traj_times[traj_num], self.targets[traj_num,:,1], label='targets')
            plt.plot(self.traj_times[traj_num], self.model_preds[traj_num,:,1], label='preds')
            plt.ylim(-2.2, 2.2)
        else:
            plt.subplot(121)
            plt.plot(self.traj_times[traj_num], self.targets[traj_num,:,0], label='targets')
            plt.plot(self.traj_times[traj_num], self.model_preds[traj_num,:,0], label='preds')

            if self.output_indices == [6]:
                plt.ylim(-1, 21)
                plt.xlabel('time (s)')
                plt.ylabel('$u_1$ (N)')
            else:
                plt.ylim(-2.2, 2.2)
                plt.xlabel('time (s)')
                plt.ylabel('$u_2$ (rad/s)')

        plt.legend()
        plt.tight_layout()
        plt.close(fig)
        return fig
