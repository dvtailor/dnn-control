from .shared_imports import *
from .util import get_identifier, create_export_path
from .preprocessing import get_train_val_datasets

import sys, multiprocessing, keras
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras import initializers, optimizers
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau
from keras import backend as K

def run_models(configurations, models_path, data_path, n_processes=None):
    pool = multiprocessing.Pool(n_processes)

    def mycallback(x):
        print(x)

    for config in configurations:
        pool.apply_async(train_net_and_export, args=(config, models_path, data_path), callback=mycallback)

    pool.close()
    pool.join()
    pool.terminate()


def train_net_and_export(config, models_path, data_path):
    mdl_id = get_identifier(config)
    mdl_path = create_export_path(mdl_id, config, models_path)

    old_stdout = sys.stdout
    sys.stdout = open(os.path.join(mdl_path, 'log.out'), 'w')

    print('dataset path:', data_path)
    print('output indices:', config['output_indices'])
    print('learning rule:', config['learning_rule'])
    print('learning rate:', config['learning_rate'])
    print('minibatch size:', config['batch_size'])
    print('hid layer dims:', config['hid_layer_dims'])
    print()

    create_and_train_network(config, data_path, exportModel=True, mdl_path=mdl_path)
    sys.stdout = old_stdout

    return mdl_id


def create_and_train_network(config, data_path, epochs=500, exportModel=False, mdl_path=None, random_seed=60317):
    data, _ = get_train_val_datasets(data_path, config['output_indices'])
    X_train, y_train, X_val, y_val, _, _ = data

    layer_dims = [5] + config['hid_layer_dims'] + [len(config['output_indices'])]
    model = create_neural_net(layer_dims, random_seed)

    additional_metrics = ['mean_absolute_error']
    set_optimiser(model, config['learning_rate'], config['learning_rule'], additional_metrics)

    early_stop_patience = 5
    start_epoch = 100 # min num epochs trained for before enable early stopping
    lr_reduce_patience = 3
    metric_to_monitor = 'val_loss' # 'val_mean_absolute_error'

    callbacks = []
    callbacks.append(get_callback_reduce_lr(metric_to_monitor, lr_reduce_patience))
    callbacks.append(get_callback_early_stopping(metric_to_monitor, early_stop_patience, start_epoch))

    if exportModel:
        callbacks.append(get_callback_export_model(mdl_path, metric_to_monitor))
        callbacks.append(get_callback_save_metrics(mdl_path, 'loss'))
        callbacks.append(get_callback_save_metrics(mdl_path, 'mean_absolute_error'))

    verbose = 2 if exportModel else 1
    model.fit(x=X_train,
                y=y_train,
                epochs=epochs,
                batch_size=config['batch_size'],
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                shuffle=True,
                verbose=verbose)


def create_neural_net(layer_dims, random_seed=None):
    model = Sequential()

    model.add(Dense(layer_dims[1],
        input_dim=layer_dims[0],
        kernel_initializer=initializers.glorot_uniform(random_seed),
        bias_initializer='zeros'))

    for i in range(2, len(layer_dims)):
        model.add(Activation('softplus')) # 'relu'

        model.add(Dense(layer_dims[i],
              kernel_initializer=initializers.glorot_uniform(random_seed),
              bias_initializer='zeros'))

    model.add(Activation('tanh'))

    return model


def set_optimiser(model, learning_rate=1e-3, learning_rule='adam', metrics=[]):
    optimizer=None
    if learning_rule == 'adam':
        optimizer = optimizers.Adam(lr=learning_rate)
    elif learning_rule == 'sgd_mom':
        optimizer = optimizers.SGD(lr=learning_rate, momentum=0.9)
    elif learning_rule == 'sgd_nest_mom':
        optimizer = optimizers.SGD(lr=learning_rate, momentum=0.9, nesterov=True)

    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=metrics)


def get_callback_export_model(path, monitor):
    save_model_path = os.path.join(path, 'best_model.h5')
    checkpoint_best = ModelCheckpoint(save_model_path,
                        monitor=monitor,
                        save_best_only=True,
                        mode='min',
                        period=1)
    return checkpoint_best


# also stores validation metrics even if not included in model.fit()
# this would simply be a list of None
class MetricHistory(Callback):
    def __init__(self, path, interval, metric):
        super(MetricHistory, self).__init__()
        self.path = path
        self.interval = interval
        self.metric = metric

    def on_train_begin(self, logs={}):
        self.train = []
        self.val = []
        self.batch = []

    def on_batch_end(self, batch, logs={}):
        self.batch_epoch.append(logs.get(self.metric))

    def on_epoch_begin(self, epoch, logs=None):
        self.batch_epoch = []

    def on_epoch_end(self, epoch, logs={}):
        self.batch.append(self.batch_epoch)

        self.train.append(logs.get(self.metric))
        self.val.append(logs.get('val_' + self.metric))

        if (epoch+1) % self.interval == 0:
            np.savez_compressed(self.path,
                **{'train_' + self.metric : np.array(self.train),
                    'val_' + self.metric : np.array(self.val),
                    'batch_' + self.metric : np.array(self.batch)})


def get_callback_save_metrics(path, metric):
    filename = 'metrics_' + metric
    return MetricHistory(os.path.join(path, filename), 1, metric)


# only applies early stopping after minimum number of epochs `start_epoch`
# has passed
class CustomStopper(EarlyStopping):
    def __init__(self, monitor='val_loss',
             min_delta=0, patience=0, verbose=0, mode='auto', start_epoch=100):
        super().__init__(monitor, min_delta, patience, verbose, mode)
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.start_epoch:
            super().on_epoch_end(epoch, logs)


def get_callback_early_stopping(metric, patience, start_epoch):
    return CustomStopper(monitor=metric,
                    min_delta=0,
                    patience=patience,
                    verbose=1,
                    mode='min',
                    start_epoch=start_epoch)


def get_callback_reduce_lr(metric, patience):
    return ReduceLROnPlateau(monitor=metric,
                             factor=0.5,
                             patience=patience,
                             mode='min',
                             min_delta=0,
                             verbose=1,
                             min_lr=1e-6)


# trick to estimate optimal starting learning rate
# https://towardsdatascience.com/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0
# increases (reference) learning rate logarithmically during training for single epoch
# stores loss after each minibatch update
def generate_loss_history(inputs, targets, learning_rule, hid_layer_dims, batch_size):
    class LRFinder(Callback):
        def __init__(self, steps_per_epoch, start, stop):
            super().__init__()

            self.min_lr = 10**start
            self.max_lr = 10**stop
            self.steps_per_epoch = steps_per_epoch
            self.lrs = np.logspace(start, stop, steps_per_epoch)

            self.history = {}
            self.iteration = 0

        def on_train_begin(self, logs={}):
            K.set_value(self.model.optimizer.lr, self.min_lr)

        def on_batch_end(self, batch, logs={}):
            # store training minibatch loss (key: 'loss')
            for k, v in logs.items():
                self.history.setdefault(k, []).append(v)

            self.iteration += 1

            if self.iteration < self.steps_per_epoch:
                K.set_value(self.model.optimizer.lr, self.lrs[self.iteration])

    # exponent of start/end learning rates (base 10)
    lr_start = -6
    lr_end = -2 # 0
    n_intervals = 397 # 595 # adjusted to include all 10^x / no. steps (single epoch)

    np.random.seed(0)
    n_instances = batch_size * n_intervals
    indices = np.random.randint(inputs.shape[0], size=n_instances)

    # data sampled to reduce size to number instances required
    X = inputs[indices,:]
    y = targets[indices,:]

    model = create_neural_net([5] + hid_layer_dims + [2], random_seed=0)
    set_optimiser(model, learning_rule=learning_rule)
    lr_finder = LRFinder(n_intervals, lr_start, lr_end)

    model.fit(X,
              y,
              epochs=1,
              batch_size=batch_size,
              callbacks=[lr_finder],
              verbose=0,
              shuffle=False)

    return np.array(lr_finder.history['loss']), lr_finder.lrs
