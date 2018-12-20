from .shared_imports import *

import itertools, json, shutil, pickle
from collections import OrderedDict


def get_identifier(config_dict):
    # replace mutable types (list) with immutable (tuple)
    model_config_tup = {}
    for key, value in config_dict.items():
        if type(value) is list:
            model_config_tup[key] = tuple(value)
        else:
            model_config_tup[key] = value
    # changed frozenset to tuple due to hash issue with different Python releases
    # not guaranteed to be consistent across versions; must check!
    mdl_id = str(hash(tuple(model_config_tup.items())))

    return 'model' + mdl_id


def create_export_path(mdl_id, config_dict, dir_path):
    mdl_path = os.path.join(dir_path, mdl_id)
    shutil.rmtree(mdl_path, ignore_errors=True)
    os.makedirs(mdl_path)
    _save_config(config_dict, mdl_path)
    return mdl_path


def _save_config(config_dict, path):
    file_path = os.path.join(path, 'config.txt')
    with open(file_path, 'w') as file:
        file.write(json.dumps(config_dict))


def read_config(path):
    return json.load(open(path), object_pairs_hook=OrderedDict)


def read_configurations(models_path):
    mdl_id_lst = os.listdir(models_path)
    configs = [0 for _ in range(len(mdl_id_lst))]

    for i, mdl_id in enumerate(mdl_id_lst):
        path_to_config = os.path.join(models_path, mdl_id, 'config.txt')
        configs[i] = read_config(path_to_config)

    return configs


def get_possible_configurations(hyper_param_grid, param_keys=None):
    if param_keys==None:
        param_keys = ['output_indices',
                        'learning_rule',
                        'learning_rate',
                        'batch_size',
                        'hid_layer_dims']
    param_args = []
    for i in range(len(param_keys)):
        param_args.append(hyper_param_grid[param_keys[i]])

    configs = list(itertools.product(*param_args))
    return [_convert_to_dict(configs[i], param_keys) for i in range(len(configs))]


def _convert_to_dict(config_tup, param_keys):
    config_dict = OrderedDict()
    for i in range(len(param_keys)):
        config_dict[param_keys[i]] = config_tup[i]
    return config_dict


def remove_model_paths(configurations, models_path):
    for config in configurations:
        path_to_model = os.path.join(models_path, get_identifier(config))
        shutil.rmtree(path_to_model, ignore_errors=True)


def pickle_controller_params(model, scalers, export_path):
    input_scaler = scalers[0]
    input_scaler_params = (input_scaler.mean_, input_scaler.scale_)

    output_normaliser = scalers[1]
    output_scaler = scalers[2]
    output_scaler_params = (output_scaler.min_, output_scaler.scale_,
                                output_normaliser.scale_, output_normaliser.mean_)

    config = model.get_config()
    weights = model.get_weights()

    params = (input_scaler_params, output_scaler_params, config, weights)
    pickle.dump(params, open(export_path, 'wb'))


def get_hid_layer_dims_combs(n_layers, layer_units):
    return [list(x) for x in
            list(itertools.combinations_with_replacement(
                sorted(layer_units, reverse=True), n_layers))]
