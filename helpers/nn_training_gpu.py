from .shared_imports import *
from .nn_training import create_neural_net, train_net_and_export

from loky import get_reusable_executor

# partition the list of config dictionaries ('configurations')
# into sub-lists where configs are allocated in order to
# minimise total complexity of the corresponding models
# ---
# complexity of model is represented by its parameter count
# generally the greater the parameter count, the longer the training time
# ---
# greedy implementation of partition problem
def partition_configs(configurations, n_procs):
    config_partition_lst = [[] for _ in range(n_procs)]

    def get_param_count(config):
        layer_dims = [5] + config['hid_layer_dims'] + [len(config['output_indices'])]
        return create_neural_net(layer_dims).count_params()

    def get_group_weight(config_paramcount_lst):
        grad = 8.3140800e-06
        intercept = 1.4184722e+02
        f = lambda x: grad * x + intercept # fitted linear function transforming parameter count to epoch time
        return sum([f(x[1]) for x in config_paramcount_lst])

    def get_group_with_smallest_weight():
        group_param_counts = [get_group_weight(group) for group in config_partition_lst]
        group_smallest_weight_idx = group_param_counts.index(min(group_param_counts))
        return config_partition_lst[group_smallest_weight_idx]

    param_count_lst = [get_param_count(config) for config in configurations]
    all_config_paramcount_tups = sorted(zip(configurations, param_count_lst), key=lambda x: x[1], reverse=True)

    for x in all_config_paramcount_tups:
        group = get_group_with_smallest_weight()
        group.append(x)

    configurations_partition = [[x[0] for x in configs_partition] for configs_partition in config_partition_lst]
    return configurations_partition


def run_on_gpu(gpu_id, configurations, models_path, data_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    for config in configurations:
        mdl_id = train_net_and_export(config, models_path, data_path)
        print(mdl_id)

    return gpu_id


def run_models_gpu(configurations, models_path, data_path, gpus_available=None):
    if gpus_available == None:
        gpus_available = [0,1,2,3,4,5,6,7,8,9] # no. gpus available in ESA server

    n_configs = len(configurations)
    n_gpus = len(gpus_available)

    n_procs = n_configs if n_configs < n_gpus else n_gpus
    gpus_use = gpus_available[:n_procs]

    configs_partition = partition_configs(configurations, n_procs)

    def mycallback(res):
        gpu_id = res.result(timeout=1)
        print('GPU #{:d} complete'.format(gpu_id))

    # using external package loky (basically wrapper of multiprocessing)
    # able to gracefully terminate worker processes
    # and therefore free allocated resouces (e.g. GPU memory) on job completion
    with get_reusable_executor(max_workers=n_procs, timeout=1) as executor:
        for idx, config_lst in enumerate(configs_partition):
            res = executor.submit(run_on_gpu, gpus_use[idx], config_lst, models_path, data_path)
            res.add_done_callback(mycallback)
