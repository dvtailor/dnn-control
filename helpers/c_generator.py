from .shared_imports import *

from string import Template

def _get_layer_dims(nn_controller):
    n_dense_layers = len(nn_controller.weights) // 2
    layer_dims = [0 for _ in range(n_dense_layers+1)]
    for i in range(n_dense_layers):
        layer_dims[i] = nn_controller.weights[i*2].shape[0]
    layer_dims[-1] = nn_controller.weights[-1].shape[0]
    return layer_dims


def _get_layer_types(nn_controller):
    n_all_layers = len(nn_controller.config)
    layer_types = ['' for _ in range(n_all_layers)]

    for i in range(n_all_layers):
        layer_type = nn_controller.config[i]['class_name']
        if layer_type == 'Activation':
            layer_types[i] = nn_controller.config[i]['config']['activation'].upper()
        else:
            layer_types[i] = 'DENSE'

    return layer_types


def _format_1d_array(arr):
    out = ['{:.9e}'.format(x) for x in arr] # 16 for double
    # start new line every 3 floats
    indices = np.arange(0, len(arr), 3)[1:]
    break_str = '\n'.ljust(5) # newline + indentation (4 spaces)
    for idx in indices:
        out[idx] = break_str + out[idx]
    return ', '.join(out)


def _format_2d_array(arr_2d):
    out = [''.join(['{', _format_1d_array(arr_1d), '}']) for arr_1d in arr_2d]
    return ',\n'.join(out)


def _format_2d_array_lst(lst_arr_2d):
    out = [''.ljust(4) + _format_2d_array(arr_2d).replace('\n', '\n'.ljust(5)) for arr_2d in lst_arr_2d]
    return ',\n'.join(['\n'.join(['{', x, '}']) for x in out])


def _indent_formatted_arr(arr_str):
    break_str = '\n'.ljust(5)
    return ''.ljust(4) + arr_str.replace('\n', break_str)


def _format_layer_dims(layer_dims):
    out = [str(x) for x in layer_dims]
    out[0] = ''.ljust(4) + out[0]

    indices = np.arange(0, len(layer_dims), 13)[1:]
    break_str = '\n'.ljust(5)
    for idx in indices:
        out[idx] = break_str + out[idx]

    return ', '.join(out)


def _format_layer_types(layer_types):
    out = list(layer_types)
    out[0] = ''.ljust(4) + out[0]

    indices = np.arange(0, len(layer_types), 10)[1:]
    break_str = '\n'.ljust(5)
    for idx in indices:
        out[idx] = break_str + out[idx]

    return ', '.join(out)


def _get_template_params(nn_controller, state_bias):
    n_all_layers = len(nn_controller.config)
    n_dense_layers = len(nn_controller.weights) // 2
    layer_dims = _get_layer_dims(nn_controller)
    layer_types = _get_layer_types(nn_controller)
    n_state_vars = layer_dims[0]

    layer_dims_str = _format_layer_dims(layer_dims)
    layer_types_str = _format_layer_types(layer_types)

    input_scaler_params_str = _indent_formatted_arr(_format_2d_array(nn_controller.input_scaler_params))
    output_scaler_params_str = _indent_formatted_arr(_format_2d_array(nn_controller.output_scaler_params))

    biases_lst = [nn_controller.weights[i] for i in np.arange(1, len(nn_controller.weights), 2)]
    biases_str = _indent_formatted_arr(_format_2d_array(biases_lst))

    weights_lst = [nn_controller.weights[i].T for i in np.arange(0, len(nn_controller.weights), 2)]
    weights_str = _indent_formatted_arr(_format_2d_array_lst(weights_lst))

    state_bias = [0. for _ in range(n_state_vars)] if state_bias is None else state_bias
    state_bias_str = _indent_formatted_arr(_format_1d_array(state_bias))

    header_params = dict(
        (('num_state_vars', str(n_state_vars)),
        ('num_control_vars', str(layer_dims[-1])),
        ('num_all_layers', str(n_all_layers)),
        ('num_dense_layers', str(n_dense_layers)),
        ('max_layer_dims', str(max(layer_dims))))
    )

    source_params = dict(
        (('layer_dims', layer_dims_str),
        ('layer_types', layer_types_str),
        ('input_scaler_params', input_scaler_params_str),
        ('output_scaler_params', output_scaler_params_str),
        ('biases', biases_str),
        ('weights', weights_str),
        ('state_bias', state_bias_str))
    )

    return header_params, source_params


def generate_c_code(nn_controller, template_dir, export_dir, state_bias=None):
    header_template_path = os.path.join(template_dir, 'nn_params_header.tpl')
    with open(header_template_path, 'r') as tpl_file:
        header_template = tpl_file.read()

    source_template_path = os.path.join(template_dir, 'nn_params_source.tpl')
    with open(source_template_path, 'r') as tpl_file:
        source_template = tpl_file.read()

    header_params, source_params = _get_template_params(nn_controller, state_bias)

    header_str = Template(header_template).substitute(header_params)
    source_str = Template(source_template).substitute(source_params)

    header_path = os.path.join(export_dir, 'nn_params.h')
    source_path = os.path.join(export_dir, 'nn_params.c')

    with open(header_path, 'w') as out_file:
        out_file.write(header_str)
    with open(source_path, 'w') as out_file:
        out_file.write(source_str)
