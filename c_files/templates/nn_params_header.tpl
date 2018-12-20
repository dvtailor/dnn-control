#ifndef _NN_PARAMS_H
#define _NN_PARAMS_H

typedef enum {DENSE, RELU, TANH, SOFTPLUS} Layer;

#define NUM_STATE_VARS ${num_state_vars}
#define NUM_CONTROL_VARS ${num_control_vars}
#define NUM_ALL_LAYERS ${num_all_layers}
#define NUM_DENSE_LAYERS ${num_dense_layers}
#define MAX_LAYER_DIMS ${max_layer_dims}

extern const unsigned short LAYER_DIMS[];
extern const Layer LAYER_TYPES[];
extern const float STATE_BIAS[NUM_STATE_VARS];
extern const float INPUT_SCALER_PARAMS[2][NUM_STATE_VARS];
extern const float OUTPUT_SCALER_PARAMS[4][NUM_CONTROL_VARS];
extern const float BIASES[NUM_DENSE_LAYERS][MAX_LAYER_DIMS];
extern const float WEIGHTS[NUM_DENSE_LAYERS][MAX_LAYER_DIMS][MAX_LAYER_DIMS];

#endif
