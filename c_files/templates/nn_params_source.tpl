#include "nn_params.h"

const unsigned short LAYER_DIMS[] = {
${layer_dims}
};

const Layer LAYER_TYPES[] = {
${layer_types}
};

const float STATE_BIAS[NUM_STATE_VARS] = {
${state_bias}  
};

const float INPUT_SCALER_PARAMS[2][NUM_STATE_VARS] = {
${input_scaler_params}
};

const float OUTPUT_SCALER_PARAMS[4][NUM_CONTROL_VARS] = {
${output_scaler_params}
};

const float BIASES[NUM_DENSE_LAYERS][MAX_LAYER_DIMS] = {
${biases}
};

const float WEIGHTS[NUM_DENSE_LAYERS][MAX_LAYER_DIMS][MAX_LAYER_DIMS] = {
${weights}
};
