#include <math.h>
#include <string.h>
#include "nn.h"
#include "nn_params.h"
#include "pd_gains.h"

void state_bias_correction(float input[]) {
    int i;
    for (i = 0; i < NUM_STATE_VARS; i++) {
        input[i] = input[i] + STATE_BIAS[i];
    }
}

void preprocess_input(float input[]) {
    int i;
    for (i = 0; i < NUM_STATE_VARS; i++) {
        input[i] = (input[i] - INPUT_SCALER_PARAMS[0][i]) / INPUT_SCALER_PARAMS[1][i];
    }
}

void postprocess_output(float control[]) {
    int i;
    float out;
    for (i = 0; i < NUM_CONTROL_VARS; i++) {
        out = (control[i] - OUTPUT_SCALER_PARAMS[0][i]) / OUTPUT_SCALER_PARAMS[1][i];
        control[i] = out * OUTPUT_SCALER_PARAMS[2][i] + OUTPUT_SCALER_PARAMS[3][i];
    }
}

void layer_dense(float input[], float output[], int dense_layer_count) {
    int i, j;
    for (i = 0; i < LAYER_DIMS[dense_layer_count+1]; i++) {
        output[i] = BIASES[dense_layer_count][i];
        for (j = 0; j < LAYER_DIMS[dense_layer_count]; j++) {
            output[i] += WEIGHTS[dense_layer_count][i][j] * input[j];
        }
    }
}

void layer_relu(float activation[], int dense_layer_count) {
    int i;
    for (i = 0; i < LAYER_DIMS[dense_layer_count]; i++) {
        activation[i] = (activation[i] < 0) ? 0 : activation[i];
    }
}

void layer_tanh(float activation[], int dense_layer_count) {
    int i;
    for (i = 0; i < LAYER_DIMS[dense_layer_count]; i++) {
        activation[i] = tanh(activation[i]);
    }
}

void layer_softplus(float activation[], int dense_layer_count) {
    int i;
    for (i = 0; i < LAYER_DIMS[dense_layer_count]; i++) {
        if (activation[i] < 30.0) {
            activation[i]  = log(exp(activation[i]) + 1);
        }
    }
}

void nn_predict(float **arr_1, float **arr_2) {
    float *prev_layer_output = *arr_1;
    float *curr_layer_output = *arr_2;
    float *tmp;
    int i, dense_layer_count = 0;

    for (i = 0; i < NUM_ALL_LAYERS; i++) {
        Layer layer = LAYER_TYPES[i];
        switch (layer) {
            case DENSE:
                layer_dense(prev_layer_output, curr_layer_output, dense_layer_count);
                dense_layer_count++;
                tmp = prev_layer_output;
                prev_layer_output = curr_layer_output;
                curr_layer_output = tmp;
                break;
            case RELU:
                layer_relu(prev_layer_output, dense_layer_count);
                break;
            case TANH:
                layer_tanh(prev_layer_output, dense_layer_count);
                break;
            case SOFTPLUS:
                layer_softplus(prev_layer_output, dense_layer_count);
                break;
        }
    }

    /* nn output goes back into arr_1 */
    *arr_1 = prev_layer_output;
    *arr_2 = curr_layer_output;
}

void compute_control(float **ptr_arr_1, float **ptr_arr_2) {
    state_bias_correction(*ptr_arr_1);
    preprocess_input(*ptr_arr_1);
    nn_predict(ptr_arr_1, ptr_arr_2);
    postprocess_output(*ptr_arr_1);
}

void nested_control(float state[], float control[]) {
    float phi = -KPP*state[0] - KDP*state[1];
    control[0] = GRAV_ACC*MASS - KPZ*state[2] - KDZ*state[3];
    control[1] = KPT * (phi - state[4]);
}

void patched_control(float state[], float control[]) {
    float dist_sq = 0, scale_factor = 0;
    float control_nested[NUM_CONTROL_VARS];

    int i;
    for (i = 0; i < NUM_STATE_VARS; i++) {
        dist_sq = dist_sq + state[i]*state[i];
    }
    scale_factor = exp(-SCALING_COEFF/dist_sq);

    nested_control(state, control_nested);

    control[0] = control[0]*scale_factor + (1-scale_factor)*control_nested[0];
    control[1] = control[1]*scale_factor + (1-scale_factor)*control_nested[1];
}

void compute_control_patched(float state[], float **ptr_arr_1, float **ptr_arr_2) {
    state_bias_correction(state);
    memcpy(*ptr_arr_1, state, NUM_STATE_VARS * sizeof(float));
    preprocess_input(*ptr_arr_1);
    nn_predict(ptr_arr_1, ptr_arr_2);
    postprocess_output(*ptr_arr_1);
    patched_control(state, *ptr_arr_1);
}

void nn(float state[NUM_STATE_VARS], float control[NUM_CONTROL_VARS]) {
    /* allocate memory for two arrays required by compute_control() */
    /* number of array elements must be at least no. of units in widest layer */
    float arr_1_tmp[MAX_LAYER_DIMS];
    float arr_2_tmp[MAX_LAYER_DIMS];
    float *arr_1 = (float *) arr_1_tmp;
    float *arr_2 = (float *) arr_2_tmp;
    /* note the block of memory pointed to by the above may swap as a result */
    /* of calling compute_control() */

    /* on function call arr_1 contains state values */
    /* on completion, arr_1 contains control values */
    memcpy(arr_1, state, NUM_STATE_VARS * sizeof(float));
    compute_control(&arr_1, &arr_2);
    memcpy(control, arr_1, NUM_CONTROL_VARS * sizeof(float));
}

void nn_stable(float state[NUM_STATE_VARS], float control[NUM_CONTROL_VARS]) {
    float state_tmp[NUM_STATE_VARS];
    float arr_1_tmp[MAX_LAYER_DIMS];
    float arr_2_tmp[MAX_LAYER_DIMS];
    float *arr_1 = (float *) arr_1_tmp;
    float *arr_2 = (float *) arr_2_tmp;

    memcpy(state_tmp, state, NUM_STATE_VARS * sizeof(float));
    compute_control_patched(state_tmp, &arr_1, &arr_2);
    memcpy(control, arr_1, NUM_CONTROL_VARS * sizeof(float));
}
