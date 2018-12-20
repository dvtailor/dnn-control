#include <stdio.h>
#include "nn.h"
#include "nn_params.h"

int main() {
    float state[NUM_STATE_VARS] = {
        0,0,0,0,0
    };
    float control[NUM_CONTROL_VARS];
    int i;

    nn_stable(state, control);
    for (i = 0; i < NUM_CONTROL_VARS; i++) {
        printf("%.12f\n", control[i]);
    }

    return 0;
}
