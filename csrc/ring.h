#pragma once

#include <cstdint>

struct Ring_fwd_params
{
    int mype;
    int npes;
    int *seqlen;
    int *h_seqlen;
    int dim;
    float softmax_scale;
    void *q_ptr;
    void *k_ptr;
    void *v_ptr;
    void *out_ptr;
    void *k_buffer;
    void *v_buffer;
};