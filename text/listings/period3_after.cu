__device__ __forceinline__ bool in_period3_bulb_f32(float cr, float ci) {
    float dr     = cr - (float)PERIOD3_CENTER_RE;
    float di_pos = ci - (float)PERIOD3_CENTER_IM;
    float di_neg = ci + (float)PERIOD3_CENTER_IM;
    return (dr * dr + di_pos * di_pos < (float)PERIOD3_RADIUS_SQ)
        || (dr * dr + di_neg * di_neg < (float)PERIOD3_RADIUS_SQ);
}
