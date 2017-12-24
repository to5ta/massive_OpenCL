
#define GX get_global_id(0)
#define GY get_global_id(1)

#define GW get_global_size(0)
#define GH get_global_size(1)

#define LX get_local_id(0)
#define LY get_local_id(1)

kernel void imgfx(global uchar4 *in,
                  global float2 *pos, global float2 *vel, global uchar4 *out, global int *values) {
    float2 max = (float2) { GW-1, GH-1 };
    in += GY*GW+GX;
    pos += GY*GW+GX; vel += GY*GW+GX;
    *pos += *vel;
    if ((*pos).s1 >= GH) *vel = (float2) { (*vel).s0, -(*vel).s1 };
    *pos = clamp(*pos+*vel, (float2) { 0, 0 }, max);
    (*vel).s1 += 0.01f*values[0];

	uchar4 outval = *in;
	outval.s0 = 128;  // set blue to 50%; .s1=green; .s2=red [all 0..255]
    out[((int)(*pos).s1)*GW+((int)(*pos).s0)] = outval;
}

kernel void imgfx_clear(global int *out) {
    out[GY*GW+GX] = 0;
}
