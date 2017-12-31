
__kernel void calcStatistic(__global uchar3 *img, __global int *result)
{
    const int lx = get_local_id(0);
    const int gx = get_global_id(0);

    __local int counts[32][256];

    float value;
    if(gx == 0)
    {
        for(int idx = 0; idx < 256; idx++)
        {
            value = (0.2126f * img[gx * 256 + lx].x +
                     0.7152f * img[gx * 256 + lx].y +
                     0.0722f * img[gx * 256 + lx].z);

            counts[lx][(int)value] += 1;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);



}
