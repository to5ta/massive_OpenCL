__kernel void calcStatistic(__global const uchar4 *img, __global int *result, int PIXEL)
{
    const int lx = get_local_id(0);
    const int wx = get_group_id(0);

    __local int counts[32][256];    // shared by 32 workitems, for 256 brightness values


    for(int i = 0; i < 32; i++)
    {
        for(int k = 0; k < 256; k++)
        {
            counts[i][k] = 0;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    
    int WG_SIZE = 32 * PIXEL;
    float value;
    int idx;
    int wgIdx;
    int strideIdx;
    int itemIdx;

    wgIdx = (wx * WG_SIZE);
    for(int i = 0; i < (PIXEL / 32); i++)
    {
        strideIdx = (i * 1024);
        for(int k = 0; k < 32; k++)
        {
            itemIdx = (k * 32);
            idx = wgIdx + strideIdx + itemIdx + lx;
            value = (0.2126f * img[idx].x +
                     0.7152f * img[idx].y +
                     0.0722f * img[idx].z);

            counts[lx][(int)value] += 1;
            // printf("counts: %d\n", counts[lx][(int)value]);

        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    int value_sum;
    for(int i = 0; i < 8; i++) // takes 8 steps of 32 to get through 256 brightness values
    {
        value_sum = 0;
        for(int k = 0; k < 32; k++)
        {
            value_sum += counts[k][i * 32 + lx];
        }
        
        result[wx * 256 + i * 32 + lx ] = value_sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}


__kernel void reduceStatistic(__global int *result, int nWG)
{
    const int lx = get_local_id(0);

    int reducedValue = 0;
    for(int k = 0; k < nWG; k++)
    {
        reducedValue += result[k * 256 + lx];
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    result[lx] = reducedValue;
    // printf("reducedValue: %d\n", reducedValue);
}
