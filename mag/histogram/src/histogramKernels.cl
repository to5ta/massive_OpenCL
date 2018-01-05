__kernel void calcStatistic(__global const uchar4 *img, __global int *result)
{
    const int lx = get_local_id(0);
    const int wx = get_group_id(0);

    __local int counts[32][256];    // shared by 32 workitems, for 256 pixels each

    for(int i = 0; i < 32; i++)
    {
        for(int k = 0; k < 256; k++)
        {
            counts[i][k] = 0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int WORK_SIZE = 8192;
    float value;
    int idx;

    for(int i = 0; i < 8; i++)
    {
        for(int k = 0; k < 32; k++)
        {
            idx = (wx * WORK_SIZE) + (i * 1024) + (k * 32) + lx;

            value = (0.2126f * img[idx].x +
                     0.7152f * img[idx].y +
                     0.0722f * img[idx].z);

            counts[lx][(int)value] += 1;
        }

    }

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    int value_sum;
    for(int i = 0; i < 8; i++)
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
    int numWorkGroups = nWG;
    int reducedValue = 0;
    for(int k=0; k < numWorkGroups; k++)
    {
        reducedValue += result[k * 256 + lx];
    }
    result[lx] = reducedValue;

}
