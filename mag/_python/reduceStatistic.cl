__kernel void reduceStatistic(__global int *result, __global char *nWG)
{
    const int lx = get_local_id(0);
//    const int gx = get_global_id(0);
//    const int wx = get_group_id(0);


    int numWorkGroups = nWG[0];
    printf("nWG: %d", numWorkGroups);


    int reducedValue = 0;
    for(int k=0; k < numWorkGroups; k++)
    {
        reducedValue += result[k * 256 + lx];
    }
    result[lx] = reducedValue;

}
