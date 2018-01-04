__kernel void test(__global char4 *img, __global char *result)
{
    const int lx = get_local_id(0);
    const int gx = get_global_id(0);
    const int wx = get_group_id(0);

    int idx = gx;
    if(gx <= 64)
    {
        printf("%d  ", idx);
    }
    result[idx] += img[idx].x + img[idx].y + img[idx].z;
}
