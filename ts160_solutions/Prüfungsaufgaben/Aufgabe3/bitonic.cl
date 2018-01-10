

int powi(int x, int e){
    return (int)(pow((float)(x),e));
}

__kernel void bitonic_kernel(int length, __global int* in, __global int* out)
{

	int gid     = get_global_id(0);
	int lid     = get_local_id(0);
	int groupid = get_group_id(0);

    if(gid>=length)
        return;

//    printf("gid: %i\n", gid);

    out[lid] = in[lid];


    int gid_observe = 5;

    //For definition of stages and steps see:
    // https://www.geeksforgeeks.org/bitonic-sort/

    float l = (float)(length);
    int total_steps = (log(l)*(log(l)+1))/2.;

    // printf("Total Steps: %i\n", total_steps);

    for(int step=0; step<=total_steps; step++){

        for(int stage=step; stage>=0; stage--){

            if(gid==gid_observe){
//                printf("Offset: %i   ", offset);
            }

            // gid  +   offset * gruppe
            int id0 = gid + powi(2,stage)*(gid/(stage+1));

            // id0 + offset
            int id1 = id0 + powi(2,stage);


            // direction?
            


            // debug only
            if(gid==gid_observe){
                printf("Step: %i Stage: %i:   GID: %2i   ID_0: %2i    ID_1: %2i \n",step, stage, gid_observe, id0, id1);
            }


        }
        if(gid==gid_observe){
            printf("\n");
        }
    }






}