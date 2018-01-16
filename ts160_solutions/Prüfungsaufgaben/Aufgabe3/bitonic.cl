
#define DEBUG_INFO 0

int powi(int x, int e){
    return (int)(pow((float)(x),e));
}

__kernel void bitonic_kernel(int length, __global int* in, __global int* out)
{

	int gid     = get_global_id(0);
	int lid     = get_local_id(0);
	int groupid = get_group_id(0);

    if(gid>=length/2)
        return;

    if(DEBUG_INFO){
        printf("GID: %i\n", gid);
    }

    out[gid*2] = in[gid*2];
    out[gid*2+1] = in[gid*2+1];

    barrier(CLK_LOCAL_MEM_FENCE);


    int gid_observe = 6;

    //For definition of stages and steps see:
    // https://www.geeksforgeeks.org/bitonic-sort/

    float l = (float)(length/2);
    int total_steps = (log(l)*(log(l)+1))/2;

    if(gid==gid_observe && DEBUG_INFO){
        printf("Total Steps: %i\n", total_steps);
        printf("Initial Values: \n");
        printf("-");
        for(int i=0; i<length; i++){
            printf("%2i", i);
            if(i<length-1) {
                printf(", ");
            }
        }
        printf("-\n[");
        for(int j=0; j<length; j++){
            printf("%2i", out[j]);
            if(j<length-1){
                printf(", ");
            }
        }
        printf("]\n");
    }

    for(int step=0; step<=total_steps; step++){

        for(int stage=step; stage>=0; stage--){

            // id0 = gid + offset*gruppe
            int id0 = gid + powi(2,stage)*(gid/(powi(2, stage)));

            // id1 = id0 + offset
            int id1 = id0 + powi(2,stage);

            // direction?
            int dir = 0;
            if((gid/powi(2,step))%2){
                dir = 1;
                int _id0 = id0;
                id0 = id1;
                id1 = _id0;
            }

            barrier(CLK_LOCAL_MEM_FENCE);

//            // swap
            if(out[id0]>out[id1]){
                int _out0 = out[id0];
                out[id0] = out[id1];
                out[id1] = _out0;
            } else {
//                out[id0] = in[id0];
//                out[id0] = in[id1];
            }

            barrier(CLK_LOCAL_MEM_FENCE);


            // debug only
            if(DEBUG_INFO){
                printf("Step: %i Stage: %i:   GID: %2i   ID_0: %2i -> %2i   Dir: %i   "\
                       " id0: %3i id1: %3i\n",step, stage, gid, id0, id1, dir, out[id0], out[id1]);

//                printf("GID: %i\n", gid);
//                printf("DIV: %i\n", powi(2,step));
//                printf("DIR: %i\n", gid/powi(2,step)%2);


            }
            if(gid==gid_observe && DEBUG_INFO){

                printf("-");
                for(int i=0; i<length; i++){
                    printf("%2i", i);
                    if(i<length-1) {
                        printf(", ");
                    }
                }
                printf("-\n");


                printf("[");
                for(int j=0; j<length; j++){
                    printf("%2i", out[j]);
                    if(j<length-1){
                        printf(", ");
                    }
                }
                printf("]\n\n");
            }

        }
        if(gid==gid_observe && DEBUG_INFO){
            printf("\n");
        }
    }






}