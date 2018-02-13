
#define DEBUG_INFO 0
#define GID_OBSERVE 0


void
printData(__global uint* data,
                   uint length,
                   uint margin);


int powi(int x, int e){
    return (int)(pow((float)(x),e));
}

__kernel void bitonic_kernel(           int     length, // must be exponent with base 2
                             __global   int*    out)
{

	int gid     = get_global_id(0);
	int lid     = get_local_id(0);
	int groupid = get_group_id(0);

    int gws     = get_global_size(0);

    if(DEBUG_INFO && gid==GID_OBSERVE){
        printf("GID: %i\n", gid);
    }

    barrier(CLK_GLOBAL_MEM_FENCE);


    //For definition of stages and steps see:
    // https://www.geeksforgeeks.org/bitonic-sort/

    float l = (float)(length);
    int total_steps = (uint)( log2(l));

    if(gid==0 && DEBUG_INFO){
        printf("Total Steps: %i\n", total_steps);
        printf("Length:      %i\n", length);
    }

//    if(gid==GID_OBSERVE && DEBUG_INFO && 0){
//        printf("Total Steps: %i\n", total_steps);
//        printf("Initial Values: \n");
//        printf("-");
//        for(int i=0; i<length; i++){
//            printf("%2i", i);
//            if(i<length-1) {
//                printf(", ");
//            }
//        }
//        printf("-\n[");
//        for(int j=0; j<length; j++){
//            printf("%2i", out[j]);
//            if(j<length-1){
//                printf(", ");
//            }
//        }
//        printf("]\n");
//    }

    for(int step=0; step<total_steps; step++){

        barrier(CLK_GLOBAL_MEM_FENCE);

        for(int stage=step; stage>=0; stage--) {

            barrier(CLK_GLOBAL_MEM_FENCE);

            int grp_offset = powi(2, stage);
            int grp_nr     = (gid / (powi(2, stage)));

            int id0 = gid + grp_offset * grp_nr;

            int id1 = id0 + grp_offset;

            int _id0;

            /// direction?
            int dir = 0;
            if ((gid / powi(2, step)) % 2) {
                dir = 1;
                _id0 = id0;
                id0 = id1;
                id1 = _id0;
            }

            barrier(CLK_GLOBAL_MEM_FENCE);

//            // swap
            if (out[id0] > out[id1]) {
                int _out0 = out[id0];
                out[id0] = out[id1];
                out[id1] = _out0;
            }

            barrier(CLK_GLOBAL_MEM_FENCE);

            // debug only
                if (DEBUG_INFO && gid == GID_OBSERVE) {
                    printf("Step: %i Stage: %i:   GID: %2i   ID_0: %2i -> %2i   Dir: %i   "\
                           " id0: %3i id1: %3i\n", step, stage, gid, id0, id1, dir, out[id0], out[id1]);

//                    printf("GID: %i\n", gid);
//                    printf("DIV: %i\n", powi(2,step));
//                    printf("DIR: %i\n", gid/powi(2,step)%2);

                }
//
//                if (gid == GID_OBSERVE && DEBUG_INFO) {
//
//                    printf("-");
//                    for (int i = 0; i < length; i++) {
//                        printf("%2i", i);
//                        if (i < length - 1) {
//                            printf(", ");
//                        }
//                    }
//                    printf("-\n");
//
//
//                    printf("[");
//                    for (int j = 0; j < length; j++) {
//                        printf("%2i", out[j]);
//                        if (j < length - 1) {
//                            printf(", ");
//                        }
//                    }
//                    printf("]\n\n");
//                }

            barrier(CLK_GLOBAL_MEM_FENCE);
        }

//        if(gid==GID_OBSERVE && DEBUG_INFO){
//            printf("\n");
//        }

        barrier(CLK_GLOBAL_MEM_FENCE);
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

//    if(gid==GID_OBSERVE){
//        printData(out, length*2, length);
//    }

    
}





void
printData(__global uint* data,
          uint length,
          uint margin) {
    printf("-");
    char skip = 0;
    for(int i=0; i<length; i++){
      printf("%3i", i);

        if(i>margin && !skip && margin*2<length){
        printf(",  ...  ");
        i=length-margin-1;
        skip=1;
        continue;
        }

        if(i<length-1) {
        printf(", ");
        }
    }
    printf("-\n");

    skip=0;

    printf("[");
    for(int i=0; i<length; i++) {
        printf("%3i", data[i]);

        if(i>margin && !skip && margin*2<length){
            printf(",  ...  ");
            i=length-margin-1;
            skip=1;
         continue;
        }

        if(i<length-1) {
            printf(", ");
        }
    }
    printf("]\n");
}
