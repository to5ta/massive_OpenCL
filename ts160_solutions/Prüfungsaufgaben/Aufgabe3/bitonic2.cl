
#define DEBUG_INFO 1
#define GID_OBSERVE 0


void
printData(__global uint* data,
                   uint length,
                   uint margin);


int powi(int x, int e){
    return (int)(pow((float)(x),e));
}

__kernel void bitonic_kernel(   __global   uint*    out,
                                           uint     step,
                                           uint     stage )
{

	int gid     = get_global_id(0);
	int lid     = get_local_id(0);
	int groupid = get_group_id(0);

    int gws     = get_global_size(0);

    if(gid==0 && DEBUG_INFO){
//        printf("  >>> KERNEL PRINT BUFFER >>>\n");
//        printData(out, gws*2, 10);
        printf("KERNEL Step: %i, Stage: %i\n", step, stage);
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    //For definition of stages and steps see:
    // https://www.geeksforgeeks.org/bitonic-sort/


    /// id0 = gid + offset_per_grp *       grp_nr
    int id0 = gid + powi(2, stage) * (gid / (powi(2, stage)));

    /// id1 = id0 +    offset
    int id1 = id0 + powi(2, stage);

    /// direction?
    int dir = 0;
    if ((gid / powi(2, step)) % 2) {
        dir = 1;
        int _id0 = id0;
        id0 = id1;
        id1 = _id0;
    }

    /// swap
    if (out[id0] > out[id1]) {
        int _out0 = out[id0];
        out[id0] = out[id1];
        out[id1] = _out0;
    }

    if(gid==0 && DEBUG_INFO){
//        printf("  <<< KERNEL PRINT BUFFER <<<\n");;
    }

//    if(gid==0 && step==18 && stage==0){
//        printf("KERNEL BUFFER\n");
//        printData(out, gws*2, 10);
//    }

    barrier(CLK_GLOBAL_MEM_FENCE);
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
