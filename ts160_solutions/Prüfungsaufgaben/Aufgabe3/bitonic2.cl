
#define DEBUG_INFO 0
#define GID_OBSERVE 0

#define MAX_ID_DATA 65536

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

    int grp_offset = powi(2, stage);
    int grp_nr     = (gid / (powi(2, stage)));

    int id0 = gid + grp_offset * grp_nr;

    int id1 = id0 + grp_offset;

//    if(id0 > MAX_ID_DATA || id1 > MAX_ID_DATA){
//        printf("GID: %3i    id0: %3i, id1: %3i    grp_offset: %3i, grp_nr: %3i\n", gid, id0, id1, grp_offset, grp_nr);
//    }

    int _id0;

    /// direction?
    int dir = 0;
    if ((gid/powi(2, step))%2==0) {
        dir = 1;
        _id0 = id0;
        id0 = id1;
        id1 = _id0;
    }

    // that was for error hunting
    if(id0 > gws*2 || id1 > gws*2 ){

//        printf("(gid:%i / powi(2, step) = %i % 2= %i\n",gid, (gid / powi(2, step)), (gid / powi(2, step))%2 );

        // kein tausch
        if(dir==0){
            printf("GID: %3i    id0: %3i, id1: %3i    grp_offset: %8i, grp_nr: %3i  ->, step: %i, stage: %i\n",
                    gid,
                    id0,
                    id1,
                    grp_offset,
                    grp_nr,
                    step,
                    stage);
        }
        // tausch
        else {
            printf("GID: %3i    id0: %3i, id1: %3i    grp_offset: %8i, grp_nr: %3i  ->, step: %i, stage: %i\n",
                    gid,
                    id0,
                    id1,
                    grp_offset,
                    grp_nr,
                    step,
                    stage);
        }
    }

/// swap
    if (out[id0] < out[id1]) {
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
