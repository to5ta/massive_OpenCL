#define PIXEL_PER_WORKITEM 	128
#define GROUP_SIZE 			32
#define DEBUG_PRINT         1

__kernel void calcStatistic_kernel(__global	unsigned char 	*rgb_global,
                            			int 		 	length,
                            __global 	unsigned int	*local_histograms){

    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int wid = get_group_id(0);

    if(gid==0 && DEBUG_PRINT)
        printf("\n[\t---KERNEL INFO BEGIN---\t]\n");

    __local unsigned char rgb_local[PIXEL_PER_WORKITEM * GROUP_SIZE * 3];
    __local unsigned int  histogram_local[GROUP_SIZE*256];


    if(gid==0){
        printf("hist[0]: %i\n", histogram_local[0]);
        printf("hist[511]: %i\n", histogram_local[511]);
    }


    if(gid==0 && DEBUG_PRINT){
//        printf("gid: %i\n", gid);
//        printf("lid: %i\n", lid);
        printf("rgb_local size[%i]\n", PIXEL_PER_WORKITEM * GROUP_SIZE);
        printf("histogram_local[%i*%i]\n", GROUP_SIZE, 256);
        printf("rgb_global length: %i (excepted)\n", length);
    }



    // copy to local
    for (int i = 0; i < PIXEL_PER_WORKITEM; ++i)
    {
        int _localID 			= (lid+i*GROUP_SIZE)*3;
        int _globalID 			= (lid+i*GROUP_SIZE + wid*PIXEL_PER_WORKITEM*GROUP_SIZE)*3;

        // ensure we only copy from valid location to valid location
//        if(_localID < PIXEL_PER_WORKITEM*GROUP_SIZE*3){
        if(_globalID < length){


            // copy rgb values
            rgb_local[_localID] 	= rgb_global[_globalID];
            rgb_local[_localID+1] 	= rgb_global[_globalID+1];
            rgb_local[_localID+2] 	= rgb_global[_globalID+2];

            if(gid==0 && DEBUG_PRINT) {
//                printf("i: %i : %i %i %i\n", i*32, rgb_local[_localID], rgb_local[_localID+1], rgb_local[_localID+2] );
//                if (_globalID == 0) {
//                    printf("\nlid     : %i\n", lid);
//                    printf("i       : %i\n", i);
//                    printf("globalID: %i\n", _globalID);
//                    printf("i %i: gid: %3i;  localID: %4i  globalID: %4i\n",i, gid, _localID, _globalID);
//                }
            }
        }
    }


    if(gid==0){
        printf("hist[0]: %i\n", histogram_local[0]);
        printf("hist[511]: %i\n", histogram_local[511]);
    }



    barrier(CLK_LOCAL_MEM_FENCE);


//    int _i = 256*32*3-3;
////    if(gid==0 && DEBUG_PRINT){
//    if(gid==0 || gid==32 && DEBUG_PRINT){
//        printf("First Pixel:\nR: %i\n", rgb_local[0]);
//        printf("G: %i\n", rgb_local[1]);
//        printf("B: %i\n", rgb_local[2]);
//
//        printf("Last Pixel:\nR: %i\n", rgb_local[_i]);
//        printf("G: %i\n", rgb_local[_i+1]);
//        printf("B: %i\n", rgb_local[_i+2]);
//    }



    // calc histogram local
    for (int i = 0; i < PIXEL_PER_WORKITEM; ++i)
    {
        int _localID  = (lid+i*GROUP_SIZE)*3;
        int _globalID = (lid+i*GROUP_SIZE + wid*PIXEL_PER_WORKITEM*GROUP_SIZE)*3;

        if(_globalID<length){
            uchar r = rgb_local[_localID];
            uchar g = rgb_local[_localID+1];
            uchar b = rgb_local[_localID+2];
            float luminance = 0.2126*r + 0.7152*g + 0.0722*b;
            if(gid==0 && DEBUG_PRINT){
//                printf("[%4i] %3i %3i %3i = lum: %3.1f\n", _localID, r, g, b,luminance);
            }
            histogram_local[lid*256 +(uchar)(luminance)]++;
        }
    }


//    if(gid==0 && DEBUG_PRINT){
//        printf("local_hist[1]: %i\n", histogram_local[1]);
//    }



    barrier(CLK_LOCAL_MEM_FENCE);


    if(gid==0)
        printf("sum: hist[0]: %i\n", histogram_local[0]);

    // sum up all local histograms
    // do this 8 times * 32 workers = for 256 bins of the histogram
    for (int k = 0; k < 8; ++k) {
        int sum=0;
        // collect each result from other work items
        for (int i = 0; i < GROUP_SIZE; ++i) {
            sum += histogram_local[i*256+lid+k*32];
        }

        if(gid==0)
            printf("sum: hist[%i]: %i\n", k*32, sum);

        barrier(CLK_LOCAL_MEM_FENCE);

        local_histograms[wid*256 + lid+k*32] = sum;
//        local_histograms[wid*256 + lid+k*32] = 123;

    }

    if(gid==0 && DEBUG_PRINT)
        printf("[\t---KERNEL INFO END---\t]\n\n");

    return;

}


__kernel void reduceStatistic_kernel(__global  int     *local_histograms,
                                        int     groups,
                              __global  int     *histogram ){

    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int wid = get_group_id(0);

    //
    for (int i = 0; i < 8; ++i) {

        uint sum = 0;
        // iterate over all local histograms
        for (int l = 0; l < groups; ++l) {
            sum += local_histograms[i*lid + l*256];
        }

        histogram[i*lid] = sum;
    }
}

