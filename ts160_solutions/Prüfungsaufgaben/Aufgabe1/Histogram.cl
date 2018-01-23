#define PIXEL_PER_WORKITEM 	256
#define GROUP_SIZE 			32
#define DEBUG_PRINT         1

__kernel void calcStatistic(__global	unsigned char 	*rgb_global,
                            			int 		 	length,
                            __global 	unsigned int	*local_histograms){

    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int wid = get_group_id(0);

    if(gid==0 && DEBUG_PRINT)
        printf("\n[\t---KERNEL INFO BEGIN---\t]\n");

    local unsigned char rgb_local[PIXEL_PER_WORKITEM * GROUP_SIZE];
    local int           histogram_local[GROUP_SIZE][256];

    if(gid==0 && DEBUG_PRINT){
        printf("gid: %i\n", gid);
        printf("lid: %i\n", lid);
        printf("rgb_local size[%i]\n", PIXEL_PER_WORKITEM * GROUP_SIZE);
        printf("histogram_local[%i][%i]\n", GROUP_SIZE, 256);
    }


    // copy to local
    for (int i = 0; i < PIXEL_PER_WORKITEM; ++i)
    {
        int _localID 			= (lid+i*GROUP_SIZE)*3;
        int _globalID 			= (lid+i*GROUP_SIZE + wid*PIXEL_PER_WORKITEM*GROUP_SIZE)*3;

        // ensure we only copy from valid location to valid location
//        if(_localID < PIXEL_PER_WORKITEM*GROUP_SIZE*3){
        if(_globalID < length){
//            if(gid==0 && DEBUG_PRINT || _globalID==length-3){
//            if(_globalID==length-3){
//                printf("lid: %i\n", lid);
//                printf("i: %i\n", i);
//                printf("globalID: %i\n", _globalID);
//                printf("localID : %i\n", _localID);
//            }

            // copy rgb values
            rgb_local[_localID] 	= rgb_global[_globalID];
            rgb_local[_localID+1] 	= rgb_global[_globalID+1];
            rgb_local[_localID+2] 	= rgb_global[_globalID+2];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);


    int _i = length-3;
    if(gid==0 && DEBUG_PRINT){
        printf("First Pixel:\nR: %i\n", rgb_local[0]);
        printf("G: %i\n", rgb_local[1]);
        printf("B: %i\n", rgb_local[2]);

        printf("Last Pixel:\nR: %i\n", rgb_local[_i]);
        printf("G: %i\n", rgb_local[_i+1]);
        printf("B: %i\n", rgb_local[_i+2]);
    }



    /*

    // calc histogram local
    for (int i = 0; i < PIXEL_PER_WORKITEM; ++i)
    {
        int _localID = (lid+i*PIXEL_PER_WORKITEM)*3;
        if(_localID<length){

            if(gid==0 && DEBUG_PRINT){
                printf("localID: %i\n", _localID);
            }

            uchar r = rgb_local[_localID];
//            uchar g = rgb_local[_localID+1];
//            uchar b = rgb_local[_localID+2];
            if(gid==0 && DEBUG_PRINT){
                printf("R: %i\n", r);
            }
//            float luminance = 0.2126*r + 0.7152*g + 0.0722*b;
//            histogram_local[lid][(uchar)luminance]++;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

     */



    barrier(CLK_LOCAL_MEM_FENCE);

    // sum up all local histograms
    // do this 8 times * 32 workers = for 256 bins of the histogram
    for (int k = 0; k < 8; ++k) {
        int sum=0;
        // collect each result from other work items
        for (int i = 0; i < GROUP_SIZE; ++i) {
            sum += histogram_local[i][lid+k*32];
        }
        local_histograms[wid*256 + lid+k*32] = sum;
    }

    if(gid==0 && DEBUG_PRINT)
        printf("[\t---KERNEL INFO END---\t]\n\n");

}


__kernel void reduceStatistic(){

}

