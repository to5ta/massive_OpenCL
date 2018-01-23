#define PIXEL_PER_WORKITEM 	256
#define GROUP_SIZE 			32

__kernel void calcStatistic(__global	unsigned char 	*rgb_global,
                            			int 		 	length,
                            __global 	unsigned int	*local_histograms){

    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int wid = get_group_id(0);

    local unsigned char rgb_local[PIXEL_PER_WORKITEM * GROUP_SIZE];
    local int           histogram_local[GROUP_SIZE][256];

    // copy to local
    for (int i = 0; i < PIXEL_PER_WORKITEM; ++i)
    {
    	int _localID 			= (lid+i*PIXEL_PER_WORKITEM)*3;
    	int _globalID 			= (lid+i*PIXEL_PER_WORKITEM + wid*PIXEL_PER_WORKITEM*GROUP_SIZE)*3;
        // copy rgb values
        rgb_local[_localID] 	= rgb_global[_globalID];
        rgb_local[_localID+1] 	= rgb_global[_globalID+1];
        rgb_local[_localID+1] 	= rgb_global[_globalID+2];

    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // calc histogram local
    for (int i = 0; i < PIXEL_PER_WORKITEM; ++i)
    {
        // rgb_local[]
        // float luminance = 0.2126*pix.R + 0.7152*pix.G + 0.0722*pix.B;
//         histogram_local[lid]
    }


    for (int k = 0; k < 8; ++k) {
        int sum=0;
        // collect each result from other work items
        for (int i = 0; i < GROUP_SIZE; ++i) {
            sum += histogram_local[i][lid+k*32];
        }
        local_histograms[wid*256 + lid+k*32] = sum;
    }



//    // copy to local_histograms
//    for (int j = 0; j < 8; ++j) {
//        int _localID = lid*8+j;
////        local_histograms[wid*PIXEL_PER_WORKITEM*GROUP_SIZE + _localID] = histogram_local[_localID];
//        local_histograms[wid*256*GROUP_SIZE + _localID] = histogram_local[lid][_localID];
//    }
//    barrier(CLK_LOCAL_MEM_FENCE);

}


__kernel void reduceStatistic(){

}

