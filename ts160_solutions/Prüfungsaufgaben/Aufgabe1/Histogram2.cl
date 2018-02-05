//#define PIXEL_PER_WORKITEM 	256
///=== build failed ===
///ptxas error   : Entry function 'calcStatistic_kernel' uses too much shared data (0xe004 bytes, 0xc000 max)

#define PIXEL_PER_WORKITEM 	64
#define GROUP_SIZE 			32

#define DEBUG_PRINT          1
#define PRINT_CONDITION wid>19 & lid==0



__kernel void calcStatistic_kernel(__global	unsigned char 	*rgb_global,
                                			int 		 	length,
                                __global 	unsigned int	*local_histograms){

    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int wid = get_group_id(0);


    if(gid==0 & DEBUG_PRINT & PRINT_CONDITION) {
        printf("\n[\t---KERNEL 'calcStatistic_kernel' INFO BEGIN---\t]\n");
    }

    if(DEBUG_PRINT & PRINT_CONDITION){
        printf("WID: %i, LID: %i\n", wid, lid);
    }

    // copy rgb_data to local memory to allow fast access
    __local unsigned char rgb_local[PIXEL_PER_WORKITEM * GROUP_SIZE * 3];

    for (int i = 0; i < PIXEL_PER_WORKITEM; i++)
    {
        // aka start-copy-index
        int _localID            = (lid+i*GROUP_SIZE) * 3;
        int _globalID           = (lid+i*GROUP_SIZE + wid*PIXEL_PER_WORKITEM*GROUP_SIZE) * 3;

        if(_globalID<length){
            rgb_local[_localID]     = rgb_global[_globalID];
            rgb_local[_localID+1]   = rgb_global[_globalID+1];
            rgb_local[_localID+2]   = rgb_global[_globalID+2];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);


//    int _i = min( PIXEL_PER_WORKITEM*32*3-3, length-3);
//    if(lid==0 & DEBUG_PRINT & PRINT_CONDITION){
//      printf("%2i/%2i/%i [%4i]:\nR: %i\nG: %i\nB: %i\n", gid, lid, wid, 0, rgb_local[0], rgb_local[1], rgb_local[2]);
//      printf("%2i/%2i/%i [%4i]:\nR: %i\nG: %i\nB: %i\n", gid, lid, wid, _i,rgb_local[_i], rgb_local[_i+1], rgb_local[_i+2]);
//    }


    // create histogram for each workitem
    __local unsigned int  workitems_histogram[GROUP_SIZE*256];


    // calc histogram per workitems (mutliples in local memory)
    for (int i = 0; i < PIXEL_PER_WORKITEM; i++)
    {
        int _localID            = (lid+i*GROUP_SIZE) * 3;
        int _globalID           = (lid+i*GROUP_SIZE + wid*PIXEL_PER_WORKITEM*GROUP_SIZE) * 3;

        if(_globalID<length){
//            uchar r = rgb_local[_localID];
//            uchar g = rgb_local[_localID+1];
//            uchar b = rgb_local[_localID+2];

            float r = (float)(rgb_local[_localID]);
            float g = (float)(rgb_local[_localID+1]);
            float b = (float)(rgb_local[_localID+2]);
            // dirty clamp
            float luminance = (0.2126*r + 0.7152*g + 0.0722*b);
//            int luminance = min(255, (int)(0.2126*r + 0.7152*g + 0.0722*b));

            workitems_histogram[lid*256 + (int)(luminance)]++;

            if(gid==0 & DEBUG_PRINT & PRINT_CONDITION){
//                printf("RGB %i %i %i =  Lum %i\n", r,g,b,luminance );
                printf("workitems_histogram[%i + %i]: %i\n", (int)(luminance), lid*256, workitems_histogram[lid*256 + (int)(luminance)]);
            }

            // workitems_histogram[lid*256 + min(255, (uchar)(luminance))]++;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);


    // summize workgroup-internal histograms & copy to buffer
    int step_size = 256 / GROUP_SIZE;  // 256%GROUP_SIZE MUST BE ZERO!

    if(gid==0 & DEBUG_PRINT & PRINT_CONDITION){
        printf("stepsize: %i\n", step_size);
    }
    for (int k = 0; k < step_size; k++) {
        int sum=0;
        int binID = k*GROUP_SIZE + lid;

        if(gid==31 & DEBUG_PRINT & PRINT_CONDITION){
            printf("Bin: %i\n", binID);
        }
        // collect each result from other work items
        for (int i = 0; i < GROUP_SIZE; ++i) {
            //                         nextHistogram + current step_offset + local offset
            sum += workitems_histogram[i*256         + binID];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        local_histograms[wid*256 + lid+k*32] = sum;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(gid==0 & DEBUG_PRINT & PRINT_CONDITION)
        printf("[\t---KERNEL 'calcStatistic_kernel' INFO END---\t]\n\n");

    return;
}


__kernel void reduceStatistic_kernel(__global  int     *local_histograms,
                                               int     groups,
                                     __global  int     *histogram ){

    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int wid = get_group_id(0);

    if(gid==0 & DEBUG_PRINT & PRINT_CONDITION)
        printf("\n[\t---KERNEL 'reduceStatistic_kernel' INFO BEGIN---\t]\n");

    if(gid==0 & DEBUG_PRINT & PRINT_CONDITION){
        printf("groups:  %i\n", groups);
    }

    int step_size = 256/GROUP_SIZE;
    if(gid==0 & DEBUG_PRINT & PRINT_CONDITION){
        printf("step_size %i\n", step_size );
    }
    for (int i = 0; i < step_size; ++i) {

        int bin = i*GROUP_SIZE + lid;

        if(gid==0 & DEBUG_PRINT & PRINT_CONDITION){
            printf("Bin %i\n", bin);
        }

        uint sum = 0;
        // iterate over all local histograms
        for (int l = 0; l < groups; l++) {
            if(gid==0 & DEBUG_PRINT & PRINT_CONDITION){
                printf("hist %i, [%i]: %i\n", l, i*step_size +lid,  local_histograms[l*256 + i*step_size +lid] );
            }
            sum += local_histograms[l*256 + bin];
        }

        if(gid==0 & DEBUG_PRINT & PRINT_CONDITION) {
          printf("Sum[%i] %i\n", i*step_size + lid, sum);
        }
        histogram[bin] = sum;
    }
    if(gid==0 & DEBUG_PRINT & PRINT_CONDITION)
        printf("[\t---KERNEL 'reduceStatistic_kernel' INFO END---\t]\n\n");

    return;
}
