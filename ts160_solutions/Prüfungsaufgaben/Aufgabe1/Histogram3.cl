#define PIXEL_PER_WORKITEM 256                   // LEAVE ENOUGH WHITESPACE FOR AUTO REPALCEMENT
#define GROUP_SIZE 32                            // LEAVE ENOUGH WHITESPACE FOR AUTO REPALCEMENT
#define DEBUG_PRINT 0                            // LEAVE ENOUGH WHITESPACE FOR AUTO REPALCEMENT
#define PRINT_CONDITION gid==0                   // LEAVE ENOUGH WHITESPACE FOR AUTO REPALCEMENT
#define GROUP_ATOMIC_ADD 1                       // LEAVE ENOUGH WHITESPACE FOR AUTO REPALCEMENT



kernel void calcStatistic_kernel(__global uchar           *rgba_global,
                                            uint 		     length,
                                   __global unsigned int	*local_histograms){

    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int wid = get_group_id(0);


    if(gid==0 & DEBUG_PRINT) {
        printf("\n[\t---KERNEL 'calcStatistic_kernel' INFO BEGIN---\t]\n");
        printf("PIXEL_PER_WORKITEM: %i\n", PIXEL_PER_WORKITEM);
        printf("GROUP_SIZE        : %i\n", GROUP_SIZE);
        printf("GROUP_ATOMIC_ADD  : %i\n", GROUP_ATOMIC_ADD);
    }


#if GROUP_ATOMIC_ADD
    for (int i = 0; i < PIXEL_PER_WORKITEM; i++) {
        // int _localID    = lid + i*GROUP_SIZE;
        int _globalID   = lid + i*GROUP_SIZE + wid * PIXEL_PER_WORKITEM*GROUP_SIZE;
        _globalID *= 3;

        // out of pixels
        if(_globalID+2>=length){
            break;
        }

        float r         = (float) ( rgba_global[_globalID] );
        float g         = (float) ( rgba_global[_globalID+1] );
        float b         = (float) ( rgba_global[_globalID+2] );
        float luminance = (0.2126*r + 0.7152*g + 0.0722*b);

        atomic_inc( &local_histograms[wid*256 + (int)(luminance)] );
        //        atomic_inc( &workitems_histogram[lid][(int)(luminance)] );
        //      workitems_histogram[lid][(int)(luminance)]++;
    }

#else
    // create histogram for each workitem AKA "counts[GROUP_SIZE=32][256]"
    __local uint volatile workitems_histogram[256][GROUP_SIZE];

    for(int i=0; i<256; i++){
        workitems_histogram[i][lid]=0;
    }

    // barrier(CLK_LOCAL_MEM_FENCE );

    // calc histogram per workitem, grab every 32th pixel
    for (int i = 0; i < PIXEL_PER_WORKITEM; i++) {
        // int _localID    = lid + i*GROUP_SIZE;
        int _globalID   = lid + i*GROUP_SIZE + wid * PIXEL_PER_WORKITEM*GROUP_SIZE;
        _globalID *= 3;

        // out of pixels
        if(_globalID+2>=length){
            break;
        }

        float r         = (float) ( rgba_global[_globalID] );
        float g         = (float) ( rgba_global[_globalID+1] );
        float b         = (float) ( rgba_global[_globalID+2] );
        float luminance = (0.2126*r + 0.7152*g + 0.0722*b);

//        atomic_inc( &workitems_histogram[(int)(luminance)][lid] );
//        atomic_inc( &workitems_histogram[lid][(int)(luminance)] );
//      workitems_histogram[lid][(int)(luminance)]++;
      workitems_histogram[(int)(luminance)][lid]++;
    }

    barrier(CLK_LOCAL_MEM_FENCE );

    // each of the 32 workers within this group shall summize 8 times the histogram bins
    // summize workgroup-internal histograms & copy to buffer
    int step_size = 256 / GROUP_SIZE;  // 256%GROUP_SIZE MUST BE ZERO!

    for (int k = 0; k < step_size; k++) {
        int sum=0;
        int binID = k*GROUP_SIZE + lid;

        // collect each result from other work items
        for (int i = 0; i < GROUP_SIZE; ++i) {
//            sum += workitems_histogram[i][binID];
            sum += workitems_histogram[binID][i];
        }

        local_histograms[wid*256 + lid+k*32] = sum;
    }

#endif




    if(gid==0 && DEBUG_PRINT && PRINT_CONDITION)
        printf("[\t---KERNEL 'calcStatistic_kernel' INFO END---\t]\n\n");

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

//        if(gid==0 & DEBUG_PRINT & PRINT_CONDITION){
//            printf("Bin %i\n", bin);
//        }

        uint sum = 0;
        // iterate over all local histograms
        for (int l = 0; l < groups; l++) {
//            if(gid==0 & DEBUG_PRINT & PRINT_CONDITION){
//                printf("hist %i, [%i]: %i\n", l, i*step_size +lid,  local_histograms[l*256 + i*step_size +lid] );
//            }
            sum += local_histograms[l*256 + bin];
        }

//        if(gid==0 & DEBUG_PRINT & PRINT_CONDITION) {
//          printf("Sum[%i] %i\n", i*step_size + lid, sum);
//        }
        histogram[bin] = sum;
    }
    if(gid==0 & DEBUG_PRINT & PRINT_CONDITION)
        printf("[\t---KERNEL 'reduceStatistic_kernel' INFO END---\t]\n\n");

    return;
}
