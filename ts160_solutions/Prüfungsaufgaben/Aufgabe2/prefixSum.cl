

__kernel void addBlockSumToPrefix_kernel(__global int *blockPrefix, // 256 x longer than blockSums
                                         __global int *blockSums,
                                         int lengthBlockPrefix,
                                         int lengthBlockSums,
                                         int blocksize) {
    int gid = get_global_id(0); // assume 0..255
    int lid = get_local_id(0);  // assume 0..255

    // __local int local_data[blocksize];

    if (gid == 0) {
        printf("lengthBlockSums:  %6i\n", lengthBlockSums);
        printf("blocksize:        %6i\n", blocksize);
        printf("lengthBlockPrefix:%6i\n", lengthBlockPrefix);
        printf("lengthBlockSums  :%6i\n", lengthBlockSums);
        printf("blocksize        :%6i\n", blocksize);
    }


    for (int sum_id = 1; sum_id < lengthBlockSums; sum_id++) {
        int prefixID = gid + sum_id * blocksize;
        if (prefixID < lengthBlockPrefix) {
            blockPrefix[prefixID] += blockSums[sum_id];
        }
    }

    // int i = 1;
    // // // add corresponding blocksum to blockwisePrefix
    // for (int block_offset = blocksize; block_offset < lengthBlockPrefix; block_offset += blocksize) {
    //     int prefixID = gid+block_offset;
    //         blockPrefix[prefixID] += blockSums[i-1];
    //     }
    //     i+=1;
    //     if(gid==0)
    //     {
    //         printf("block_offset: %i\n", block_offset );
    //     }
    // }


    //     // local_data[lid] = blockPrefix[gid+block_offset];

    //     for (int field = 0; field < blocksize; field++) {
    //         blockPrefix[gid * blocksize + field + block_offset * blocksize] += blockSums[i + block_offset];
    //     }
    //     i += 1;
    // }
}


__kernel void calcBlockSum_kernel(__global int *inputData,
                                  __global int *blockPrefix,
                                  __global int *blockSums,
                                  int lengthInputData,
                                  int lengthBlockSums,
                                  int blocksize) {
    int gid = get_global_id(0); // assume 0..255
    int lid = get_local_id(0);  // assume 0..255

    // if (gid == 0) {
    //     printf("lengthInputData %i\n", lengthInputData);
    // }


    // sum each last block element of inputData and blockPrefix, store in blockSums
    for (int block_offset = 0; block_offset < lengthBlockSums; block_offset += blocksize) {

        //              last block element + offset if a work item has to do calc more than one sum, implies we have more than 256**2 = 65536 in inputData
        int blockIndex = ((gid + 1) * blocksize - 1) + block_offset * blocksize;

        // half-full or empty block
        // if((gid+block_offset)*blocksize > lengthInputData) {
        if (blockIndex > lengthInputData) {

            if (blockIndex - lengthInputData < blocksize) {
                // printf("GID %i, blockIndex %i\n", gid, blockIndex );
                // printf("inputData[lengthInputData-1]: %i\n", inputData[lengthInputData-1]);
                // printf("blockPrefix[lengthInputData-1]: %i\n", blockPrefix[lengthInputData-1] );
                blockSums[gid + block_offset] = inputData[lengthInputData - 1] + blockPrefix[lengthInputData - 1];
            } else {
                blockSums[gid + block_offset] = 0;
            }
        } else {
            blockSums[gid + block_offset] = inputData[blockIndex] + blockPrefix[blockIndex];
        }
    }
}


__kernel void prefixBlockwise_kernel(__global int *inputData,
                                     __global int *blockPrefix,
                                     int lengthInputData,
                                     int blocksize) {

    int gid = get_global_id(0);
    int lid = get_local_id(0);

    // printf("GID: %i LID: %i\n", gid, lid );

//    if(gid==0){
//        printf("blocksize:       %6i\n", blocksize );
//        printf("lengthInputData: %6i\n", lengthInputData );
//
//    }

    // input is multiple of 256, execute each 256-block
    for (int block_offset = 0; block_offset < lengthInputData; block_offset += blocksize) {

        // __local int local_data[blocksize];
        __local int local_data[256];

        int k = 8;  // depth of tree: log2(256)
        int d, i;

//        if(gid==0) {
//            printf("gid0: block_offset: %i\n", block_offset);
//        }

        // copy to local memory
        local_data[lid] = inputData[gid + block_offset];
        barrier(CLK_LOCAL_MEM_FENCE);
//
//        if(gid==0){
//            printf("Block_Offset: %i\n", block_offset);
//            for(int j=0; j<blocksize; j+=32){
//                printf("GID %i: %i \n",j+block_offset, blockPrefix[j]);
//            }
//        }

        // up sweep
        int no_items_that_work = blocksize / 2;
        int offset = 1;
        for (d = 0; d < k; d++, no_items_that_work >>= 1, offset <<= 1) {
            if (lid < no_items_that_work) {
                i = lid * (offset << 1) + offset - 1;
                local_data[i + offset] = local_data[i] + local_data[i + offset];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }


        // down sweep
        if (lid == blocksize - 1)
            local_data[blocksize - 1] = 0;
        no_items_that_work = 1;
        offset = blocksize / 2;
        for (d = 0; d < k; d++, no_items_that_work <<= 1, offset >>= 1) {
            if (lid < no_items_that_work) {
                i = lid * (offset << 1) + offset - 1;
                int tmp = local_data[i];
                local_data[i] = local_data[i + offset];
                local_data[i + offset] = tmp + local_data[i + offset];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        // write result to global memory
        blockPrefix[gid + block_offset] = local_data[lid];


        barrier(CLK_LOCAL_MEM_FENCE);

//        if(gid==0){
//            printf("Block_Offset: %i\n", block_offset);
//            for(int j=0; j<blocksize; j+=32){
//                printf("GID %i: %i \n",j+block_offset, blockPrefix[j]);
//            }
//        }
    }
}




