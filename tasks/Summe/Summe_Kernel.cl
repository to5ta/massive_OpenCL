__kernel void summe_kernel(__global int* in, __global int* out)
{

	int gid     = get_global_id(0);
	int lid     = get_local_id(0);
	int groupid = get_group_id(0);

    __local int localArray[256];

    // per work-item=thread='this function': sum two input numbers together, store in local memory
    // consider this as the first 'loop'
    localArray[lid] = in[gid*2] + in[gid*2+1];
    barrier(CLK_LOCAL_MEM_FENCE);

    int sum;
    int lid_max = 128;

    while(lid_max>1) {
    	if(lid<lid_max) {
            // pre-calc sum
            sum = localArray[lid*2] + localArray[lid*2+1];
            // sync threads=work items
            barrier(CLK_LOCAL_MEM_FENCE);
            // overwrite with sum
            localArray[lid] = sum;
        }
        lid_max = lid_max/2;
    }
    if(lid==0){
        // make sure everything is done now
        barrier(CLK_LOCAL_MEM_FENCE);

        // finally write result back to global out buffer
        out[groupid] = localArray[0] + localArray[1];
    }
    return;

}