/**********************************************************************
Copyright �2013 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

�	Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
�	Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/
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
thun
    if(lid==0){
        // make sure everything is done now
        barrier(CLK_LOCAL_MEM_FENCE);

        // finally write result back to global out buffer
        out[groupid] = localArray[0] + localArray[1];
    }
    return;

}