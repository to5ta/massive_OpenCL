/**********************************************************************
 Copyright ©2013 Advanced Micro Devices, Inc. All rights reserved.
 
 Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 
 ï    Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 ï    Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 ********************************************************************/


#define GX get_global_id(0)
#define LX get_local_id(0)

#define NUM_BANKS 32
#define DOUBLE_LOG_NUM_BANKS 10
#define CONFLICT_FREE_OFFSET(n) ((n)>>NUM_BANKS+(n)>>DOUBLE_LOG_NUM_BANKS)
#define CONFLICT_FREE_INDEX(n) ((n)+CONFLICT_FREE_OFFSET(n))
//#define CONFLICT_FREE_INDEX(n) (n)

__kernel void calcPrefix256(__global int* in, __global int* out)    // needs exactly 256 elements
{
    __local localArray[256];
    
    localArray[LX] = in[GX];
    
    barrier(CLK_LOCAL_MEM_FENCE);

# UPSWEEP
    uint numPasses = 8;
    uint offset = 1;
    uint numWorkers = 128;
    
    for(uint k = 0; k < numPasses; k++)
    {
        if(LX < numWorkers)
        {
            uint i = LX * (2 << k+1);
            localArray[i] += localArray[i + offset];
        }
        
        offset = offset << 1;
        numWorkers = numWorkers >> 1;
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
# DOWNSWEEP
    numPasses = 8; #log()
    offset = 256;
    numWorkers = 1;
    
    localArray[0] = 0;
    
    for(uint k = 1; k < numPasses; k++)
    {
        uint tmp;
        if(LX < numWorkers)
        {
            uint i = LX * (2 << k+1);
            tmp = localArray[i]
            
            localArray[i] += localArray[i + offset];
        }
        offset = offset >> 1;
        numWorkers = numWorkers << 1;
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
}

