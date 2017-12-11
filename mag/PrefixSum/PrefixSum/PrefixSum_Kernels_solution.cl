/**********************************************************************
Copyright ©2013 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

•	Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
•	Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
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

__kernel void calcPrefix256(__global int* in, __global int* out)	// needs exactly 256 elements
{
	__local int localArray[256];
	
	int k=8;	// depth of tree: log2(256)
	int d, i, i1, i2;

	// copy to local memory
	localArray[CONFLICT_FREE_INDEX(LX)] = in[GX];
	barrier(CLK_LOCAL_MEM_FENCE);

	// Up-Sweep
	int noItemsThatWork=128;
	int offset = 1;
	for (d=0 ; d<k ; d++,noItemsThatWork>>=1,offset<<=1) {
		if (LX < noItemsThatWork) {
			i1 = LX*(offset<<1)+offset-1;
			i2 = i1+offset;
			i1 = CONFLICT_FREE_INDEX(i1);
			i2 = CONFLICT_FREE_INDEX(i2);
			localArray[i2] = localArray[i1] + localArray[i2];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	

	// Down-Sweep
	if (LX==255)
		localArray[CONFLICT_FREE_INDEX(255)] = 0;
	noItemsThatWork = 1;
	offset = 128;
	for (d=0 ; d<k ; d++,noItemsThatWork<<=1,offset>>=1)
	{
		if (LX < noItemsThatWork) {
			i1 = LX*(offset<<1)+offset-1;
			i2 = i1+offset;
			i1 = CONFLICT_FREE_INDEX(i1);
			i2 = CONFLICT_FREE_INDEX(i2);
			int tmp = localArray[i1];
			localArray[i1] = localArray[i2];
			localArray[i2] = tmp + localArray[i2];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// write result to global memory
	out[GX] = localArray[CONFLICT_FREE_INDEX(LX)];
    printf("GX: %d\n", out[GX]);
}
