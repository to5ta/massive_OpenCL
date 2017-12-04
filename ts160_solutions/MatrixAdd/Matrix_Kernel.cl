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

// Kernel code
// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
	int width;
	int height;
	__global float* elements;
} Matrix;

// Thread block size
#define BLOCK_SIZE 16
// Matrix multiplication function called by MatMulKernel()

// Matrix addition kernel called by MatAddHost()
__kernel void MatAddKernel( int Awidth, int Aheight, __global float* Aelements,
							int Bwidth, int Bheight, __global float* Belements,
							int Cwidth, int Cheight, __global float* Celements)
{
	Matrix A = { Awidth, Aheight, Aelements };
	Matrix B = { Bwidth, Bheight, Belements };
	Matrix C = { Cwidth, Cheight, Celements };
	
	int col = get_global_id(0);
	int row = get_global_id(1);

    Celements[col + Cwidth*row] = Aelements[row+Cwidth*col] + Belements[row+Cwidth*col];

}


// Matrix addition kernel called by MatAddHost()
__kernel void MatMulKernel( int Awidth, int Aheight, __global float* Aelements,
                            int Bwidth, int Bheight, __global float* Belements,
                            int Cwidth, int Cheight, __global float* Celements)
{
    Matrix A = { Awidth, Aheight, Aelements };
    Matrix B = { Bwidth, Bheight, Belements };
    Matrix C = { Cwidth, Cheight, Celements };

	// __local float sum = 0;
	// barrier(CLK_LOCAL_MEM_FENCE);

    int col = get_global_id(0);
    int row = get_global_id(1);

    float res=0.f;
	if(col<Cwidth && row<Cheight){
		Celements[col+row*Cwidth] = 0.f;
		for(int index=0; index<Awidth; index++) {
	    	res += Aelements[index+row*Cwidth] * Belements[col+Bwidth*index];
	    }
        Celements[col+row*Cwidth] = res;
    }
}

// Matrix addition kernel called by MatAddHost()
__kernel void MatMulSharedKernel( int Awidth, int Aheight, __global float* Aelements,
                                  int Bwidth, int Bheight, __global float* Belements,
                                  int Cwidth, int Cheight, __global float* Celements)
{
    Matrix A = { Awidth, Aheight, Aelements };
    Matrix B = { Bwidth, Bheight, Belements };
    Matrix C = { Cwidth, Cheight, Celements };

    // __local float sum = 0;
    // barrier(CLK_LOCAL_MEM_FENCE);

    int grp_id0 = get_group_id(0);
    int grp_id1 = get_group_id(1);

    int grp_max_0 = get_num_groups(0);
    int grp_max_1 = get_num_groups(1);

    int lid0 = get_local_id(0);
    int lid1 = get_local_id(1);
    int l_max_0 = get_local_size(0);
    int l_max_1 = get_local_size(1);

    int col = get_global_id(0);
    int row = get_global_id(1);

//    // global id does not correspond to width or height of C
//    if(!col<Cwidth || !row<Cheight){
//        return;
//    }

    __local float patchA[256];
    __local float patchB[256];

//    __local float patchC[256];
//    patchC[lid0 + lid1*l_max_0] = 0.f;

    barrier(CLK_LOCAL_MEM_FENCE);

    float Cresult = 0.f;

    for(int patch_id=0; patch_id<(Awidth-1)/BLOCK_SIZE+1; patch_id++){

        int Ax = patch_id*BLOCK_SIZE+lid0;
        int Ay = grp_id1*BLOCK_SIZE+lid1;
//        patchA[lid0 + lid1*l_max_0] = Aelements[Ax + Awidth*Ay];
        // memory safety considerations
        if(Ax<Awidth && Ay<Aheight) {
            patchA[lid0 + lid1*l_max_0] = Aelements[Ax + Awidth*Ay];
        } else {
            patchA[lid0 + lid1*l_max_0] = 0.f;
        }

        int By = patch_id*BLOCK_SIZE+lid1;
        int Bx = grp_id0*BLOCK_SIZE+lid0;
//        patchB[lid0 + lid1*l_max_0] = Belements[Bx + Bwidth*By];
        // memory safety considerations
        if(Bx<Bwidth && By<Bheight) {
            patchB[lid0 + lid1*l_max_0] = Belements[Bx + Bwidth*By];
        } else {
            patchB[lid0 + lid1*l_max_0] = 0.f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        float sum = 0.f;
        for(int index=0; index<l_max_0; index++) {
            sum += patchA[index + l_max_0*lid1] * patchB[lid0 + l_max_0*index];
        }

//        patchC[lid0 + lid1*l_max_0] += sum;
        Cresult += sum;
        barrier(CLK_LOCAL_MEM_FENCE);

    }

    // global id does not correspond to width or height of C
    if(col<Cwidth && row<Cheight){
        Celements[col+row*Cwidth] = Cresult;
    }
}



//
//    // loop over all patches in C
//    for(int gr=0; gr<grp_max_1; gr++){
//        for(int gc=0; gc<grp_max_0; gc++) {
//
//            if(col<Cwidth && row<Cheight){
//                // buffer patch from A
//                patchA[lid0 + lid1*l_max_0] = Aelements[col + row*Awidth];
//
//                // buffer patch from B
//                patchB[lid0 + lid1*l_max_0] = Belements[col + row*Bwidth];
//
//                barrier(CLK_LOCAL_MEM_FENCE);
//
//                float sum = 0.f;
//                for(int index=0; index<l_max_0; index++) {
//                    sum += patchA[index + l_max_0*lid1] * patchB[lid0 + l_max_0*index];
//                }
//                Celements[col+row*Cwidth] += sum;
//            }
//
//            barrier(CLK_LOCAL_MEM_FENCE);
//        }
//    }