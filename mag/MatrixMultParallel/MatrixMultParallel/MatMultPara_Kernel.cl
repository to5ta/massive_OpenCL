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

#define TS 8

__kernel void matMultGlob_kernel( int M, int N, int K,
                                 __global int* matA, __global int* matB, __global int* matOut)
{
    const int row = get_local_id(0);
    const int col = get_local_id(1);
	const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
    // const int TS = 8;
    
    // Local memory to fit a tile of TS*TS elements of A and B
    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];
    
    // Initialise the accumulation register
    float acc = 0;
    
    // Loop over all tiles
    const int numTiles = K/TS;
    for (int t=0; t<numTiles; t++) {
        
        // Load one tile of A and B into local memory
        const int tiledRow = TS*t + row;
        const int tiledCol = TS*t + col;
        Asub[col][row] = matA[tiledCol*M + globalRow];
        Bsub[col][row] = matB[globalCol*K + tiledRow];
        
        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Perform the computation for a single tile
        for (int k=0; k<TS; k++) {
            acc += Asub[k][row] * Bsub[col][k];
        }
        
        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Store the final result in C
    matOut[globalCol*M + globalRow] = acc;
}
