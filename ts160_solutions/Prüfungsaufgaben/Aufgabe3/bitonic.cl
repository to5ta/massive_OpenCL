__kernel void bitonic_kernel(__global int* in, __global int* out)
{

	int gid     = get_global_id(0);
	int lid     = get_local_id(0);
	int groupid = get_group_id(0);


	out[gid] = in[gid]+1;
}