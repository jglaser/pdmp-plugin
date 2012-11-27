#include <hoomd/hoomd_config.h>
#include <hoomd/PotentialPairGPU.cuh>

cudaError_t gpu_compute_pair_forces_pdmp(float4 *d_force,
              float *d_virial,
              const unsigned int virial_pitch,
              const unsigned int N,
              const unsigned int n_ghost,
              const Scalar4 *_d_pos,
              const BoxDim& box,
              const unsigned int *d_n_neigh,
              const unsigned int *d_nlist,
              const Index2D& nli,
              const float *d_rcutsq, 
              const unsigned int ntypes,
              const unsigned int block_size,
              const unsigned int compute_virial,
              const Scalar *d_params);
 
