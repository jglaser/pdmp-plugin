#include "PotentialPairPDMPGPU.cuh"
#include <hoomd/HOOMDMath.h>
#include <hoomd/Index1D.h>
#include <hoomd/ParticleData.cuh>

/*! \file PotentialPairPDMPGPU.cuh
    \brief Defines templated GPU kernel code for calculating the pair forces.
*/

#ifdef NVCC
template<unsigned int compute_virial>
__global__ void gpu_compute_pair_forces_pdmp_kernel(float4 *d_force,
                                               float *d_virial,
                                               const unsigned int virial_pitch,
                                               const unsigned int N,
                                               const Scalar4 *d_pos,
                                               const BoxDim box,
                                               const unsigned int *d_n_neigh,
                                               const unsigned int *d_nlist,
                                               const Index2D nli,
                                               const Scalar *d_params,
                                               const float *d_rcutsq,
                                               const unsigned int ntypes)
    {
    Index2D typpair_idx(ntypes);
    const unsigned int num_typ_parameters = typpair_idx.getNumElements();

    // shared arrays for per type pair parameters
    extern __shared__ char s_data[];
    Scalar *s_params = (Scalar *)(&s_data[0]);
    float *s_rcutsq = (float *)(&s_data[num_typ_parameters*sizeof(Scalar)]);

    // load in the per type pair parameters
    for (unsigned int cur_offset = 0; cur_offset < num_typ_parameters; cur_offset += blockDim.x)
        {
        if (cur_offset + threadIdx.x < num_typ_parameters)
            {
            s_rcutsq[cur_offset + threadIdx.x] = d_rcutsq[cur_offset + threadIdx.x];
            s_params[cur_offset + threadIdx.x] = d_params[cur_offset + threadIdx.x];
            }
        }
    __syncthreads();

    // start by identifying which particle we are to handle
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
        return;

    // load in the length of the neighbor list (MEM_TRANSFER: 4 bytes)
    unsigned int n_neigh = d_n_neigh[idx];

    // read in the position of our particle.
    // (MEM TRANSFER: 16 bytes)
    float4 postypei = tex1Dfetch(pdata_pos_tex, idx);
    float3 posi = make_float3(postypei.x, postypei.y, postypei.z);


    // initialize the force to 0
    float4 force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float virialxx = 0.0f;
    float virialxy = 0.0f;
    float virialxz = 0.0f;
    float virialyy = 0.0f;
    float virialyz = 0.0f;
    float virialzz = 0.0f;

    // prefetch neighbor index
    unsigned int cur_j = 0;
    unsigned int next_j = d_nlist[nli(idx, 0)];

    // loop over neighbors
    // on pre Fermi hardware, there is a bug that causes rare and random ULFs when simply looping over n_neigh
    // the workaround (activated via the template paramter) is to loop over nlist.height and put an if (i < n_neigh)
    // inside the loop
    #if (__CUDA_ARCH__ < 200)
    for (int neigh_idx = 0; neigh_idx < nli.getH(); neigh_idx++)
    #else
    for (int neigh_idx = 0; neigh_idx < n_neigh; neigh_idx++)
    #endif
        {
        #if (__CUDA_ARCH__ < 200)
        if (neigh_idx < n_neigh)
        #endif
            {
            // read the current neighbor index (MEM TRANSFER: 4 bytes)
            // prefetch the next value and set the current one
            cur_j = next_j;
            next_j = d_nlist[nli(idx, neigh_idx+1)];

            // get the neighbor's position (MEM TRANSFER: 16 bytes)
            float4 postypej = tex1Dfetch(pdata_pos_tex, cur_j);
            float3 posj = make_float3(postypej.x, postypej.y, postypej.z);

            // calculate dr (with periodic boundary conditions) (FLOPS: 3)
            float3 dx = posi - posj;

            // apply periodic boundary conditions: (FLOPS 12)
            dx = box.minImage(dx);

            // access the per type pair parameters
            unsigned int typpair = typpair_idx(__float_as_int(postypei.w), __float_as_int(postypej.w));
            float rcutsq = s_rcutsq[typpair];
            Scalar param = s_params[typpair];

            Scalar3 f = make_scalar3(0.0,0.0,0.0);
            // evaluate the potential

            // dimensions of cubic overlap volume
            Scalar Lsq= rcutsq/Scalar(3.0);
            if (dx.x*dx.x < Lsq && dx.y*dx.y < Lsq && dx.z*dx.z < Lsq)
                {
                Scalar L = sqrtf(Lsq);

                Scalar3 pe_factors = make_scalar3(0.0,0.0,0.0);
                pe_factors.x = (Scalar(1.0)-copysignf(Scalar(1.0),dx.x)*dx.x/L);
                pe_factors.y = (Scalar(1.0)-copysignf(Scalar(1.0),dx.y)*dx.y/L);
                pe_factors.z = (Scalar(1.0)-copysignf(Scalar(1.0),dx.z)*dx.z/L);

                Scalar max_energy = param;


                f.x = copysignf(Scalar(1.0),dx.x)/L*max_energy*pe_factors.y*pe_factors.z;
                f.y = copysignf(Scalar(1.0),dx.y)/L*max_energy*pe_factors.x*pe_factors.z;
                f.z = copysignf(Scalar(1.0),dx.z)/L*max_energy*pe_factors.x*pe_factors.y;
                force.x += f.x;
                force.y += f.y;
                force.z += f.z;
                force.w += max_energy*pe_factors.x*pe_factors.y*pe_factors.z;
                }

            // calculate the virial
            if (compute_virial)
                {
                virialxx +=  f.x*dx.x/Scalar(2.0);
                virialxy +=  (f.x*dx.y+f.y*dx.x)/Scalar(4.0);
                virialxz +=  (f.x*dx.z+f.z*dx.x)/Scalar(4.0);
                virialyy +=  f.y*dx.y/Scalar(2.0);
                virialyz +=  (f.y*dx.z+f.z*dx.y)/Scalar(4.0);
                virialzz +=  f.z*dx.z/Scalar(2.0);
                }
            }
        }

    // potential energy per particle must be halved
    force.w *= 0.5f;
    // now that the force calculation is complete, write out the result (MEM TRANSFER: 20 bytes)
    d_force[idx] = force;

    if (compute_virial)
        {
        d_virial[0*virial_pitch+idx] = virialxx;
        d_virial[1*virial_pitch+idx] = virialxy;
        d_virial[2*virial_pitch+idx] = virialxz;
        d_virial[3*virial_pitch+idx] = virialyy;
        d_virial[4*virial_pitch+idx] = virialyz;
        d_virial[5*virial_pitch+idx] = virialzz;
        }
    }

//! Kernel driver that computes lj forces on the GPU for LJForceComputeGPU
/*! \param pair_args Other arugments to pass onto the kernel
    \param d_params Parameters for the potential, stored per type pair
    
    This is just a driver function for gpu_compute_pair_forces_kernel(), see it for details.
*/
cudaError_t gpu_compute_pair_forces_pdmp(float4 *d_force,
              float *d_virial,
              const unsigned int virial_pitch,
              const unsigned int N,
              const unsigned int n_ghost,
              const Scalar4 *d_pos,
              const BoxDim& box,
              const unsigned int *d_n_neigh,
              const unsigned int *d_nlist,
              const Index2D& nli,
              const float *d_rcutsq, 
              const unsigned int ntypes,
              const unsigned int block_size,
              const unsigned int compute_virial,
              const Scalar *d_params)
    {
    assert(d_params);
    
    // setup the grid to run the kernel
    dim3 grid( N / block_size + 1, 1, 1);
    dim3 threads(block_size, 1, 1);

    // bind the position texture
    pdata_pos_tex.normalized = false;
    pdata_pos_tex.filterMode = cudaFilterModePoint;
    cudaError_t error = cudaBindTexture(0, pdata_pos_tex, d_pos, sizeof(Scalar4)*(N+n_ghost));
    if (error != cudaSuccess)
        return error;

    Index2D typpair_idx(ntypes);
    unsigned int shared_bytes = (2*sizeof(float) + sizeof(Scalar))
                                * typpair_idx.getNumElements();
    
    // run the kernel
    if (compute_virial)
        gpu_compute_pair_forces_pdmp_kernel<1>
          <<<grid, threads, shared_bytes>>>(d_force, d_virial, virial_pitch, N, d_pos, box, d_n_neigh, d_nlist, nli, d_params, d_rcutsq, ntypes);
    else
        gpu_compute_pair_forces_pdmp_kernel<0>
          <<<grid, threads, shared_bytes>>>(d_force, d_virial, virial_pitch, N, d_pos, box, d_n_neigh, d_nlist, nli, d_params, d_rcutsq, ntypes);
        
    return cudaSuccess;
    }
#endif

