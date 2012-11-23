#ifndef __POTENTIAL_PAIR_PDMP_GPU_H__
#define __POTENTIAL_PAIR_PDMP_GPU_H__


#include <boost/bind.hpp>

#include "PotentialPairPDMP.h"
#include "PotentialPairPDMPGPU.cuh"

/*! \file PotentialPairPDMPGPU.h
    \brief Defines the template class for standard pair potentials on the GPU
    \note This header cannot be compiled by nvcc
*/

#ifdef ENABLE_CUDA

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

class PotentialPairPDMPGPU : public PotentialPairPDMP
    {
    public:
        //! PDMPruct the pair potential
        PotentialPairPDMPGPU(boost::shared_ptr<SystemDefinition> sysdef,
                         boost::shared_ptr<NeighborList> nlist,
                         const std::string& log_suffix="");
        //! Destructor
        virtual ~PotentialPairPDMPGPU() {}
        
        //! Set the block size to execute on the GPU
        /*! \param block_size Size of the block to run on the device
            Performance of the code may be dependant on the block size run
            on the GPU. \a block_size should be set to be a multiple of 32.
        */
        void setBlockSize(int block_size)
            {
            m_block_size = block_size;
            }
    protected:
        unsigned int m_block_size;  //!< Block size to execute on the GPU
        
        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);

    };

void export_PotentialPairPDMPGPU();
#endif
#endif
