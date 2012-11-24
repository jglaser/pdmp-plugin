#ifdef ENABLE_CUDA
#include "PotentialPairPDMPGPU.h"

PotentialPairPDMPGPU::PotentialPairPDMPGPU(boost::shared_ptr<SystemDefinition> sysdef,
                                           boost::shared_ptr<NeighborList> nlist,
                                           const std::string& log_suffix)
    : PotentialPairPDMP(sysdef, nlist, log_suffix), m_block_size(64)
    {
    // can't run on the GPU if there aren't any GPUs in the execution configuration
    if (!this->exec_conf->isCUDAEnabled())
        {
        this->m_exec_conf->msg->error() << "Creating a PotentialPairPDMPGPU with no GPU in the execution configuration" 
                  << std::endl;
        throw std::runtime_error("Error initializing PotentialPairPDMPGPU");
        }
    }

void PotentialPairPDMPGPU::computeForces(unsigned int timestep)
    {
    // start by updating the neighborlist
    this->m_nlist->compute(timestep);
    
    // start the profile
    if (this->m_prof) this->m_prof->push(this->exec_conf, this->m_prof_name);
    
    // The GPU implementation CANNOT handle a half neighborlist, error out now
    bool third_law = this->m_nlist->getStorageMode() == NeighborList::half;
    if (third_law)
        {
        this->m_exec_conf->msg->error() << "PotentialPairPDMPGPU cannot handle a half neighborlist" 
                  << std::endl;
        throw std::runtime_error("Error computing forces in PotentialPairPDMPGPU");
        }
        
    // access the neighbor list
    ArrayHandle<unsigned int> d_n_neigh(this->m_nlist->getNNeighArray(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_nlist(this->m_nlist->getNListArray(), access_location::device, access_mode::read);
    Index2D nli = this->m_nlist->getNListIndexer();
    
    // access the particle data
    ArrayHandle<Scalar4> d_pos(this->m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_diameter(this->m_pdata->getDiameters(), access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_charge(this->m_pdata->getCharges(), access_location::device, access_mode::read);

    BoxDim box = this->m_pdata->getBox();
    
    // access parameters
    ArrayHandle<Scalar> d_ronsq(this->m_ronsq, access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_rcutsq(this->m_rcutsq, access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_params(this->m_params, access_location::device, access_mode::read);
    
    ArrayHandle<Scalar4> d_force(this->m_force, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar> d_virial(this->m_virial, access_location::device, access_mode::readwrite);
    
    // access flags
    PDataFlags flags = this->m_pdata->getFlags();

    gpu_compute_pair_forces_pdmp(d_force.data,
                         d_virial.data,
                         this->m_virial.getPitch(),
                         this->m_pdata->getN(),
                         this->m_pdata->getNGhosts(),
                         d_pos.data,
                         box,
                         d_n_neigh.data,
                         d_nlist.data,
                         nli,
                         d_rcutsq.data,
                         this->m_pdata->getNTypes(),
                         m_block_size,
                         flags[pdata_flag::pressure_tensor] || flags[pdata_flag::isotropic_virial],
             d_params.data);
    
    if (this->exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
   
    if (this->m_prof) this->m_prof->pop(this->exec_conf);
    }

void export_PotentialPairPDMPGPU()
    {
     boost::python::class_<PotentialPairPDMPGPU, boost::shared_ptr<PotentialPairPDMPGPU>, boost::python::bases<PotentialPairPDMP>, boost::noncopyable >
              ("PotentialPairPDMPGPU", boost::python::init< boost::shared_ptr<SystemDefinition>, boost::shared_ptr<NeighborList>, const std::string& >())
              .def("setBlockSize", &PotentialPairPDMPGPU::setBlockSize)
              ;
    }
#endif
