#include <iostream>
#include <stdexcept>
#include <boost/shared_ptr.hpp>
#include <boost/python.hpp>

#include "PotentialPairPDMP.h"

#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

/*! \param sysdef System to compute forces on
    \param nlist Neighborlist to use for computing the forces
    \param log_suffix Name given to this instance of the force
*/
PotentialPairPDMP::PotentialPairPDMP(boost::shared_ptr<SystemDefinition> sysdef,
                                       boost::shared_ptr<NeighborList> nlist,
                                       const std::string& log_suffix)
    : ForceCompute(sysdef), m_nlist(nlist), m_typpair_idx(m_pdata->getNTypes())
    {
    m_exec_conf->msg->notice(5) << "PDMPructing PotentialPairPDMP" << endl;

    assert(m_pdata);
    assert(m_nlist);
    
    GPUArray<Scalar> rcutsq(m_typpair_idx.getNumElements(), exec_conf);
    m_rcutsq.swap(rcutsq);
    GPUArray<Scalar> params(m_typpair_idx.getNumElements(), exec_conf);
    m_params.swap(params);
    
    // initialize name
    m_prof_name = std::string("Pair PDMP");
    m_log_name = std::string("pair_pdmp_energy") + log_suffix;

    // initialize memory for per thread reduction
    allocateThreadPartial();
    }

PotentialPairPDMP::~PotentialPairPDMP()
    {
    m_exec_conf->msg->notice(5) << "Destroying PotentialPairPDMP" << endl;
    }

/*! \param typ1 First type index in the pair
    \param typ2 Second type index in the pair
    \param param Parameter to set
    \note When setting the value for (\a typ1, \a typ2), the parameter for (\a typ2, \a typ1) is automatically
          set.
*/
void PotentialPairPDMP::setParams(unsigned int typ1, unsigned int typ2, const Scalar& param)
    {
    if (typ1 >= m_pdata->getNTypes() || typ2 >= m_pdata->getNTypes())
        {
        this->m_exec_conf->msg->error() << "pair.pdmp: Trying to set pair params for a non existant type! "
                  << typ1 << "," << typ2 << std::endl;
        throw std::runtime_error("Error setting parameters in PotentialPairPDMP");
        }
    
    ArrayHandle<Scalar> h_params(m_params, access_location::host, access_mode::readwrite);
    h_params.data[m_typpair_idx(typ1, typ2)] = param;
    h_params.data[m_typpair_idx(typ2, typ1)] = param;
    }

/*! \param typ1 First type index in the pair
    \param typ2 Second type index in the pair
    \param rcut Cuttoff radius to set
    \note When setting the value for (\a typ1, \a typ2), the parameter for (\a typ2, \a typ1) is automatically
          set.
*/
void PotentialPairPDMP::setRcut(unsigned int typ1, unsigned int typ2, Scalar rcut)
    {
    if (typ1 >= m_pdata->getNTypes() || typ2 >= m_pdata->getNTypes())
        {
        this->m_exec_conf->msg->error() << "pair.pdmp: Trying to set rcut for a non existant type! "
                  << typ1 << "," << typ2 << std::endl;
        throw std::runtime_error("Error setting parameters in PotentialPairPDMP");
        }
    
    ArrayHandle<Scalar> h_rcutsq(m_rcutsq, access_location::host, access_mode::readwrite);
    h_rcutsq.data[m_typpair_idx(typ1, typ2)] = rcut * rcut;
    h_rcutsq.data[m_typpair_idx(typ2, typ1)] = rcut * rcut;
    }

/*! PotentialPairPDMP provides:
     - \c pair_pdmp_energy
    where "name" is replaced with evaluator::getName()
*/
std::vector< std::string > PotentialPairPDMP::getProvidedLogQuantities()
    {
    vector<string> list;
    list.push_back(m_log_name);
    return list;
    }

/*! \param quantity Name of the log value to get
    \param timestep Current timestep of the simulation
*/
Scalar PotentialPairPDMP::getLogValue(const std::string& quantity, unsigned int timestep)
    {
    if (quantity == m_log_name)
        {
        compute(timestep);
        return calcEnergySum();
        }
    else
        {
        this->m_exec_conf->msg->error() << "pair.pdmp : " << quantity << " is not a valid log quantity" 
                  << std::endl;
        throw std::runtime_error("Error getting log value");
        }
    }

/*! \post The pair forces are computed for the given timestep. The neighborlist's compute method is called to ensure
    that it is up to date before proceeding.

    \param timestep specifies the current time step of the simulation
*/
void PotentialPairPDMP::computeForces(unsigned int timestep)
    {
    // start by updating the neighborlist
    m_nlist->compute(timestep);
    
    // start the profile for this compute
    if (m_prof) m_prof->push(m_prof_name);
    
    // depending on the neighborlist settings, we can take advantage of newton's third law
    // to reduce computations at the cost of memory access complexity: set that flag now
    bool third_law = m_nlist->getStorageMode() == NeighborList::half;
    
    // access the neighbor list, particle data, and system box
    ArrayHandle<unsigned int> h_n_neigh(m_nlist->getNNeighArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_nlist(m_nlist->getNListArray(), access_location::host, access_mode::read);
    Index2D nli = m_nlist->getNListIndexer();

    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::read);


    //force arrays
    ArrayHandle<Scalar4> h_force(m_force,access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar>  h_virial(m_virial,access_location::host, access_mode::overwrite);


    const BoxDim& box = m_pdata->getBox();
    ArrayHandle<Scalar> h_ronsq(m_ronsq, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_rcutsq(m_rcutsq, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_params(m_params, access_location::host, access_mode::read);
    
    PDataFlags flags = this->m_pdata->getFlags();
    bool compute_virial = flags[pdata_flag::pressure_tensor] || flags[pdata_flag::isotropic_virial];

#pragma omp parallel
    {
    #ifdef ENABLE_OPENMP
    int tid = omp_get_thread_num();
    #else
    int tid = 0;
    #endif

    // need to start from a zero force, energy and virial
    memset(&m_fdata_partial[m_index_thread_partial(0,tid)] , 0, sizeof(Scalar4)*m_pdata->getN());
    memset(&m_virial_partial[6*m_index_thread_partial(0,tid)] , 0, 6*sizeof(Scalar)*m_pdata->getN());

    // for each particle
#pragma omp for schedule(guided)
    for (int i = 0; i < (int)m_pdata->getN(); i++)
        {
        // access the particle's position and type (MEM TRANSFER: 4 scalars)
        Scalar3 pi = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);
        unsigned int typei = __scalar_as_int(h_pos.data[i].w);
        // sanity check
        assert(typei < m_pdata->getNTypes());
        
        // initialize current particle force, potential energy, and virial to 0
        Scalar3 fi = make_scalar3(0, 0, 0);
        Scalar pei = 0.0;
        Scalar virialxxi = 0.0;
        Scalar virialxyi = 0.0;
        Scalar virialxzi = 0.0;
        Scalar virialyyi = 0.0;
        Scalar virialyzi = 0.0;
        Scalar virialzzi = 0.0;
        
        // loop over all of the neighbors of this particle
        const unsigned int size = (unsigned int)h_n_neigh.data[i];
        for (unsigned int k = 0; k < size; k++)
            {
            // access the index of this neighbor (MEM TRANSFER: 1 scalar)
            unsigned int j = h_nlist.data[nli(i, k)];
            assert(j < m_pdata->getN() + m_pdata->getNGhosts());
            
            // calculate dr_ji (MEM TRANSFER: 3 scalars / FLOPS: 3)
            Scalar3 pj = make_scalar3(h_pos.data[j].x, h_pos.data[j].y, h_pos.data[j].z);
            Scalar3 dx = pi - pj;
            
            // access the type of the neighbor particle (MEM TRANSFER: 1 scalar)
            unsigned int typej = __scalar_as_int(h_pos.data[j].w);
            assert(typej < m_pdata->getNTypes());
            
            // apply periodic boundary conditions
            dx = box.minImage(dx);
                
            // get parameters for this type pair
            unsigned int typpair_idx = m_typpair_idx(typei, typej);
            Scalar param = h_params.data[typpair_idx];
            Scalar rcutsq = h_rcutsq.data[typpair_idx];

            Scalar3 force = make_scalar3(0.0,0.0,0.0);
            Scalar pe = Scalar(0.0);
            if (dx.x*dx.x < rcutsq && dx.y*dx.y < rcutsq && dx.z*dx.z < rcutsq)
                {
                Scalar rcut = sqrtf(rcutsq);

                Scalar3 pe_factors = make_scalar3(0.0,0.0,0.0);
                pe_factors.x = (Scalar(1.0)-fabs(dx.x)/rcut);
                pe_factors.y = (Scalar(1.0)-fabs(dx.y)/rcut);
                pe_factors.z = (Scalar(1.0)-fabs(dx.z)/rcut);

                Scalar max_energy = param*Scalar(8.0)*rcut*rcut*rcut;

                pe = max_energy*pe_factors.x*pe_factors.y*pe_factors.z;

                force.x = copysign(1.0,dx.x)/rcut*max_energy*pe_factors.y*pe_factors.z;
                force.y = copysign(1.0,dx.y)/rcut*max_energy*pe_factors.x*pe_factors.z;
                force.z = copysign(1.0,dx.z)/rcut*max_energy*pe_factors.x*pe_factors.y;
                
                fi += force;
                pei += pe*Scalar(0.5);
                }

            // calculate the virial
            if (compute_virial)
                {
                virialxxi +=  force.x*dx.x/Scalar(2.0);
                virialxyi +=  (force.x*dx.y+force.y*dx.x)/Scalar(4.0);
                virialxzi +=  (force.x*dx.z+force.z*dx.x)/Scalar(4.0);
                virialyyi +=  force.y*dx.y/Scalar(2.0);
                virialyzi +=  (force.y*dx.z+force.z*dx.y)/Scalar(4.0);
                virialzzi +=  force.z*dx.z/Scalar(2.0);
                }
 
            // add the force to particle j if we are using the third law (MEM TRANSFER: 10 scalars / FLOPS: 8)
            // only add force to local particles
            if (third_law && j < m_pdata->getN())
                {
                unsigned int mem_idx = m_index_thread_partial(j,tid);
                m_fdata_partial[mem_idx].x -= force.x;
                m_fdata_partial[mem_idx].y -= force.y;
                m_fdata_partial[mem_idx].z -= force.z;
                m_fdata_partial[mem_idx].w += pe * Scalar(0.5);
                if (compute_virial)
                    {
                    m_virial_partial[0+6*mem_idx] += force.x*dx.x/Scalar(2.0);
                    m_virial_partial[1+6*mem_idx] += (force.x*dx.y+force.y*dx.x)/Scalar(4.0);
                    m_virial_partial[2+6*mem_idx] += (force.x*dx.z+force.z*dx.x)/Scalar(4.0);
                    m_virial_partial[3+6*mem_idx] += force.y*dx.y/Scalar(2.0);
                    m_virial_partial[4+6*mem_idx] += (force.y*dx.z+force.z*dx.y)/Scalar(4.0);
                    m_virial_partial[5+6*mem_idx] += force.z*dx.z/Scalar(2.0);
                    }
                }
            }
            
        // finally, increment the force, potential energy and virial for particle i
        unsigned int mem_idx = m_index_thread_partial(i,tid);
        m_fdata_partial[mem_idx].x += fi.x;
        m_fdata_partial[mem_idx].y += fi.y;
        m_fdata_partial[mem_idx].z += fi.z;
        m_fdata_partial[mem_idx].w += pei;
        if (compute_virial)
            {
            m_virial_partial[0+6*mem_idx] += virialxxi;
            m_virial_partial[1+6*mem_idx] += virialxyi;
            m_virial_partial[2+6*mem_idx] += virialxzi;
            m_virial_partial[3+6*mem_idx] += virialyyi;
            m_virial_partial[4+6*mem_idx] += virialyzi;
            m_virial_partial[5+6*mem_idx] += virialzzi;
            }
        }
#pragma omp barrier
    
    // now that the partial sums are complete, sum up the results in parallel
#pragma omp for
    for (int i = 0; i < (int) m_pdata->getN(); i++)
        {
        // assign result from thread 0
        h_force.data[i].x = m_fdata_partial[i].x;
        h_force.data[i].y = m_fdata_partial[i].y;
        h_force.data[i].z = m_fdata_partial[i].z;
        h_force.data[i].w = m_fdata_partial[i].w;

        for (int j = 0; j < 6; j++)
            h_virial.data[j*m_virial_pitch+i] = m_virial_partial[j+6*i];

        #ifdef ENABLE_OPENMP
        // add results from other threads
        int nthreads = omp_get_num_threads();
        for (int thread = 1; thread < nthreads; thread++)
            {
            unsigned int mem_idx = m_index_thread_partial(i,thread);
            h_force.data[i].x += m_fdata_partial[mem_idx].x;
            h_force.data[i].y += m_fdata_partial[mem_idx].y;
            h_force.data[i].z += m_fdata_partial[mem_idx].z;
            h_force.data[i].w += m_fdata_partial[mem_idx].w;
            for (int j = 0; j < 6; j++)
                h_virial.data[j*m_virial_pitch+i] += m_virial_partial[j+6*mem_idx];
            }
        #endif
        }
    } // end omp parallel

    if (m_prof) m_prof->pop();
    }

//! Export this pair potential to python
void export_PotentialPairPDMP()
    {
    boost::python::class_<PotentialPairPDMP, boost::shared_ptr<PotentialPairPDMP>, boost::python::bases<ForceCompute>, boost::noncopyable >
          ("PotentialPairPDMP", boost::python::init< boost::shared_ptr<SystemDefinition>, boost::shared_ptr<NeighborList>, const std::string& >())
          .def("setParams", &PotentialPairPDMP::setParams)
          .def("setRcut", &PotentialPairPDMP::setRcut)
          .def("setRon", &PotentialPairPDMP::setRon)
          ;
    }


