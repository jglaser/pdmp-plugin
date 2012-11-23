// Include boost.python to do the exporting
#include <boost/python.hpp>
using namespace boost::python;

#include "PotentialPairPDMP.h"

#ifdef ENABLE_CUDA
#include "PotentialPairPDMPGPU.h"
#endif

// specify the python module. Note that the name must expliclty match the PROJECT() name provided in CMakeLists
// (with an underscore in front)
BOOST_PYTHON_MODULE(_pdmp)
    {
    // export pair potential
    export_PotentialPairPDMP();

    #ifdef ENABLE_CUDA
    export_PotentialPairPDMPGPU();
    #endif
    }
