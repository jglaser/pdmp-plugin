from hoomd_plugins.pdmp import _pdmp

# Next, since we are extending an pair potential, we need to bring in the base class and some other parts from
# hoomd_script
from hoomd_script import pair
from hoomd_script import util
from hoomd_script import globals
import hoomd
import math

class pdmp(pair.pair):
    ## Specify the Lennard-Jones %pair %force
    #
    # This method creates the pair force using the c++ classes exported in module.cc. When creating a new pair force,
    # one must update the referenced classes here.
    def __init__(self, r_cut, name=None):
        util.print_status_line();

        # tell the base class how we operate

        # initialize the base class
        pair.pair.__init__(self, r_cut, name);

        # update the neighbor list
        neighbor_list = pair._update_global_nlist(r_cut);
        neighbor_list.subscribe(lambda: self.log*self.get_max_rcut())

        # create the c++ mirror class
        if not globals.exec_conf.isCUDAEnabled():
            self.cpp_force = _pdmp.PotentialPairPDMP(globals.system_definition, neighbor_list.cpp_nlist, self.name);
            self.cpp_class = _pdmp.PotentialPairPDMP;
        else:
            neighbor_list.cpp_nlist.setStorageMode(hoomd.NeighborList.storageMode.full);
            self.cpp_force = _pdmp.PotentialPairPDMPGPU(globals.system_definition, neighbor_list.cpp_nlist, self.name);
            self.cpp_class = _pdmp.PotentialPairPDMPGPU;
            # you can play with the block size value, set it to any multiple of 32 up to 1024. Use the
            # lj.benchmark() command to find out which block size performs the fastest
            self.cpp_force.setBlockSize(512);

        globals.system.addCompute(self.cpp_force, self.force_name);

        # setup the coefficent options
        self.required_coeffs = ['A'];

    ## Process the coefficients
    #
    # The coefficients that the user specifies need not be the same coefficients that get passed as paramters
    # into your Evaluator. This method processes the named coefficients and turns them into the parameter struct
    # for the Evaluator.
    #
    def process_coeff(self, coeff):
        A =  coeff['A'];

        return float(A);
