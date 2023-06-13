from functions import numerical_hessian, forward_diff_hessian, analytical_hessian
import numpy as np
import matplotlib.pyplot as plt
import time
from openmm import System, VerletIntegrator
from openmm.app.topology import Topology
from openmm.app.element import Element
from openmm.app import Simulation
from openmm.openmm import NonbondedForce
from openmm.unit import nanometers, picosecond, amu
import numpy as np
import matplotlib.pyplot as plt


def get_openmm_potential_function(cluster):
    natoms,ndims = cluster.shape
    masses    = np.ones(natoms)
    mass_unit = amu
    sigma     = nanometers
    time_unit = picosecond
    epsilon   = (mass_unit * sigma**2)/(time_unit**2)
    system    = System()
    topology  = Topology()
    chain     = topology.addChain()
    residue   = topology.addResidue(name='argon',chain=chain)
    element_object = Element.getByAtomicNumber(18)
    forces    = NonbondedForce()
    for mass in masses:
        system.addParticle(mass=mass*mass_unit)
        forces.addParticle(charge=0,sigma=1*sigma, epsilon=1*epsilon)
        topology.addAtom(name='argon',element=element_object,residue=residue)
    system.addForce(force=forces)
    simulation = Simulation(topology, system, VerletIntegrator(0.001*time_unit))
    def add_all_interatomic_potentials(coordinates):
        simulation.context.setPositions(coordinates*sigma)
        state = simulation.context.getState(getEnergy=True)
        energy= state.getPotentialEnergy()/epsilon
        return energy
    return add_all_interatomic_potentials

datasets= {
  77 : np.load('cambridge.npy'),
  100: np.load('lj100.npy'),
   310: np.load('lj310.npy'),
   500: np.load('lj500.npy'),  
}
performance = {}

fig, ax = plt.subplots(nrows = 2, ncols = len(datasets), figsize=(10,5))
plt.subplots_adjust(wspace=0.3, hspace=0.3)

for counter, (size, dataset) in enumerate(datasets.items()):
    add_all_interatomic_potentials = get_openmm_potential_function(dataset)
    start = time.time()
    myH   = forward_diff_hessian(coordinates=dataset,func=add_all_interatomic_potentials,diff=0.01)
    duration = time.time() - start
    trueH    = analytical_hessian(dataset)
    ax[0][counter].plot(myH.flatten())
    ax[0][counter].set(title=f"{size} atoms\nNumerical {duration:.2f} sec")
    ax[1][counter].plot(trueH.flatten())
    ax[1][counter].set(title="Analytical")
    performance[size] = { 'time':duration, 'analytical': trueH, 'numerical': myH}

plt.show()

    