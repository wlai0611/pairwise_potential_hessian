from openmm import System, VerletIntegrator
from openmm.app.topology import Topology
from openmm.app.element import Element
from openmm.app import Simulation
from openmm.openmm import NonbondedForce
from openmm.unit import nanometers, picosecond, amu
import numpy as np
import matplotlib.pyplot as plt
import functions
import time

#The OpenMM section of this document adapted from :
#https://github.com/molmod/openmm-tutorial-msbs/blob/main/01_first_steps/02_lennard_jones.ipynb
#Give the mass, element and forces acting on each particle
cluster = np.load('cambridge.npy')
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

#Create a function that would calculate the potential of the above system given certain coordinates
def add_all_interatomic_potentials(coordinates):
    simulation.context.setPositions(coordinates*sigma)
    state = simulation.context.getState(getEnergy=True)
    energy= state.getPotentialEnergy()/epsilon
    return energy

#Evaluate Performance of Numerical Hessian
trueH = functions.analytical_hessian(cluster)
start = time.time()
myH   = functions.numerical_hessian(coordinates=cluster,func=add_all_interatomic_potentials,diff=0.01)
runtime = time.time()-start

fig,ax=plt.subplots(nrows=2)
ax[0].plot(trueH.flatten())
ax[0].set(title='Analytical')
ax[1].plot(myH.flatten())
ax[1].set(title=f'Numerical, Runtime {runtime:.2f} s')
plt.show()

errors = trueH-myH
fig, ax = plt.subplots(ncols=2)
errors = ax[0].imshow(errors)
ax[0].set(title='Errors')
fig.colorbar(errors)
actualH = ax[1].imshow(trueH - np.diag(np.diag(trueH)))
ax[1].set(title='True H without diagonal')
fig.colorbar(actualH)
plt.show()

print()