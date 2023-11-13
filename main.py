import numpy as np

seed = 7
np.random.seed = seed

import matplotlib.pyplot as plt
import copy

#%matplotlib inline

from qiskit import QuantumRegister, QuantumCircuit, BasicAer
from qiskit.circuit.library import TwoLocal

from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit_machine_learning.algorithms import NumPyDiscriminator, QGAN

algorithm_globals.random_seed = seed

import my_utils

#########################################################
################### Load training data ##################
#########################################################

# Number training data samples
N = 2000

# Load data samples from log-normal distribution with mean=1 and standard deviation=1
mu = 1
sigma = 1
#real_data = np.random.lognormal(mean=mu, sigma=sigma, size=N)
#real_data = np.random.multivariate_normal(mean=np.array([1.5,1.5]), cov = np.array([[1, 0.0],[0.0, 1]]), size=N)
#real_data = np.random.uniform(0.0, 3.0, (N, 2))
#real_data = np.random.lognormal(mean=mu, sigma=sigma, size=(N, 2))
#real_data = np.random.normal(3.5, 2, size=N)
real_data1 = np.random.normal(0.5, 1, int(N/2)) # Gaussian1
real_data2 = np.random.normal(3.5, 0.5, int(N/2)) # Gaussian2
real_data = np.concatenate((real_data1, real_data2), axis=0) # Gaussian mixture
np.random.shuffle(real_data)

# Set the data resolution
# Set upper and lower data values as list of k min/max data values [[min_0,max_0],...,[min_k-1,max_k-1]]
bounds = np.array([0.0, 7.0])
#bounds = np.array([0.0, 15.0])
#bounds = np.array([[0.0, 3.0], [0.0, 3.0]])
# Set number of qubits per data dimension as list of k qubit values[#q_0,...,#q_k-1]
num_qubits = [3]
#num_qubits = [2, 2]
k = len(num_qubits)

real_data_round = my_utils.bound_data(real_data, bounds)

########################################################
################### Initialize QGAN ####################
########################################################

# Set number of training epochs
# Note: The algorithm's runtime can be shortened by reducing the number of training epochs.
num_epochs = 100
# Batch size
batch_size = 100

# Initialize qGAN
qgan = QGAN(real_data_round, bounds, num_qubits, batch_size, num_epochs, snapshot_dir=None)
qgan.seed = 1
# Set quantum instance to run the quantum generator
quantum_instance = QuantumInstance(
    backend=BasicAer.get_backend("statevector_simulator"), seed_transpiler=seed, seed_simulator=seed
)

# Set an initial state for the generator circuit as a uniform distribution
# This corresponds to applying Hadamard gates on all qubits
init_dist = QuantumCircuit(sum(num_qubits))
init_dist.h(init_dist.qubits)

# Set the ansatz circuit
ansatz = TwoLocal(int(np.sum(num_qubits)), "ry", "cz", entanglement="circular", reps=1)
ansatz.decompose().draw(output='mpl')

# Set generator's initial parameters - in order to reduce the training time and hence the
# total running time for this notebook
#init_params = [3.0, 1.0, 0.6, 1.6]

# You can increase the number of training epochs and use random initial parameters.
# init_params = np.random.rand(ansatz.num_parameters_settable) * 2 * np.pi
init_params = np.random.rand(ansatz.num_parameters_settable) * 2 * np.pi
#init_params = (2*np.random.rand(ansatz.num_parameters_settable)-1)*0.1

# Set generator circuit by adding the initial distribution infront of the ansatz
g_circuit = ansatz.compose(init_dist, front=True)
#print(g_circuit)
g_circuit.decompose().draw(output='mpl')

# Set quantum generator
qgan.set_generator(generator_circuit=g_circuit, generator_init_params=init_params)
# The parameters have an order issue that following is a temp. workaround
qgan._generator._free_parameters = sorted(g_circuit.parameters, key=lambda p: p.name)
# Set classical discriminator neural network
discriminator = NumPyDiscriminator(len(num_qubits))
qgan.set_discriminator(discriminator)

#######################################################
# Run qGAN
#######################################################
result = qgan.run(quantum_instance)

samples_g, prob_g = qgan.generator.get_output(qgan.quantum_instance, shots=10000)
samples_g = np.array(samples_g)

print("Training results:")
for key, value in result.items():
    print(f"  {key} : {value}")

from plotter import plotter
plotter_obj = plotter(num_epochs, qgan)

plotter_obj.plot_dist(real_data_round, bounds, samples_g, prob_g, len(num_qubits))
plotter_obj.plot_rel_entropy()
plotter_obj.plot_loss()

brkpnt1 = 1

