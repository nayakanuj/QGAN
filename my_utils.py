import copy
import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import AerSimulator

# Import from Qiskit Aer noise module
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise import QuantumError, ReadoutError
from qiskit.providers.aer.noise import pauli_error
from qiskit.providers.aer.noise import depolarizing_error
from qiskit.providers.aer.noise import thermal_relaxation_error


def bound_data(real_data, bounds):
    # real_data_round = copy.deepcopy(real_data)
    real_data_round = []
    if len(bounds.shape) == 1:
        for ind in range(len(real_data)):
            if real_data[ind] > bounds[1] + 0.5:
                continue  # real_data_round[ind] = bounds[1]
            elif real_data[ind] < bounds[0] - 0.5:
                continue  # real_data_round[ind] = bounds[0]
            else:
                real_data_round.append(round(real_data[ind]))
                # real_data_round[ind] = real_data[ind]
        real_data_round = np.array(real_data_round)
    else:
        for ind_dim in range(len(bounds)):
            for ind in range(len(real_data[:, ind_dim])):
                if real_data[ind, ind_dim] > bounds[ind_dim, 1]:
                    real_data_round[ind, ind_dim] = bounds[ind_dim, 1]
                elif real_data[ind, ind_dim] < bounds[ind_dim, 0]:
                    real_data_round[ind, ind_dim] = bounds[ind_dim, 0]
                else:
                    real_data_round[ind, ind_dim] = round(real_data[ind, ind_dim])
                    # real_data_round[ind, ind_dim] = real_data[ind, ind_dim]

    return real_data_round


def save_res(save_filename, num_epochs, real_data_round, bounds, samples_g, prob_g, num_dim, rel_entr, g_loss, d_loss):
    np.savez(save_filename, num_epochs=num_epochs, real_data_round=real_data_round, bounds=bounds, samples_g=samples_g,
             prob_g=prob_g, num_dim=num_dim, rel_entr=rel_entr, g_loss=g_loss, d_loss=d_loss)
    return


def save_plots():
    pass


def get_init_params(ckt_depth, num_qubits):
    init_params = None
    if ckt_depth == 3 and num_qubits[0] == 3:
        # init_params = np.array([-0.07003108647808957,
        #                0.09197932889888075,
        #                0.005516208006221746,
        #                0.01586714796957971,
        #                0.024541510544921555,
        #                -0.013265246579549328,
        #                0.07542876050200559,
        #                -0.07540192316749171,
        #                0.0607616759482843,
        #                -0.08812957585509607,
        #                -0.003737006980968838,
        #                -0.010205733348182111]) # works well - wide mixture of Gaussians

        init_params = np.array([-0.006603370080803847,
                                0.13707048162815721,
                                0.01632777760110912,
                                0.07105380804307698,
                                0.06450801647281212,
                                0.06938057219611285,
                                0.17364277980005394,
                                0.018231022773473937,
                                0.09717326981672378,
                                -0.01632267904906194,
                                0.05281204913075593,
                                0.08347748684910486])#-0.02  # narrow mixture of Gaussians

        #init_params += np.random.rand(len(init_params)) * 0.07

        # init_params = np.ones(init_params.shape)
        # init_params -= 0.05
        # # one more option - entanglement_map = [[0, 2], [1, 2]]
        # 0.07856141188850466
        # 0.01456309201874948
        # 0.06750739975255825
        # -0.04600938837508337
        # 0.07198297556646598
        # -0.08857422983222346
        # 0.09342560398752298
        # -0.074947923032872
        # 0.024163863440186263
        # 0.031788611505715526
        # 0.08576708557706073
        # 0.05378239459936525
    elif ckt_depth == 2 and num_qubits[0] == 3:
        # init_params = [0.055975456796131855,
        #                0.08370096718371078,
        #                0.0817910830029216,
        #                0.036798741206601364,
        #                0.012055491412813635,
        #                0.07771387568521687,
        #                0.008011976864023796,
        #                -0.04298936900270938,
        #                0.0388253291900059] # 2nd best

        init_params = np.array([-0.023795802736555374,
                                0.08732582974491965,
                                0.0021732017290951866,
                                0.09138854653380664,
                                0.08109199547971827,
                                0.016581968204232744,
                                0.05556760287966161,
                                0.027333722725535206,
                                0.07860410871441154])  # Best option
        init_params += np.random.rand(len(init_params)) * 0.2

    elif ckt_depth == 1 and num_qubits[0] == 3:
        init_params = np.array([0.06755085193219557,
                                -0.07424685993934532,
                                0.0981186061833168,
                                0.07526282855255213,
                                0.009301221283617878,
                                -0.028679546234033085])  # best option
        # init_params = np.array([0.08755085193219557,
        #                         -0.07424685993934532,
        #                         0.0981186061833168,
        #                         0.07526282855255213,
        #                         0.009301221283617878,
        #                         -0.028679546234033085])
        init_params += np.random.rand(len(init_params)) * 0.3
    elif ckt_depth == 4 and num_qubits[0] == 3:
        # init_params = np.array([0.04834643106384673,
        #                         -0.0768409260238931,
        #                         0.03981593964240935,
        #                         0.07655019605325804,
        #                         -0.030589737276267993,
        #                         0.05194571824059831,
        #                         0.07408029659040885,
        #                         -0.0509221986058954,
        #                         0.01034197081583923,
        #                         0.06953266485970643,
        #                         0.0983669904591638,
        #                         -0.09109144754141647,
        #                         -0.008387379093382252,
        #                         -0.06160166052497223,
        #                         0.08582806191656896]) # 3rd best
        # init_params = np.array([0.07367080514866796,
        #                         -0.17685929642020456,
        #                         0.05663342524747918,
        #                         0.13007775444198083,
        #                         -0.08418223045699748,
        #                         0.0808929824350778,
        #                         0.12515578743464517,
        #                         -0.12484285124500862,
        #                         0.046952988633992686,
        #                         0.16511025400214546,
        #                         0.17373552688028682,
        #                         -0.20519358109370572,
        #                         -0.03977875696374927,
        #                         -0.14620177508760887,
        #                         0.20672411165937452]) # 2nd Best option

        init_params = np.array([0.08070634699734354,
                                -0.16946703471473173,
                                0.06364279822729892,
                                0.13712928471460173,
                                -0.07716297507249743,
                                0.08790235540040847,
                                0.1321775055668289,
                                -0.11783347827808403,
                                0.08328726922000317,
                                0.2008898885217805,
                                0.18074489984763173,
                                -0.19816014538231197,
                                -0.03275592366889741,
                                -0.13919240211976863,
                                0.30710109522364587])

        #init_params += 0.02
    else:
        return

    return init_params

def get_noise_model(p_noise):
    # p_reset = 0.0001
    # p_meas = 0.0001
    # p_gate1 = 0.0001 #p_noise

    # p_reset = 0.5
    # p_meas = 0.5
    # p_gate1 = 0.5  # p_noise

    # p_reset = 0.0
    # p_meas = 0.0
    # p_gate1 = 0.5  # p_noise

    p_reset = 0.0
    p_meas = 0.0
    p_gate1 = p_noise

    error_reset = pauli_error([('X', p_reset), ('I', 1 - p_reset)])
    error_meas = pauli_error([('X', p_meas), ('I', 1 - p_meas)])
    error_gate1 = pauli_error([('X', p_gate1), ('I', 1 - p_gate1)])
    error_gate2 = error_gate1.tensor(error_gate1)

    # Add errors to noise model
    noise_bit_flip = NoiseModel()
    noise_bit_flip.add_all_qubit_quantum_error(error_reset, "reset")
    noise_bit_flip.add_all_qubit_quantum_error(error_meas, "measure")
    noise_bit_flip.add_all_qubit_quantum_error(error_gate1, ["ry","rz","rx","u1","u2","u3"])
    noise_bit_flip.add_all_qubit_quantum_error(error_gate2, ["cz","cy","cx"])

    return noise_bit_flip
