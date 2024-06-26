"""
About
=====

This Python script implements the analytical estimates for directional
flow induced by peristalsis in perivascular networks described in:

  Gjerde, Ingeborg G., Marie E. Rognes, and Antonio L. Sánchez. "The
  directional flow generated by peristalsis in perivascular
  networks—Theoretical and numerical reduced-order descriptions."
  Journal of Applied Physics 134.17 (2023).
  
The code is based on an original implementation by Gjerde and Rognes
available at https://github.com/scientificcomputing/perivascular-peristalsis (extracted Feb 1 2024).

Dependencies
============

* Python3
* NumPy

How to run
==========

$ python3 pvs_network_netflow.py # Runs tests

or in a Python script:

  from . import estimate_net_flow
  network = ... # Define input data, as per below
  mean_flow_rates, mean_velocities = estimate_net_flow(network)

Sharing and copyright
=====================

This script is licensed under the MIT License (see LICENSE).

More information
================

Data structures for the network and parameters:

*Convention*: All nodes n (except orphan nodes) is numbered equal to
 its mother edge/element e.

* indices (list): list of three-tuples representing the network via
  its bifurcations. For each bifurcation/junction, give the mother
  (e0) and daughter edge indices (e1, e2) as (e0, e1, e2).

* paths (list): list of root-to-leaf-paths, one tuple (n0, n1, n2,
  ..., nl) for each leaf node where n0 is the root node, nl is the
  leaf node, and n1, n2, ... are the bifurcation nodes on the path
  between n0 and nl.

The remaining parameters can be given in any dimensions, but sample
units are given here for illustration purposes:

* f (float):             Wave frequency (Hz)
* omega (float):         Angular frequency (Hz)
* lmbda (float):         Wave length (mm)
* varepsilon (float):    Relative wave amplitude (AU)
* r_o (list of floats):  Inner radii (mm) for each element/edge
* r_e (list of floats):  Outer radii (mm) for each element/edge
* L (list of floats):    Length (mm) of each element/edge

Idealized networks are included for testing

"""

import numpy as np
import numpy.linalg

DEBUG_MODE = False
def debug(x):
    if DEBUG_MODE:
        print(x)
    
# Bunch of helper functions for ensuring close match between code and
# paper presentation.
def _invR(beta):
    "Helper function for evaluating \mathcal{R}^{-1}(beta), eq. (25)."
    val = np.pi/8*(beta**4 - 1 - (beta**2 - 1)**2/np.log(beta))
    return val

def _R(beta):
    "Helper function for evaluating \mathcal{R}(beta)."
    return 1.0/_invR(beta)

def _delta(beta):
    "Helper function for evaluating Delta(beta), eq. (26)"
    d_dividend = (2 - (beta**2-1)/np.log(beta))**2
    d_divisor = beta**4 - 1 - (beta**2 - 1)**2/np.log(beta)
    delt = d_dividend/d_divisor
    return delt
    
def _beta(r_e, r_o):
    "beta is the ratio of outer-to-inner radius (r_e/r_o) for an annular PVS cross-section."
    return r_e/r_o

def _xi(l):
    "Helper function for expressions in eq. (38)."
    z = 1j
    xi1 = (np.exp(z*l) - 1)/l
    return xi1

def _alpha(l, P, R):
    "Helper function for expressions in eq. (39)."
    z = 1j
    A1 = (0.5 - (1 - np.cos(l))/(l**2))
    A2 = P*(1 - np.exp(z*l))/(2*l**2*R)
    return (A1 + A2.real)

def avg_Q_1_n(P, dP, ell, R, Delta):  
    "Evaluate <Q_1_n> as per eq. (34)"
    z = 1j
    val = (- dP/(R*ell) + Delta*(1./2 - (1 - np.cos(ell))/(ell**2)) 
           + Delta/(2*ell**2*R)*(P*(1 - np.exp(z*ell))).real)
    return val

def solve_for_P(indices, paths, R, ell, gamma):
    """Main function #1: Solving for the P_i, eq. (38)."""
    
    # Recall: A bifurcating tree has N junctions with n = 2N + 1
    # edges. The number of junctions N plus number of downstream ends
    # (N+1) is also 2N + 1.

    # Form the n x n system of linear equations for determining P: A P
    # = b (complex):
    n = len(R)
    A = np.zeros((n, n), dtype=complex)
    b = np.zeros(n, dtype=complex)
    
    # The complex number i
    z = 1j

    for (I, (i, j, k)) in enumerate(indices):

        # Set the correct matrix columns for this junction constraint, eq. (38)
        A[I, i] = np.exp(z*ell[i])*gamma[i]/(R[i]*ell[i])
        A[I, j] = - gamma[j]/(R[j]*ell[j])
        A[I, k] = - gamma[k]/(R[k]*ell[k])

        # Set the correct vector row for this junction constraint
        b[I] = gamma[i]*_xi(ell[i]) - gamma[j]*_xi(-ell[j]) \
            - gamma[k]*_xi(-ell[k]) + z*(- gamma[i] + gamma[j] + gamma[k])

    # Apply the additional constraints given in terms of the
    # root-to-leaf paths
    I = len(indices)
    for (k, path) in enumerate(paths):
        x_n = 0.0
        for n in path:
            A[I+k, n] = np.exp(-z*x_n)/gamma[n] # - compared to eq. (40)
            x_n += ell[n]

    # Solve the linear systems for real and imaginary parts of P:
    P = np.linalg.solve(A, b)
    return P

def solve_for_dP(R, ell, Delta, gamma, indices, paths, P):
    "Main function #2: Solving for the dP_i."

    # n x n system of linear (real) equations for determining dP: A dP = b 
    n = len(P)
    A = np.zeros((n, n))
    b = np.zeros(n)
    
    # Define the real linear system A P = b 
    for (I, (i, j, k)) in enumerate(indices):
        
        # Set right matrix columns for this junction constraint
        A[I, i] = gamma[i]/(R[i]*ell[i])
        A[I, j] = - gamma[j]/(R[j]*ell[j])
        A[I, k] = - gamma[k]/(R[k]*ell[k])

        # Set right vector row for this junction constraint
        b[I] = (gamma[i]*Delta[i]*_alpha(ell[i], P[i], R[i]) 
                - gamma[j]*Delta[j]*_alpha(ell[j], P[j], R[j])
                - gamma[k]*Delta[k]*_alpha(ell[k], P[k], R[k]))

    # Apply additional constraints given in terms of the root-to-leaf paths
    I = len(indices)
    for (k, path) in enumerate(paths):
        for n in path:
            A[I+k, n] = 1.0/gamma[n]
            
    # Solve the linear systems for dP:
    dP = np.linalg.solve(A, b)
    
    return dP

def solve_bifurcating_tree(network):

    # Extract individual parameters from argument list (for readability)
    (indices, paths, r_o, r_e, L, k, omega, varepsilon) = network

    # Compute scaled parameters for solver functions
    beta = [_beta(re, ro) for (re, ro) in zip(r_e, r_o)]
    ell = [k*l for l in L]

    # Computation of dimension-less parameters
    R = [_R(b) for b in beta]
    Delta = [_delta(b) for b in beta]
    gamma = [(r_o_n/r_o[0])**2 for r_o_n in r_o]
    
    debug("Solving for P")
    P = solve_for_P(indices, paths, R, ell, gamma)

    debug("Solving for dP")
    dP = solve_for_dP(R, ell, Delta, gamma, indices, paths, P)

    debug("Evaluating non-dimensional average flow rates <Q_1_n>")
    avg_Q_1 = np.zeros(len(dP))
    for (n, _) in enumerate(dP):
        avg_Q_1[n] = avg_Q_1_n(P[n], dP[n], ell[n], R[n], Delta[n])

    return (P, dP, avg_Q_1)

def Qprime(Q, varepsilon, omega, k, r_o):
    """Dimensionalize Q to yield Q', see eq. (5)."""
    alpha = 2*np.pi*varepsilon*omega*np.power(r_o, 2)/k
    val = alpha*Q
    return val

def avg_u(Q, r_e, r_o):
    """Compute the average cross-section velocity for a given flux Q,
under the assumption of an annular cross-section defined by inner radius r_o and outer radius r_e."""

    # Definition of annular cross-section area
    A = np.pi*(np.power(r_e, 2) - np.power(r_o, 2))

    # Mean velocity is flux divided by area
    return Q/A
    
def estimate_net_flow(network):

    """Estimate the net directional flow in a given bifurcating tree. Input parameters should be such that

      (indices, paths, r_o, r_e, L, k, omega, varepsilon) = network

    """
    
    # Compute the non-dimensional, first-order contribution to the
    # average mean flow avg_Q_1 (P, dP are auxiliary byproducts)
    (P, dP, avg_Q_1) = solve_bifurcating_tree(network)

    # Extract individual parameters from argument list (for readability)
    (indices, paths, r_o, r_e, L, k, omega, varepsilon) = network

    # Mean flow rates in each branch n (see text after eq. (34))
    avg_Q = varepsilon*avg_Q_1

    # Dimensionalize <Q>_n to yield <Q'>_n for each branch n, see eq. (5)
    avg_Q_prime = Qprime(avg_Q, varepsilon, omega, k, r_o)

    avg_u_prime = avg_u(avg_Q_prime, r_e, r_o)
    
    return avg_Q_prime, avg_u_prime

# ==============================================================================
# Helper functions for setting up idealized networks including:
# 
# * single_bifurcation_data
# * three_junction_data
# * murray_tree_data
#
# ==============================================================================

def single_bifurcation_data(lmbda=1.0):
    """Data for a single bifurcation test case. Data for network
    consisting of three edges and a single bifurcation, with inner and
    outer radius and lengths of the daughter vessels identical to
    those of the mother vessel. 

    Test case used for Table 1c (left) in Gjerde, Sanchez, Rognes
    (2023), parametrized by peristaltic wavelength lmbda (given in
    mm).

    """
    indices = [(0, 1, 2),]
    paths = [(0, 1), (0, 2)]
    
    # Peristaltic wave parameters: wave length lmbda and (angular) wave number k
    f = 1.0                 # frequency (Hz = 1/s)
    omega = 2*np.pi*f       # Angular frequency (Hz)
    k = 2*np.pi/lmbda       # wave number (1/mm)
    varepsilon = 0.1        # AU 
        
    r_o = [0.1, 0.1, 0.1]  # Inner radii (mm) for each element/edge
    r_e = [0.3, 0.2, 0.2]  # Outer radii (mm) for each element/edge
    L = [2.0, 1.0, 1.0]        # Element lengths (mm)
        
    data = (indices, paths, r_o, r_e, L, k, omega, varepsilon)
    return data
    
def three_junction_data():
    """Data for a three bifurcation test case. Data for network consisting
    of seven edges and three bifurcation. Inner and outer radius of
    the daughter vessels are identical to that of the mother vessel in
    order to compare with analytical expression.

    """
    
    indices = [(0, 1, 2), (1, 3, 4), (2, 5, 6)]
    paths = [(0, 1, 3), (0, 1, 4), (0, 2, 5), (0, 2, 6)]
    
    f = 1.0                 # frequency (Hz = 1/s)
    omega = 2*np.pi*f       # Angular frequency (Hz)
    lmbda = 2.0             # mm    
    k = 2*np.pi/lmbda       # wave number (1/mm)
    varepsilon = 0.1        # AU 
    ro0 = 0.1               # Base inner radius (mm)
    re0 = 0.2               # Base outer radius (mm)
        
    r_o = [ro0, ro0, ro0, ro0, ro0, ro0, ro0]  # Inner radii (mm) for each element/edge
    r_e = [re0, re0, re0, re0, re0, re0, re0]  # Outer radii (mm) for each element/edge
    L = [1.0, 0.5, 0.5, 0.25, 0.25, 0.25, 0.25]    # Element lengths (mm)
        
    data = (indices, paths, r_o, r_e, L, k, omega, varepsilon)
    return data
    
def murray_tree_data(m=1, r=0.1, gamma=1.0, beta=2.0, L0=10):
    """Generate a Murray tree of m (int) generations. 

    Default corresponds to the single bifurcation test data.

    Starting one branch for m = 0. Given r is the radius of the mother
    branch. Radii of daughter branches (j, k) with parent i is
    goverend by gamma, such that

    r_o_j + r_o_k = r_o_i.

    and

    r_o_j/r_o_k = gamma

    i.e 

    r_o_j = gamma r_o_k
    """
    
    # Compute the number of elements/branches
    N = int(sum([2**i for i in range(m+1)]))

    # Create the generations
    generations = [[0,]]
    for i in range(1, m+1):
        _N = int(sum([2**j for j in range(i)]))
        siblings = list(range(_N, _N + 2**i))
        generations.append(siblings)

    # Create the indices (bifurcations)
    indices = []
    for (i, generation) in enumerate(generations[:-1]):
        children = generations[i+1]
        for (j, parent) in enumerate(generation):
            (sister, brother) = (children[2*j], children[2*j+1])
            indices += [(parent, sister, brother)]
    n = len(indices)
    assert N == (2*n+1), "N = 2*n+1"

    # Iterate through generations from children and upwards by
    # reversing the generations list for creating the paths
    generations.reverse()

    # Begin creating the paths
    ends = generations[0]
    paths = [[i,] for i in ends]
    for (i, generation) in enumerate(generations[1:]):
        for (j, path) in enumerate(paths):
            end_node = path[0]
            index = j // (2**(i+1))
            path.insert(0, generations[i+1][index])

    # Reverse it back
    generations.reverse()

    # Create inner radii based on Murray's law with factor gamma
    r_o = numpy.zeros(N)
    r_o[0] = r  
    for (g, generation) in enumerate(generations[1:]):
        for index in generation[::2]:
            ri = r_o[0]/(2**g)
            rk = 1/(1+gamma)*ri
            rj = gamma*rk
            r_o[index] = rk
            r_o[index+1] = rj
            
    L = L0*r_o
    r_e = beta*r_o
    
    # Peristaltic wave parameters: wave length lmbda and (angular) wave number k
    f = 1.0                 # frequency (Hz = 1/s)
    omega = 2*np.pi*f          # Angular frequency (Hz)
    lmbda = 2.0             # mm    
    k = 2*np.pi/lmbda          # wave number (1/mm)
    varepsilon = 0.1        # AU 
    
    data = (indices, paths, r_o, r_e, L, k, omega, varepsilon)
    return data

def run_single_bifurcation_test(lmbda):

    print("Running single bifurcation test case via general solution algorithm")
    data = single_bifurcation_data(lmbda=lmbda)
    (indices, paths, r_o, r_e, L, k, omega, varepsilon) = data
    
    (P, dP, avg_Q_1) = solve_bifurcating_tree(data)

    print("P = ", P)
    print("dP = ", dP)
    print("<Q_1> = ", avg_Q_1)
    print("eps*<Q_1_0> = %.3e" % (varepsilon*avg_Q_1[0]))
    
    Q10 = varepsilon*avg_Q_1[0]
    Qp = Qprime(Q10, varepsilon, omega, k, r_o[0])
    print("eps*<Q_1_0>' (mm^3/s) = %.3e" % Qp)

    return Qp

def simple_asymmetric_data(lmbda=1.0):
    """Data for a simple asymetric test case. Data for network consisting
    of 5 edges and two bifurcation, with inner and outer radius and
    lengths of the daughter vessels identical to those of the mother
    vessel.
    """
    indices = [(0, 1, 2), (2, 3, 4)]
    paths = [(0, 1), (0, 2, 3), (0, 2, 4)]
    r_o = [0.1, 0.1, 0.1, 0.1, 0.1]
    r_e =  [0.2, 0.2, 0.2, 0.2, 0.2]
    L = [1.0, 1.0, 1.0, 1.0, 1.0]
    
    # Peristaltic wave parameters: wave length lmbda and (angular) wave number k
    f = 1.0                 # frequency (Hz = 1/s)
    omega = 2*np.pi*f       # Angular frequency (Hz)
    k = 2*np.pi/lmbda       # wave number (1/mm)
    varepsilon = 0.1        # AU 
        
    data = (indices, paths, r_o, r_e, L, k, omega, varepsilon)
    return data

def run_simple_asymmetric_test():

    print("Running simple asymmetric test case via general solution algorithm")
    data = simple_asymmetric_data(lmbda=0.1)
    (indices, paths, r_o, r_e, L, k, omega, varepsilon) = data
    
    (P, dP, avg_Q_1) = solve_bifurcating_tree(data)

    print("P = ", P)
    print("dP = ", dP)
    print("<Q_1> = ", avg_Q_1)
    print("eps*<Q_1_0> = %.3e" % (varepsilon*avg_Q_1[0]))
    
    Q10 = varepsilon*avg_Q_1[0]
    Qp = Qprime(Q10, varepsilon, omega, k, r_o[0])
    print("eps*<Q_1_0>' (mm^3/s) = %.3e" % Qp)

    return True

def test_murray_data():

    debug("Murray tree data (m=1)")
    data = murray_tree_data(m=1, r=0.1, gamma=1.0, beta=2.0, L0=10)
    #debug(data)

    debug("Single bifurcation data")
    data = single_bifurcation_data()
    #debug(data)

    debug("Murray tree data (m=2)")
    data = murray_tree_data(m=2, r=0.1, gamma=1.0, beta=2.0, L0=10)
    #debug(data)

    debug("Three junction data")
    data = three_junction_data()
    #debug(data)

    debug("Murray tree data (m=3)")
    data = murray_tree_data(m=3, r=0.1, gamma=1.0, beta=2.0, L0=10)
    #debug(data)

    return True
    
def run_murray_tree():
    print("Solving Murray tree.")

    data = murray_tree_data(m=2, r=0.1, gamma=1.0, beta=2.0, L0=10)
    (indices, paths, r_o, r_e, L, k, omega, varepsilon) = data

    (P, dP, avg_Q_1) = solve_bifurcating_tree(data)
    
    print("P = ", P)
    print("dP = ", dP)
    print("<Q_1> = ", avg_Q_1)
    print("eps*<Q_1_0> = %.3e" % (varepsilon*avg_Q_1[0]))
    
    Q10 = varepsilon*avg_Q_1[0]
    Qp = Qprime(Q10, varepsilon, omega, k, r_o[0])
    print("eps*<Q_1_0>' (mm^3/s) = %.3e" % Qp)

    return True

def simple_schematic_data(alpha=1.0):
    """Data for a single bifurcation test case. Data for network
    consisting of three edges and a single bifurcation, with varying
    thickness and length between the branches. Scale length by alpha
    """
    indices = [(0, 1, 2),]
    paths = [(0, 1), (0, 2)]
    
    # Peristaltic wave parameters: wave length lmbda and (angular) wave number k
    lmbda = 10.0
    f = 1.0                 # frequency (Hz = 1/s)
    omega = 2*np.pi*f       # Angular frequency (Hz)
    k = 2*np.pi/lmbda       # wave number (1/mm)
    varepsilon = 0.1        # AU 

    beta=3.0
    r_o = [0.3, 0.2, 0.1]     # Inner radii (mm) for each element/edge
    r_e = [beta*r for r in r_o]  # Outer radii (mm) for each element/edge
    L = [alpha*l for l in [2.0, 1.0, 2.0]]       # Element lengths (mm)
        
    data = (indices, paths, r_o, r_e, L, k, omega, varepsilon)
    return data
    
def test_estimate_net_flow():

    print("Running simple schematic test case via estimate_net_flow")
    data = simple_schematic_data(1.0)
    
    avg_Q_prime, avg_u_prime = estimate_net_flow(data)

    print("<Q'>_n (mm^3/s) = ", avg_Q_prime)
    print("<u'>_n (mm/s) = ", avg_u_prime)

def test():
    print("")
    success = run_simple_asymmetric_test()

    # Compare against published reference including  numerical tests
    print("")
    Qp = run_single_bifurcation_test(0.1)
    diff = (Qp - 1.341e-04)/1.341e-04
    assert(diff < 0.001), "WARNING: Check correctness of results! (%r)" % diff
    Qp = run_single_bifurcation_test(1.0)
    diff = (Qp - 1.341e-03)/1.341e-03
    assert(diff < 0.001), "WARNING: Check correctness of results! (%r)" % diff

    print("")
    success = test_murray_data()

    print("")
    success = run_murray_tree()

    print("")
    success = test_estimate_net_flow()

if __name__ == "__main__":
    test()
