"""Independent dense/BFS checks of reductions and certificate hypotheses."""
import unittest

import networkx as nx
import numpy as np

from src.reachability_reference import inverse_closeness, reachability_readout
from src.reachability_theory import (barbell, path_copies, certificate_beta,
                                    centrality_error_bound, state_jacobian_bound, sigmoid)


class TheoryTests(unittest.TestCase):
    def test_barbell_reduction_matches_explicit_graph(self):
        for size in (2, 3, 6, 10):
            graph = nx.barbell_graph(size, 0)
            adjacency = nx.to_numpy_array(graph)
            for steps in (1, 2, 3, 6):
                for beta in (4., 8., 20.):
                    with self.subTest(size=size, steps=steps, beta=beta):
                        reduced = barbell(size, steps, beta)
                        prediction = np.repeat(reduced['prediction'], reduced['orbit_sizes'].astype(int))
                        np.testing.assert_allclose(prediction, inverse_closeness(adjacency, steps, beta=beta),
                                                   rtol=1e-12, atol=1e-12)
            exact = inverse_closeness(adjacency)
            np.testing.assert_allclose(exact, np.repeat(reduced['target'], reduced['orbit_sizes'].astype(int)))

    def test_path_reduction_matches_explicit_graph(self):
        for copies in (1, 2, 5, 7):
            graph = nx.disjoint_union_all([nx.path_graph(3) for _ in range(copies)])
            adjacency = nx.to_numpy_array(graph)
            for steps in (1, 2, 3, 5):
                for beta in (4., 8., 20.):
                    reduced = path_copies(copies, steps, beta)
                    np.testing.assert_allclose(np.tile(reduced['prediction'], copies),
                                               inverse_closeness(adjacency, steps, beta=beta),
                                               rtol=1e-12, atol=1e-12)

    def test_barbell_temperature_critical_window(self):
        size = 10**120
        critical = 2*np.log(2)
        for offset, expected in ((critical-1, .5), (critical, .25), (critical+1, 0)):
            result = barbell(size, 3, 2*np.log(float(size))+offset)
            self.assertAlmostEqual(result['mae'], expected, places=9)
        self.assertAlmostEqual(barbell(10**12, 3, 12)['mae'], .5, places=9)

    def test_path_state_error_and_readout_error_separate(self):
        copies = 10**120
        n = 3*float(copies)
        result = path_copies(copies, 2, 1.5*np.log(n))
        self.assertLess(max(result['max_state_errors']), 1e-80)
        self.assertAlmostEqual(result['mae'], 1/3, places=12)
        for offset in (-2., 0., 2.):
            result = path_copies(copies, 2, 2*np.log(n)+offset)
            leakage = np.exp(-offset/2)
            self.assertAlmostEqual(result['mae'], leakage/(3*(2+leakage)), places=12)
        self.assertLess(path_copies(copies, 2, 2.5*np.log(n))['mae'], 1e-12)

    def test_adversarial_perturbations_stay_inside_certificate(self):
        graphs = [nx.path_graph(8), nx.barbell_graph(5, 0),
                  nx.disjoint_union(nx.cycle_graph(6), nx.path_graph(3)),
                  nx.gnp_random_graph(12, .25, seed=18)]
        for graph in graphs:
            adjacency = nx.to_numpy_array(graph)
            closed = adjacency+np.eye(len(graph))
            degree = int(adjacency.sum(1).max())
            for eta in (0., .1, .3):
                delta = 1e-3
                beta = certificate_beta(degree, delta, eta)
                hidden, exact = np.eye(len(graph)), np.eye(len(graph))
                states = [hidden]
                for _ in range(20):
                    exact = (closed@exact > 0).astype(float)
                    # Perturb toward the wrong Boolean value at every step.
                    perturbation = eta*(1-2*exact)
                    hidden = sigmoid(beta*(closed@hidden+perturbation-.5))
                    states.append(hidden)
                    self.assertLessEqual(np.abs(hidden-exact).max(), delta+1e-14)
                prediction = reachability_readout(states, isolated_mask=adjacency.sum(1)==0)
                for idx, node in enumerate(graph):
                    distances = nx.single_source_shortest_path_length(graph, node)
                    m = len(distances)-1
                    if m:
                        target = sum(distances.values())/m
                        bound = centrality_error_bound(len(graph), m, 20, delta)
                        self.assertLessEqual(abs(prediction[idx]-target), bound+1e-12)

    def test_jacobian_bound_is_conditional_state_bound(self):
        adjacency = nx.to_numpy_array(nx.path_graph(5))
        closed = adjacency+np.eye(5)
        delta, eta, theta = .001, .1, 1.5
        beta = certificate_beta(2, delta, eta)
        hidden = np.eye(5)
        output = sigmoid(beta*(closed@hidden+eta*np.tanh(theta*hidden)-.5))
        bound = state_jacobian_bound(2, beta, delta, eta*abs(theta))
        for i in range(5):
            for j in range(5):
                derivative_sum = beta*output[i,j]*(1-output[i,j])*(
                    closed[i].sum()+eta*theta*(1-np.tanh(theta*hidden[i,j])**2))
                self.assertLessEqual(derivative_sum, bound+1e-12)

    def test_invalid_certificate_is_rejected(self):
        for args in ((10, .1, 0), (2, .01, .5), (2, 0, 0)):
            with self.assertRaises(ValueError): certificate_beta(*args)
        with self.assertRaises(ValueError): centrality_error_bound(100, 2, 3, .03)
        with self.assertRaises(ValueError): barbell(1, 3, 10)

    def test_robust_temperature_leading_coefficient(self):
        # Independently construct the four-orbit recurrence with adversarial
        # perturbations; check both sides of Theorem 4b's asymptotic boundary.
        s = 1e80
        counts = np.array([[s-1,1,0,0], [s-1,1,1,0],
                           [0,1,1,s-1], [0,0,1,s-1]])
        sizes = np.array([s-1,1,1,s-1])
        first_exact = (counts > 0).astype(float)
        for eta in (0., .1, .3):
            mu = .5-eta
            for factor in (.8, 1.2):
                beta = factor*np.log(s+1)/mu
                q = float(sigmoid(-beta*mu))
                hidden = np.where(first_exact > 0, 1-q, q)
                exact = first_exact.copy()
                sums = [np.ones(4), hidden@sizes]
                errors = [float(np.abs(hidden-exact).max())]
                for _ in range(2):
                    exact = (counts@exact > 0).astype(float)
                    hidden = sigmoid(beta*(counts@hidden+eta*(1-2*exact)-.5))
                    errors.append(float(np.abs(hidden-exact).max()))
                    sums.append(hidden@sizes)
                prediction = (3*sums[3]-sum(sums[:3]))/(sums[3]-1)
                if factor < 1:
                    self.assertGreater(errors[1], .99)
                    self.assertAlmostEqual(prediction[0], 1.5, places=10)
                else:
                    self.assertLess(max(errors), 1e-50)
                    self.assertAlmostEqual(prediction[0], 2., places=10)
                    delta = 2*q
                    self.assertGreaterEqual(beta, certificate_beta(int(s), delta, eta))

    def test_bounded_residual_lower_bound_attained(self):
        s = 1e80
        counts = np.array([[s-1,1,0,0], [s-1,1,1,0],
                           [0,1,1,s-1], [0,0,1,s-1]])
        sizes = np.array([s-1,1,1,s-1])
        for eta in (0., .1, 1., 3.):
            beta = 4.
            hidden = sigmoid(beta*((counts>0).astype(float)-eta-.5))
            sums = [np.ones(4), hidden@sizes]
            for _ in range(2):
                hidden = sigmoid(beta*(counts@hidden-eta-.5))
                sums.append(hidden@sizes)
            prediction = (3*sums[3]-sum(sums[:3]))/(sums[3]-1)
            lower = float((sigmoid(beta*(.5-eta))+sigmoid(-beta*(.5+eta)))/2)
            self.assertAlmostEqual(2-prediction[0], lower, places=12)


if __name__ == '__main__':
    unittest.main()
