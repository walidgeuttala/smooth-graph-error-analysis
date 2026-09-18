"""Independent BFS oracle and counterexamples for the paper's recurrence."""

import unittest

import networkx as nx
import numpy as np

from src.reachability_reference import inverse_closeness, reachability_states


def bfs_target(graph):
    values = []
    for node in graph.nodes:
        distances = nx.single_source_shortest_path_length(graph, node)
        other = [distance for target, distance in distances.items() if target != node]
        values.append(sum(other) / len(other) if other else 0.0)
    return np.asarray(values)


class ReachabilityReferenceTests(unittest.TestCase):
    def test_boolean_matches_independent_bfs(self):
        graphs = [nx.empty_graph(0), nx.empty_graph(1), nx.empty_graph(5),
                  nx.path_graph(8), nx.cycle_graph(9), nx.complete_graph(6),
                  nx.disjoint_union(nx.path_graph(5), nx.empty_graph(2))]
        graphs.extend(nx.gnp_random_graph(15, p, seed=seed)
                      for p in (0.05, 0.2, 0.7) for seed in range(4))
        for graph in graphs:
            with self.subTest(nodes=len(graph), edges=graph.number_of_edges()):
                np.testing.assert_allclose(inverse_closeness(nx.to_numpy_array(graph)),
                                           bfs_target(graph), atol=1e-14)

    def test_truncation_is_mean_of_discovered_distances(self):
        a = nx.to_numpy_array(nx.path_graph(7))
        result = inverse_closeness(a, steps=2)
        self.assertAlmostEqual(result[0], 1.5)
        self.assertAlmostEqual(inverse_closeness(a)[0], 3.5)
        self.assertTrue(np.all(result < inverse_closeness(a)))

    def test_permutation_equivariance_hard_and_soft(self):
        a = nx.to_numpy_array(nx.gnp_random_graph(9, 0.3, seed=12))
        permutation = np.random.default_rng(8).permutation(len(a))
        for beta in (None, 20.0):
            original = inverse_closeness(a, beta=beta)
            relabeled = inverse_closeness(a[np.ix_(permutation, permutation)], beta=beta)
            np.testing.assert_allclose(relabeled, original[permutation], atol=1e-12)

    def test_legacy_threshold_disproves_exact_recovery(self):
        a = nx.to_numpy_array(nx.path_graph(2))
        for beta in (10.0, 100.0, 1000.0):
            states = reachability_states(a, 1, beta=beta, threshold=1.0)
            np.testing.assert_array_equal(states[1], np.full((2, 2), 0.5))
            # The manuscript's epsilon convention produces zero, not truth=1.
            np.testing.assert_array_equal(
                inverse_closeness(a, 1, beta=beta, threshold=1.0, epsilon=1e-8),
                np.zeros(2))
        with self.assertRaises(ValueError):
            inverse_closeness(a, 1, beta=100, threshold=1.0)

    def test_corrected_soft_threshold_approaches_bfs(self):
        graph = nx.disjoint_union(nx.path_graph(8), nx.empty_graph(1))
        a = nx.to_numpy_array(graph)
        np.testing.assert_allclose(inverse_closeness(a, beta=100), bfs_target(graph), atol=1e-12)
        # A finite, less steep relaxation is not claimed to be exactly BFS.
        self.assertGreater(np.max(np.abs(inverse_closeness(a, beta=10) - bfs_target(graph))), 1e-6)

    def test_boolean_saturates_after_component_diameter(self):
        a = nx.to_numpy_array(nx.path_graph(5))
        states = reachability_states(a, 8)
        np.testing.assert_array_equal(states[4], states[8])
        np.testing.assert_allclose(inverse_closeness(a, 4), inverse_closeness(a, 8))

    def test_invalid_inputs_and_zero_horizon(self):
        for invalid in (np.ones((2, 3)), [[0, 1], [0, 0]], [[1]], [[0, 0.5], [0.5, 0]]):
            with self.assertRaises(ValueError):
                inverse_closeness(invalid)
        with self.assertRaises(ValueError):
            inverse_closeness([[0, 1], [1, 0]], steps=0)
        np.testing.assert_array_equal(inverse_closeness(np.zeros((2, 2)), steps=0), [0, 0])


if __name__ == "__main__":
    unittest.main()
