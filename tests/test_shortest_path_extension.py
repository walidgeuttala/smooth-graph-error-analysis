import unittest
import itertools
import numpy as np
import networkx as nx
from src.shortest_path_extension import (trajectory_distances, soft_bellman_ford,
                                          distance_outputs, error_metrics,
                                          differentiable_soft_bellman_ford)
from src.reachability_reference import reachability_states


class ExtensionTests(unittest.TestCase):
    def test_boolean_full_matrix_matches_bfs(self):
        for g in (nx.path_graph(7), nx.barbell_graph(4, 0),
                  nx.disjoint_union(nx.path_graph(3), nx.empty_graph(2))):
            d = np.asarray(nx.floyd_warshall_numpy(g))
            h = reachability_states(nx.to_numpy_array(g), len(g)-1)
            np.testing.assert_array_equal(trajectory_distances(h, np.isfinite(d)), d)

    def test_weighted_hard_matches_dijkstra_and_floyd(self):
        g = nx.gnp_random_graph(9, .3, seed=4)
        for i, (u, v) in enumerate(g.edges): g[u][v]['weight'] = .2+(i % 7)/3
        w = nx.to_numpy_array(g, nonedge=np.inf)
        d, _, _ = soft_bellman_ford(w, len(g)-1)
        np.testing.assert_allclose(d, nx.floyd_warshall_numpy(g))
        for u, targets in nx.all_pairs_dijkstra_path_length(g):
            for v, cost in targets.items(): self.assertAlmostEqual(d[u,v], cost)

    def test_soft_partition_matches_enumerated_walks(self):
        w = np.array([[np.inf, 1., 3.], [1., np.inf, 1.], [3., 1., np.inf]])
        for tau in (.1, .7):
            for normalize in (False, True):
                d, _, counts = soft_bellman_ford(w, 4, tau, normalize)
                for s, t in itertools.permutations(range(3), 2):
                    costs = []
                    for k in range(1,5):
                        for middle in itertools.product(range(3), repeat=k-1):
                            path = (s,)+middle+(t,)
                            cost = sum(w[a,b] for a,b in zip(path,path[1:]))
                            if np.isfinite(cost): costs.append(cost)
                    expected = -tau*np.logaddexp.reduce(-np.array(costs)/tau)
                    if normalize: expected += tau*np.log(len(costs))
                    self.assertAlmostEqual(d[s,t],expected)
                    self.assertAlmostEqual(counts[s,t],np.log(len(costs)))

    def test_mean_stability_and_entropy_bounds(self):
        w = np.full((6,6),1.); np.fill_diagonal(w,np.inf)
        hard, _, _ = soft_bellman_ford(w,5)
        for tau in (.01,.1,1.):
            soft, _, count = soft_bellman_ford(w,5,tau)
            normalized, _, _ = soft_bellman_ford(w,5,tau,True)
            mask = ~np.eye(6,dtype=bool)
            self.assertTrue(np.all(hard[mask]-soft[mask] <= tau*count[mask]+1e-12))
            self.assertTrue(np.all(normalized[mask] >= hard[mask]-1e-12))
            metrics=error_metrics(soft,hard)
            self.assertLessEqual(metrics['mean_max'],metrics['distance_max']+1e-12)

    def test_shrinking_triangle_reciprocal_instability(self):
        for n in (30,300,3000):
            a=1/n; w=np.array([[np.inf,a,2*a],[a,np.inf,a],[2*a,a,np.inf]])
            exact,_,_=soft_bellman_ford(w,2)
            soft,_,_=soft_bellman_ford(w,2,1/n**2)
            m=error_metrics(soft,exact)
            self.assertAlmostEqual(m['distance_max']*n*n,np.log(2),places=7)
            self.assertAlmostEqual(m['closeness_mae'],4*np.log(2)/27,delta=.003)

    def test_triangle_reduction_matches_explicit_components(self):
        n=9;a=1/n
        block=np.array([[np.inf,a,2*a],[a,np.inf,a],[2*a,a,np.inf]])
        w=np.full((n,n),np.inf)
        for i in range(0,n,3):w[i:i+3,i:i+3]=block
        for normalized in (False,True):
            whole,_,_=soft_bellman_ford(w,2,1/n**2,normalized)
            local,_,_=soft_bellman_ford(block,2,1/n**2,normalized)
            exact,_,_=soft_bellman_ford(w,2)
            target,_,_=soft_bellman_ford(block,2)
            for key,value in error_metrics(local,target).items():
                self.assertAlmostEqual(error_metrics(whole,exact)[key],value)

    def test_invalid_inputs_and_outputs(self):
        with self.assertRaises(ValueError): soft_bellman_ford(np.eye(2),2)
        w=np.array([[np.inf,1.],[1.,np.inf]])
        for tau in (-1,np.nan):
            with self.assertRaises(ValueError): soft_bellman_ford(w,2,tau)
        mean,c,valid=distance_outputs(np.array([[0.,-1.],[-1.,0.]]),np.ones((2,2),bool))
        self.assertFalse(valid.any());self.assertTrue(np.isnan(c).all())

    def test_autograd_matches_finite_differences_with_absent_edges(self):
        import torch
        w=np.array([[np.inf,1.,2.5,np.inf], [1.,np.inf,1.,np.inf],
                    [2.5,1.,np.inf,np.inf], [np.inf,np.inf,np.inf,np.inf]])
        for normalized in (False,True):
            tensor=torch.tensor(w,dtype=torch.float64,requires_grad=True)
            output=differentiable_soft_bellman_ford(tensor,3,.2,normalized)
            expected,_,_=soft_bellman_ford(w,3,.2,normalized)
            np.testing.assert_allclose(output.detach().numpy(),expected)
            output[0,2].backward()
            self.assertTrue(torch.isfinite(tensor.grad).all())
            eps=1e-6
            plus=w.copy();minus=w.copy();plus[0,1]+=eps;minus[0,1]-=eps
            fd=(soft_bellman_ford(plus,3,.2,normalized)[0][0,2]-
                soft_bellman_ford(minus,3,.2,normalized)[0][0,2])/(2*eps)
            self.assertAlmostEqual(tensor.grad[0,1].item(),fd,places=7)


if __name__ == '__main__': unittest.main()
