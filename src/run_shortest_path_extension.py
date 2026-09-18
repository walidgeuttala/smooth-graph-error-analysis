"""CPU sweeps with independent exact targets and retained negative findings."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time
import networkx as nx
import numpy as np
from shortest_path_extension import (trajectory_distances, soft_bellman_ford,
                                     error_metrics)
from reachability_reference import reachability_states, reachability_readout

ROOT = Path(__file__).resolve().parents[1]


def save(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def unweighted():
    rows=[]
    for size in (4,12,32,100):
        graphs=[('paths',nx.disjoint_union_all([nx.path_graph(3) for _ in range(size)]),2),
                ('barbell',nx.barbell_graph(size,0),3)]
        if size<=32:
            graphs += [('cycle',nx.cycle_graph(size),size//2),
                       ('star',nx.star_graph(size-1),2)]
        for family,g,k in graphs:
            a=nx.to_numpy_array(g); d=np.asarray(nx.floyd_warshall_numpy(g))
            exact=reachability_states(a,k)
            np.testing.assert_allclose(trajectory_distances(exact,np.isfinite(d)),d)
            for schedule in ('fixed8','degree','order','order_strong'):
                n=len(g);degree=max(dict(g.degree()).values())
                beta={'fixed8':8.,'degree':3*np.log(degree+1),
                      'order':1.5*np.log(n),'order_strong':3*np.log(n)}[schedule]
                h=reachability_states(a,k,beta=beta)
                dh=trajectory_distances(h,np.isfinite(d))
                metrics=error_metrics(dh,d)
                truth=reachability_readout(exact,isolated_mask=a.sum(1)==0)
                raw=reachability_readout(h,isolated_mask=a.sum(1)==0)
                rows.append(dict(family=family,n=n,degree=degree,K=k,schedule=schedule,beta=beta,
                                 state_max=float(np.abs(h-exact).max()),
                                 original_readout_mae=float(np.abs(raw-truth).mean()),**metrics))
    return rows


def weighted():
    rows=[]
    for n in (6,12,24):
        for density in (.2,.6):
            for seed in (0,1,2):
                g=nx.gnp_random_graph(n,density,seed=seed)
                g.add_edges_from(nx.path_graph(n).edges())
                for distribution in ('unit','uniform','loguniform'):
                    rng=np.random.default_rng(seed)
                    for u,v in g.edges():
                        g[u][v]['weight']={'unit':1.,'uniform':rng.uniform(.5,1.5),
                                          'loguniform':np.exp(rng.uniform(-4,1))}[distribution]
                    w=nx.to_numpy_array(g,nonedge=np.inf)
                    d=np.asarray(nx.floyd_warshall_numpy(g));k=n-1
                    hard,hard_states,_=soft_bellman_ford(w,k)
                    np.testing.assert_allclose(hard,d,atol=1e-12)
                    for u,ds in nx.all_pairs_dijkstra_path_length(g):
                        for v,c in ds.items(): assert abs(d[u,v]-c)<1e-10
                    paths=nx.shortest_simple_paths(g,0,n-1,weight='weight')
                    costs=[]
                    for _ in range(2):
                        path=next(paths,None)
                        if path is not None: costs.append(sum(g[u][v]['weight'] for u,v in zip(path,path[1:])))
                    gap=costs[1]-costs[0] if len(costs)==2 else None
                    degree=max(dict(g.degree()).values())
                    for schedule in ('fixed.1','fixed.01','size_scaled'):
                        tau={'fixed.1':.1,'fixed.01':.01,'size_scaled':1/(n*k*np.log(max(degree,2)))}[schedule]
                        for normalized in (False,True):
                            dh,states,counts=soft_bellman_ford(w,k,tau,normalized)
                            finite=np.isfinite(hard_states)
                            state_error=float(np.abs(states[finite]-hard_states[finite]).max())
                            metrics=error_metrics(dh,d)
                            assert metrics['mean_max']<=metrics['distance_max']+1e-10
                            rows.append(dict(n=n,degree=degree,density=density,seed=seed,
                                             distribution=distribution,K=k,schedule=schedule,tau=tau,
                                             normalized=normalized,probe_gap=gap,
                                             state_max=state_error,
                                             entropy_bound=float(tau*counts[np.isfinite(counts)].max()),**metrics))
    return rows


def controlled():
    rows=[]
    # Directed two-hop paths: one optimum of cost 2, m-1 alternatives cost 2+gap.
    for m in (2,8,32,128):
        for gap in (0.,.01,.1,1.):
            w=np.full((m+2,m+2),np.inf)
            w[0,1:m+1]=1.;w[1:m+1,-1]=1.+gap;w[1,-1]=1.
            for tau in (.01,.1,1.,.1/np.log(m+1)):
                for normalized in (False,True):
                    d,_,_=soft_bellman_ford(w,2,tau,normalized)
                    formula=2-tau*np.log1p((m-1)*np.exp(-gap/tau))
                    if normalized: formula+=tau*np.log(m)
                    assert abs(d[0,-1]-formula)<1e-12
                    rows.append(dict(family='parallel_paths',n=m+2,m=m,gap=gap,tau=tau,
                                     normalized=normalized,prediction=float(d[0,-1]),
                                     absolute_error=abs(float(d[0,-1])-2)))
    # Exact component reduction: r disjoint weighted triangles, n=3r.
    for n in (30,90,300,900,3000,9000):
        for scale in ('fixed','shrinking'):
            a=1. if scale=='fixed' else 1/n
            w=np.array([[np.inf,a,2*a],[a,np.inf,a],[2*a,a,np.inf]])
            exact,_,_=soft_bellman_ford(w,2)
            for schedule in ('n^-2','n^-3'):
                tau=n**(-2. if schedule=='n^-2' else -3.)
                for normalized in (False,True):
                    dh,_,_=soft_bellman_ford(w,2,tau,normalized)
                    rows.append(dict(family='triangle_copies',n=n,represented_only=True,
                                     weight_scale=a,scale=scale,schedule=schedule,tau=tau,
                                     normalized=normalized,**error_metrics(dh,exact)))
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,
                          default=ROOT/'results/sofsem2027_shortest_paths')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    u=unweighted();save(args.output/'unweighted.csv',u)
    w=weighted();save(args.output/'weighted.csv',w)
    c=controlled();save(args.output/'controlled.csv',c)
    # Retain full matrices for inspection in addition to aggregate CSV errors.
    g=nx.disjoint_union_all([nx.path_graph(3) for _ in range(10)])
    exact=np.asarray(nx.floyd_warshall_numpy(g))
    states=reachability_states(nx.to_numpy_array(g),2,beta=1.5*np.log(len(g)))
    np.savez_compressed(args.output/'unweighted_matrices.npz',exact=exact,states=states,
                        relaxed=trajectory_distances(states,np.isfinite(exact)),support=np.isfinite(exact))
    weights=np.array([[np.inf,1/300,2/300],[1/300,np.inf,1/300],[2/300,1/300,np.inf]])
    exact,_,_=soft_bellman_ford(weights,2)
    relaxed,states,_=soft_bellman_ford(weights,2,1/300**2)
    np.savez_compressed(args.output/'weighted_matrices.npz',weights=weights,exact=exact,
                        relaxed=relaxed,states=states,represented_nodes=300)
    candidates=[r for r in c if r['family']=='triangle_copies' and
                r['distance_max']<1e-5 and r['closeness_mae'] is not None and r['closeness_mae']>.05]
    manifest=dict(status='complete',seconds=time.monotonic()-start,gpu_used=False,
                  unweighted_cases=len(u),weighted_cases=len(w),controlled_cases=len(c),
                  numpy=np.__version__,networkx=nx.__version__,
                  candidate_rule='distance_max < 1e-5 and closeness_mae > .05; controlled family only',
                  candidates=candidates,
                  weighted_invalid_cases=sum(r['invalid_closeness_nodes']>0 for r in w),
                  mean_stability_violations=sum(r['mean_max']>r['distance_max']+1e-10 for r in w))
    manifest['source_hashes']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (Path(__file__),ROOT/'src/shortest_path_extension.py',ROOT/'src/reachability_reference.py')}
    manifest['output_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in args.output.iterdir() if p.suffix in ('.csv','.npz')}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('candidates','source_hashes','output_hashes')}))


if __name__=='__main__': main()
