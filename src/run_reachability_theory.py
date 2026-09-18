"""CPU-only tests of temperature thresholds and sufficient stability bounds."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import time

os.environ.setdefault('MPLCONFIGDIR', '/tmp/sofsem2027-theory-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from reachability_reference import reachability_readout
from reachability_theory import (barbell, path_copies, certificate_beta,
                                centrality_error_bound, state_jacobian_bound, sigmoid)

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def scaling_sweep(quick):
    rows = []
    exponents = (1, 3, 8) if quick else (1, 2, 3, 4, 6, 8, 12, 20, 40, 80, 120)
    for family in ('barbell', 'path_copies'):
        for exponent in exponents:
            size = 10**exponent
            scale = float(size if family == 'barbell' else 3*size)
            for label, beta in [(f'fixed_{b}', float(b)) for b in (4, 8, 12, 20, 100)] + [
                    (f'coefficient_{c}', c*np.log(scale)) for c in (1.5, 1.9, 2., 2.1, 2.5, 3.)]:
                result = barbell(size, 3, beta) if family == 'barbell' else path_copies(size, 2, beta)
                rows.append({'family': family, 'size_parameter': size, 'represented_nodes': (2 if family=='barbell' else 3)*size,
                             'schedule': label, 'beta': beta, 'mae': result['mae'],
                             'max_state_error_all_steps': max(result['max_state_errors']),
                             'max_state_error_final': result['max_state_errors'][-1]})
    return rows


def critical_sweep(quick):
    rows = []
    for family in ('barbell', 'path_copies'):
        for exponent in ((2, 8) if quick else (2, 4, 8, 24, 80)):
            size = 10**exponent
            for offset in np.linspace(-2, 4, 13 if quick else 121):
                scale = float(size if family == 'barbell' else 3*size)
                beta = 2*np.log(scale)+offset
                result = barbell(size, 3, beta) if family == 'barbell' else path_copies(size, 2, beta)
                leak = np.exp(-offset/2)
                limit = (0.5 if offset<2*np.log(2) else 0.) if family=='barbell' else leak/(3*(2+leak))
                rows.append({'family': family, 'size_parameter': size, 'offset': float(offset),
                             'beta': beta, 'mae': result['mae'], 'limit_mae': float(limit)})
    return rows


def perturbation_sweep(quick):
    graphs = [('path8', nx.path_graph(8)), ('barbell4', nx.barbell_graph(4, 0))]
    if not quick:
        graphs += [(f'path{n}', nx.path_graph(n)) for n in (16, 32)]
        graphs += [(f'barbell{s}', nx.barbell_graph(s, 0)) for s in (8, 16)]
        graphs += [(f'random_{p}_{seed}', nx.gnp_random_graph(32, p, seed=seed))
                   for p in (.1, .3, .7) for seed in (0, 1, 2)]
    steps = 12 if quick else 64
    rows = []
    for name, graph in graphs:
        adjacency = nx.to_numpy_array(graph)
        closed = adjacency+np.eye(len(graph))
        degree = int(adjacency.sum(1).max())
        isolated = adjacency.sum(1) == 0
        targets, reachable = [], []
        for node in graph:
            lengths = nx.single_source_shortest_path_length(graph, node)
            reachable.append(len(lengths)-1)
            targets.append(sum(lengths.values())/(len(lengths)-1) if len(lengths)>1 else 0.)
        for eta in (0., .1, .3):
            for kind in ('adversarial', 'equivariant_residual'):
                for multiplier in (.5, 1., 1.5):
                    delta = .001
                    required = certificate_beta(degree, delta, eta)
                    beta = multiplier*required
                    hidden = np.eye(len(graph))
                    exact = hidden.copy()
                    states, state_error, largest_jacobian = [hidden], 0., 0.
                    for _ in range(steps):
                        exact = (closed@exact > 0).astype(float)
                        if kind == 'adversarial':
                            perturbation = eta*(1-2*exact)
                            residual_derivative = 0.
                        else:
                            perturbation = eta*np.tanh(1.5*hidden)
                            residual_derivative = 1.5*eta*(1-np.tanh(1.5*hidden)**2)
                        output = sigmoid(beta*(closed@hidden+perturbation-.5))
                        jacobian_norm = np.max(beta*output*(1-output)*(
                            closed.sum(1)[:,None]+residual_derivative))
                        largest_jacobian = max(largest_jacobian, float(jacobian_norm))
                        hidden = output
                        states.append(hidden)
                        state_error = max(state_error, float(np.abs(hidden-exact).max()))
                    predicted = reachability_readout(states, isolated_mask=isolated)
                    error = float(np.abs(predicted-targets).max())
                    bound = max(centrality_error_bound(len(graph), m, steps, delta) for m in reachable if m)
                    jbound = state_jacobian_bound(degree, beta, delta,
                                                  1.5*eta if kind=='equivariant_residual' else 0.)
                    certified = multiplier >= 1
                    if certified:
                        assert state_error <= delta+1e-12, (name, eta, beta, state_error)
                        assert error <= bound+1e-12
                        assert largest_jacobian <= jbound+1e-12
                    rows.append({'graph': name, 'nodes': len(graph), 'max_degree': degree, 'steps': steps,
                                 'perturbation': kind, 'eta': eta, 'delta': delta, 'beta': beta,
                                 'beta_multiplier': multiplier, 'certified': certified,
                                 'max_state_error': state_error, 'max_ic_error': error, 'ic_bound': bound,
                                 'max_local_state_jacobian_norm': largest_jacobian,
                                 'conditional_jacobian_bound': jbound})
    return rows


def precision_checks():
    """Independent scalar formulas at 80 decimal digits, not matrix reductions."""
    import mpmath as mp
    rows = []
    with mp.workdps(80):
        for copies in (10**6, 10**12):
            n = 3*copies
            for offset in (-2., 0., 2.):
                beta_float = 2*np.log(float(n))+offset
                beta = mp.mpf(beta_float)
                logistic = lambda x: 1/(1+mp.exp(-x))
                q = logistic(-beta/2)
                a = 1-q
                b1 = 2+(n-4)*q
                b2 = 2*logistic(beta*(2*a-mp.mpf('.5')))+a+(n-3)*logistic(beta*(2*q-mp.mpf('.5')))
                endpoint = (2*b2-1-b1)/(b2-1)
                observed = path_copies(copies, 2, beta_float)['prediction'][0]
                difference = float(abs(mp.mpf(observed)-endpoint))
                assert difference < 1e-12
                rows.append({'copies': copies, 'beta': beta_float, 'float64_endpoint': float(observed),
                             'mpmath_endpoint': str(endpoint), 'absolute_difference': difference})
    return rows


def figures(directory, scaling, critical, perturbations):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.1), constrained_layout=True)
    for ax, family, label in zip(axes[:2], ('barbell','path_copies'),
                                 ('Connected barbell, K=3', 'Disjoint three-node paths, K=2')):
        for exponent in sorted({int(round(np.log10(float(r['size_parameter'])))) for r in critical}):
            selected = [r for r in critical if r['family']==family and r['size_parameter']==10**exponent]
            if selected:
                ax.plot([r['offset'] for r in selected], [r['mae'] for r in selected], label=f'size $10^{{{exponent}}}$')
        if family=='barbell':
            ax.axvline(2*np.log(2), color='black', linestyle=':', label=r'$c=2\log 2$')
        else:
            x = np.linspace(-2,4,150)
            ax.plot(x, np.exp(-x/2)/(3*(2+np.exp(-x/2))), 'k--', label='proved limit')
        ax.set(xlabel=r'Offset $c$ in $\beta=2\log(\mathrm{scale})+c$', ylabel='Node MAE', title=label)
        ax.legend(fontsize=7)
    for metric, label in (('mae', 'IC error'), ('max_state_error_all_steps', 'State error')):
        selected = [r for r in scaling if r['family']=='path_copies' and r['schedule']=='coefficient_1.5']
        axes[2].semilogy([np.log10(float(r['represented_nodes'])) for r in selected],
                         [max(r[metric], 1e-300) for r in selected], 'o-', label=label)
    axes[2].set(xlabel=r'$\log_{10}(n)$', ylabel='Error', title=r'State/readout separation: $\beta=1.5\log n$')
    axes[2].legend()
    fig.suptitle('Exact symmetry reductions, evaluated in float64; large graphs are not materialized', fontsize=11)
    fig.savefig(directory/'temperature_thresholds.png', dpi=180)
    fig.savefig(directory/'temperature_thresholds.pdf')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.2,4), constrained_layout=True)
    for multiplier in (.5, 1., 1.5):
        selected = [r for r in perturbations if r['beta_multiplier']==multiplier]
        ax.scatter([r['max_degree'] for r in selected],
                   [max(r['max_state_error']/r['delta'], 1e-16) for r in selected],
                   s=18, alpha=.6, label=f'{multiplier:g} × sufficient beta')
    ax.axhline(1, color='black', linestyle='--', label='certificate error limit')
    ax.set(yscale='log', xlabel='Maximum degree', ylabel='Observed state error / certified tolerance',
           title='Bounded-perturbation stress tests')
    ax.legend(fontsize=8)
    fig.savefig(directory/'stability_certificate.png', dpi=180)
    fig.savefig(directory/'stability_certificate.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--render-only', action='store_true', help='Use existing numerical CSVs; do not rerun experiments')
    parser.add_argument('--output', type=Path, default=ROOT/'results/sofsem2027_theory')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=args.render_only)
    started = time.perf_counter()
    manifest = {'status': 'running', 'quick': args.quick, 'gpu_used': False,
                'python': platform.python_version(), 'numpy': np.__version__, 'networkx': nx.__version__,
                'note': 'Large graph families simulated via proved symmetry reductions; not runtime scalability measurements',
                'code_hashes': {name: sha256(ROOT/'src'/name) for name in (
                    'run_reachability_theory.py', 'reachability_theory.py', 'reachability_reference.py')}}
    if args.render_only:
        manifest = json.loads((args.output/'manifest.json').read_text())
        manifest['render_script_sha256'] = sha256(__file__)
        def read(name):
            with (args.output/f'{name}.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            strings = {'family', 'schedule', 'graph', 'perturbation', 'mpmath_endpoint'}
            integers = {'size_parameter', 'represented_nodes', 'nodes', 'steps', 'max_degree', 'copies'}
            return [{key: value if key in strings else int(value) if key in integers else
                          value == 'True' if key == 'certified' else float(value)
                     for key, value in row.items()} for row in rows]
        scaling, critical, perturbations, precision = [read(name) for name in (
            'scaling', 'critical_windows', 'perturbations', 'precision_checks')]
    else:
        (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
        scaling, critical = scaling_sweep(args.quick), critical_sweep(args.quick)
        perturbations, precision = perturbation_sweep(args.quick), precision_checks()
        for name, rows in (('scaling',scaling), ('critical_windows',critical), ('perturbations',perturbations), ('precision_checks',precision)):
            save_csv(args.output/f'{name}.csv', rows)
    figures(args.output, scaling, critical, perturbations)
    manifest.update(status='complete',
                    scaling_cases=len(scaling), critical_cases=len(critical), perturbation_cases=len(perturbations),
                    certified_cases=sum(r['certified'] for r in perturbations), certificate_violations=0,
                    precision_checks=len(precision), max_precision_discrepancy=max(r['absolute_difference'] for r in precision))
    manifest['rendering_seconds' if args.render_only else 'elapsed_seconds'] = time.perf_counter()-started
    manifest['output_hashes'] = {p.name: sha256(p) for p in args.output.iterdir() if p.name != 'manifest.json'}
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
