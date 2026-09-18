"""Small CPU controls isolating leakage, component masking, and exact-zero gates.

Run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m
src.run_reachability_escape_controls. Dense reference timings are not benchmarks.
"""
import hashlib
import json
from pathlib import Path

import networkx as nx
import numpy as np

from src.reachability_reference import reachability_readout, reachability_states


def exact_zero_states(adjacency, steps):
    """C1 smoothstep gate: zero below 0, one above 1, cubic in between.

    On integer Boolean aggregates this executes OR exactly. It is not strictly
    positive, analytic, or a demonstration of successful gradient training.
    """
    states = [np.eye(len(adjacency))]
    for _ in range(steps):
        z = np.clip(adjacency @ states[-1] + states[-1], 0, 1)
        states.append(z*z*(3-2*z))
    return np.stack(states)


def measure(graph, steps, beta, family):
    adjacency = nx.to_numpy_array(graph, nodelist=list(range(len(graph))))
    labels = np.empty(len(graph), dtype=int)
    target = np.zeros(len(graph))
    for label, component in enumerate(nx.connected_components(graph)):
        for node in component:
            labels[node] = label
            distances = nx.single_source_shortest_path_length(graph, node)
            if len(distances) > 1:
                target[node] = sum(distances.values())/(len(distances)-1)
    mask = labels[:, None] == labels[None, :]
    isolates = adjacency.sum(1) == 0
    hard = reachability_states(adjacency, steps)
    soft = reachability_states(adjacency, steps, beta=beta)
    zero = exact_zero_states(adjacency, steps)
    # Independent graph-distance oracle checks the entire output, not just rank.
    np.testing.assert_array_equal(zero, hard)
    rows = []
    for name, states in [('sigmoid', soft), ('sigmoid_masked_readout', soft),
                         ('smoothstep_exact_zero', zero), ('boolean', hard)]:
        readout_states = states*mask if name == 'sigmoid_masked_readout' else states
        prediction = reachability_readout(readout_states, isolated_mask=isolates)
        if name in ('boolean', 'smoothstep_exact_zero'):
            np.testing.assert_allclose(prediction, target, rtol=0, atol=1e-12)
        rows.append(dict(family=family, nodes=len(graph), steps=steps, beta=beta,
                         method=name, mae=float(np.abs(prediction-target).mean()),
                         max_state_error=float(np.abs(states-hard).max()),
                         max_cross_component_row_mass=float((states[-1]*(~mask)).sum(1).max())))
    return rows


def main():
    rows = []
    for copies in (1, 10, 100):
        graph = nx.disjoint_union_all([nx.path_graph(3) for _ in range(copies)])
        for beta in (8., 12.):
            rows.extend(measure(graph, 2, beta, 'P3_copies'))
    for size in (28, 100, 278):
        rows.extend(measure(nx.barbell_graph(size, 0), 3, 8., 'barbell'))
    # Explicit isolate convention and multi-component control.
    rows.extend(measure(nx.disjoint_union(nx.path_graph(3), nx.empty_graph(2)),
                        2, 8., 'P3_and_isolates'))
    root = Path(__file__).resolve().parents[1]
    output = root/'results/sofsem2027_theory/escape_controls.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(description='Explicit small dense graphs; CPU controls, not training or scalability results',
                   source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (Path(__file__).resolve(), root/'src/reachability_reference.py')},
                   numpy_version=np.__version__, networkx_version=nx.__version__, rows=rows)
    output.write_text(json.dumps(payload, indent=2)+'\n')
    print(output)
    for row in rows:
        if row['nodes'] in (300, 556):
            print(row)


if __name__ == '__main__':
    main()
