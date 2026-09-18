"""Render the submission figure from the recorded, unchanged numerical CSVs."""
import csv
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/sofsem2027-paper-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'results/sofsem2027_theory'
OUT = ROOT / 'docs/figures'


def main():
    with (DATA / 'critical_windows.csv').open() as stream:
        critical = list(csv.DictReader(stream))
    with (DATA / 'scaling.csv').open() as stream:
        scaling = list(csv.DictReader(stream))
    plt.rcParams.update({'font.size': 9, 'axes.labelsize': 9,
                         'legend.fontsize': 8, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8), layout='constrained')
    for exponent, style in ((2, '-'), (8, '--'), (80, ':')):
        rows = [r for r in critical if r['family'] == 'barbell'
                and int(r['size_parameter']) == 10**exponent]
        axes[0].plot([float(r['offset']) for r in rows],
                     [float(r['mae']) for r in rows], style,
                     label=rf'$s=10^{{{exponent}}}$', linewidth=1.6)
    axes[0].axvline(2*np.log(2), color='0.55', linewidth=.8)
    axes[0].plot([2*np.log(2)], [.25], 'ko', markersize=3)
    axes[0].set(xlabel=r'Offset $c$ in $\beta=2\log s+c$',
                ylabel='Node MAE', title='(a) Connected barbell', ylim=(-.01, .52))
    axes[0].legend(loc='lower left', frameon=False)
    rows = [r for r in scaling if r['family'] == 'path_copies'
            and r['schedule'] == 'coefficient_1.5']
    x = [np.log10(float(r['represented_nodes'])) for r in rows]
    axes[1].semilogy(x, [float(r['mae']) for r in rows], 'o-',
                     label='Scalar MAE', markersize=3)
    axes[1].semilogy(x, [float(r['max_state_error_all_steps']) for r in rows],
                     's--', label='Maximum state error', markersize=3)
    axes[1].axhline(1/3, color='0.55', linewidth=.8, linestyle=':')
    axes[1].set(xlabel=r'$\log_{10} n$', ylabel='Error',
                title=r'(b) Replicated paths, $\beta=1.5\log n$')
    axes[1].legend(loc='lower left', frameon=False)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / 'sofsem_thresholds.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
