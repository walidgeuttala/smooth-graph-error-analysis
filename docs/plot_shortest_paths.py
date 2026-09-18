"""Plot the recorded extension sweeps without recomputing experiments."""
import csv
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/sofsem-extension-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]


def main():
    data=ROOT/'results/sofsem2027_shortest_paths'
    u=list(csv.DictReader((data/'unweighted.csv').open()))
    c=list(csv.DictReader((data/'controlled.csv').open()))
    plt.rcParams.update({'font.size':8,'legend.fontsize':7,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(9,2.8),layout='constrained')
    rows=[r for r in u if r['family']=='paths' and r['schedule']=='order']
    for key,label in [('state_max','Reachability error'),('distance_max','Distance max error'),
                      ('original_readout_mae','Original readout MAE'),('mean_mae','Fixed-support mean MAE')]:
        axes[0].loglog([float(r['n']) for r in rows],[float(r[key]) for r in rows],'.-',label=label)
    axes[0].set(title='(a) Unweighted path copies',xlabel='Explicit vertices n',ylabel='Error')
    rows=[r for r in c if r['family']=='triangle_copies' and r['scale']=='shrinking'
          and r['normalized']=='False' and r['schedule']=='n^-2']
    for key,label in [('distance_max','Distance max error'),('closeness_mae','Closeness MAE')]:
        axes[1].loglog([float(r['n']) for r in rows],[float(r[key]) for r in rows],'.-',label=label)
    remedy=[r for r in c if r['family']=='triangle_copies' and r['scale']=='shrinking'
            and r['normalized']=='False' and r['schedule']=='n^-3']
    axes[1].loglog([float(r['n']) for r in remedy],[float(r['closeness_mae']) for r in remedy],'.--',label=r'Closeness, $\tau=n^{-3}$')
    axes[1].set(title=r'(b) Weighted triangles, $\tau=n^{-2}$',xlabel='Represented vertices n',ylabel='Error')
    for gap in ('0.0','0.1','1.0'):
        rows=[r for r in c if r['family']=='parallel_paths' and r['gap']==gap
              and r['tau']=='0.1' and r['normalized']=='False']
        axes[2].semilogx([float(r['m']) for r in rows],[float(r['absolute_error']) for r in rows],'.-',label=f'Gap {gap}')
    axes[2].set(title=r'(c) Competing paths, $\tau=0.1$',xlabel='Number of paths m',ylabel='Source-target distance error')
    for ax in axes: ax.legend(frameon=False);ax.spines[['top','right']].set_visible(False)
    fig.savefig(ROOT/'docs/figures/shortest_path_extension.pdf')
    fig.savefig(data/'shortest_path_extension.png',dpi=180)
    plt.close(fig)

    # Main-paper panels use the physical LNCS text width so labels remain legible.
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.35), layout='constrained')
    rows = [r for r in u if r['family'] == 'paths' and r['schedule'] == 'order']
    for key, label in [('state_max', 'State max'), ('distance_max', 'Distance max'),
                       ('original_readout_mae', 'Original mean MAE'),
                       ('mean_mae', 'Exact-support mean MAE')]:
        axes[0].loglog([float(r['n']) for r in rows],
                      [float(r[key]) for r in rows], '.-', label=label)
    axes[0].set(title='(a) Unweighted paths', xlabel='Explicit vertices n', ylabel='Error')
    rows = [r for r in c if r['family'] == 'triangle_copies' and r['scale'] == 'shrinking'
            and r['normalized'] == 'False' and r['schedule'] == 'n^-2']
    for key, label in [('distance_max', 'Distance max'), ('closeness_mae', 'Closeness MAE')]:
        axes[1].loglog([float(r['n']) for r in rows],
                      [float(r[key]) for r in rows], '.-', label=label)
    axes[1].loglog([float(r['n']) for r in remedy],
                  [float(r['closeness_mae']) for r in remedy], '.--',
                  label=r'Closeness, $\tau=n^{-3}$')
    axes[1].set(title=r'(b) Triangles, $\tau=n^{-2}$', xlabel='Represented vertices n')
    for ax in axes:
        ax.legend(frameon=False)
        ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(ROOT/'docs/figures/shortest_path_main.pdf')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.8, 2.1), layout='constrained')
    for gap in ('0.0', '0.1', '1.0'):
        rows = [r for r in c if r['family'] == 'parallel_paths' and r['gap'] == gap
                and r['tau'] == '0.1' and r['normalized'] == 'False']
        ax.semilogx([float(r['m']) for r in rows],
                    [float(r['absolute_error']) for r in rows], '.-', label=f'Gap {gap}')
    ax.set(xlabel='Number of paths m', ylabel='Source-target distance error')
    ax.legend(frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(ROOT/'docs/figures/shortest_path_gaps.pdf')
    plt.close(fig)


if __name__=='__main__':main()
