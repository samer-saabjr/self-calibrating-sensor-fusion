"""Regenerate the two four-row sensor traces for manuscript Figure 1.

From the root of the extracted supplementary ZIP, run:
    python code/generate_figure_1_traces.py

Saves: figures/trace_binary.pdf and figures/trace_counting.pdf
No camera images/video are used or distributed.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
OUT = ROOT / 'figures'
OUT.mkdir(exist_ok=True)


def binary_states(events):
    m = pd.Series(events, dtype=int)
    m = m.mask((m == 1) & (m.shift(1, fill_value=0) == 1), 0)
    return (m.cumsum() % 2).to_numpy(dtype=int)


def counting_states(events):
    states = np.zeros(len(events), dtype=int)
    active = 0
    for k, event in enumerate(events):
        event = int(event)
        if event in (1, 2):
            active = 0 if event == active else event
        states[k] = active
    return states


def shade_occupied(ax, x, gt, color='0.88'):
    occupied = (gt > 0).astype(int)
    # Extend boundaries by one sampling interval to show full labelled seconds.
    transitions = np.diff(np.r_[0, occupied, 0])
    for first, last in zip(np.flatnonzero(transitions == 1),
                           np.flatnonzero(transitions == -1)):
        lo = float(x[first]) - 0.5
        hi = float(x[last - 1]) + 0.5
        ax.axvspan(lo, hi, color=color, alpha=0.8, zorder=0)


def draw_task(df, gt, expert, start, end, title, output, labels):
    window = slice(start, end + 1)
    xs = df['elapsed_seconds'].to_numpy(dtype=int)[window]
    gt = np.asarray(gt)[window]
    expert = np.asarray(expert)[window]
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10.2, 6.8),
                             gridspec_kw={'height_ratios': [1, 1, .48, .48]})
    for ax in axes:
        shade_occupied(ax, xs, gt)
        ax.grid(True, alpha=0.25)
    for ax, col, label in zip(axes[:2], ('dBFS_mic1','dBFS_mic2'),
                               ('Inexpert 1\n[dBFS]','Inexpert 2\n[dBFS]')):
        ax.plot(xs, df[col].to_numpy(dtype=float)[window], lw=1.1,
                color='#3875a9', zorder=3)
        ax.set_ylabel(label)
    axes[2].step(xs, expert, where='post', lw=1.35,
                 color='#d68d47', zorder=3)
    axes[2].set_ylabel('Expert\noutput')
    axes[3].step(xs, gt, where='post', lw=1.45,
                 color='#43995b', zorder=3)
    axes[3].set_ylabel('State')
    axes[2].set_yticks(labels[0])
    axes[3].set_yticks(labels[1]); axes[3].set_yticklabels(labels[2])
    axes[3].set_xlim(start, end)
    axes[3].set_xlabel('Sample k (s)')
    axes[0].set_title(title, fontweight='bold', fontsize=12)
    fig.tight_layout(h_pad=.45)
    fig.savefig(output, bbox_inches='tight')
    plt.close(fig)
    print(f'Created {output.relative_to(ROOT)} (samples {start}–{end})')


def main():
    b = pd.read_csv(DATA / 'binary_walking_1851.csv')
    assert len(b) == 1851
    binary_gt = binary_states(b['mouse'].to_numpy(dtype=int))
    assert tuple(np.bincount(binary_gt, minlength=2)) == (1168, 683)
    draw_task(b, binary_gt, b['human'].to_numpy(dtype=int), 0, 200,
              'Binary walking detection', OUT/'trace_binary.pdf',
              ([0, 1], [0, 1], ['quiet', 'walking']))

    c = pd.read_csv(DATA / 'person_counting_2163.csv')
    assert len(c) == 2163
    count_gt = counting_states(c['mouse_event'].to_numpy(dtype=int))
    assert tuple(np.bincount(count_gt, minlength=3)) == (860, 559, 744)
    draw_task(c, count_gt, c['human_count'].to_numpy(dtype=int), 1440, 1640,
              'Person counting', OUT/'trace_counting.pdf',
              ([0, 1, 2], [0, 1, 2], ['0', '1', '2']))


if __name__ == '__main__':
    main()
