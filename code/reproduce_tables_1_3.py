"""Regenerate the experimental data summaries in manuscript Tables 1–3.

Run from the extracted supplementary archive root:
    python code/reproduce_tables_1_3.py

The raw microphone/label columns in the release CSVs are not modified.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'


def binary_gt(events):
    m = pd.Series(events, dtype=int)
    m = m.mask((m == 1) & (m.shift(1, fill_value=0) == 1), 0)
    return np.where((m.cumsum() % 2).to_numpy() == 1, 1, -1)


def counting_gt(events):
    states = np.zeros(len(events), dtype=int)
    current = 0
    for k, e in enumerate(events):
        if e == 1:
            current = 0 if current == 1 else 1
        elif e == 2:
            current = 0 if current == 2 else 2
        states[k] = current
    return states


def close(x, expected, tol=0.006):
    if abs(x-expected) > tol:
        raise AssertionError(f"Mismatch: calculated {x:.6f}, expected {expected:.6f}")


def summarize(task, path, labels, expert_col, event_col, counts_expected,
              expert_error, recall_expected, mean_sd_expected, coeff_expected):
    data = pd.read_csv(DATA / path)
    x = binary_gt(data[event_col]) if task == 'Binary' else counting_gt(data[event_col])
    assert len(x) == len(data)
    counts = [int((x == cl).sum()) for cl in labels]
    assert counts == counts_expected, (counts, counts_expected)
    pred = data[expert_col].to_numpy(dtype=int)
    if task == 'Binary':
        pred = np.where(pred == 1, 1, -1)
    error = 100*(pred != x).mean()
    close(error, expert_error)
    recall = [100*(pred[x == cl] == cl).mean() for cl in labels]
    for got, exp in zip(recall, recall_expected):
        close(got, exp, tol=0.055)
    print('\n' + task + ':')
    print(f'  Table 1: N={len(x)}, class counts={counts}, expert error={error:.2f}%')
    print('  Table 1: expert recalls=' + ', '.join(f'{r:.1f}%' for r in recall))
    for col_idx in (1, 2):
        v = data[f'dBFS_mic{col_idx}'].to_numpy(dtype=float)
        means_sd = [(v[x == cl].mean(), v[x == cl].std(ddof=1)) for cl in labels]
        for (mean, sd), (expect_mean, expect_sd) in zip(means_sd, mean_sd_expected[col_idx]):
            close(mean, expect_mean); close(sd, expect_sd)
        print(f'  Table 2: microphone {col_idx}: '
              + ', '.join(f'{m:.2f} +/- {s:.2f}' for m, s in means_sd) + ' dBFS')
        H = np.column_stack([x.astype(float), np.ones(len(x))])
        coupling, offset = np.linalg.lstsq(H, v, rcond=None)[0]
        resid_var = np.var(v - H @ [coupling, offset], ddof=0)
        targets = coeff_expected[col_idx]
        for got, expected in zip((offset, coupling, resid_var), targets):
            close(got, expected, tol=0.001)
        print(f'  Table 3: microphone {col_idx}: offset={offset:.3f}, '
              f'coupling={coupling:.3f}, residual variance={resid_var:.3f}')
    return data


def main():
    summarize('Binary', 'binary_walking_1851.csv', [-1, 1], 'human', 'mouse',
              [1168, 683], 28.47, [100.0, 22.8],
              {1:[(-34.00,1.02),(-25.34,5.39)],
               2:[(-33.22,0.75),(-20.50,4.36)]},
              {1:(-29.668,4.332,11.366),2:(-26.860,6.360,7.357)})
    summarize('Counting', 'person_counting_2163.csv', [0, 1, 2],
              'human_count', 'mouse_event', [860, 559, 744],
              44.24, [100.0,31.1,23.1],
              {1:[(-40.33,2.10),(-31.56,6.83),(-26.39,7.68)],
               2:[(-35.81,1.45),(-26.89,4.47),(-21.61,4.11)]},
              {1:(-39.900,7.007,34.673),2:(-35.377,7.137,12.410)})
    print('\nPASS: Tables 1–3 are reproduced within published rounding precision.')


if __name__ == '__main__':
    main()
