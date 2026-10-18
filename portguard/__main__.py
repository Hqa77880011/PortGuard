import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import j0


def covariance(c):
    x = np.linspace(0, c['length'], c['ports'])
    distance = abs(x[:, None] - x[None, :])
    return x, j0(2 * np.pi * distance) * np.exp(-distance / c['correlation'])


def prepare(c, out):
    rng = np.random.default_rng(c['seed'])
    x, k = covariance(c)
    eig, vec = np.linalg.eigh(k)
    root = vec @ np.diag(np.sqrt(np.maximum(eig, 0)))
    innovation = (rng.normal(size=(c['slots'], c['ports'])) +
                  1j * rng.normal(size=(c['slots'], c['ports']))) @ root.T / np.sqrt(2)
    rho = j0(2 * np.pi * c['doppler'] * c['slot_seconds'])
    h = np.empty_like(innovation)
    h[0] = innovation[0]
    for t in range(1, len(h)):
        h[t] = rho * h[t - 1] + np.sqrt(max(0, 1 - rho ** 2)) * innovation[t]
    noise = np.sqrt(c['noise'] / 2) * (
        rng.normal(size=h.shape) + 1j * rng.normal(size=h.shape))
    np.savez_compressed(out / 'channel.npz', h=h, noisy=h + noise, x=x)


def posterior(y, probes, k, c, compensate=True):
    gain = np.linalg.solve(k[np.ix_(probes, probes)] + c['noise'] * np.eye(len(probes)),
                           k[probes]).T
    mu = gain @ y
    sigma = k - gain @ k[probes]
    rho = j0(2 * np.pi * c['doppler'] * c['slot_seconds']) ** c['delay'] if compensate else 1
    return rho * mu, np.maximum(np.diag(rho ** 2 * sigma + (1 - rho ** 2) * k), 0)


def run(c, out):
    z = np.load(out / 'channel.npz')
    h, observed, x = z['h'], z['noisy'], z['x']
    _, k = covariance(c)
    methods = ['PortGuard', 'fixed', 'random', 'scan', 'stale-greedy', 'mean-only',
               'no-cost', 'no-hysteresis', 'no-risk', 'uniform-probes', 'no-delay', 'oracle']
    rows = []
    for method in methods:
        rng = np.random.default_rng(c['seed'] + 1000)
        previous = 0
        probes = np.linspace(0, c['ports'] - 1, c['probes'], dtype=int)
        for t in range(c['delay'], len(h)):
            start = time.perf_counter()
            mu, variance = posterior(observed[t - c['delay'], probes], probes, k, c,
                                     method != 'no-delay')
            draw = mu + np.sqrt(variance / 2) * (
                rng.normal(size=(c['draws'], c['ports'])) +
                1j * rng.normal(size=(c['draws'], c['ports'])))
            rates = np.log2(1 + 10 ** (c['snr_db'] / 10) * abs(draw) ** 2)
            tail_n = max(1, int(np.ceil(c['alpha'] * len(rates))))
            lower_tail = np.sort(rates, axis=0)[:tail_n].mean(0)
            score = np.quantile(rates, c['beta'], axis=0) + c['eta'] * lower_tail
            if method in ['mean-only', 'no-risk']:
                score = rates.mean(0)
            movement = abs(x - x[previous])
            costs = c['startup'] * (np.arange(len(x)) != previous) + c['distance_cost'] * movement
            if method != 'no-cost':
                score -= c['switch_weight'] * costs
            candidate = int(np.argmax(score))
            threshold = 0 if method == 'no-hysteresis' else c['hysteresis']
            action = candidate if score[candidate] - score[previous] > threshold else previous
            if method == 'fixed':
                action = 0
            elif method == 'random':
                action = int(rng.integers(len(x)))
            elif method == 'scan':
                action = int(probes[np.argmax(abs(observed[t - c['delay'], probes]))])
            elif method == 'stale-greedy':
                action = int(probes[np.argmax(abs(observed[t - c['delay'], probes]))])
            elif method == 'oracle':
                action = int(np.argmax(abs(h[t])))
            true_rates = np.log2(1 + 10 ** (c['snr_db'] / 10) * abs(h[t]) ** 2)
            rows.append(dict(method=method, slot=t, port=action, rate=true_rates[action],
                             oracle_rate=true_rates.max(), switched=int(action != previous),
                             distance=abs(x[action] - x[previous]), latency_ms=1000 * (time.perf_counter() - start)))
            if method in ['uniform-probes', 'scan']:
                probes = (np.linspace(0, len(x) - 1, c['probes'], dtype=int) + t) % len(x)
            else:
                local = [action, *np.argsort(abs(x - x[action]))[1:3]]
                order = local + list(np.argsort(-variance))
                probes = np.array(list(dict.fromkeys(order))[:c['probes']])
            previous = action
    pd.DataFrame(rows).to_csv(out / 'decisions.csv', index=False)


def evaluate(c, out):
    data = pd.read_csv(out / 'decisions.csv')
    rows = []
    for method, g in data.groupby('method', sort=False):
        n = max(1, int(np.ceil(c['alpha'] * len(g))))
        rows.append(dict(method=method, spectral_efficiency=g.rate.mean(),
                         outage=float((g.rate < c['target_rate']).mean()),
                         cvar_throughput=np.sort(g.rate)[:n].mean(), switching_rate=g.switched.mean(),
                         movement=g.distance.mean(), near_oracle=float((g.rate >= .95 * g.oracle_rate).mean()),
                         latency_ms=g.latency_ms.mean()))
    pd.DataFrame(rows).to_csv(out / 'metrics.csv', index=False)


def plot(out):
    d = pd.read_csv(out / 'metrics.csv')
    fig, ax = plt.subplots(figsize=(8, 5))
    for row in d.itertuples():
        ax.scatter(row.switching_rate, row.cvar_throughput)
        ax.annotate(row.method, (row.switching_rate, row.cvar_throughput), fontsize=8)
    ax.set(xlabel='Switching fraction', ylabel='Worst-tail spectral efficiency')
    fig.tight_layout()
    fig.savefig(out / 'tradeoff.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'run', 'evaluate', 'plot'])
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--output', default='runs/default')
    args = parser.parse_args()
    c = json.loads(Path(args.config).read_text(encoding='utf-8'))
    if not 0 < c['probes'] <= c['ports'] or c['slots'] <= c['delay']:
        raise ValueError('Require 0 < probes <= ports and slots > delay')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    settings = out / 'settings.json'
    settings.write_text(json.dumps(c, indent=2))
    if args.command == 'plot':
        plot(out)
    else:
        globals()[args.command](c, out)


if __name__ == '__main__':
    main()
