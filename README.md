# PortGuard

Implementation of **Risk- and switching-aware fluid-antenna port selection with delayed CSI.**

## Install

Use Python 3.11 in an isolated environment, then run from the repository root:

```bash
python -m pip install -e .
```

## Data

`prepare` generates a complex spatially correlated channel tape and independent measurement noise. The covariance combines a Jakes/Bessel term with exponential distance decay. Every controller sees the same tape; only the full-CSI oracle accesses current complete channels. There is no external dataset requirement for this simulation.

## Workflow

```bash
python -m portguard prepare
python -m portguard run
python -m portguard evaluate
python -m portguard plot
```

Main commands accept `--config config.json` and `--output runs/default`. Keep each seed/condition in its own output directory. The resolved configuration is written to `settings.json`.

## Configuration

`ports,length,correlation` define geometry/covariance. `doppler,slot_seconds,delay,noise` define aging/estimation. `probes` bounds sparse acquisition; `draws` controls posterior rate sampling. `alpha` is lower-tail probability mass; `beta` is a rate quantile. `eta,switch_weight,startup,distance_cost,hysteresis` price tail loss and movement. `target_rate` fixes outage. Training epochs, batch size, learning rate and seed use the defaults shown in `config.json` where applicable.

## Outputs

`channel.npz` is the common tape. `decisions.csv` records actions, realized rate, oracle rate, switching, movement and host time. `metrics.csv` reports spectral efficiency, outage, worst-tail throughput, switching fraction, mean movement and near-oracle accuracy. `tradeoff.png` plots tail reliability against switching.

## Implementation and controls

Spatial conditioning retains uncertainty; temporal propagation uses Jakes correlation. The score combines a lower rate quantile, lower-tail mean and displacement/startup cost. Hysteresis requires additional advantage over staying. Next-slot probes retain local support and add high-variance ports. The generated temporal channel uses an AR process matched at one slot; multistep inference uses the declared Jakes correlation, so covariance/aging mismatch can be studied explicitly.

Controls: `PortGuard`, `fixed`, `random`, `scan`, `stale-greedy`, `mean-only`, `no-cost`, `no-hysteresis`, `no-risk`, `uniform-probes`, `no-delay`, `oracle`. The run command evaluates the listed controls together.

```bash
python -m portguard.compare --inputs "runs/*/metrics.csv" --reference PortGuard --metrics spectral_efficiency outage cvar_throughput switching_rate --unit seed
```

The comparison command averages repeated rows within the specified independent unit, bootstraps units and applies paired Wilcoxon/Holm comparisons per metric.

