# PortGuard

Implementation of **Risk- and switching-aware fluid-antenna port selection with delayed CSI.**

## Installation

Use Python 3.10+. Create and activate a virtual environment before installing:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

In PowerShell, replace the activation line with `.venv\Scripts\Activate.ps1`.

## Data and preprocessing

`prepare` generates a complex spatially correlated channel tape and independent measurement noise. The covariance combines a Jakes/Bessel term with exponential distance decay. Every controller sees the same tape; only the full-CSI oracle accesses current complete channels. There is no external dataset requirement for this simulation.

```bash
python -m portguard prepare --config config.json --output runs/default
```

## Training and inference

```bash
python -m portguard run --config config.json --output runs/default
```

## Evaluation

```bash
python -m portguard evaluate --config config.json --output runs/default
python -m portguard plot --config config.json --output runs/default
```

## Outputs

`channel.npz` is the common tape. `decisions.csv` records actions/rates/movement/timing; `metrics.csv` summarizes throughput/outage/tail/switching; `tradeoff.png` visualizes the trade-off.

Keep `config.json` with the run; it contains model/data settings and the seed where the CLI does not expose one. Use separate output directories for different seeds/conditions.

Source inspection: Python 3.13.5, numpy 2.1.3, scipy 1.15.3, pandas 2.2.3, matplotlib 3.10.0 (installed versions inspected). Checks covered Python syntax, configuration parsing and CLI arguments. No training, full evaluation or CI was run in this update.
