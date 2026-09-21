# AI-Based Traffic Optimization in a Digital Twin of Girona

A reproducible research project that evaluates whether reinforcement learning can reduce urban traffic congestion compared with traditional traffic-light control in a SUMO-based digital twin of Girona, Spain.

## Research questions

1. Can reinforcement learning reduce congestion relative to fixed-time and actuated traffic-light control in a simulated real-world urban network?
2. Does a controller trained under one traffic-demand distribution generalize to unseen traffic conditions?

## Scope

The initial digital twin models a compact area around Plaça Catalunya and the Eixample district of Girona. Keeping the network to approximately 4–10 intersections enables controlled, repeatable experiments before scaling the system.

## Project status

- [x] Development environment: Python 3.11, SUMO 1.27.1, TraCI
- [ ] Digital twin: import and validate the Girona road network
- [ ] Traffic scenarios: normal, low, peak, variable demand, incidents
- [ ] Prediction: persistence baseline, tree model, sequence model
- [ ] Traffic control: fixed-time, actuated, RL controller
- [ ] Evaluation: repeated experiments, statistical summaries, plots

## Evaluation metrics

- Average waiting time
- Queue length
- Average travel time
- Throughput
- Number of stops
- SUMO emissions where applicable

## Repository layout

```text
├── app/                      # Optional visualisation or demo
├── configs/                  # Reproducible experiment configuration
├── data/osm/                 # OpenStreetMap source data
├── experiments/              # Experiment definitions and tracked results
├── models/                   # Locally generated model checkpoints
├── notebooks/                # Exploratory analysis only
├── report/                   # Research report and figures
├── simulation/               # SUMO networks, routes and configurations
├── src/                      # Reusable Python source code
│   └── simulation/
├── tests/                    # Automated tests
└── requirements.txt          # Python dependencies
```

## Setup

### Prerequisites

- Python 3.11
- SUMO 1.27.1 or compatible, including `sumo`, `sumo-gui`, `netconvert`, and the SUMO tools

### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
sumo --version
```

If `sumo` is not found after installation, restart the terminal so it reloads the system `PATH`.

## Reproducibility principles

Experiments use explicit configurations, deterministic random seeds where possible, fixed baselines, repeated runs across scenarios, and versioned source networks. Generated model checkpoints and high-volume simulation outputs are excluded from Git.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
