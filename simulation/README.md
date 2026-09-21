# Girona Gran Via–Eixample SUMO scenario

## MVP area

The first digital twin covers a compact road network around Gran Via de Jaume I and the Eixample district of Girona.

- Source bounding box in WGS84: `2.8175,41.9790,2.8255,41.9855` (`west,south,east,north`)
- OpenStreetMap source: `data/osm/girona_gran_via_eixample.osm`
- Network conversion retains the largest connected component usable by passenger vehicles.
- The generated network contains 261 edges, 150 junctions, and 14 traffic-light controllers.
- Fase 2 provides five 3,600-second, seeded passenger-demand scenarios: low, normal, peak, variable, and incident.

The network intentionally stays small enough for repeatable baseline and reinforcement-learning experiments. The OSM import is a structural model: OSM geometry, lane data, and signal metadata require later manual validation before the model can support real-world claims.

## Generate the network

From the repository root in PowerShell:

```powershell
$env:SUMO_HOME = 'C:\Program Files (x86)\Eclipse\Sumo'
& "$env:SUMO_HOME\bin\netconvert.exe" `
  --osm-files data\osm\girona_gran_via_eixample.osm `
  --output-file simulation\girona_gran_via_eixample.net.xml `
  --type-files "$env:SUMO_HOME\data\typemap\osmNetconvert.typ.xml" `
  --geometry.remove --ramps.guess --junctions.join `
  --tls.guess-signals --tls.discard-simple --tls.join --proj.utm `
  --keep-edges.by-vclass passenger --keep-edges.components 1 `
  --keep-edges.postload --remove-edges.isolated
```

## Generate scenario demand

Scenario definitions are versioned in `configs/scenarios/`. Generate an input deterministically with:

```powershell
.\.venv\Scripts\python.exe src\simulation\scenarios.py normal
```

Pass `--force` after changing the network or a scenario JSON. The variable scenario combines low, peak, and low demand blocks. The incident scenario additionally creates `simulation/additional/incident.add.xml`, which temporarily closes an edge and reroutes passenger vehicles.

## Run a scenario

Headless fixed-time baseline:

```powershell
.\.venv\Scripts\python.exe src\simulation\run_simulation.py --scenario normal
```

Visual inspection:

```powershell
.\.venv\Scripts\python.exe src\simulation\run_simulation.py --scenario incident --gui --no-save
```

The runner appends locally generated metrics to `experiments/baseline/fixed_time_metrics.csv`; this reproducible output is excluded from Git. See `experiments/baseline/README.md` for the baseline table and metric definitions.

## Validation checklist

1. Open the GUI run and confirm that vehicles enter, traverse, and leave the network without teleportation.
2. Confirm the incident closure reroutes vehicles between simulation seconds 1,200 and 1,800.
3. Inspect every imported traffic-light controller in NetEdit before defining the fixed-time baseline.
4. Record any changed lane, connection, or signal-plan assumption in this directory or an experiment configuration.
