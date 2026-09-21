# Girona Gran Via–Eixample SUMO scenario

## MVP area

The first digital twin covers a compact road network around Gran Via de Jaume I and the Eixample district of Girona.

- Source bounding box in WGS84: `2.8175,41.9790,2.8255,41.9855` (`west,south,east,north`)
- OpenStreetMap source: `data/osm/girona_gran_via_eixample.osm`
- Network conversion retains the largest connected component usable by passenger vehicles.
- The generated network contains 261 edges, 150 junctions, and 14 traffic-light controllers.
- The first demand scenario lasts 1,800 seconds, uses seed `42`, and generates 360 passenger trips at a nominal period of 5 seconds.

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

## Generate the initial demand

```powershell
$env:SUMO_HOME = 'C:\Program Files (x86)\Eclipse\Sumo'
python "$env:SUMO_HOME\tools\randomTrips.py" `
  -n simulation\girona_gran_via_eixample.net.xml `
  -r simulation\girona_gran_via_eixample.rou.xml `
  -b 0 -e 1800 -p 5 --fringe-factor max --min-distance 250 `
  --seed 42 --validate --remove-loops
```

## Run

Headless validation:

```powershell
.\.venv\Scripts\python.exe src\simulation\run_simulation.py
```

Visual inspection:

```powershell
.\.venv\Scripts\python.exe src\simulation\run_simulation.py --gui
```

## Validation checklist

1. Open the GUI run and confirm that vehicles enter, traverse, and leave the network without teleportation.
2. Inspect every imported traffic-light controller in NetEdit before defining the fixed-time baseline.
3. Record any changed lane, connection, or signal-plan assumption in this directory or an experiment configuration.
