# Fixed-time baseline results

`fixed_time_metrics.csv` is generated locally by running each configured scenario with SUMO's imported static traffic-light programs. It is intentionally excluded from Git because results are reproducible from the versioned network, scenario definitions, code, SUMO version, and seed.

## Generate the baseline table

From the repository root in PowerShell:

```powershell
$scenarios = 'low', 'normal', 'peak', 'variable', 'incident'
Remove-Item experiments\baseline\fixed_time_metrics.csv -ErrorAction Ignore
foreach ($scenario in $scenarios) {
  .\.venv\Scripts\python.exe src\simulation\run_simulation.py --scenario $scenario
}
```

## Latest validation run

| Scenario | Completed trips | Avg. waiting time (s) | Avg. travel time (s) | Avg. queue length | Max. queue length | Throughput (veh/h) | Teleports |
|---|---:|---:|---:|---:|---:|---:|---:|
| Low | 360 | 29.34 | 171.05 | 2.72 | 11 | 360 | 0 |
| Normal | 720 | 28.54 | 167.33 | 5.22 | 18 | 720 | 0 |
| Peak | 1,800 | 34.40 | 180.77 | 15.34 | 43 | 1,800 | 0 |
| Variable | 840 | 31.87 | 180.22 | 6.69 | 40 | 840 | 0 |
| Incident | 720 | 29.47 | 170.37 | 5.38 | 22 | 720 | 0 |

These are single-seed structural-validation results, not inferential comparisons. Future controller comparisons must use the same scenario definition and multiple seeds.

## Metric definitions

- **Completed trips**: vehicles that reach their destination, including after the nominal 3,600-second demand horizon.
- **Average waiting time**: SUMO `tripinfo` waiting time averaged over completed trips.
- **Average travel time**: SUMO `tripinfo` duration averaged over completed trips.
- **Average queue length**: time average of halted passenger vehicles across non-internal lanes, sampled once per simulation second.
- **Max. queue length**: maximum of that queue measure over the run.
- **Throughput**: completed trips normalized by the 3,600-second demand horizon.
- **Teleports**: vehicles beginning a SUMO teleport; zero is required for a valid baseline run.

## Incident definition

The incident scenario temporarily closes edge `-166571453#2` to passenger traffic from 1,200 to 1,800 seconds. It uses a SUMO rerouter on three upstream notification edges. The closed edge was selected because all affected route pairs retain a feasible passenger-vehicle detour in the directed network; this makes the experiment a rerouting and congestion test rather than a disconnected-network failure.
