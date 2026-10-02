# Sample input files

Small example trips for trying the analyzer without downloading VED:

```bash
python scripts/run_pipeline.py --log data/samples/vacuum_leak.csv --dtc-file data/samples/vacuum_leak_dtc.csv
```

| File | Content | Fault code |
|---|---|---|
| `normal_trip.csv` | Healthy trip | - |
| `vacuum_leak.csv` | Vacuum leak | P0171 |
| `maf_drift.csv` | Dirty MAF sensor | P0101 |
| `rich_injector.csv` | Leaking injector | P0172 |
| `misfire.csv` | Misfire | P0300 |
| `speed_sensor_failure.csv` | Failing speed sensor | P0500 |
| `hv_battery_degradation.csv` | Hybrid battery wear | P0A7F |

Each `<name>_dtc.csv` holds the fault code and the time it was set (`code,time_ms`).

These are real VED trips with **simulated** faults injected (see `../README.md`), picked from trips the
model diagnoses correctly, so they demonstrate the output rather than measure accuracy. Regenerate them
with `python scripts/make_samples.py`.

Source: [Vehicle Energy Dataset (VED)](https://github.com/gsoh/VED), Apache License 2.0.
