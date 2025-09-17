# Timing Reporter

## Overview
Timing Reporter converts CSV timing exports into shareable visual reports for endurance or sprint racing sessions. The `lap_analysis.py` script parses each driver's laps and sector splits, detects pit-in/pit-out laps, and assembles interactive dashboards that highlight who is fastest and how consistent each stint was.【F:lap_analysis.py†L190-L275】【F:lap_analysis.py†L401-L520】

The tool can operate fully from the command line or via graphical file pickers. By default it generates an interactive Plotly HTML report and can optionally export a multi-page PDF that mirrors the same charts for offline distribution.【F:lap_analysis.py†L520-L723】【F:lap_analysis.py†L784-L833】

## Key Features
- Parses raw timing CSV files with lap time and sector columns, handling decimal commas or dots automatically.【F:lap_analysis.py†L270-L303】
- Automatically assigns drivers, tags pit in/out laps based on a configurable threshold, and computes per-driver summaries (best lap, theoretical lap, rolling averages, etc.).【F:lap_analysis.py†L220-L266】【F:lap_analysis.py†L333-L384】
- Builds interactive Plotly dashboards with lap-time evolution, sector traces, distribution boxplots, scatter comparisons, and a sortable summary table colored per driver with category filters.【F:lap_analysis.py†L520-L723】
- Optional PDF export that renders the same figures using Plotly + Kaleido and stitches them into a single file with Pillow.【F:lap_analysis.py†L723-L833】
- Windows helper script (`run_report.bat`) that bootstraps a virtual environment, installs dependencies, runs the generator, and opens the resulting report automatically.【F:run_report.bat†L1-L33】

## Requirements
- Python 3.10 or later (the codebase relies on modern type-hint syntax such as `str | None`).
- Core dependencies listed in `requirements.txt`: pandas, numpy, plotly, kaleido, and Pillow. Plotly and Kaleido are required for interactive charts, while Pillow is only needed for PDF export.【F:requirements.txt†L1-L5】
- Tkinter must be available for the optional file dialogs (included with most desktop Python distributions).

## Installation
1. (Optional) Create and activate a virtual environment.
2. Install the Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure your timing CSV file follows the expected format (see [Input data expectations](#input-data-expectations)).

On Windows you can simply run `run_report.bat`; it creates a `.venv`, installs requirements, executes the report generator, and opens the output folder once finished.【F:run_report.bat†L1-L33】

## Usage
```bash
python lap_analysis.py [options]
```
If `--csv` is omitted the script opens a file picker (or falls back to a console prompt). The HTML output path is also chosen through a save dialog unless the program is running headless.【F:lap_analysis.py†L835-L915】

### Common options
| Option | Description |
| --- | --- |
| `--csv PATH` | Path to the timing CSV file. Leaving it blank triggers a file picker. |
| `--outdir DIR` | Output directory used to seed the save dialog (defaults to `timing_report`).【F:lap_analysis.py†L856-L864】 |
| `--max-lap-sec SECONDS` | Visual filter that hides laps slower than the given threshold (default `200`).【F:lap_analysis.py†L844-L850】【F:lap_analysis.py†L520-L555】 |
| `--max-sector-sec SECONDS` | Similar visual filter for sector charts (default `100`).【F:lap_analysis.py†L844-L850】【F:lap_analysis.py†L520-L555】 |
| `--pit-gap-sec SECONDS` | Gap above the driver median used to flag pit-in laps and the following out-lap (default `25`).【F:lap_analysis.py†L848-L856】【F:lap_analysis.py†L303-L333】 |
| `--hide-out-in` | Excludes pit in/out laps from the interactive charts and optional PDF.【F:lap_analysis.py†L848-L856】【F:lap_analysis.py†L520-L723】【F:lap_analysis.py†L723-L833】 |
| `--invert-y` | Reverses the Y axis so faster laps appear higher on the charts.【F:lap_analysis.py†L848-L856】【F:lap_analysis.py†L545-L555】【F:lap_analysis.py†L777-L805】 |
| `--interactive` | Reserved flag for switching between static and interactive output. The current version always produces the interactive HTML workflow.【F:lap_analysis.py†L838-L850】【F:lap_analysis.py†L520-L723】 |
| `--pdf` | Additionally export a PDF copy of the report (requires Kaleido and Pillow).【F:lap_analysis.py†L851-L916】【F:lap_analysis.py†L723-L833】 |

Example:
```bash
python lap_analysis.py --csv "CARRERA - TREINO OFICIAL 1 - laptimes.csv" \
  --outdir timing_report --hide-out-in --invert-y --pdf
```

## Input data expectations
The parser expects CSV exports where:
- Each driver name appears in the `Time of Day` column on a row where the `Lap` field is blank, matching the format exported by many MyLaps-based timing systems.【F:lap_analysis.py†L270-L283】
- Lap data rows contain numeric `Lap` identifiers plus `Lap Tm`, `S1 Tm`, `S2 Tm`, and `S3 Tm` columns (sector columns are optional but recommended).【F:lap_analysis.py†L283-L303】
- Decimal values may use commas or periods; both are converted automatically.【F:lap_analysis.py†L244-L266】

If your timing software exports a different schema, adjust the column mapping in `parse_lap_data` accordingly.【F:lap_analysis.py†L270-L303】

## Outputs
- **Interactive HTML report** containing five Plotly charts (lap times, three sectors, and distributions) plus a sortable summary table with per-driver metrics and category filters.【F:lap_analysis.py†L520-L723】
- **Optional PDF report** mirroring the charts for offline sharing.【F:lap_analysis.py†L723-L833】
- Intermediate PNGs and static HTML helpers are available through `plot_all`/`build_html` if you want to integrate the library in other workflows.【F:lap_analysis.py†L404-L463】

The script prints the final file locations to stdout once generation is complete.【F:lap_analysis.py†L903-L916】

## Troubleshooting
- Make sure Plotly, Kaleido, and Pillow are installed if you request PDF output; missing packages raise descriptive runtime errors.【F:lap_analysis.py†L730-L742】【F:lap_analysis.py†L520-L555】
- On Linux servers without Tkinter or a GUI, provide `--csv` and precreate the destination paths so the script does not fall back to console prompts.
- If lap or sector charts are empty, review the filtering thresholds (`--max-lap-sec` / `--max-sector-sec`) and ensure the CSV contains numeric data.
