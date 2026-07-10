from __future__ import annotations

import argparse
from pathlib import Path

from tandem_rlvr.analysis import generate_all_figures, generate_reports, load_available_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Stage 7 research figures and reports from saved TandemRLVR outputs.")
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    result = run_stage7_generate_report(args.outputs_dir, args.report_dir)
    print("Stage 7 report generation complete", flush=True)
    print(f"Outputs directory: {result['outputs_dir']}", flush=True)
    print(f"Report directory: {result['report_dir']}", flush=True)
    if result["missing_files"]:
        print("Missing optional files:", flush=True)
        for missing in result["missing_files"]:
            print(f"  - {missing}", flush=True)
    if result["figures"]:
        print("Generated figures:", flush=True)
        for name, path in result["figures"].items():
            print(f"  - {name}: {path}", flush=True)
    else:
        print("No figures generated because no compatible Stage 5 or Stage 6 data was found.", flush=True)
    print("Generated reports:", flush=True)
    for name, path in result["reports"].items():
        print(f"  - {name}: {path}", flush=True)


def run_stage7_generate_report(outputs_dir: str | Path, report_dir: str | Path) -> dict[str, object]:
    outputs_path = Path(outputs_dir)
    report_path = Path(report_dir)
    figures_dir = report_path / "figures"
    results = load_available_results(outputs_path)
    figures = generate_all_figures(results, figures_dir)
    reports = generate_reports(results, report_path, figures)
    return {
        "outputs_dir": outputs_path,
        "report_dir": report_path,
        "missing_files": results.missing_files,
        "figures": figures,
        "reports": reports,
    }


if __name__ == "__main__":
    main()
