"""Stage legacy analysis scripts OUTSIDE the public repository; never run models implicitly."""
from __future__ import annotations
import argparse, hashlib, json, shutil
from pathlib import Path

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parent
    work = args.work_dir.expanduser().resolve()
    if work == repo or repo in work.parents or work in repo.parents:
        raise SystemExit("Choose a private workspace outside and not containing the public repository.")
    if work.exists() and any(work.iterdir()):
        raise SystemExit("Workspace is not empty; choose a new directory. Existing analyses are never overwritten.")
    work.mkdir(parents=True, exist_ok=True)
    plan = []
    def stage(src: Path, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        plan.append({"source": str(src.relative_to(repo)), "staged": str(dest.relative_to(work)),
                     "sha256": hashlib.sha256(dest.read_bytes()).hexdigest()})
    for src in (repo / "src").glob("*.py"):
        if src.name == "analyze_signal_audit.py":
            dst = work / "03_stage05_signal_audit/00_scripts" / src.name
        elif src.name == "run_prewrite_analysis.py":
            dst = work / "09_prewrite_work/00_scripts" / src.name
        elif src.name == "run_level_only_comparator.py":
            dst = work / "12_final_presubmission_v2_2_work" / src.name
        else:
            dst = work / "04_formal_modeling/00_scripts" / src.name
        stage(src, dst)
    stage(repo / "SQL/extract_cohort_flow.ps1", work / "03_stage0/extract_cohort_flow.ps1")
    stage(repo / "SQL/extract_signal_data.ps1", work / "03_stage05_signal_audit/00_scripts/extract_signal_data.ps1")
    stage(repo / "SQL/extract_formal_data.ps1", work / "04_formal_modeling/00_scripts/extract_formal_data.ps1")
    for name in ("mimic_clinical_characteristics.sql", "sicdb_clinical_characteristics.sql"):
        stage(repo / "SQL" / name, work / "clinical_characteristics" / name)
    # Legacy figure assembly expects the earlier source-tree layout. This copy contains only public aggregates.
    for src in (repo / "figures/source_data").glob("*.csv"):
        stage(src, work / "11_delivery_cc_handoff/FINAL_WRITING_HANDOFF_CC/figures/source_data" / src.name)
    (work / "STAGING_MANIFEST.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print("PRIVATE_WORKSPACE_STAGED", work)
    print("No database access, model fitting, or result regeneration was performed.")
    print("Follow REPRODUCIBILITY.md and retain all patient-level products in this private workspace.")
if __name__ == "__main__":
    main()
