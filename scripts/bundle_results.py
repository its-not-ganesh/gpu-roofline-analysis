"""Archive one machine's measurements and plots for independent analysis."""
import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results_final")
    parser.add_argument("--env", required=True, help="Machine directory under --input")
    parser.add_argument("--figures", default="figures_final", help="Plots for this machine only")
    parser.add_argument("--output", default="experiment_data_bundle.zip")
    parser.add_argument("--omit-nsight-reports", action="store_true")
    args = parser.parse_args()
    if Path(args.env).name != args.env or args.env in {".", ".."}:
        parser.error("--env must be a single machine directory name")
    machine = Path(args.input) / args.env
    figures = Path(args.figures)
    files = {}
    for source, prefix in ((machine, f"results/{args.env}"), (figures, "figures")):
        if not source.is_dir():
            parser.error(f"Missing evidence directory: {source}")
        for path in source.rglob("*"):
            if path.is_symlink() or not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix == ".ncu-rep" and args.omit_nsight_reports:
                continue
            if suffix in {".json", ".csv", ".txt", ".png", ".ncu-rep"}:
                if ".pending." not in path.name:
                    files[f"{prefix}/{path.relative_to(source).as_posix()}"] = path
    if not files:
        parser.error("No experiment evidence found")
    with ZipFile(args.output, "x", ZIP_DEFLATED) as archive:
        for name, path in sorted(files.items()):
            archive.write(path, name)
    with ZipFile(args.output) as archive:
        corrupt = archive.testzip()
        if corrupt:
            raise SystemExit(f"Archive verification failed: {corrupt}")
    print(f"Verified {len(files)} files in {args.output}")


if __name__ == "__main__":
    main()
