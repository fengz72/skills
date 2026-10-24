#!/usr/bin/env python3
"""Check OPT coverage and local report links, not experiment truth or acceptance."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from urllib.parse import unquote

OPT = re.compile(r"OPT-\d{3,}")
LINK = re.compile(r"\[(OPT-\d{3,})\]\(([^)]+)\)")


def table_entries(text):
    entries = []
    for line in text.splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cells = re.split(r"(?<!\\)\|", line.strip())[1:-1]
        if not cells:
            continue
        first = cells[0].strip()
        link = LINK.fullmatch(first)
        if OPT.fullmatch(first):
            entries.append((first, None, cells))
        elif link:
            entries.append((link.group(1), link.group(2), cells))
    return entries


def audit(progress, experiments, report):
    progress, experiments, report = Path(progress), Path(experiments), Path(report)
    allocated = table_entries(progress.read_text())
    summary = table_entries(report.read_text())
    issues = []
    allocated_counts = Counter(row[0] for row in allocated)
    summary_counts = Counter(row[0] for row in summary)
    records = {path.stem: path for path in experiments.glob("OPT-*.md")}
    explicit_zero = re.search(r"^优化尝试数[：:]\s*0\s*$", progress.read_text(), re.M)
    if not allocated and not explicit_zero:
        issues.append("no OPT allocation table found in progress; coverage cannot be established")
    if allocated and explicit_zero:
        issues.append("progress declares zero attempts but allocates OPT records")
    for label, counts in [("progress", allocated_counts), ("report", summary_counts)]:
        for opt, count in counts.items():
            if count != 1:
                issues.append("%s contains %s %d times" % (label, opt, count))
    expected = set(allocated_counts)
    for label, observed in [("record files", set(records)), ("report rows", set(summary_counts))]:
        for opt in sorted(expected - observed):
            issues.append("%s missing %s" % (label, opt))
        for opt in sorted(observed - expected):
            issues.append("%s has unallocated %s" % (label, opt))
    for opt, path in records.items():
        lines = path.read_text().splitlines()
        if not lines or not re.match(r"^# " + re.escape(opt) + r"(?:[：:\s]|$)", lines[0]):
            issues.append("%s header does not identify its OPT" % path.name)
        sections = re.findall(r"^## ([1-7])\.", path.read_text(), re.M)
        if sections != list("1234567"):
            issues.append("%s does not keep the seven numbered record sections" % path.name)
    for opt, target, cells in summary:
        if len(cells) != 7 or any(not cell.strip() for cell in cells):
            issues.append("%s summary must fill all seven columns" % opt)
        if not target or "://" in target or target.startswith("#"):
            issues.append("%s summary needs a local record link" % opt)
            continue
        destination = (report.parent / unquote(target.split("#", 1)[0])).resolve()
        required = (experiments / (opt + ".md")).resolve()
        if destination != required or not destination.is_file():
            issues.append("%s summary link does not resolve to its record" % opt)
    return {"record_coverage_complete": not issues,
            "allocated": sorted(expected), "records": sorted(records),
            "summary": sorted(summary_counts), "issues": issues,
            "not_checked": ["metric values and changes", "real execution/precision/performance evidence",
                            "adoption agrees with final artifact", "placeholder content"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--experiments", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if not args.experiments.is_dir():
        parser.error("experiments must be an existing directory")
    try:
        result = audit(args.progress, args.experiments, args.report)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["record_coverage_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
