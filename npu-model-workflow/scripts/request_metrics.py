#!/usr/bin/env python3
"""Summarize one finalized request cohort; not a model acceptance checker."""

import argparse
import json
import math
from pathlib import Path


def percentile(values, fraction):
    """Nearest-rank quantile. All timestamps in a cohort use one clock."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def summarize(rows, *, window_start_ns, window_end_ns, expected_requests,
              min_requests, p99_limit_ms, max_send_lag_ms,
              latency_start="scheduled_ns", max_error_rate=0.0):
    if (type(window_start_ns) is not int or type(window_end_ns) is not int
            or window_end_ns <= window_start_ns):
        raise ValueError("window must have integer ns timestamps and positive duration")
    if (type(expected_requests) is not int or expected_requests < 1
            or type(min_requests) is not int or min_requests < 1):
        raise ValueError("expected_requests and min_requests must be positive integers")
    for name, value in [("p99_limit_ms", p99_limit_ms),
                        ("max_send_lag_ms", max_send_lag_ms),
                        ("max_error_rate", max_error_rate)]:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(name + " must be finite")
    if p99_limit_ms <= 0 or max_send_lag_ms < 0 or not 0 <= max_error_rate <= 1:
        raise ValueError("invalid latency, send lag, or error-rate limit")
    if latency_start not in {"scheduled_ns", "sent_ns", "received_ns"}:
        raise ValueError("unsupported latency start")

    issues, latencies, lags = [], [], []
    ids = set()
    counts = {"measured": 0, "success": 0, "error": 0, "timeout": 0,
              "pending": 0, "success_completed_in_window": 0,
              "outstanding_at_window_end": 0}
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            issues.append("row %d is not an object" % index)
            continue
        if row.get("phase") == "warmup":
            continue
        if row.get("phase") != "measure":
            issues.append("row %d must explicitly identify phase" % index)
            continue
        counts["measured"] += 1
        prefix = "row %d: " % index
        rid = row.get("request_id")
        if not isinstance(rid, str) or not rid or rid in ids:
            issues.append(prefix + "request_id must be nonempty and unique")
        else:
            ids.add(rid)
        status = row.get("status")
        if status not in {"success", "error", "timeout", "pending"}:
            issues.append(prefix + "unknown status")
            continue
        counts[status] += 1
        scheduled, sent = row.get("scheduled_ns"), row.get("sent_ns")
        if type(scheduled) is not int or not window_start_ns <= scheduled < window_end_ns:
            issues.append(prefix + "scheduled_ns outside measurement window or not integer")
            continue
        if type(sent) is not int or sent < scheduled:
            issues.append(prefix + "sent_ns missing or before scheduled_ns")
            continue
        lags.append((sent - scheduled) / 1e6)
        received = row.get("received_ns")
        if received is not None and (type(received) is not int or received < sent):
            issues.append(prefix + "received_ns before sent_ns or not integer")
            continue
        if status == "pending":
            counts["outstanding_at_window_end"] += 1
            issues.append(prefix + "request not finalized; drain or record its timeout")
            continue
        completed, started = row.get("completed_ns"), row.get(latency_start)
        earliest = received if received is not None else sent
        if (type(completed) is not int or completed < earliest
                or type(started) is not int or completed < started):
            issues.append(prefix + "completion/start timestamps missing or out of order")
            continue
        latencies.append((completed - started) / 1e6)
        if completed >= window_end_ns:
            counts["outstanding_at_window_end"] += 1
        elif status == "success":
            counts["success_completed_in_window"] += 1

    if counts["measured"] != expected_requests:
        issues.append("measured count differs from expected requests; missing/extra arrivals")
    if counts["measured"] < min_requests:
        issues.append("insufficient request count for the declared contract")
    if len(latencies) != counts["measured"]:
        issues.append("not all measured requests have a usable finalized latency")
    lag_max = max(lags) if lags else None
    if lag_max is None or lag_max > max_send_lag_ms:
        issues.append("load generator send lag exceeds contract or is unavailable")
    valid = not issues
    # Never display a deceptively small p99 for only the completed subset.
    p99 = percentile(latencies, .99) if valid else None
    error_rate = ((counts["error"] + counts["timeout"]) / counts["measured"]
                  if counts["measured"] else None)
    duration_s = (window_end_ns - window_start_ns) / 1e9
    numeric_met = bool(valid and p99 <= p99_limit_ms and error_rate <= max_error_rate)
    return {
        "measurement_valid": valid,
        "numeric_constraints_met": numeric_met,
        "invalid_reasons": issues,
        "counts": counts,
        "duration_s": duration_s,
        "latency_start": latency_start,
        "quantile_method": "nearest-rank; includes finalized errors/timeouts",
        "p99_ms": p99,
        "max_send_lag_ms": lag_max,
        "error_rate": error_rate,
        "successful_completions_in_window_per_s": counts["success_completed_in_window"] / duration_s,
        "cohort_successes_per_window_s": counts["success"] / duration_s if valid else None,
        "requires_separate_evidence": [
            "steady-state queue/backlog and representative workload",
            "configured timing boundary and real output completion",
            "independent frozen-config runs, correctness pressure and stability",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", required=True, type=Path, help="finalized request JSONL")
    parser.add_argument("--window-start-ns", type=int, required=True)
    parser.add_argument("--window-end-ns", type=int, required=True)
    parser.add_argument("--expected-requests", type=int, required=True)
    parser.add_argument("--min-requests", type=int, required=True)
    parser.add_argument("--p99-limit-ms", type=float, required=True)
    parser.add_argument("--max-send-lag-ms", type=float, required=True)
    parser.add_argument("--max-error-rate", type=float, default=0)
    parser.add_argument("--latency-start", choices=["scheduled_ns", "sent_ns", "received_ns"],
                        default="scheduled_ns")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    try:
        rows = [json.loads(line) for line in args.requests.read_text().splitlines() if line.strip()]
        result = summarize(rows, window_start_ns=args.window_start_ns,
                           window_end_ns=args.window_end_ns,
                           expected_requests=args.expected_requests, min_requests=args.min_requests,
                           p99_limit_ms=args.p99_limit_ms, max_send_lag_ms=args.max_send_lag_ms,
                           max_error_rate=args.max_error_rate, latency_start=args.latency_start)
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    output = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.out:
        with args.out.open("x", encoding="utf-8") as stream:
            stream.write(output)
    else:
        print(output, end="")
    return 0 if result["numeric_constraints_met"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
