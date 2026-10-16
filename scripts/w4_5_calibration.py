"""Report fixed C/D live value calibration; never select or modify checkpoints.

The input benchmark must reference FULL battles.jsonl logs, not smoke runs.
Phase boundaries count zero-based neural decisions, not simulator turns. Ties
are excluded because the existing binary value metric contract requires 0/1.

Usage: ``python scripts/w4_5_calibration.py --benchmark
reports/w4_5_ablation.json --final-holdout reports/w4_5_test_results.json``.
Library entry points are ``load_benchmark`` and ``calibration_report``.
Each weighting/phase reports Brier, ROC AUC, 10-bin ECE, reliability,
population probability standard deviation, and mean absolute P(win)-0.5.
Trajectory-equal metrics give each nonempty trajectory unit mass in that
phase; they do NOT score an averaged prediction. Reliability count is the
raw number of decisions; weight_fraction carries the metric's weighting.
Bootstrap draws resample eligible battles, retaining all their decisions.
Scalar CIs omit undefined one-class AUC draws and report their counts.
Brier/dispersion/ECE use battle sufficient statistics; AUC sorts only once.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from pokeai.training.value_metrics import value_metrics

SCALARS = ("brier", "roc_auc", "ece", "std_probability", "mean_absolute_p_minus_half")
PHASES = {"all": (0, None), "0-5": (0, 6), "6-15": (6, 16),
          "16-30": (16, 31), "31+": (31, None),
          "first": (0, 1), "last": (-1, None)}


def _read_json(path):
    return json.loads(Path(path).read_text())


def load_benchmark(benchmark):
    """Load C/D full logs using the benchmark's explicit per-arm paths."""
    report = _read_json(benchmark)
    groups = {}
    sources = {}
    for opponent, entry in report["opponents"].items():
        groups[opponent] = {}
        sources[opponent] = {}
        for arm in ("C", "D"):
            spec = entry["arms"][arm]
            path = Path(spec["battles_path"])
            if not path.is_file() and not path.is_absolute():
                path = Path(benchmark).parent / path
            with path.open() as handle:
                records = [json.loads(line) for line in handle if line.strip()]
            if not records:
                raise ValueError(f"Empty full benchmark log: {path}")
            groups[opponent][arm] = records
            sources[opponent][arm] = str(path)
    if not groups:
        raise ValueError("Benchmark has no opponents")
    return groups, sources


def _trajectories(records):
    trajectories, excluded = [], {"ties": 0}
    seen = set()
    for row in records:
        if row.get("error") is not None:
            raise ValueError(f"Failed benchmark battle: {row['battle_id']}")
        index = row["matchup_index"]
        if index in seen:
            raise ValueError(f"Duplicate matchup_index: {index}")
        seen.add(index)
        if row["winner"] == "tie":
            excluded["ties"] += 1
            continue
        if row["winner"] not in ("player", "opponent"):
            raise ValueError(f"Unresolved benchmark battle: {row['battle_id']}")
        p = np.asarray(row["values"]["player"], dtype=np.float64)
        y = int(row["winner"] == "player")
        value_metrics(p, np.full(p.size, y))  # Shared binary-input validation.
        if not p.size:
            raise ValueError(f"Missing live values: {row['battle_id']}")
        trajectories.append((p, y))
    return trajectories, excluded


class _Distribution:
    """Sorted decisions plus per-battle sufficient statistics for resampling."""

    def __init__(self, trajectories, phase, equally_weighted):
        lower, upper = PHASES[phase]
        selected = [(p[lower:upper], y) for p, y in trajectories]
        selected = [(p, y) for p, y in selected if p.size]
        self.battles = len(selected)
        lengths = np.asarray([p.size for p, _ in selected], dtype=int)
        self.p = np.concatenate([p for p, _ in selected]) if selected else np.array([])
        self.y = np.concatenate([np.full(p.size, y) for p, y in selected]) if selected else np.array([])
        self.ids = np.repeat(np.arange(self.battles), lengths)
        self.weights = (1 / lengths[self.ids] if equally_weighted
                        else np.ones(self.p.size))
        self.bins = np.minimum((self.p * 10).astype(int), 9)
        self.order = np.argsort(self.p, kind="stable")
        sorted_p = self.p[self.order]
        self.starts = np.r_[0, np.flatnonzero(np.diff(sorted_p)) + 1] if self.p.size else np.array([], dtype=int)
        # Brier and dispersion require only these battle sums in every replicate.
        columns = [self.weights, self.weights * self.p,
                   self.weights * self.p ** 2,
                   self.weights * (self.p - self.y) ** 2,
                   self.weights * np.abs(self.p - 0.5)]
        self.sums = np.column_stack([
            np.bincount(self.ids, weights=c, minlength=self.battles)
            for c in columns
        ])
        self.bin_mass = np.zeros((self.battles, 10))
        self.bin_residual = np.zeros((self.battles, 10))
        np.add.at(self.bin_mass, (self.ids, self.bins), self.weights)
        np.add.at(self.bin_residual, (self.ids, self.bins),
                  self.weights * (self.p - self.y))

    def scalars(self, multiplicities):
        totals = multiplicities @ self.sums
        mass = totals[0]
        if not mass:
            return dict.fromkeys(SCALARS)
        mean = totals[1] / mass
        weights = (self.weights * multiplicities[self.ids])[self.order]
        labels = self.y[self.order]
        positives = np.add.reduceat(weights * labels, self.starts)
        negatives = np.add.reduceat(weights * (1 - labels), self.starts)
        pos, neg = positives.sum(), negatives.sum()
        auc = (float(np.dot(positives, np.cumsum(negatives) - negatives / 2)
                     / (pos * neg)) if pos and neg else None)
        return {
            "brier": float(totals[3] / mass), "roc_auc": auc,
            "ece": float(np.abs(multiplicities @ self.bin_residual).sum() / mass),
            "std_probability": float(np.sqrt(max(0, totals[2] / mass - mean ** 2))),
            "mean_absolute_p_minus_half": float(totals[4] / mass),
        }

    def report(self, rng, samples):
        base = value_metrics(self.p, self.y)
        unit = np.ones(self.battles)
        base.update(self.scalars(unit))
        mass = self.weights.sum()
        base["battles"] = self.battles
        base["mean_probability"] = float(np.dot(self.weights, self.p) / mass) if mass else None
        base["base_rate"] = float(np.dot(self.weights, self.y) / mass) if mass else None
        rate = base["base_rate"]
        base["references"] = {
            "constant_0.5": {"probability": .5, "brier": .25 if mass else None},
            "empirical_base_rate": {
                "probability": rate,
                "brier": float(rate * (1 - rate)) if rate is not None else None,
                "interpretation": "in-sample descriptive constant, not a fitted predictor or holdout claim",
            },
        }
        # Replace all weighting-sensitive fields, not just the requested scalars.
        if mass:
            clipped = np.clip(self.p, 1e-15, 1 - 1e-15)
            base["bce"] = float(np.dot(self.weights, -self.y * np.log(clipped)
                                      - (1 - self.y) * np.log1p(-clipped)) / mass)
            base["accuracy"] = float(np.dot(self.weights, (self.p >= .5) == self.y) / mass)
        for index, row in enumerate(base["reliability"]):
            mask = self.bins == index
            bin_mass = self.weights[mask].sum()
            row["weight_fraction"] = float(bin_mass / mass) if mass else None
            row["mean_probability"] = float(np.dot(self.weights[mask], self.p[mask]) / bin_mass) if bin_mass else None
            row["observed_rate"] = float(np.dot(self.weights[mask], self.y[mask]) / bin_mass) if bin_mass else None
        draws = {key: [] for key in SCALARS}
        if self.battles:
            for _ in range(samples):
                counts = rng.multinomial(self.battles, np.full(self.battles, 1 / self.battles))
                for key, value in self.scalars(counts).items():
                    if value is not None:
                        draws[key].append(value)
        base["ci95"] = {key: _interval(values, samples) for key, values in draws.items()}
        return base


def _interval(values, samples):
    limits = np.quantile(values, [.025, .975]).tolist() if len(values) else [None, None]
    return {"lower": limits[0], "upper": limits[1],
            "defined_replicates": len(values), "requested_replicates": samples}


def _summarize(records, rng, samples):
    trajectories, excluded = _trajectories(records)
    return {"input_battles": len(records), "excluded": excluded,
            "decision_weighted": {
                phase: _Distribution(trajectories, phase, False).report(rng, samples)
                for phase in PHASES},
            "trajectory_equally_weighted": {
                phase: _Distribution(trajectories, phase, True).report(rng, samples)
                for phase in PHASES}}


def _brier_difference(groups, rng, samples):
    # Resample matched scheduled battles within opponent strata. Outcomes may
    # differ between arms; only the schedule/team pairing is shared.
    strata = []
    for opponent, arms in groups.items():
        indexed = [{row["matchup_index"]: row for row in arms[arm]} for arm in ("C", "D")]
        if indexed[0].keys() != indexed[1].keys():
            raise ValueError(f"Unpaired C/D schedules for {opponent}")
        paired = []
        for index in sorted(indexed[0]):
            c, d = [rows[index] for rows in indexed]
            for key in ("player_team_sha256", "opponent_team_sha256"):
                if not c.get(key) or c[key] != d.get(key):
                    raise ValueError(f"C/D team pairing mismatch: {opponent}/{index}")
            if "tie" in (c["winner"], d["winner"]):
                continue
            values = []
            for row in (c, d):
                p = np.asarray(row["values"]["player"])
                loss = (p - int(row["winner"] == "player")) ** 2
                values.extend((loss.sum(), len(loss), loss.mean()))
            paired.append(values)
        if paired:
            strata.append(np.asarray(paired))
    def estimate(totals):
        if not totals[1] or not totals[4]:
            return {"decision_weighted": None, "trajectory_equally_weighted": None}
        return {"decision_weighted": float(totals[0] / totals[1] - totals[3] / totals[4]),
                "trajectory_equally_weighted": float((totals[2] - totals[5]) / pairs)}
    pairs = sum(len(s) for s in strata)
    point = estimate(sum((s.sum(axis=0) for s in strata), np.zeros(6)))
    draws = {key: [] for key in point}
    for _ in range(samples if pairs else 0):
        total = np.zeros(6)
        for stratum in strata:
            counts = rng.multinomial(len(stratum), np.full(len(stratum), 1 / len(stratum)))
            total += counts @ stratum
        for key, value in estimate(total).items():
            draws[key].append(value)
    return {"paired_non_tie_battles": pairs, "estimate": point,
            "ci95": {key: _interval(values, samples) for key, values in draws.items()},
            "direction": "C minus D; negative favors C",
            "tie_policy": "exclude a pair if either arm tied"}


def calibration_report(groups, *, samples=5000, seed=20261016, final_holdout=None):
    """Build JSON-safe live calibration metrics with seeded battle-cluster CIs."""
    if samples < 5000:
        raise ValueError("At least 5000 battle-bootstrap replicates are required")
    rng = np.random.default_rng(seed)
    opponents = {}
    pooled = {arm: [] for arm in ("C", "D")}
    for opponent, arms in sorted(groups.items()):
        opponents[opponent] = {arm: _summarize(arms[arm], rng, samples) for arm in ("C", "D")}
        # Keep matchup IDs unique for validation when pooling separate runs.
        for arm, rows in pooled.items():
            start = len(rows)
            rows.extend(dict(row, matchup_index=start + i)
                        for i, row in enumerate(arms[arm]))
    return {
        "schema_version": 1, "arms": {"C": "probe production candidate", "D": "joint ablation only"},
        "bootstrap": {"unit": "battle (all decisions retained together)",
                      "replicates": samples, "seed": seed, "interval": "95% percentile",
                      "undefined_auc": "one-class replicates omitted; counts reported"},
        "phase_definition": "zero-based neural decision index, NOT simulator turn",
        "weighting": {"decision_weighted": "each decision has unit mass",
                      "trajectory_equally_weighted": "each nonempty trajectory has unit mass within each phase"},
        "tie_policy": "exclude ties from binary live metrics; counts reported",
        "opponents": opponents,
        "pooled": {arm: _summarize(rows, rng, samples) for arm, rows in pooled.items()},
        "c_minus_d_brier": {"pooled": _brier_difference(groups, rng, samples),
                            "opponents": {opp: _brier_difference({opp: arms}, rng, samples)
                                          for opp, arms in sorted(groups.items())}},
        "final_holdout": final_holdout,
        "selection": "descriptive fixed-arm evaluation only; no model selection",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=Path("reports/w4_5_ablation.json"))
    parser.add_argument("--final-holdout", type=Path)
    parser.add_argument("--out", type=Path, default=Path("reports/w4_5_value_calibration.json"))
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261016)
    args = parser.parse_args()
    groups, sources = load_benchmark(args.benchmark)
    report = calibration_report(groups, samples=args.bootstrap_samples, seed=args.seed,
                                final_holdout=_read_json(args.final_holdout) if args.final_holdout else None)
    report["sources"] = {"benchmark": str(args.benchmark), "battles": sources,
                         "final_holdout": str(args.final_holdout) if args.final_holdout else None}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
