"""The benchmark-global, REGION-level label-access finalization pass.

Extracted so the full split generator and the narrow label-access migration run byte-identical
science. This module deliberately does NOT import or expose any assignments writer: it can only ever
produce ``label_access.csv`` and the label-access block of the per-leaf summaries.
"""

from __future__ import annotations

from evals import split_artifacts as SA
from evals import split_spec


def finalize(root, benchmark, candidates, *, audit_only):
    """The benchmark-global label-access pass. The unit of decision is the TARGET REGION, not the
    (region, seed) cell, in this order:

      1. structural ELIGIBILITY -- a region enters the candidate set only when EVERY one of its seed
         cells qualifies (sizing only, no peeking at results)
      2. ``B_d = min(B_max,d, min-cell N_source, min-cell N_target)`` over every seed cell belonging to
         those eligible regions
      3. construct the frozen order and the five allocation sets from that B_d
      4. AUDIT the realized sets; a failure in ANY seed demotes the WHOLE region across all seeds
      5. B_d is NOT recomputed after the audit
      6. a benchmark left with fewer than ``MIN_HEADLINE_TARGETS`` surviving regions loses the headline
         allocation aggregate

    Region-level is the only defensible granularity: the contrasts average seeds within a region and
    bootstrap regions, so a region present at seeds 0 and 2 but absent at seed 1 would silently become a
    two-seed mean sitting beside three-seed means. Keeping B_d fixed across step 4 matters for the same
    reason -- recomputing it after demotion would let the audit outcome move the budget, so the surviving
    regions would be measured at a budget chosen partly by which regions failed.

    Writes each survivor's label_access.csv and records the decision -- B_d, eligibility, exclusions,
    fractions, realized counts -- on the per-leaf summaries that become ``splits.json``."""
    if not candidates:
        return
    spec = split_spec.ALL_SPECS[benchmark]
    unit = SA.label_access_unit(benchmark)

    by_region: dict[str, list] = {}
    for c in candidates:
        by_region.setdefault(c["holdout"], []).append(c)
    for cells in by_region.values():
        cells.sort(key=lambda c: int(c["seed"]))

    # 1. structural eligibility, ALL-SEEDS-OR-NONE
    eligible: dict[str, list] = {}
    excluded: dict[str, list[str]] = {}
    for region, cells in sorted(by_region.items()):
        reasons = [
            f"seed={c['seed']}: {w}"
            for c in cells
            for w in SA.label_access_eligibility(n_source=c["n_source"], n_target_pool=c["n_target_pool"])
        ]
        if reasons:
            excluded[region] = reasons
        else:
            eligible[region] = cells
    if not eligible:
        raise SA.SplitArtifactError(
            f"{benchmark}: no geographic_ood target region is eligible for the label-access contract "
            f"at every seed -- {len(excluded)} region(s) failed the predeclared sizing rules"
        )

    # 2. B_d over EVERY seed cell of the eligible regions (never over a partial region)
    b_d = SA.benchmark_budget([c for cells in eligible.values() for c in cells], spec.max_label_budget)

    # 3 + 4. build, then audit the realized sets; any seed failing demotes the whole region
    built: dict[str, list] = {}
    for region, cells in sorted(eligible.items()):
        problems: list[str] = []
        staged: list = []
        for c in cells:
            where = f"{benchmark}/{SA.LABEL_ACCESS_REGIME}/{c['seed']}/{region}/{SA.LABEL_ACCESS_FILENAME}"
            rows = SA.build_label_access_rows(
                seed=c["seed"], source_ids=c["source_ids"], target_pool_ids=c["pool_ids"],
                target_test_ids=c["test_ids"], where=where,
            )
            SA.validate_label_access_rows(
                rows, source_ids=c["source_ids"], target_pool_ids=c["pool_ids"],
                target_test_ids=c["test_ids"], where=where,
            )
            src_ranked, tgt_ranked = SA.ranked_ids(rows)
            cls = c["class_of"]
            problems += [
                f"seed={c['seed']}: {p}"
                for p in SA.audit_allocation(
                    source_classes=[cls[s] for s in src_ranked],
                    target_classes=[cls[s] for s in tgt_ranked],
                    budget=b_d, where=where,
                )
            ]
            staged.append((c, rows))
        if problems:
            excluded[region] = problems          # 5. B_d stays as computed in step 2
            continue
        for c, rows in staged:
            c["rows"] = rows
        built[region] = [c for c, _ in staged]

    # 6. headline status is a REGION count over regions surviving at every seed
    headline_targets = sorted(built)
    headline_ok = len(headline_targets) >= SA.MIN_HEADLINE_TARGETS
    fractions = [
        {
            "percent": int(f),
            "n_target_labels": SA.allocation_target_count(f, b_d),
            "n_source_labels": int(b_d) - SA.allocation_target_count(f, b_d),
            "n_total_labels": int(b_d),
        }
        for f in SA.ALLOCATION_PERCENTS
    ]
    for region in headline_targets:
        for c in built[region]:
            c["summary"]["label_access"] = {
                "headline_eligible": bool(headline_ok),
                "benchmark_budget": int(b_d),
                "max_label_budget": (None if spec.max_label_budget is None else int(spec.max_label_budget)),
                "unit": unit,
                "n_source_pool": int(c["n_source"]),
                "n_target_pool": int(c["n_target_pool"]),
                "allocation_fractions": fractions,
                "additive_counts": list(SA.LABEL_ACCESS_COUNTS),
                "headline_targets": headline_targets,
            }
    # Exclusion metadata is REGION-CONSISTENT: every seed of an excluded region carries the identical
    # reason list, so a reader cannot conclude the region was usable at some seeds and not others.
    for region, reasons in sorted(excluded.items()):
        for c in by_region[region]:
            c["summary"]["label_access"] = {
                "headline_eligible": False,
                "excluded": True,
                "exclusion_reasons": list(reasons),
                "benchmark_budget": int(b_d),
                "unit": unit,
            }

    if not audit_only:
        for region in headline_targets:
            for c in built[region]:
                # The frozen label DRAW is bound as tightly as the frozen partitions: a different valid
                # permutation passes every structural check but silently changes every allocation point.
                la_path, la_sha = SA.write_label_access(root, benchmark, c["seed"], region, c["rows"])
                c["summary"]["label_access_csv"] = str(la_path.relative_to(root))
                c["summary"]["label_access_sha256"] = la_sha
    print(
        f"  [label-access] {benchmark}: B_d={b_d} {unit} over {len(eligible)} eligible region(s); "
        f"{len(headline_targets)} surviving, {len(excluded)} excluded"
        f"{'' if headline_ok else f' -- FEWER THAN {SA.MIN_HEADLINE_TARGETS}, no headline aggregate'}",
        flush=True,
    )


