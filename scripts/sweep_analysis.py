#!/usr/bin/env python3
"""Analysis for the keep-ratio sweep: table, prune-overhead isolation via
linear fit, speedup curves, fraction-of-theoretical-ceiling (H1), and
where the TTFT_vlm-vs-keep curve bends (H2 candidate).

Prune-overhead isolation (per the refined methodology): fit
encode_ms = A + B*K over the PRUNED cells only (keep < 1.0 -- keep=1.0
runs a structurally different code path, the pruning branch is gated off
entirely there, so it doesn't belong in a fit meant to characterize that
branch's cost). Extrapolate the fit to K=576 and subtract the MEASURED
keep=1.0 encode_ms; since the two quantities differ only by the presence
of the scoring/top-K/gather ops, that difference is the pruning branch's
cost. Reported with residuals so the fit's validity is visible, not
assumed.
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_cells(tag_prefix):
    cells = {}
    for f in glob.glob(str(REPO_ROOT / "results" / f"*_{tag_prefix}_keep*.json")):
        d = json.loads(Path(f).read_text())
        keep = float(d["config"]["extra_args"][1])
        cells[keep] = d
        cells[keep]["_source_file"] = f
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag-prefix", default="p2_sweep")
    args = ap.parse_args()

    cells = load_cells(args.tag_prefix)
    if not cells:
        raise SystemExit(f"no cells found for prefix {args.tag_prefix}")
    ratios = sorted(cells.keys(), reverse=True)
    print(f"[analysis] found {len(ratios)} cells: {ratios}")

    platform_tags = sorted({c.get("platform_tag", "(unset)") for c in cells.values()})
    if len(platform_tags) > 1:
        print(f"[analysis] WARNING: cells span multiple platform_tags {platform_tags} -- "
              f"this sweep is not internally comparable, results were likely mixed by "
              f"accident (e.g. a tag_prefix collision across platforms)", flush=True)

    # ---- table ----
    table = []
    for r in ratios:
        c = cells[r]
        a = c["aggregate"]
        n_img_tokens = max(1, round(576 * r))
        n_prompt_tokens = int(round(np.mean([run["n_prompt_tokens"] for run in c["runs"]])))
        ttft_vlm_runs = [run["encode_ms"] + run["prompt_eval_ms"] for run in c["runs"]]
        row = {
            "keep": r, "K_image_tokens": n_img_tokens, "n_prompt_tokens": n_prompt_tokens,
            "encode_ms_mean": a["encode_ms"]["mean"], "encode_ms_std": a["encode_ms"]["std"],
            "ttft_llm_ms_mean": a["ttft_llm_ms"]["mean"], "ttft_llm_ms_std": a["ttft_llm_ms"]["std"],
            "ttft_vlm_ms_mean": a["ttft_vlm_ms"]["mean"], "ttft_vlm_ms_std": a["ttft_vlm_ms"]["std"],
            "ttft_vlm_ms_min": min(ttft_vlm_runs), "ttft_vlm_ms_max": max(ttft_vlm_runs),
            "encoder_fraction_mean": a["encoder_fraction"]["mean"],
            "decode_tok_per_s_mean": a["decode_tok_per_s"]["mean"],
            "max_rss_mib_mean": a.get("max_rss_mib", {}).get("mean"),
            "peak_footprint_mib_mean": a.get("peak_footprint_mib", {}).get("mean"),
            "kv_buffer_mib": c.get("kv_buffer_mib"),
            "generated_text": c["determinism"]["generated_text_first_run"],
            "identical_output": c["determinism"]["identical_output_across_timed_runs"],
        }
        table.append(row)

    # ---- prune-overhead isolation: fit encode_ms = A + B*K over PRUNED cells ----
    # Two fits reported: the original range (keep>=0.05, K>=29, where the fit was
    # excellent) and the full extended range (all keep<1.0), so the fit's breakdown
    # when extended is visible rather than silently replacing the earlier result.
    keep1_encode_measured = next(row["encode_ms_mean"] for row in table if row["keep"] == 1.0)

    def fit_range(rows, label):
        K = np.array([row["K_image_tokens"] for row in rows], dtype=float)
        Y = np.array([row["encode_ms_mean"] for row in rows], dtype=float)
        B, A = np.polyfit(K, Y, 1)
        fitted = A + B * K
        residuals = Y - fitted
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((Y - Y.mean()) ** 2))
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        fit_at_576 = A + B * 576
        return {
            "label": label,
            "A_intercept_ms": float(A), "B_slope_ms_per_token": float(B),
            "r_squared": float(r_squared),
            "residuals_ms": [float(x) for x in residuals],
            "residual_max_abs_ms": float(np.max(np.abs(residuals))),
            "residual_std_ms": float(np.std(residuals)),
            "points_K": [float(x) for x in K], "points_encode_ms": [float(x) for x in Y],
            "fitted_encode_ms_at_K576": float(fit_at_576),
            "scoring_selection_overhead_ms": float(fit_at_576 - keep1_encode_measured),
        }

    pruned_original = [row for row in table if 0.0 < row["keep"] < 1.0 and row["K_image_tokens"] >= 29]
    pruned_extended = [row for row in table if row["keep"] < 1.0]
    fit_original = fit_range(pruned_original, "original range (keep 0.75-0.05, K 29-432)")
    fit_result = fit_range(pruned_extended, "extended range (keep 0.75-0.01, K 6-432)")
    fit_result["comparison_original_range_fit"] = fit_original
    A, B = fit_result["A_intercept_ms"], fit_result["B_slope_ms_per_token"]
    fit_at_576 = fit_result["fitted_encode_ms_at_K576"]
    scoring_selection_overhead_ms = fit_result["scoring_selection_overhead_ms"]
    fit_result["measured_encode_ms_at_keep1"] = float(keep1_encode_measured)
    fit_result["model"] = "encode_ms = A + B*K, extended fit over ALL pruned cells (keep<1.0, K=6..432)"
    fit_result["note_extended"] = (
        f"Extended fit R²={fit_result['r_squared']:.4f} (max|resid|="
        f"{fit_result['residual_max_abs_ms']:.1f}ms) vs the original range's R²="
        f"{fit_original['r_squared']:.4f} (max|resid|={fit_original['residual_max_abs_ms']:.1f}ms) - "
        "the linear model that fit K=29..432 excellently does not extrapolate to K<29; "
        "report both, don't silently replace the good fit with the degraded one."
    )
    fit_result["note"] = (
        "overhead = fit(K=576) - measured(keep=1.0); these differ only by the "
        "presence of the scoring/top-K/gather ops, so the difference isolates "
        "their cost. A negative value would mean the fit doesn't extrapolate "
        "cleanly to the unpruned case -- report as-is either way."
    )

    # ---- speedup curves + fraction of theoretical ceiling (H1) ----
    base = next(row for row in table if row["keep"] == 1.0)
    for row in table:
        row["speedup_ttft_llm"] = base["ttft_llm_ms_mean"] / row["ttft_llm_ms_mean"]
        row["speedup_ttft_vlm"] = base["ttft_vlm_ms_mean"] / row["ttft_vlm_ms_mean"]
        # theoretical TTFT_llm if prefill scaled perfectly linearly with token count
        theoretical_ttft_llm = base["ttft_llm_ms_mean"] * (row["n_prompt_tokens"] / base["n_prompt_tokens"])
        row["theoretical_ttft_llm_ms"] = theoretical_ttft_llm
        achieved_reduction = base["ttft_llm_ms_mean"] - row["ttft_llm_ms_mean"]
        theoretical_max_reduction = base["ttft_llm_ms_mean"] - theoretical_ttft_llm
        row["fraction_of_theoretical_ceiling_h1"] = (
            achieved_reduction / theoretical_max_reduction if theoretical_max_reduction > 0 else None
        )

    # ---- where the TTFT_vlm-vs-keep curve bends (H2 candidate): finite-difference slopes ----
    bends = []
    for i in range(len(table) - 1):
        r0, r1 = table[i]["keep"], table[i + 1]["keep"]
        dv = table[i + 1]["ttft_vlm_ms_mean"] - table[i]["ttft_vlm_ms_mean"]
        dr = r1 - r0
        bends.append({"between_keep": [r0, r1], "d_ttft_vlm_ms_per_d_keep": dv / dr if dr != 0 else None})

    out = {
        "tag_prefix": args.tag_prefix,
        "platform_tags": platform_tags,
        "table": table,
        "prune_overhead_fit": fit_result,
        "bend_slopes_ttft_vlm_vs_keep": bends,
    }
    # named by tag_prefix so a different platform/sweep's analysis (e.g.
    # p2_sweep_x86) never overwrites another's (e.g. the default p2_sweep, M4)
    out_path = REPO_ROOT / "results" / f"{args.tag_prefix}_analysis.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[analysis] wrote {out_path}")

    # console summary
    print()
    print(f"{'keep':>7} {'K':>4} {'encode_ms':>12} {'ttft_llm_ms':>14} {'ttft_vlm_ms (min-max)':>28} "
          f"{'enc_frac':>9} {'dec_tok/s':>10} {'speedup_llm':>12} {'frac_ceil_h1':>13}")
    for row in table:
        fc = row["fraction_of_theoretical_ceiling_h1"]
        vlm_range = f"({row['ttft_vlm_ms_min']:.0f}-{row['ttft_vlm_ms_max']:.0f})"
        print(f"{row['keep']:>7.3f} {row['K_image_tokens']:>4d} "
              f"{row['encode_ms_mean']:>8.1f}±{row['encode_ms_std']:>3.0f} "
              f"{row['ttft_llm_ms_mean']:>10.1f}±{row['ttft_llm_ms_std']:>3.0f} "
              f"{row['ttft_vlm_ms_mean']:>8.1f}±{row['ttft_vlm_ms_std']:>4.0f} {vlm_range:>15} "
              f"{row['encoder_fraction_mean']*100:>8.2f}% {row['decode_tok_per_s_mean']:>10.2f} "
              f"{row['speedup_ttft_llm']:>11.3f}x {f'{fc*100:.1f}%' if fc is not None else 'n/a':>13}")
    print()
    print(f"Original-range fit (K=29..432): A={fit_original['A_intercept_ms']:.2f}ms "
          f"B={fit_original['B_slope_ms_per_token']:.4f}ms/token R²={fit_original['r_squared']:.4f} "
          f"max|resid|={fit_original['residual_max_abs_ms']:.2f}ms")
    print(f"Extended fit (K=6..432):        A={A:.2f}ms B={B:.4f}ms/token "
          f"R²={fit_result['r_squared']:.4f} max|resid|={fit_result['residual_max_abs_ms']:.2f}ms")
    print(f"Extended fit at K=576: {fit_at_576:.2f}ms; measured keep=1.0 encode_ms: "
          f"{keep1_encode_measured:.2f}ms; scoring+selection overhead: "
          f"{scoring_selection_overhead_ms:+.2f}ms")

    # ---- H2: does TTFT_vlm turn around anywhere in the tested range? ----
    min_row = min(table, key=lambda row: row["ttft_vlm_ms_mean"])
    print()
    print(f"H2 check: TTFT_vlm minimum (mean) is at keep={min_row['keep']:g} "
          f"({min_row['ttft_vlm_ms_mean']:.1f}ms). Ratios below that keep value:")
    for row in table:
        if row["keep"] < min_row["keep"]:
            delta = row["ttft_vlm_ms_mean"] - min_row["ttft_vlm_ms_mean"]
            print(f"  keep={row['keep']:>6.3f} (K={row['K_image_tokens']:>3d}): "
                  f"{row['ttft_vlm_ms_mean']:.1f}ms mean ({delta:+.1f}ms vs minimum), "
                  f"range ({row['ttft_vlm_ms_min']:.0f}-{row['ttft_vlm_ms_max']:.0f})ms")


if __name__ == "__main__":
    main()
