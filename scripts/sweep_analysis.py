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

    # ---- table ----
    table = []
    for r in ratios:
        c = cells[r]
        a = c["aggregate"]
        n_img_tokens = max(1, round(576 * r))
        n_prompt_tokens = int(round(np.mean([run["n_prompt_tokens"] for run in c["runs"]])))
        row = {
            "keep": r, "K_image_tokens": n_img_tokens, "n_prompt_tokens": n_prompt_tokens,
            "encode_ms_mean": a["encode_ms"]["mean"], "encode_ms_std": a["encode_ms"]["std"],
            "ttft_llm_ms_mean": a["ttft_llm_ms"]["mean"], "ttft_llm_ms_std": a["ttft_llm_ms"]["std"],
            "ttft_vlm_ms_mean": a["ttft_vlm_ms"]["mean"], "ttft_vlm_ms_std": a["ttft_vlm_ms"]["std"],
            "encoder_fraction_mean": a["encoder_fraction"]["mean"],
            "decode_tok_per_s_mean": a["decode_tok_per_s"]["mean"],
            "max_rss_mib_mean": a.get("max_rss_mib", {}).get("mean"),
            "peak_footprint_mib_mean": a.get("peak_footprint_mib", {}).get("mean"),
            "kv_buffer_mib": c.get("kv_buffer_mib"),
            "generated_text": c["determinism"]["generated_text_first_run"],
            "identical_output": c["determinism"]["identical_output_across_timed_runs"],
        }
        table.append(row)

    # ---- prune-overhead isolation: fit encode_ms = A + B*K over PRUNED cells only ----
    pruned = [row for row in table if row["keep"] < 1.0]
    K = np.array([row["K_image_tokens"] for row in pruned], dtype=float)
    Y = np.array([row["encode_ms_mean"] for row in pruned], dtype=float)
    B, A = np.polyfit(K, Y, 1)  # Y = A + B*K
    fitted = A + B * K
    residuals = Y - fitted
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((Y - Y.mean()) ** 2))
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    fit_at_576 = A + B * 576
    keep1_encode_measured = next(row["encode_ms_mean"] for row in table if row["keep"] == 1.0)
    scoring_selection_overhead_ms = fit_at_576 - keep1_encode_measured

    fit_result = {
        "model": "encode_ms = A + B*K, fit over pruned cells only (keep<1.0)",
        "A_intercept_ms": float(A), "B_slope_ms_per_token": float(B),
        "r_squared": float(r_squared),
        "residuals_ms": [float(x) for x in residuals],
        "residual_max_abs_ms": float(np.max(np.abs(residuals))),
        "residual_std_ms": float(np.std(residuals)),
        "points_K": [float(x) for x in K], "points_encode_ms": [float(x) for x in Y],
        "fitted_encode_ms_at_K576": float(fit_at_576),
        "measured_encode_ms_at_keep1": float(keep1_encode_measured),
        "scoring_selection_overhead_ms": float(scoring_selection_overhead_ms),
        "note": "overhead = fit(K=576) - measured(keep=1.0); these differ only by the "
                "presence of the scoring/top-K/gather ops, so the difference isolates "
                "their cost. A negative value would mean the fit doesn't extrapolate "
                "cleanly to the unpruned case -- report as-is either way.",
    }

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
        "table": table,
        "prune_overhead_fit": fit_result,
        "bend_slopes_ttft_vlm_vs_keep": bends,
    }
    out_path = REPO_ROOT / "results" / "p2_sweep_analysis.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[analysis] wrote {out_path}")

    # console summary
    print()
    print(f"{'keep':>6} {'K':>4} {'encode_ms':>12} {'ttft_llm_ms':>14} {'ttft_vlm_ms':>14} "
          f"{'enc_frac':>9} {'dec_tok/s':>10} {'speedup_llm':>12} {'frac_ceil_h1':>13}")
    for row in table:
        fc = row["fraction_of_theoretical_ceiling_h1"]
        print(f"{row['keep']:>6.2f} {row['K_image_tokens']:>4d} "
              f"{row['encode_ms_mean']:>8.1f}±{row['encode_ms_std']:>3.0f} "
              f"{row['ttft_llm_ms_mean']:>10.1f}±{row['ttft_llm_ms_std']:>3.0f} "
              f"{row['ttft_vlm_ms_mean']:>10.1f}±{row['ttft_vlm_ms_std']:>3.0f} "
              f"{row['encoder_fraction_mean']*100:>8.2f}% {row['decode_tok_per_s_mean']:>10.2f} "
              f"{row['speedup_ttft_llm']:>11.3f}x {f'{fc*100:.1f}%' if fc is not None else 'n/a':>13}")
    print()
    print(f"Prune-overhead fit: A={A:.2f}ms B={B:.4f}ms/token R²={r_squared:.4f} "
          f"max|resid|={fit_result['residual_max_abs_ms']:.2f}ms")
    print(f"Fitted encode_ms at K=576: {fit_at_576:.2f}ms; measured keep=1.0 encode_ms: "
          f"{keep1_encode_measured:.2f}ms; scoring+selection overhead: "
          f"{scoring_selection_overhead_ms:+.2f}ms")


if __name__ == "__main__":
    main()
