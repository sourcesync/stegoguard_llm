"""Export a compact JSON bundle of the frozen pilot_v0 results.

The bundle collects what an interactive results explorer (or any quick
analysis) needs in one file:

  op       out-of-fold detector predictions for the 104 matched parents
           (416 records) for the deployed detector and each feature-family
           variant: [score, y, channel, pred_f1, pred_fpr_05, pred_fpr_01]
  parents  the 48 adaptive-replay parents: cell, prompt, clean answer,
           candidate counts, and how StegoGuard routed the clean answer
           in each round under each rewriter
  trajs    all 288 adaptive trajectories, rounds R0-R3: chosen candidate,
           selection reason, detector votes, original and delivered text,
           payload length, exact recovery, decoded bits, decoder error,
           slot details
  texts    deduplicated text table referenced by index from parents/trajs

Channel codes: c clean, r rank, s synonym, u unicode.

Inputs are read from outputs/pilot_v0 at the paths recorded in the freeze
manifests (the JSON/JSONL data bundle must be unpacked first). Run from the
repository root:

    python scripts/export_explorer_bundle.py
    python scripts/export_explorer_bundle.py --out path/to/bundle.json

Added after the experiments; not covered by the freeze manifests.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ATK = {"clean": "c", "rank_encoding": "r", "synonym_slot_codebook": "s", "unicode_hidden_channel": "u"}
POL = {"frozen_no_feedback": "frozen", "fixed_channel_feedback": "fixed", "channel_migration_feedback": "migration"}
DEF = {"deterministic_rewrite": "det", "randomized_rewrite": "rand"}
VARIANTS = [
    "full_deployed_with_unicode_hygiene", "full_fusion", "distributional", "cross_model", "semantic",
    "distributional_cross_model", "distributional_semantic", "cross_model_semantic",
]


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="outputs/pilot_v0", help="frozen output directory")
    ap.add_argument("--out", default="outputs/explorer/explorer_bundle.json")
    args = ap.parse_args()
    root = Path(args.root)
    step10 = root / "adaptive" / "step10"

    # ---------- detector operating points ----------
    op: dict[str, list] = {v: [] for v in VARIANTS}
    for r in rows(root / "detection" / "step8" / "predictions.jsonl"):
        if r["experiment"] == "primary_common_all" and r["variant"] in op:
            op[r["variant"]].append([
                round(float(r["score"]), 6), int(r["y"]), ATK[r["attack_name"]],
                int(r["pred_f1"]), int(r["pred_fpr_05"]), int(r["pred_fpr_01"]),
            ])
    for v, recs in op.items():
        if len(recs) != 416:
            raise ValueError(f"{v}: expected 416 records, found {len(recs)}")

    # ---------- adaptive replay ----------
    texts: list[str] = []
    index: dict[str, int] = {}

    def tid(s: str | None) -> int:
        s = s or ""
        if s not in index:
            index[s] = len(texts)
            texts.append(s)
        return index[s]

    prompts: dict[str, str] = {}
    for r in rows(root / "attacks" / "full" / "stego_samples_full.jsonl"):
        prompts.setdefault(r["parent_sample_id"], r["prompt"])

    defense, payload, ctrl = {}, {}, {}
    for k in range(4):
        for r in rows(step10 / f"round_{k}" / "defense_outputs.jsonl"):
            if r["record_role"] == "clean_control":
                c = ctrl.setdefault(r["clean_parent_id"], {})
                c["_clean"] = tid(r["original_text"])
                c[f"{DEF[r['defender_policy']]}:{k}"] = [int(bool(r["detected"])), r["flagged_fold_count"], tid(r["final_text"])]
            else:
                defense[(r["trajectory_id"], k)] = r
        for r in rows(step10 / f"round_{k}" / "payload_results.jsonl"):
            payload[(r["trajectory_id"], k)] = r

    parents, trajs = {}, []
    for m in rows(step10 / "trajectory_manifest.jsonl"):
        p = m["clean_parent_id"]
        if p not in parents:
            parents[p] = {
                "model": "Qwen" if "Qwen" in m["source_model"] else "Mistral",
                "domain": m["domain"], "len": int(m["nominal_target_length"]), "topic": m["topic_id"],
                "prompt": tid(prompts.get(p, "")), "clean": ctrl[p]["_clean"],
                "counts": {ATK[a]: n for a, n in m["candidate_counts"].items()},
                "ctrl": {k: v for k, v in ctrl[p].items() if k != "_clean"},
            }
        rounds = []
        for k in range(4):
            d, pr = defense[(m["trajectory_id"], k)], payload[(m["trajectory_id"], k)]
            rounds.append({
                "a": ATK[d["attack_name"]], "s": int(d["strategy_index"]), "why": d["selection_reason"],
                "det": int(bool(d["detected"])), "fold": d["flagged_fold_count"],
                "score": round(float(d["ensemble_mean_score"]), 4),
                "o": tid(d["original_text"]), "f": tid(d["final_text"]), "seed": d["applied_rewrite_seed"],
                "bits": int(d["payload_bit_length"]), "ok": int(bool(pr["exact_payload_recovery"])),
                "dec": pr["decoded_payload"] or "", "err": pr["decoder_error"] or "",
                "slots": [[x["slot_id"], x["expected_bit"], x["decoded_bit"], x["status"]]
                          for x in (pr.get("slot_details") or [])],
            })
        trajs.append({"id": m["trajectory_id"], "p": p, "pol": POL[m["attacker_condition"]],
                      "def": DEF[m["defender_policy"]], "init": ATK[m["initial_attack_name"]], "r": rounds})

    # ---------- candidate catalogue (Pipeline tab) ----------
    det_dir = step10 / "detection"
    feat_dir = det_dir / "features"

    def detector(d: dict) -> dict:
        return {"mean": round(float(d["ensemble_mean_score"]), 4), "n": d["flagged_fold_count"],
                "flag": int(bool(d["detected_f1"])), "uni": int(bool(d["unicode_hygiene_flag"])),
                "folds": [[round(float(f["score"]), 4), round(float(f["threshold_f1"]), 4)]
                          for f in d["fold_results"]]}

    scores = {d["sample_id"]: detector(d) for d in rows(det_dir / "detector_scores.jsonl")}
    feats: dict[str, list] = {}
    for role, fname in (("a", "evaluator_a_distributional.jsonl"), ("b", "evaluator_b_distributional.jsonl")):
        for r in rows(feat_dir / fname):
            feats.setdefault(r["sample_id"], [None] * 13)[0 if role == "a" else 4:(4 if role == "a" else 8)] = [
                round(float(r[k]), 4) for k in ("ppl", "mean_logp", "logp_var", "sentence_logp_var")]
    for r in rows(feat_dir / "semantic_features.jsonl"):
        feats[r["sample_id"]][8:10] = [round(float(r["visible_instruction_retention"]), 4),
                                       round(float(r["paraphrase_embedding_similarity"]), 4)]
    for r in rows(feat_dir / "unicode_features.jsonl"):
        feats[r["sample_id"]][10:13] = [r["nonprinting_count"], r["zero_width_count"], r["unicode_tag_count"]]

    used: dict[tuple, list] = {}
    for t in trajs:
        for k, rd in enumerate(t["r"]):
            used.setdefault((t["p"], rd["a"], rd["s"]), []).append(f"{t['pol']}:{t['def']}:{k}")

    catalogue: dict[str, dict] = {
        p: {"det": scores[p], "feat": feats[p], "cands": [], "fails": []} for p in parents}
    for r in rows(step10 / "candidate_samples.jsonl"):
        p, a, cp = r["parent_sample_id"], ATK[r["attack_name"]], r["channel_parameters"] or {}
        c = {"a": a, "s": int(r["strategy_index"]), "id": r["strategy_id"], "pay": r["payload"],
             "bits": int(r["payload_bit_length"]), "ok": int(bool(r["exact_match"])), "t": tid(r["text"]),
             "tok": r["stego_actual_token_length"], "bpt": round(float(r["bits_per_token"]), 4),
             "tries": r["encoding_attempts"], "src": r["candidate_source"],
             "det": scores[r["sample_id"]], "feat": feats[r["sample_id"]],
             "used": used.get((p, a, int(r["strategy_index"])), [])}
        if a == "s":
            c["elig"] = cp.get("eligible_slot_ids", [])
            c["subs"] = [[x["slot_id"], x["bit"], x["observed_term_before"], x["selected_term"], int(x["text_changed"])]
                         for x in cp.get("substitutions", [])]
        else:
            c["topk"], c["nonce"] = cp.get("top_k"), cp.get("encoding_nonce")
        catalogue[p]["cands"].append(c)
    for r in rows(step10 / "candidate_failures.jsonl"):
        msg = r["error_message"]
        catalogue[r["clean_parent_id"]]["fails"].append(
            {"a": ATK[r["attack_name"]], "s": int(r["strategy_index"]),
             "why": msg.split("Last error: ", 1)[-1]})
    for c in catalogue.values():
        c["cands"].sort(key=lambda x: (x["a"], x["s"]))
        c["fails"].sort(key=lambda x: (x["a"], x["s"]))

    # ---------- clean corpus (Corpus tab): prompts and their generations only ----------
    corpus: dict[str, dict] = {}
    for r in rows(root / "clean" / "clean_samples_full.jsonl"):
        c = corpus.setdefault(r["prompt_id"], {
            "domain": r["domain"], "topic": r["topic_seed"], "len": int(r["nominal_target_length"]),
            "prompt": tid(r["prompt"]), "gens": []})
        c["gens"].append({"model": "Qwen" if "Qwen" in r["source_model"] else "Mistral",
                          "rep": int(r["replicate_index"]), "tok": int(r["actual_token_length"]),
                          "id": r["sample_id"], "text": tid(r["text"])})
    if sum(len(c["gens"]) for c in corpus.values()) != 160:
        raise ValueError("expected 160 clean generations")
    for c in corpus.values():
        c["gens"].sort(key=lambda g: (g["model"], g["rep"]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"op": op, "parents": parents, "trajs": trajs, "cat": catalogue,
                               "corpus": corpus, "texts": texts},
                              separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    n_c = sum(len(c["cands"]) for c in catalogue.values())
    n_f = sum(len(c["fails"]) for c in catalogue.values())
    print(f"wrote {out}: {len(parents)} parents, {len(trajs)} trajectories, {n_c} candidates, "
          f"{n_f} failed candidates, {len(texts)} texts, {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
