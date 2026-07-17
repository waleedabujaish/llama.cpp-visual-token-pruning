#!/usr/bin/env python3
"""TextVQA accuracy simulation of FasterVLM-style [CLS]-attention pruning.

llava-hf/llava-1.5-7b-hf, greedy decoding, on a fixed ~200-sample slice of the
TextVQA validation set (lmms-lab/textvqa, first N samples in streaming order).
For each sample the vision tower runs ONCE (eager attention so probs are
available); each keep ratio then selects its token subset, projects, splices
into the language embeddings, and generates.

Pruning semantics = FasterVLM: scores = attentions[-2][:, :, 0, 1:].mean(heads),
features = hidden_states[-2][:, 1:], top-K via boolean mask (original spatial
order preserved).

Accuracy = standard soft VQA accuracy: min(1, matches/3) against the 10 human
answers, with VQAv2-style normalization.

Writes a timestamped JSON to results/ with per-sample predictions.
"""

import argparse
import datetime
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# ---------------- VQA answer normalization (VQAv2 eval conventions) ----------
CONTRACTIONS = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
    "couldnt": "couldn't", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't",
    "hadnt": "hadn't", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd",
    "hes": "he's", "im": "i'm", "isnt": "isn't", "itd": "it'd", "itll": "it'll",
    "its": "it's", "lets": "let's", "shes": "she's", "shouldve": "should've",
    "shouldnt": "shouldn't", "thats": "that's", "theres": "there's", "theyd": "they'd",
    "theyll": "they'll", "theyre": "they're", "theyve": "they've", "wasnt": "wasn't",
    "werent": "weren't", "whats": "what's", "wheres": "where's", "wholl": "who'll",
    "whos": "who's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't",
    "youd": "you'd", "youll": "you'll", "youre": "you're", "youve": "you've"}
NUM_MAP = {"none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
           "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
ARTICLES = {"a", "an", "the"}
PUNCT = re.compile(r"[;/\[\]\"{}()=+\\_\-><@`,?!.']")


def vqa_normalize(ans: str) -> str:
    ans = ans.lower().strip()
    ans = PUNCT.sub("", ans)
    words = []
    for w in ans.split():
        w = NUM_MAP.get(w, w)
        w = CONTRACTIONS.get(w, w)
        if w not in ARTICLES:
            words.append(w)
    return " ".join(words)


def vqa_accuracy(pred: str, answers: list) -> float:
    # official VQAv2/TextVQA form: leave-one-out average over the 10 answers
    pred_n = vqa_normalize(pred)
    ans_n = [vqa_normalize(a) for a in answers]
    accs = [min(1.0, sum(o == pred_n for o in (ans_n[:i] + ans_n[i + 1:])) / 3.0)
            for i in range(len(ans_n))]
    return float(sum(accs) / len(accs)) if accs else 0.0


# ---------------- pruning ----------------------------------------------------
def prune_features(feats: torch.Tensor, scores: torch.Tensor, keep: float) -> torch.Tensor:
    """feats (576,d), scores (576,) -> (K,d) in original order (FasterVLM)."""
    n = feats.shape[0]
    k = max(1, round(n * keep))
    if k >= n:
        return feats
    idx = torch.topk(scores, k).indices
    mask = torch.zeros(n, dtype=torch.bool, device=feats.device)
    mask[idx] = True
    return feats[mask]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument("--ratios", default="1.0,0.5,0.25,0.1")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--model-id", default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--tag", default="p1_textvqa_prune_sim")
    ap.add_argument("--save-images", type=int, default=0,
                    help="save first N sample images to assets/phase1/textvqa/ for the viz step")
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    ratios = [float(r) for r in args.ratios.split(",")]
    device = args.device
    dtype = torch.float16 if device == "mps" else torch.float32

    print(f"[sim] loading {args.model_id} on {device} ({dtype}) ...", flush=True)
    processor = AutoProcessor.from_pretrained(args.model_id)
    # vision tower must be eager (we need attention probs); keep the 7B LM on
    # sdpa for generation throughput, falling back to all-eager if the dict
    # form isn't supported.
    try:
        model = LlavaForConditionalGeneration.from_pretrained(
            args.model_id, dtype=dtype,
            attn_implementation={"text_config": "sdpa", "vision_config": "eager"}).to(device)
        attn_impl = "text=sdpa, vision=eager"
    except (ValueError, TypeError) as e:
        print(f"[sim] per-submodule attn_implementation unsupported ({e}); using eager everywhere")
        model = LlavaForConditionalGeneration.from_pretrained(
            args.model_id, dtype=dtype, attn_implementation="eager").to(device)
        attn_impl = "eager"
    model.eval()
    tok = processor.tokenizer
    image_processor = processor.image_processor

    vision = model.model.vision_tower if hasattr(model.model, "vision_tower") else model.vision_tower
    projector = (model.model.multi_modal_projector
                 if hasattr(model.model, "multi_modal_projector") else model.multi_modal_projector)
    embed = model.get_input_embeddings()

    print("[sim] streaming lmms-lab/textvqa validation ...", flush=True)
    ds = load_dataset("lmms-lab/textvqa", split="validation", streaming=True)

    img_dir = REPO_ROOT / "assets/phase1/textvqa"
    if args.save_images:
        img_dir.mkdir(parents=True, exist_ok=True)

    results = {f"{r:g}": [] for r in ratios}
    records = []
    t0 = time.time()
    n_done = 0
    n_with_ocr = 0
    for sample in ds:
        if n_done >= args.n_samples:
            break
        img = sample["image"].convert("RGB")
        question = sample["question"]
        answers = sample["answers"]

        if args.save_images and n_done < args.save_images:
            img.save(img_dir / f"textvqa_{n_done:03d}.png")

        pix = image_processor(images=img, return_tensors="pt")["pixel_values"].to(device, dtype)
        with torch.no_grad():
            out = vision(pix, output_hidden_states=True, output_attentions=True)
        feats = out.hidden_states[-2][0, 1:, :]              # (576, 1024)
        attn = out.attentions[-2][0]                          # (heads, 577, 577)
        scores = attn[:, 0, 1:].mean(dim=0)                   # (576,)

        # vicuna_v1 conversation format, as used by the official LLaVA-1.5 TextVQA
        # eval (--conv-mode vicuna_v1) and therefore by FasterVLM's tables; the
        # official eval also appends the OCR reference line to the question.
        prompt_pre = ("A chat between a curious user and an artificial intelligence assistant. "
                      "The assistant gives helpful, detailed, and polite answers to the user's "
                      "questions. USER: ")
        ocr = sample.get("ocr_tokens") or []
        n_with_ocr += bool(ocr)
        ocr_line = f"\nReference OCR token: {', '.join(ocr)}" if ocr else ""
        prompt_post = (f"\n{question}{ocr_line}\n"
                       f"Answer the question using a single word or phrase. ASSISTANT:")
        ids_pre = tok(prompt_pre, return_tensors="pt", add_special_tokens=True).input_ids.to(device)
        ids_post = tok(prompt_post, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

        rec = {"i": n_done, "question": question, "answers": answers, "pred": {}}
        for r in ratios:
            kept = prune_features(feats, scores, r)
            with torch.no_grad():
                img_embeds = projector(kept.unsqueeze(0))     # (1, K, 4096)
                ie = torch.cat([embed(ids_pre), img_embeds.to(embed.weight.dtype),
                                embed(ids_post)], dim=1)
                am = torch.ones(ie.shape[:2], dtype=torch.long, device=device)
                gen = model.generate(inputs_embeds=ie, attention_mask=am,
                                     max_new_tokens=args.max_new_tokens, do_sample=False,
                                     pad_token_id=tok.pad_token_id or tok.eos_token_id)
            pred = tok.decode(gen[0], skip_special_tokens=True).strip()
            acc = vqa_accuracy(pred, answers)
            results[f"{r:g}"].append(acc)
            rec["pred"][f"{r:g}"] = {"text": pred, "acc": acc}
        records.append(rec)
        n_done += 1
        if n_done % 10 == 0:
            el = time.time() - t0
            accs = {k: f"{np.mean(v)*100:.1f}" for k, v in results.items()}
            print(f"[sim] {n_done}/{args.n_samples} ({el/60:.1f} min) acc so far: {accs}", flush=True)

    summary = {k: {"acc_mean": float(np.mean(v)), "n": len(v)} for k, v in results.items()}
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = {
        "tag": args.tag, "timestamp": ts,
        "model": args.model_id, "device": device, "dtype": str(dtype),
        "dataset": "lmms-lab/textvqa validation, first n in streaming order",
        "n_samples": n_done, "n_with_ocr_line": n_with_ocr, "ratios": ratios,
        "max_new_tokens": args.max_new_tokens, "decoding": "greedy (deterministic, no seed)",
        "attn_implementation": attn_impl,
        "prompt_format": "vicuna_v1 system + USER: <image>\\n<q>[\\nReference OCR token: ...]\\n"
                         "Answer the question using a single word or phrase. ASSISTANT:",
        "pruning": "FasterVLM: attentions[-2] CLS->patch mean-over-heads, topk, original order",
        "metric": "soft VQA accuracy min(1, matches/3), VQAv2 normalization",
        "summary": summary,
        "records": records,
    }
    # write raw results FIRST; retention is derived post-hoc so a bug in it
    # can never lose the run
    path = REPO_ROOT / "results" / f"{ts}_{args.tag}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"[sim] wrote {path}")

    base_entry = summary.get(f"{1.0:g}")
    base = base_entry["acc_mean"] if base_entry else None
    for k, s in summary.items():
        s["retention_vs_keep100"] = float(s["acc_mean"] / base) if base else None
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
