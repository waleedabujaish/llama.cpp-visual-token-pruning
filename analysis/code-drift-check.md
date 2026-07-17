# Code drift check vs upstream master (2026-07-17)

- Local pin: `e8f19cc0ad70a243c8012bf17b4be601abfc8ea2`
- `upstream/master` (github.com/ggml-org/llama.cpp, fetched 2026-07-17):
  **`e8f19cc0ad70a243c8012bf17b4be601abfc8ea2` — identical to the pin.**
- `git log e8f19cc0..upstream/master` → **0 commits** (0 touching
  `tools/mtmd/`). No drift is possible; every citation below was additionally
  re-verified verbatim against the working tree.

Permalink base: `https://github.com/ggml-org/llama.cpp/blob/e8f19cc0ad70a243c8012bf17b4be601abfc8ea2/`

## Bug A citations (all verified verbatim, old line == new line)

| Citation | Content | Permalink |
|---|---|---|
| `models/llava.cpp:36` | CLS concat after patches | [tools/mtmd/models/llava.cpp#L36](https://github.com/ggml-org/llama.cpp/blob/e8f19cc0ad70a243c8012bf17b4be601abfc8ea2/tools/mtmd/models/llava.cpp#L36) |
| `models/llava.cpp:7` | `n_pos = n_patches + (class_embedding?1:0)` | [#L7](https://github.com/ggml-org/llama.cpp/blob/e8f19cc0ad70a243c8012bf17b4be601abfc8ea2/tools/mtmd/models/llava.cpp#L7) |
| `models/llava.cpp:151-160` | patches input + ggml_get_rows | [#L151-L160](https://github.com/ggml-org/llama.cpp/blob/e8f19cc0ad70a243c8012bf17b4be601abfc8ea2/tools/mtmd/models/llava.cpp#L151-L160) |
| `clip.cpp:4095-4099` | identity positions fill | [tools/mtmd/clip.cpp#L4095-L4099](https://github.com/ggml-org/llama.cpp/blob/e8f19cc0ad70a243c8012bf17b4be601abfc8ea2/tools/mtmd/clip.cpp#L4095-L4099) |
| `clip.cpp:4101-4108` | patches = [1..576] fill | [#L4101-L4108](https://github.com/ggml-org/llama.cpp/blob/e8f19cc0ad70a243c8012bf17b4be601abfc8ea2/tools/mtmd/clip.cpp#L4101-L4108) |
| `clip.cpp:4080-4088` | GLM_EDGE separate positions-only case | [#L4080-L4088](https://github.com/ggml-org/llama.cpp/blob/e8f19cc0ad70a243c8012bf17b4be601abfc8ea2/tools/mtmd/clip.cpp#L4080-L4088) |
| `clip.cpp:3590` | n_pos definition at encode | [#L3590](https://github.com/ggml-org/llama.cpp/blob/e8f19cc0ad70a243c8012bf17b4be601abfc8ea2/tools/mtmd/clip.cpp#L3590) |
| `clip.cpp:971-978` | dispatch of 5 projector types to clip_graph_llava | [#L971-L978](https://github.com/ggml-org/llama.cpp/blob/e8f19cc0ad70a243c8012bf17b4be601abfc8ea2/tools/mtmd/clip.cpp#L971-L978) |
| `clip.cpp:1841` | optional class_embedding load | [#L1841](https://github.com/ggml-org/llama.cpp/blob/e8f19cc0ad70a243c8012bf17b4be601abfc8ea2/tools/mtmd/clip.cpp#L1841) |
| `clip.cpp:1962-1965` | Yi-VL MLP→MLP_NORM promotion | [#L1962-L1965](https://github.com/ggml-org/llama.cpp/blob/e8f19cc0ad70a243c8012bf17b4be601abfc8ea2/tools/mtmd/clip.cpp#L1962-L1965) |
| `legacy .../convert_image_encoder_to_gguf.py:60` | rename chain → v.class_embd | [#L60](https://github.com/ggml-org/llama.cpp/blob/e8f19cc0ad70a243c8012bf17b4be601abfc8ea2/tools/mtmd/legacy-models/convert_image_encoder_to_gguf.py#L60) |
| `.../convert_image_encoder_to_gguf.py:164-172` | CLIP vs SigLIP encoder classes | [#L164-L172](https://github.com/ggml-org/llama.cpp/blob/e8f19cc0ad70a243c8012bf17b4be601abfc8ea2/tools/mtmd/legacy-models/convert_image_encoder_to_gguf.py#L164-L172) |
| `models/internvl.cpp:10-13` | same CLS-last pattern (unverified lead) | [#L10-L13](https://github.com/ggml-org/llama.cpp/blob/e8f19cc0ad70a243c8012bf17b4be601abfc8ea2/tools/mtmd/models/internvl.cpp#L10-L13) |
| pre-refactor CLS-first code | `ggml_acc` at offset 0 | [ffc72720.../tools/mtmd/clip.cpp#L1129-L1137](https://github.com/ggml-org/llama.cpp/blob/ffc727203af1061fdeb49efef30f76171722e403/tools/mtmd/clip.cpp#L1129-L1137) (parent of 32916a490) |

## Bug B citations (all verified verbatim)

| Citation | Content | Permalink |
|---|---|---|
| `models/llava.cpp:15` | `il_last = hparams.n_layer - 1` | [#L15](https://github.com/ggml-org/llama.cpp/blob/e8f19cc0ad70a243c8012bf17b4be601abfc8ea2/tools/mtmd/models/llava.cpp#L15) |
| `models/llava.cpp:18-20` | MINICPMV/GLM_EDGE `il_last += 1` | [#L18-L20](https://github.com/ggml-org/llama.cpp/blob/e8f19cc0ad70a243c8012bf17b4be601abfc8ea2/tools/mtmd/models/llava.cpp#L18-L20) |
| `models/llava.cpp:24-29` | feature_layers override (granite) | [#L24-L29](https://github.com/ggml-org/llama.cpp/blob/e8f19cc0ad70a243c8012bf17b4be601abfc8ea2/tools/mtmd/models/llava.cpp#L24-L29) |
| `models/llava.cpp:56` | layer loop bound | [#L56](https://github.com/ggml-org/llama.cpp/blob/e8f19cc0ad70a243c8012bf17b4be601abfc8ea2/tools/mtmd/models/llava.cpp#L56) |
| `.../convert_image_encoder_to_gguf.py:278-282` | `block_count = n-1` for llava projector | [#L278-L282](https://github.com/ggml-org/llama.cpp/blob/e8f19cc0ad70a243c8012bf17b4be601abfc8ea2/tools/mtmd/legacy-models/convert_image_encoder_to_gguf.py#L278-L282) |
| `.../convert_image_encoder_to_gguf.py:349-353` | last-layer pop before export | [#L349-L353](https://github.com/ggml-org/llama.cpp/blob/e8f19cc0ad70a243c8012bf17b4be601abfc8ea2/tools/mtmd/legacy-models/convert_image_encoder_to_gguf.py#L349-L353) |
| `.../convert_image_encoder_to_gguf.py:108` | projector-type choices mlp/ldp/ldpv2 | [#L108](https://github.com/ggml-org/llama.cpp/blob/e8f19cc0ad70a243c8012bf17b4be601abfc8ea2/tools/mtmd/legacy-models/convert_image_encoder_to_gguf.py#L108) |
| `.../minicpmv-convert...py:587,710` | full block_count written | [#L587](https://github.com/ggml-org/llama.cpp/blob/e8f19cc0ad70a243c8012bf17b4be601abfc8ea2/tools/mtmd/legacy-models/minicpmv-convert-image-encoder-to-gguf.py#L587), [#L710](https://github.com/ggml-org/llama.cpp/blob/e8f19cc0ad70a243c8012bf17b4be601abfc8ea2/tools/mtmd/legacy-models/minicpmv-convert-image-encoder-to-gguf.py#L710) |
| `.../minicpmv-convert...py:594-597` | v1 fallback block_count=26 quirk | [#L594-L597](https://github.com/ggml-org/llama.cpp/blob/e8f19cc0ad70a243c8012bf17b4be601abfc8ea2/tools/mtmd/legacy-models/minicpmv-convert-image-encoder-to-gguf.py#L594-L597) |
| `.../glmedge-convert...py:212` | full block_count written | [#L212](https://github.com/ggml-org/llama.cpp/blob/e8f19cc0ad70a243c8012bf17b4be601abfc8ea2/tools/mtmd/legacy-models/glmedge-convert-image-encoder-to-gguf.py#L212) |
| `conversion/llava.py:41-42` | modern converter rejects non-pixtral llava | [#L41-L42](https://github.com/ggml-org/llama.cpp/blob/e8f19cc0ad70a243c8012bf17b4be601abfc8ea2/conversion/llava.py#L41-L42) |
| `gguf-py/gguf/constants.py:4667-4711` | VisionProjectorType lacks mlp/ldp/adapter | [#L4667-L4711](https://github.com/ggml-org/llama.cpp/blob/e8f19cc0ad70a243c8012bf17b4be601abfc8ea2/gguf-py/gguf/constants.py#L4667-L4711) |

## Shared citations (tests/CI)

| Citation | Content | Permalink |
|---|---|---|
| `tools/mtmd/tests.sh:94-97` | llava-graph models in the manual matrix | [#L94-L97](https://github.com/ggml-org/llama.cpp/blob/e8f19cc0ad70a243c8012bf17b4be601abfc8ea2/tools/mtmd/tests.sh#L94-L97) |
| `tools/mtmd/tests.sh:178` | `--temp 0 -n 128` | [#L178](https://github.com/ggml-org/llama.cpp/blob/e8f19cc0ad70a243c8012bf17b4be601abfc8ea2/tools/mtmd/tests.sh#L178) |
| `tools/mtmd/tests.sh:206-209` | substring pass criterion | [#L206-L209](https://github.com/ggml-org/llama.cpp/blob/e8f19cc0ad70a243c8012bf17b4be601abfc8ea2/tools/mtmd/tests.sh#L206-L209) |

## Referenced commits (permalink form)

- `370359e5b` (#3436, 2023-10) original llava support — introduces Bug B.
- `32916a490` (#13321, 2025-05) graph refactor — introduces Bug A
  (pre-refactor parent: `ffc727203af1061fdeb49efef30f76171722e403`).
- `053367d14` (InternVL), `92ecdcc06` (Llama 4), `bacddc049` (CogVLM) —
  copies of the CLS-after-patches pattern into other graphs.
- `e39a2ce66` (2025-12-12) moved the llava graph verbatim into
  `models/llava.cpp`.
