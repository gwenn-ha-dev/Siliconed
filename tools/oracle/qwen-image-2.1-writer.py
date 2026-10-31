#!/usr/bin/env python3
"""Qwen3-VL-8B writes: the goldens of `siliconed check qwen-image-2.1 writer` — greedy `generate`, fp32 on the CPU.

    qwen-image-2.1-writer.py [CASE …] [--max-new N] [--goldens DIR]

CASE is one of `system` (a rewrite with a system turn), `plain` (the same prompt, no system turn) and
`describe` (one 512x512 picture and an instruction); none given: all three, in one process (the model is
loaded once). Each writes `qwen21/goldens-qwen21-writer-CASE.{safetensors,json}`.

The reference is transformers' `Qwen3VLForConditionalGeneration.generate(do_sample=False)` of the Turbo's
`text_encoder/` (the very file the engine's head map is cut from), with the Turbo's `generation_config.json`
(stop on `<|im_end|>` and `<|endoftext|>`) and the Turbo's `processor/` and chat template
(`apply_chat_template`). **The reference code does the computation**: this script transcribes the loading
(the 36 layers and `lm_head` streamed through hooks, as `qwen-image-2.1.py` does for the encoder, so that fp32
fits in 16 GB: ~9 GB, ~10 s a decoding step) and the picture's size (the engine's rule, `smart_resize` on 32
then Pillow's Lanczos, so that the processor's own resize is the identity).

What a golden holds, for the check:
  ids, pixel_values, image_grid_thw   the prompt as the processor builds it
  generated                           the tokens the reference chose (the stop token kept, as `generate` keeps it)
  logits [steps, vocabulary]          the raw logits at every step: the engine, fed these very tokens
                                      (teacher forcing), is judged on them; and the argmax of each row is the
                                      reference's token, asserted here
  gaps [steps]                        top-1 minus top-2 of each row: where it is tiny, a different token is a
                                      tie, not a fault (the check says so)
  head_sha256 (manifest)              the bits of `lm_head.weight` and `model.language_model.norm.weight` in the
                                      published file, which the engine's head map must reproduce

Greedy is the check's choice, not the publisher's: the published `generation_config.json` samples
(`do_sample: true`, temperature 0.7, top-k 20, top-p 0.8); `do_sample=False` with those left unset is argmax.
"""
import argparse
import hashlib
import json
import os
import resource
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open

sys.dont_write_bytecode = True      # the repository carries no caches
import common  # noqa: E402
from common import Checkpoint, Streamed  # noqa: E402

REPOSITORY = "Qwen/Qwen-Image-2.1-Turbo"
PROMPT = "woman posing in a library"
#: The Lab's two instructions are the check's own, written here and in `QwenImageWriterStages.swift` to the
#: character (the check compares the manifest's with its own).
SYSTEM = ("You write prompts for a text-to-image model. Rewrite the user's short prompt as one vivid, detailed "
          "paragraph: subject, pose, clothing, setting, light, mood. Answer with the prompt only.")
REWRITE_PLAIN = ("Rewrite this short prompt for a text-to-image model as one vivid, detailed paragraph, "
                 "and answer with the prompt only: ")
DESCRIBE = ("Describe this picture as a prompt for a text-to-image model: one vivid, detailed paragraph. "
            "Answer with the prompt only.")
PICTURE = "border-collie-beach.png"        # 1024x1024 in `tools/oracle/inputs/`; the case reads it at 512x512
PICTURE_512 = "writer-describe-512.png"
MAX_PIXELS = 1 << 20
MAX_NEW = 48
CASES = ("system", "plain", "describe")

torch.set_grad_enabled(False)

_SNAP: Path | None = None


def snap() -> Path:
    global _SNAP
    if _SNAP is None:
        _SNAP = common.snapshot(REPOSITORY, ["text_encoder", "processor"])
    return _SNAP


def check_keys(module, checkpoint):
    wanted, published = set(module.state_dict()), set(checkpoint.keys)
    if published != wanted:
        missing, extra = sorted(wanted - published), sorted(published - wanted)
        raise SystemExit(f"keys: {len(missing)} missing {missing[:4]}, {len(extra)} extra {extra[:4]}")


def check_resident(module, streamed: tuple[str, ...]):
    for name, t in list(module.named_parameters()) + list(module.named_buffers()):
        if t.is_meta and not name.startswith(streamed):
            raise SystemExit(f"still on meta: {name}")


class CompactMap:
    """The language model of an installed **Compact map** (`qwen3-vl-8b-encoder.compact.v0.silicon`: int8
    convrot linears with a scale per output row, bf16 for the rest), read as a `Checkpoint`: every linear is
    DEQUANTIZED by the publisher's own code (comfy_kitchen's `dequantize_int8_convrot_weight`, fp32,
    `(q * s)` rotated back by the Hadamard of its group), so that the reference computes with the very values
    the engine's map holds. No download: the map is the installed one, and Comfy's functions are the pinned
    files of `Tests/SiliconedTests/Fixtures/quant/comfy/` (`quant_fixtures.py fetch_comfy` writes them).

    The map is the forge's layout, undone here: a linear is stored transposed, `[K, N]`; `q_proj` and `k_proj`
    (rows) and `q_norm`, `k_norm` are permuted per head (`0, Dh/2, 1, Dh/2+1, …`) for an interleaved RoPE.
    The vision tower, the final norm and `lm_head` are not read from the map: the Turbo's, as for the Standard
    (the cases of a Compact check are the text ones). `check` compares layer 0 with the Turbo's bf16 weights, to
    show the undoing is right: a quantization step apart, not a permutation or a transposition."""

    PERMUTED = ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight")

    def __init__(self, path: Path, turbo: Checkpoint, comfy_dir: Path):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import quant_fixtures as qf
        qf.COMFY_DIR = str(comfy_dir)
        _, self._dequantize = qf.comfy_functions()
        self.path, self.turbo = path, turbo
        with open(path, "rb") as f:
            assert f.read(7) == b"SILICON"
            f.read(1)
            (length,) = struct.unpack("<Q", f.read(8))
            self.header = json.loads(f.read(length))
        self.data = np.memmap(path, mode="r", dtype=np.uint8)
        self.entries = self.header["tensors"]
        self.group = {e["rotation"]["group"] for e in self.entries.values() if "rotation" in e}
        assert self.group == {256} and self.header["linear_weights_transposed"] is True
        self.keys = turbo.keys

    @staticmethod
    def mapped(key: str) -> str | None:
        prefix = "model.language_model."
        return key[len(prefix):] if key.startswith(prefix) and not key.startswith(prefix + "norm.") else None

    def _raw(self, entry: dict, dtype, offset: int, count: int):
        return np.frombuffer(self.data, dtype=dtype, count=count, offset=offset)

    def tensor(self, key: str, dtype: torch.dtype) -> torch.Tensor:
        name = self.mapped(key)
        if name is None:
            return self.turbo.tensor(key, dtype)
        entry = self.entries[name]
        shape = entry["shape"]
        if entry["dtype"] == "int8":
            k, n = shape                                           # stored transposed
            q = torch.from_numpy(self._raw(entry, np.int8, entry["offset"], k * n).reshape(k, n).copy()).t().contiguous()
            scale = torch.from_numpy(self._raw(entry, np.float32, entry["offset"] + entry["scale"]["offset"], n).copy())
            value = self._dequantize(q, scale[:, None], int(entry["rotation"]["group"]))
        else:
            assert entry["dtype"] == "bfloat16" and not entry.get("transposed")
            raw = self._raw(entry, np.int16, entry["offset"], int(np.prod(shape))).reshape(shape).copy()
            value = torch.from_numpy(raw).view(torch.bfloat16).float()
        if name.endswith(self.PERMUTED):
            heads_dim = 128
            index = torch.arange(value.shape[0]).view(-1, heads_dim)
            half = heads_dim // 2
            order = torch.empty_like(index)
            order[:, :half] = index[:, 0::2]                       # old p     sits at new 2p
            order[:, half:] = index[:, 1::2]                       # old p+Dh/2 sits at new 2p+1
            value = value[order.reshape(-1)]
        return value.to(dtype) if value.is_floating_point() else value

    def under(self, prefix: str, dtype: torch.dtype) -> dict:
        n = len(prefix)
        return {k[n:]: self.tensor(k, dtype) for k in self.keys if k.startswith(prefix)}

    def check(self, comfy_note: str = "") -> dict:
        """Layer 0, every linear, against the Turbo's bf16: the relative error per output row, worst and median."""
        out = {}
        for name in ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
                     "self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.down_proj.weight",
                     "self_attn.q_norm.weight"):
            key = "model.language_model.layers.0." + name
            ours, theirs = self.tensor(key, torch.float32), self.turbo.tensor(key, torch.float32)
            if ours.dim() == 1:
                out[name] = {"max_abs_diff": float((ours - theirs).abs().max())}
                continue
            r = (ours - theirs).double().norm(dim=1) / theirs.double().norm(dim=1)
            out[name] = {"worst_row": float(r.max()), "median_row": float(r.median())}
        print("  Compact against the Turbo's bf16, layer 0:", json.dumps(out))
        return out


def load_model(compact: Path | None = None):
    """`Qwen3VLForConditionalGeneration` on `meta` (SDPA, fp32): the embedding table, the final norm and the
    vision tower resident, the 36 layers and `lm_head` streamed; the Turbo's `generation_config.json` attached
    as `from_pretrained` would. With `compact`, the language model's weights are that map's, dequantized."""
    from transformers import AutoConfig, GenerationConfig, Qwen3VLForConditionalGeneration
    config = AutoConfig.from_pretrained(snap() / "text_encoder")
    with torch.device("meta"):
        model = Qwen3VLForConditionalGeneration._from_config(config, attn_implementation="sdpa", dtype=torch.float32)
    model.eval()
    checkpoint = Checkpoint(snap() / "text_encoder", "model.safetensors")
    compact_check = None
    if compact:
        comfy = Path(os.environ.get("SILICONED_COMFY_DIR") or Path(__file__).resolve().parents[2]
                     / "Tests" / "SiliconedTests" / "Fixtures" / "quant" / "comfy")
        if not (comfy / "kitchen").exists():
            sys.exit(f"{comfy}/kitchen is missing: `python tools/oracle/quant_fixtures.py` fetches Comfy's files "
                     "(or SILICONED_COMFY_DIR names a checkout that has them)")
        checkpoint = CompactMap(compact, checkpoint, comfy)
        compact_check = checkpoint.check()
    check_keys(model, checkpoint)
    streamed = ("model.language_model.layers.", "lm_head.")
    model.load_state_dict({k: checkpoint.tensor(k, torch.float32) for k in checkpoint.keys
                           if not k.startswith(streamed)}, strict=False, assign=True)
    lm, visual = model.model.language_model, model.model.visual
    # Non-persistent buffers built on `meta`: redone on the CPU.
    lm.rotary_emb = type(lm.rotary_emb)(config=config.text_config)
    visual.rotary_pos_emb = type(visual.rotary_pos_emb)(visual.config)
    for i, layer in enumerate(lm.layers):
        Streamed(layer, f"model.language_model.layers.{i}.", checkpoint, torch.float32)
    Streamed(model.lm_head, "lm_head.", checkpoint, torch.float32)
    check_resident(model, streamed)
    model.generation_config = GenerationConfig.from_pretrained(snap() / "text_encoder")
    print(f"  tie_word_embeddings {config.tie_word_embeddings}, stops {model.generation_config.eos_token_id}, "
          f"attention {model.config._attn_implementation}")
    return model, compact_check


def load_processor():
    """The Turbo's `Qwen3VLProcessor`, torchvision image processor: its rescale and normalization are fused,
    as the engine's (the Pillow fallback rounds twice and moves every value by an ulp)."""
    from transformers import Qwen3VLProcessor
    from transformers.image_processing_backends import TorchvisionBackend
    processor = Qwen3VLProcessor.from_pretrained(snap() / "processor")
    if not isinstance(processor.image_processor, TorchvisionBackend):
        sys.exit(f"the image processor is {type(processor.image_processor).__name__}, not the torchvision one: "
                 "install torchvision (tools/oracle/requirements.txt)")
    return processor


def head_hashes() -> dict:
    """sha256 of the raw bf16 bytes of the two tensors the head map keeps, as published."""
    out = {}
    with safe_open(str(snap() / "text_encoder" / "model.safetensors"), framework="pt") as f:
        for name in ("lm_head.weight", "model.language_model.norm.weight"):
            t = f.get_tensor(name)
            assert t.dtype == torch.bfloat16
            out[name] = hashlib.sha256(t.contiguous().view(torch.int16).numpy().tobytes()).hexdigest()
    return out


def picture(goldens: Path) -> Image.Image:
    """The case's 512x512 picture: the shipped 1024x1024 one by Pillow's Lanczos, written beside the goldens
    where the check reads the very same file; then the engine's size rule (the identity at 512x512)."""
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    inputs = common.provide_inputs(goldens)
    target = inputs / PICTURE_512
    if not target.exists():
        Image.open(inputs / PICTURE).convert("RGB").resize((512, 512), Image.LANCZOS).save(target)
    image = Image.open(target).convert("RGB")
    height, width = smart_resize(image.height, image.width, factor=32, min_pixels=65536, max_pixels=MAX_PIXELS)
    assert (width, height) == (512, 512) == image.size
    return image


def messages_of(case: str) -> list:
    if case == "system":
        return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": PROMPT}]
    if case == "plain":
        return [{"role": "user", "content": REWRITE_PLAIN + PROMPT}]
    return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": DESCRIBE}]}]


def expected_text(case: str) -> str:
    """The template by hand, as `Qwen3VLChat.template` writes it: the check of the check."""
    slot = "<|vision_start|><|image_pad|><|vision_end|>"
    if case == "system":
        return f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{PROMPT}<|im_end|>\n<|im_start|>assistant\n"
    if case == "plain":
        return f"<|im_start|>user\n{REWRITE_PLAIN}{PROMPT}<|im_end|>\n<|im_start|>assistant\n"
    return f"<|im_start|>user\n{slot}{DESCRIBE}<|im_end|>\n<|im_start|>assistant\n"


def versions() -> dict:
    import importlib.metadata as metadata
    return {"transformers": metadata.version("transformers"), "torchvision": metadata.version("torchvision"),
            "pillow": metadata.version("pillow")}


class Steps:
    """Counts `lm_head` calls (one a decoding step) and says where the run is."""

    def __init__(self):
        self.n, self.started = 0, time.time()

    def hook(self, _module, _inputs, _output):
        self.n += 1
        if self.n % 8 == 0:
            print(f"  step {self.n}, {time.time() - self.started:.0f} s", flush=True)


def run_case(case: str, model, processor, goldens: Path, max_new: int, hashes: dict, compact: Path | None = None,
             compact_check: dict | None = None):
    print(f"== {case}", flush=True)
    messages = messages_of(case)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    assert text == expected_text(case), f"chat template: {text!r}"
    image = picture(goldens) if case == "describe" else None
    batch = processor(text=[text], images=[image] if image else None, return_tensors="pt")
    n = batch["input_ids"].shape[1]
    print(f"  prompt: {n} tokens" + (f", grid {batch['image_grid_thw'][0].tolist()}" if image else ""))
    if image:
        same = processor.image_processor(images=[image], do_resize=False, return_tensors="pt")["pixel_values"]
        assert torch.equal(same, batch["pixel_values"]), "the processor resized a picture already at its size"

    steps = Steps()
    handle = model.lm_head.register_forward_hook(steps.hook)
    started = time.time()
    out = model.generate(**batch, do_sample=False, temperature=None, top_p=None, top_k=None,
                         max_new_tokens=max_new, output_logits=True, return_dict_in_generate=True)
    seconds = time.time() - started
    handle.remove()
    generated = out.sequences[0, n:].tolist()
    stops = set(model.generation_config.eos_token_id)
    end = next((t + 1 for t, token in enumerate(generated) if token in stops), len(generated))
    generated = generated[:end]
    logits = torch.stack([l[0] for l in out.logits])[:end].float()
    assert logits.argmax(-1).tolist() == generated, "greedy is not the argmax of the logits"
    top = logits.topk(2, dim=-1)
    gaps = (top.values[:, 0] - top.values[:, 1]).contiguous()
    written = processor.tokenizer.decode(generated, skip_special_tokens=True)
    print(f"  {len(generated)} tokens in {seconds:.0f} s, smallest gap {float(gaps.min()):.4f} "
          f"(step {int(gaps.argmin())})\n    « {written} »", flush=True)

    payload = {"ids": batch["input_ids"][0].to(torch.int32), "generated": torch.tensor(generated, dtype=torch.int32),
               "logits": logits, "gaps": gaps}
    if image:
        payload["pixel_values"] = batch["pixel_values"].float()
        payload["image_grid_thw"] = batch["image_grid_thw"][0].to(torch.int32)
    common.write_golden(goldens / "qwen21", f"goldens-qwen21-writer-{case}" + ("-compact" if compact else ""), payload, {
        "map": compact.name if compact else None, "map_bytes": compact.stat().st_size if compact else None,
        "dequantized_by": "comfy_kitchen dequantize_int8_convrot_weight, fp32" if compact else None,
        "layer0_against_turbo": compact_check,
        "model": f"{REPOSITORY}@{common.REVISIONS[REPOSITORY]}", **versions(), "case": case,
        "system": SYSTEM if case == "system" else None, "prompt": PROMPT if case != "describe" else None,
        "rewrite_plain": REWRITE_PLAIN if case == "plain" else None,
        "describe": DESCRIBE if case == "describe" else None, "picture": PICTURE_512 if image else None,
        "max_new_tokens": max_new, "max_pixels": MAX_PIXELS, "stops": sorted(stops), "stopped": generated[-1] in stops,
        "text": written, "seconds": seconds, "head_sha256": hashes,
        "max_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9})


def main() -> None:
    common.unbuffered()
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("cases", nargs="*", choices=CASES)
    p.add_argument("--max-new", type=int, default=MAX_NEW, help="tokens at most per case (default 48: ~10 s each)")
    p.add_argument("--map", help="judge a Compact: the installed int8 convrot encoder map, dequantized for the reference "
                                 "(the text cases only; goldens `…-compact`)")
    p.add_argument("--goldens", help="output folder (default: $SILICONED_GOLDENS, else ./goldens)")
    args = p.parse_args()
    goldens = common.goldens_directory(args.goldens)
    (goldens / "qwen21").mkdir(parents=True, exist_ok=True)
    started = time.time()
    hashes = head_hashes()
    compact = Path(args.map).resolve() if args.map else None
    cases = args.cases or [c for c in CASES if not (compact and c == "describe")]
    if compact and "describe" in cases:
        p.error("--map: the picture's case needs the vision tower, which the Compact check does not dequantize")
    processor = load_processor()
    model, compact_check = load_model(compact)
    for case in cases:
        run_case(case, model, processor, goldens, args.max_new, hashes, compact, compact_check)
    print(f"  total {time.time() - started:.0f} s")


if __name__ == "__main__":
    main()
