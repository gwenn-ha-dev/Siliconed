#!/usr/bin/env python3
"""Qwen3-VL-8B writes: the goldens of `siliconed check qwen-image-2.1 writer` — greedy `generate`, fp32 on the CPU.

    qwen-image-2.1-writer.py [CASE …] [--max-new N] [--goldens DIR]
    qwen-image-2.1-writer.py [CASE …] --bf16              the bf16 witness of each case, teacher-forced on its golden
    qwen-image-2.1-writer.py system --variant compact     the Compact's published encoder, dequantized

CASE is one of `system` (a rewrite with a system turn), `plain` (the same prompt, no system turn) and
`describe` (one 512x512 picture and an instruction); none given: all three, in one process (the model is
loaded once). Each writes `qwen21/goldens-qwen21-writer-CASE.{safetensors,json}`.

The reference is transformers' `Qwen3VLForConditionalGeneration.generate(do_sample=False)` of the Turbo's
`text_encoder/`, with the Turbo's `generation_config.json`
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

Greedy is the check's choice, not the publisher's: the published `generation_config.json` samples
(`do_sample: true`, temperature 0.7, top-k 20, top-p 0.8); `do_sample=False` with those left unset is argmax.
"""
import argparse
import json
import resource
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from safetensors import safe_open

sys.dont_write_bytecode = True      # the repository carries no caches
import common  # noqa: E402
from common import Checkpoint, Streamed  # noqa: E402

REPOSITORY = "Qwen/Qwen-Image-2.1-Turbo"
#: The Compact's encoder, where the app reads it: Comfy-Org's int8 convrot, the pin of `QwenImage21Modules.swift`.
COMPACT = dict(repo="Comfy-Org/Qwen-Image-2.1", revision="aacf31a6d48e707f352ecba714cee0680da3479b",
               file="text_encoders/qwen3vl_8b_int8_convrot.safetensors",
               sha256="8bfd0f6e12abf2d2d697ecc888e5e90b0d6741d6708f05799f53afa560452e8f")
PROMPT = "woman posing in a library"
#: The Lab's two instructions are the check's own, written here and in `QwenImage21WriterStages.swift` to the
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


class Compact:
    """The Compact's published encoder (`COMPACT`: the linears, the embedding table and the head in int8 convrot with
    a scale per output row, the rest in bf16), read as a `Checkpoint`: every int8 tensor is DEQUANTIZED by the
    publisher's own code (comfy_kitchen's `dequantize_int8_convrot_weight`, fp32, `(q * s)` rotated back by the
    Hadamard of its group), so that the reference computes with the values the file holds.

    The file names the language model without transformers' `language_model.` (`model.layers.N.…`,
    `model.embed_tokens.weight`, `model.norm.weight`); `model.visual.…` and `lm_head.weight` are named alike.
    `check` compares layer 0 with the Turbo's bf16 weights: a quantization step apart."""

    def __init__(self, turbo: Checkpoint):
        import quantized
        self.path = quantized.pinned(COMPACT)
        _, self._dequantize = quantized.comfy_functions()
        self._file = safe_open(str(self.path), framework="pt")
        names = set(self._file.keys())
        self._quantized = {n for n in names if n.endswith(".weight") and n[:-len(".weight")] + ".comfy_quant" in names}
        self.turbo, self.keys = turbo, turbo.keys
        missing = sorted(k for k in self.keys if self.published(k) not in names)
        if missing:
            raise SystemExit(f"{self.path.name}: {len(missing)} of the reference's tensors are not in it, {missing[:4]}")
        self.layer0 = self.check()

    @staticmethod
    def published(key: str) -> str:
        """transformers' name -> the file's."""
        prefix = "model.language_model."
        return "model." + key[len(prefix):] if key.startswith(prefix) else key

    def tensor(self, key: str, dtype: torch.dtype) -> torch.Tensor:
        name = self.published(key)
        if name not in self._quantized:
            value = self._file.get_tensor(name)
            return value.to(dtype) if value.is_floating_point() else value
        layer = name[:-len(".weight")]
        config = json.loads(bytes(self._file.get_tensor(layer + ".comfy_quant").tolist()).decode())
        assert config.get("format") == "int8_tensorwise" and config.get("convrot") is True, config
        group = int(config.get("convrot_groupsize", 256))
        return self._dequantize(self._file.get_tensor(name), self._file.get_tensor(layer + ".weight_scale"), group).to(dtype)

    def under(self, prefix: str, dtype: torch.dtype) -> dict:
        n = len(prefix)
        return {k[n:]: self.tensor(k, dtype) for k in self.keys if k.startswith(prefix)}

    def check(self) -> dict:
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

    def describe(self) -> dict:
        """For the golden's manifest: the file, and how it was read."""
        return {"variant": "compact", "repository": COMPACT["repo"], "revision": COMPACT["revision"],
                "file": COMPACT["file"], "sha256": COMPACT["sha256"],
                "dequantized_by": "comfy_kitchen dequantize_int8_convrot_weight, fp32",
                "layer0_against_turbo": self.layer0}


def load_model(compact: bool = False, dtype: torch.dtype = torch.float32):
    """`Qwen3VLForConditionalGeneration` on `meta` (SDPA, fp32): the embedding table, the final norm and the
    vision tower resident, the 36 layers and `lm_head` streamed; the Turbo's `generation_config.json` attached
    as `from_pretrained` would. With `compact`, the weights are the Compact's published file's, dequantized.
    Returns the model and, for a Compact, its reader."""
    from transformers import AutoConfig, GenerationConfig, Qwen3VLForConditionalGeneration
    config = AutoConfig.from_pretrained(snap() / "text_encoder")
    with torch.device("meta"):
        model = Qwen3VLForConditionalGeneration._from_config(config, attn_implementation="sdpa", dtype=dtype)
    model.eval()
    checkpoint = Checkpoint(snap() / "text_encoder", "model.safetensors")
    if compact:
        checkpoint = Compact(checkpoint)
    check_keys(model, checkpoint)
    streamed = ("model.language_model.layers.", "lm_head.")
    model.load_state_dict({k: checkpoint.tensor(k, dtype) for k in checkpoint.keys
                           if not k.startswith(streamed)}, strict=False, assign=True)
    lm, visual = model.model.language_model, model.model.visual
    # Non-persistent buffers built on `meta`: redone on the CPU.
    lm.rotary_emb = type(lm.rotary_emb)(config=config.text_config)
    visual.rotary_pos_emb = type(visual.rotary_pos_emb)(visual.config)
    for i, layer in enumerate(lm.layers):
        Streamed(layer, f"model.language_model.layers.{i}.", checkpoint, dtype)
    Streamed(model.lm_head, "lm_head.", checkpoint, dtype)
    check_resident(model, streamed)
    model.generation_config = GenerationConfig.from_pretrained(snap() / "text_encoder")
    print(f"  tie_word_embeddings {config.tie_word_embeddings}, stops {model.generation_config.eos_token_id}, "
          f"attention {model.config._attn_implementation}")
    return model, checkpoint if compact else None


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


def run_case(case: str, model, processor, goldens: Path, max_new: int, compact: Compact | None = None):
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
        "weights": compact.describe() if compact else None,
        "model": f"{REPOSITORY}@{common.REVISIONS[REPOSITORY]}", **versions(), "case": case,
        "system": SYSTEM if case == "system" else None, "prompt": PROMPT if case != "describe" else None,
        "rewrite_plain": REWRITE_PLAIN if case == "plain" else None,
        "describe": DESCRIBE if case == "describe" else None, "picture": PICTURE_512 if image else None,
        "max_new_tokens": max_new, "max_pixels": MAX_PIXELS, "stops": sorted(stops), "stopped": generated[-1] in stops,
        "text": written, "seconds": seconds,
        "max_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9})


def run_witness(case: str, model, processor, goldens: Path):
    """The bf16 witness of a case: the model in bf16 as `from_pretrained(torch_dtype=bfloat16)` runs it, one forward
    over the prompt and the fp32 golden's own answer (teacher-forced), the logits that follow each row before a
    token of the answer. What an engine in bf16 is judged beside: its distance to the fp32 logits is bf16's own."""
    print(f"== {case} (bf16, teacher-forced)", flush=True)
    stem = goldens / "qwen21" / f"goldens-qwen21-writer-{case}.safetensors"
    if not stem.exists():
        sys.exit(f"{stem} is missing: the fp32 run writes it first")
    with safe_open(str(stem), framework="pt") as f:
        generated, reference = f.get_tensor("generated").long(), f.get_tensor("logits")
    text = processor.apply_chat_template(messages_of(case), tokenize=False, add_generation_prompt=True)
    image = picture(goldens) if case == "describe" else None
    batch = processor(text=[text], images=[image] if image else None, return_tensors="pt")
    ids = torch.cat([batch["input_ids"], generated[None, :-1]], dim=1)
    inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    if image:
        inputs["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)
        inputs["image_grid_thw"] = batch["image_grid_thw"]
        # The answer's rows are text: type 0 after the processor's own.
        types = batch["mm_token_type_ids"]
        inputs["mm_token_type_ids"] = torch.cat([types, types.new_zeros(1, len(generated) - 1)], dim=1)
    started = time.time()
    logits = model(**inputs).logits[0, -len(generated):].float()
    picks = logits.argmax(-1)
    same = int((picks == generated).sum())
    relative = float((logits - reference).norm() / reference.norm())
    print(f"  {len(generated)} rows in {time.time() - started:.0f} s; {same} picks are the fp32 run's; "
          f"relative L2 to the fp32 logits {relative:.3e}", flush=True)
    common.write_golden(goldens / "qwen21", f"witness-bf16-qwen21-writer-{case}", {"logits": logits, "picks": picks.to(torch.int32)}, {
        "model": f"{REPOSITORY}@{common.REVISIONS[REPOSITORY]}", **versions(), "case": case, "dtype": "bfloat16",
        "teacher_forced_on": stem.name, "same_picks": same, "relative_l2": relative})


def main() -> None:
    common.unbuffered()
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("cases", nargs="*", choices=CASES)
    p.add_argument("--max-new", type=int, default=MAX_NEW, help="tokens at most per case (default 48: ~10 s each)")
    p.add_argument("--variant", choices=("compact",), help="the Compact's published encoder (int8 convrot), dequantized "
                   "for the reference: the `system` case, the one its check reads (`goldens-qwen21-writer-system-compact`)")
    p.add_argument("--bf16", action="store_true", help="write the bf16 witness of each case, teacher-forced on its fp32 "
                                                      "golden (`witness-bf16-qwen21-writer-…`)")
    p.add_argument("--goldens", help="output folder (default: $SILICONED_GOLDENS, else ./goldens)")
    args = p.parse_args()
    goldens = common.goldens_directory(args.goldens)
    (goldens / "qwen21").mkdir(parents=True, exist_ok=True)
    started = time.time()
    compact = args.variant == "compact"
    cases = args.cases or (["system"] if compact else list(CASES))
    if compact and cases != ["system"]:
        p.error("--variant compact: the `system` case, the one the Compact's check reads")
    processor = load_processor()
    if args.bf16:
        if compact:
            p.error("--bf16 is the Standard's witness")
        model, _ = load_model(dtype=torch.bfloat16)
        for case in cases:
            run_witness(case, model, processor, goldens)
        print(f"  total {time.time() - started:.0f} s")
        return
    model, weights = load_model(compact)
    for case in cases:
        run_case(case, model, processor, goldens, args.max_new, weights)
    print(f"  total {time.time() - started:.0f} s")


if __name__ == "__main__":
    main()
