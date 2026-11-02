#!/usr/bin/env python3
"""Anima Turbo v1.1's goldens: what `siliconed check anima <stage>` judges the engine against.

    anima.py tokenizer                      the T5 tokenizer on a corpus of edge cases -> anima/t5-corpus.json
    anima.py text [--long]                  both tokenizers, Qwen3-0.6B, the conditioner -> goldens-anima-text[-long]
    anima.py dit --size 512 [1024 768x512]  one DiT evaluation at the first sigma       -> goldens-anima-dit-<size>
    anima.py bf16 --size 1024               the same evaluation in bfloat16             -> witness-bf16-anima-dit-<size>
    anima.py dit|bf16 --size 512 --variant compact|light   the same, for the Compact's or the Light's published DiT
                                            (GGUF), dequantized by gguf-py -> goldens-anima-dit-<variant>-<size>
    anima.py trajectory --size 512 [1024]   the 8-step Euler loop                       -> goldens-anima-trajectory-<size>
    anima.py vae --size 512 [1024 768x512]  the decoder on the final latent             -> goldens-anima-vae-<size>
    anima.py witness --size 512 --step 0    one guided evaluation in float64            -> witness-anima-f64-<size>-step<n>
    anima.py vae-witness --size 512         the decoder in float64                      -> witness-anima-vae-f64-<size>
    anima.py img2img --size 512             img2img at strength 0.6: the picture encoded, the start, the
                                            truncated loop (steps 3-7)                  -> goldens-anima-img2img-<size>
    anima.py all [--size 512 1024 768x512]  all of the above, in an order that satisfies their dependencies

`bf16` is the bar of a half-precision path (the ANE's fp16 products): the `dit` evaluation with the
reference in bfloat16, on the `dit` golden's inputs. It is not part of `all` (costly, and only that path reads it).

Goldens go in `<goldens>/anima/`; `--goldens DIR`, else `SILICONED_GOLDENS`, else `./goldens`. Each stage
rereads what an earlier one wrote (`dit` the text, `trajectory` the text, `vae` the trajectory, a rectangle's
`vae` the 1024 trajectory), so run them in the order above. `all` at the default size takes about
two minutes on an M1 Pro, and about ten with `--size 512 1024 768x512`, which writes everything
`siliconed check anima` reads. The witness of the DiT is the first step's (the only one the check reads) and
exists for the squares: a rectangle has no trajectory to take its starting latent from.

The reference is diffusers' `CosmosTransformer3DModel`, `AnimaTextConditioner` and
`AutoencoderKLQwenImage`, with transformers' Qwen3 and T5 tokenizer, in float32 on the CPU. The Turbo
weights are only published in ComfyUI format (one file, DiT and conditioner mixed): `turbo_state_dicts` renames
them to diffusers' names, and loading is **strict**: a missing or extra tensor makes the oracle fail rather
than skew it. The published weights are bf16 and are WIDENED to fp32 before loading, otherwise the module
would silently become bf16 and the oracle would be a bf16 oracle.

What the reference does that a port by analogy would miss:
  - the text is not one encoder: Qwen3-0.6B's `last_hidden_state` (final norm INCLUDED, unlike Z-Image) and
    the T5 token identifiers both enter a conditioner, which maps the first onto the second's vocabulary;
    its output is padded to 512 with zeros. No chat template, "longest" padding (hence none for one prompt).
  - the step is `x += (sigma[i+1] - sigma[i]) * v`, with no negation (Z-Image's pipeline negates).
  - the schedule is `linspace(1, 1/N, N)` through a static shift of 3.
  - diffusers forces RoPE to float32 (`x.float()` in `apply_rotary_emb`, `dtype=torch.float32` in
    `CosmosRotaryPosEmbed`): the witness replaces both by their transcription in double, otherwise it would be
    no more precise than what it judges.

Setup:  in a fresh virtual environment, pip install -r tools/oracle/requirements.txt
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

sys.dont_write_bytecode = True      # the repository carries no caches
import common  # noqa: E402
import quantized  # noqa: E402

BASE_REPOSITORY = "circlestone-labs/Anima-Base-v1.0-Diffusers"
TURBO_REPOSITORY = "circlestone-labs/Anima"
TURBO_FILE = "split_files/diffusion_models/anima-turbo-v1.1.safetensors"
PROMPT = "a 30 year old woman posing in a library"      # the reference prompt
SEED = 42
STEPS = 8            # the Turbo card: "CFG 1, 8-12 steps"; the low end
SHIFT = 3.0          # scheduler_config.json
MAX_SEQUENCE_LENGTH = 512

torch.set_grad_enabled(False)

OUT: Path
_ROOTS: dict = {}


def base() -> Path:
    if "base" not in _ROOTS:
        # The base repository's transformer and conditioner weights are not read (the Turbo file has both,
        # in ComfyUI's format): only their configurations.
        _ROOTS["base"] = common.snapshot(BASE_REPOSITORY, ["tokenizer", "t5_tokenizer", "text_encoder",
                                                           "scheduler", "vae"],
                                         files=["transformer/config.json", "text_conditioner/config.json"])
    return _ROOTS["base"]


def turbo() -> Path:
    if "turbo" not in _ROOTS:
        _ROOTS["turbo"] = common.snapshot(TURBO_REPOSITORY, [], files=[TURBO_FILE]) / TURBO_FILE
    return _ROOTS["turbo"]


# ------------------------------------------------------------------ ComfyUI -> diffusers, strict

BLOCK_RULES = [
    (r"adaln_modulation_self_attn\.1\.", "norm1.linear_1."),
    (r"adaln_modulation_self_attn\.2\.", "norm1.linear_2."),
    (r"adaln_modulation_cross_attn\.1\.", "norm2.linear_1."),
    (r"adaln_modulation_cross_attn\.2\.", "norm2.linear_2."),
    (r"adaln_modulation_mlp\.1\.", "norm3.linear_1."),
    (r"adaln_modulation_mlp\.2\.", "norm3.linear_2."),
    (r"self_attn\.q_proj\.", "attn1.to_q."), (r"self_attn\.k_proj\.", "attn1.to_k."),
    (r"self_attn\.v_proj\.", "attn1.to_v."), (r"self_attn\.output_proj\.", "attn1.to_out.0."),
    (r"self_attn\.q_norm\.", "attn1.norm_q."), (r"self_attn\.k_norm\.", "attn1.norm_k."),
    (r"cross_attn\.q_proj\.", "attn2.to_q."), (r"cross_attn\.k_proj\.", "attn2.to_k."),
    (r"cross_attn\.v_proj\.", "attn2.to_v."), (r"cross_attn\.output_proj\.", "attn2.to_out.0."),
    (r"cross_attn\.q_norm\.", "attn2.norm_q."), (r"cross_attn\.k_norm\.", "attn2.norm_k."),
    (r"mlp\.layer1\.", "ff.net.0.proj."), (r"mlp\.layer2\.", "ff.net.2."),
]
TOP_RULES = {
    "x_embedder.proj.1.weight": "patch_embed.proj.weight",
    "t_embedder.1.linear_1.weight": "time_embed.t_embedder.linear_1.weight",
    "t_embedder.1.linear_2.weight": "time_embed.t_embedder.linear_2.weight",
    "t_embedding_norm.weight": "time_embed.norm.weight",
    "final_layer.adaln_modulation.1.weight": "norm_out.linear_1.weight",
    "final_layer.adaln_modulation.2.weight": "norm_out.linear_2.weight",
    "final_layer.linear.weight": "proj_out.weight",
}


def diffusers_name(name: str) -> str:
    """A DiT name of the ComfyUI file (`blocks.3.self_attn.q_proj.weight`) -> its diffusers name."""
    if name in TOP_RULES:
        return TOP_RULES[name]
    if m := re.match(r"blocks\.(\d+)\.(.*)", name):
        rest = m.group(2)
        for pattern, target in BLOCK_RULES:
            if re.match(pattern, rest):
                return f"transformer_blocks.{m.group(1)}." + re.sub(pattern, target, rest, count=1)
        raise SystemExit(f"block key without a rule: {name}")
    raise SystemExit(f"key without a rule: {name}")


def turbo_state_dicts() -> tuple[dict, dict]:
    """The Turbo file, renamed and WIDENED to fp32: (DiT, conditioner)."""
    dit, adapter = {}, {}
    with safe_open(str(turbo()), framework="pt") as f:
        for key in f.keys():
            name = key.removeprefix("model.diffusion_model.")
            tensor = f.get_tensor(key).float()
            if name.startswith("llm_adapter."):
                adapter[name.removeprefix("llm_adapter.")] = tensor
            else:
                dit[diffusers_name(name)] = tensor
    return dit, adapter


def load_transformer(variant: str | None = None):
    """The fp32 DiT. With a `variant`, the weights are the Compact's or the Light's published GGUF, dequantized
    by gguf-py (`quantized.Variant`); only `transformer/config.json` is read from the base repository, and the
    text adapter the file also holds is left out."""
    from diffusers import CosmosTransformer3DModel
    if variant:
        root = common.snapshot(BASE_REPOSITORY, [], files=["transformer/config.json"])
        config = CosmosTransformer3DModel.load_config(root / "transformer")
    else:
        config = CosmosTransformer3DModel.load_config(base() / "transformer")
    with torch.device("meta"):
        model = CosmosTransformer3DModel.from_config(config)
    if variant:
        published = quantized.Variant("anima", variant)
        dit = {k: published.tensor(k, torch.float32) for k in published.keys}
        model.published = published
    else:
        dit, _ = turbo_state_dicts()
    model.load_state_dict(dit, strict=True, assign=True)
    # `rope` has no weights: its possible buffers remain to be materialized on the CPU.
    model = model.to(torch.float32)
    return model.eval()


def load_conditioner():
    from diffusers import AnimaTextConditioner
    config = AnimaTextConditioner.load_config(base() / "text_conditioner")
    model = AnimaTextConditioner.from_config(config).to(torch.float32)
    _, adapter = turbo_state_dicts()
    model.load_state_dict(adapter, strict=True)
    return model.eval()


def write(stem: str, payload: dict, extra: dict | None = None):
    common.write_golden(OUT, stem, payload,
                        {"prompt": PROMPT, "seed": SEED, "steps": STEPS, "shift": SHIFT,
                         "turbo": f"{TURBO_REPOSITORY}@{common.REVISIONS[TURBO_REPOSITORY]}:{TURBO_FILE}",
                         **(extra or {})})


def read(stem: str, producer: str) -> dict:
    return common.read_golden(OUT, stem, producer)


# ------------------------------------------------------------------------------------ stages

#: The T5 tokenizer's edge cases. The reference prompt only touches ASCII: the `Precompiled` normalizer
#: (nmt_nfkc), the merging of unknowns, a `▁` typed by the user, special tokens in the text and the
#: truncation at 512 only show up here.
CASES = [
    "", " ", "   ", "a", "masterpiece, best quality, 1girl",
    "  leading and trailing  ", "tab\tand\nnewline\r\nmixed", "multiple    spaces",
    "Ｆｕｌｌｗｉｄｔｈ ＡＢＣ １２３", "ﬁ ligature ﬂ", "café naïve résumé", "cafe\u0301 combining",
    "①②③ ⅳ ㎏ ℃", "日本語のテキスト", "한국어 텍스트", "中文提示词，逗号。", "emoji 😀 👩‍👩‍👧 🇫🇷",
    "zero\u200bwidth", "nbsp\u00a0space", "ideographic\u3000space", "soft\u00adhyphen",
    "▁already▁marked", "▁", "a▁▁b", "</s> inside", "<pad><unk></s>", "<extra_id_0> sentinel <extra_id_99>",
    "score_9, score_8_up, rating:safe, (masterpiece:1.2)", "Ω≈ç√∫˜µ≤≥÷", "\u0000control\u0007chars",
    "tilde ~ caret ^ backtick ` pipe |", "UPPER lower MiXeD", "numbers 3.14159 2,718 1e-10",
    "Ⅻ Ⓐ ⒜ ㊙", "ｶﾀｶﾅ half-width", "ß ẞ ǅ ǆ", "Ǆemo", "\ufeffBOM", "‘quotes’ “double” «guillemets»",
    "— – ‑ dashes", "…ellipsis", "x" * 40, "long " * 700,
]


def stage_tokenizer():
    """Each case through `T5TokenizerFast` (truncated at 512), plus 300 random strings drawn from the
    alphabets that matter (Latin, combining marks, punctuation, CJK, full-width forms). No weights."""
    from transformers import T5TokenizerFast
    tokenizer = T5TokenizerFast.from_pretrained(base() / "t5_tokenizer")
    rng = random.Random(0)
    alphabet = [chr(c) for c in list(range(32, 127)) + list(range(0xA0, 0x250)) + list(range(0x300, 0x370))
                + list(range(0x2000, 0x2070)) + list(range(0x3000, 0x3100)) + list(range(0xFF00, 0xFFF0))]
    cases = list(CASES)
    for _ in range(300):
        cases.append("".join(rng.choice(alphabet) for _ in range(rng.randint(1, 40))))
    out = []
    for text in cases:
        ids = tokenizer([text], max_length=512, truncation=True).input_ids[0]
        out.append({"text": text, "ids": ids})
    (OUT / "t5-corpus.json").write_text(json.dumps(out, ensure_ascii=False))
    print(f"-> {OUT / 't5-corpus.json'} ({len(out)} cases)")


def stage_text(prompt: str = PROMPT, length: int = MAX_SEQUENCE_LENGTH, stem: str = "goldens-anima-text"):
    """Reproduces `AnimaTextEncoderStep` + `AnimaConditionStep`: no chat template, "longest" padding (hence
    none for a single prompt), `last_hidden_state` (final norm INCLUDED, unlike Z-Image), then the
    conditioner, padded to 512 with zeros (a minimum: a longer text keeps its rows). `--long`:
    `common.LONG_PROMPT`, `max_sequence_length` raised for both tokenizers."""
    from transformers import AutoTokenizer, Qwen3Model
    from transformers import T5TokenizerFast
    tokenizer = AutoTokenizer.from_pretrained(base() / "tokenizer")
    t5 = T5TokenizerFast.from_pretrained(base() / "t5_tokenizer")
    qwen_inputs = tokenizer([prompt], padding="longest", max_length=length,
                            truncation=True, return_tensors="pt")
    t5_inputs = t5([prompt], padding="longest", max_length=length,
                   truncation=True, return_tensors="pt")

    encoder = Qwen3Model.from_pretrained(base() / "text_encoder", torch_dtype=torch.float32).eval()
    assert next(encoder.parameters()).dtype == torch.float32
    out = encoder(input_ids=qwen_inputs.input_ids, attention_mask=qwen_inputs.attention_mask,
                  output_hidden_states=True)
    qwen = out.last_hidden_state * qwen_inputs.attention_mask.unsqueeze(-1).float()
    probes = {f"qwen_layer{i}_out": out.hidden_states[i + 1][0] for i in (0, 13, 26)}
    del encoder

    conditioner = load_conditioner()
    taps = {}
    for i in (0, 5):
        conditioner.blocks[i].register_forward_hook(common.tap(taps, f"cond_block{i}_out"))
    embeds = conditioner(source_hidden_states=qwen, target_input_ids=t5_inputs.input_ids,
                         target_attention_mask=t5_inputs.attention_mask,
                         source_attention_mask=qwen_inputs.attention_mask)
    payload = {"qwen_ids": qwen_inputs.input_ids[0].to(torch.int32),
               "t5_ids": t5_inputs.input_ids[0].to(torch.int32),
               "qwen_last_hidden": qwen[0], "prompt_embeds": embeds[0], **probes, **taps}
    for inputs in (qwen_inputs, t5_inputs):
        assert inputs.input_ids.shape[1] < length, f"{inputs.input_ids.shape[1]} tokens: the prompt was cut at {length}"
    write(stem, payload, {
        "prompt": prompt, "max_sequence_length": length,
        "qwen_tokens": int(qwen_inputs.input_ids.shape[1]),
        "t5_tokens": int(t5_inputs.input_ids.shape[1])})


def schedule(steps: int = STEPS):
    """`FlowMatchEulerDiscreteScheduler` under `sigmas = linspace(1, 1/N, N)`, static shift 3."""
    from diffusers import FlowMatchEulerDiscreteScheduler
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(base() / "scheduler")
    scheduler.set_timesteps(sigmas=np.linspace(1.0, 1 / steps, steps), device="cpu")
    return scheduler


def initial_latent(height: int, width: int) -> torch.Tensor:
    from diffusers.utils.torch_utils import randn_tensor
    generator = torch.Generator("cpu").manual_seed(SEED)
    return randn_tensor((1, 16, 1, height // 8, width // 8), generator=generator,
                        device=torch.device("cpu"), dtype=torch.float32)


def probe_dit(model) -> dict:
    """The boundaries the engine is judged at: patch embedding, time embedding, blocks 0, 1, 13, 27 and
    block 0's three branches. Returns the dict the hooks fill during the forward pass."""
    taps = {}
    model.patch_embed.register_forward_hook(common.tap(taps, "patch_embed_out", lambda o: o[0].flatten(0, 2)))

    def time_hook(_m, _a, o):
        taps["temb"], taps["embedded_timestep"] = o[0][0].clone(), o[1][0].clone()
    model.time_embed.register_forward_hook(time_hook)
    for i in (0, 1, 13, 27):
        model.transformer_blocks[i].register_forward_hook(common.tap(taps, f"block{i}_out"))
    block0 = model.transformer_blocks[0]
    block0.attn1.register_forward_hook(common.tap(taps, "block0_attn1_out"))
    block0.attn2.register_forward_hook(common.tap(taps, "block0_attn2_out"))
    block0.ff.register_forward_hook(common.tap(taps, "block0_ff_out"))
    return taps


def stage_dit_variant(size: str, variant: str):
    """`dit` on the Compact's or the Light's published weights: the Standard's golden inputs (same latent, same
    text, same timestep), the same probes. Writes `goldens-anima-dit-<variant>-<size>`."""
    height, width, name = common.parse_size(size)
    standard = read(f"goldens-anima-dit-{name}", f"anima.py dit --size {name}")
    model = load_transformer(variant)
    taps = probe_dit(model)
    timestep = standard["timestep"]
    out = model(hidden_states=standard["latent_in"][None, :, None], timestep=timestep,
                encoder_hidden_states=standard["prompt_embeds"][None], padding_mask=torch.zeros(1, 1, height, width),
                return_dict=False)[0]
    payload = {"latent_in": standard["latent_in"], "timestep": timestep, "prompt_embeds": standard["prompt_embeds"],
               "model_out": out[0, :, 0], **taps}
    write(quantized.stem(f"goldens-anima-dit-{name}", variant), payload,
          {"inputs": f"goldens-anima-dit-{name}", "weights": model.published.describe()})


def stage_dit(size: str):
    height, width, name = common.parse_size(size)
    embeds = read("goldens-anima-text", "anima.py text")["prompt_embeds"][None]
    model = load_transformer()
    scheduler = schedule()
    latent = initial_latent(height, width)
    t = scheduler.timesteps[0]
    timestep = (t / scheduler.config.num_train_timesteps).expand(1).float()
    padding = torch.zeros(1, 1, height, width)

    taps = probe_dit(model)

    out = model(hidden_states=latent, timestep=timestep, encoder_hidden_states=embeds,
                padding_mask=padding, return_dict=False)[0]
    payload = {"latent_in": latent[0, :, 0], "timestep": timestep, "prompt_embeds": embeds[0],
               "model_out": out[0, :, 0], **taps}
    write(f"goldens-anima-dit-{name}", payload, {"sigma": float(scheduler.sigmas[0])})


def stage_bf16(size: str, variant: str | None = None):
    """`dit` with diffusers in bfloat16, as `from_pretrained(torch_dtype=torch.bfloat16)` runs it: weights,
    latent, text and timestep in bf16 (the published weights are bf16, so the cast is exact). The inputs are
    the dit golden's own; the boundaries are widened to float32 (exact) under the golden's keys, minus its
    inputs. Writes `witness-bf16-anima-dit-<size>`."""
    height, width, name = common.parse_size(size)
    golden = read(f"goldens-anima-dit-{name}", f"anima.py dit --size {name}")
    half = torch.bfloat16
    model = load_transformer(variant).to(half)
    taps = probe_dit(model)
    out = model(hidden_states=golden["latent_in"][:, None][None].to(half), timestep=golden["timestep"].to(half),
                encoder_hidden_states=golden["prompt_embeds"][None].to(half),
                padding_mask=torch.zeros(1, 1, height, width, dtype=half), return_dict=False)[0]
    taps["model_out"] = out[0, :, 0]
    write(quantized.stem(f"witness-bf16-anima-dit-{name}", variant), {k: v.float() for k, v in taps.items()},
          {"dtype": "bfloat16", "inputs": f"goldens-anima-dit-{name}",
           **({"weights": model.published.describe()} if variant else {})})


def stage_trajectory(size: str):
    resolution = int(size)
    embeds = read("goldens-anima-text", "anima.py text")["prompt_embeds"][None]
    model = load_transformer()
    scheduler = schedule()
    scheduler.set_begin_index(0)
    latent = initial_latent(resolution, resolution)
    padding = torch.zeros(1, 1, resolution, resolution)
    payload = {"latent_init": latent[0, :, 0].clone(), "sigmas": scheduler.sigmas.float()}
    for i, t in enumerate(scheduler.timesteps):
        timestep = (t / scheduler.config.num_train_timesteps).expand(1).float()
        velocity = model(hidden_states=latent, timestep=timestep, encoder_hidden_states=embeds,
                         padding_mask=padding, return_dict=False)[0]
        latent = scheduler.step(velocity, t, latent, return_dict=False)[0]
        payload[f"velocity_{i}"] = velocity[0, :, 0].clone()
        payload[f"latent_{i + 1}"] = latent[0, :, 0].clone()
        print(f"  step {i}: sigma = {float(scheduler.sigmas[i]):.4f}", flush=True)
    payload["latent_final"] = latent[0, :, 0].clone()
    write(f"goldens-anima-trajectory-{resolution}", payload)


def stage_img2img(size: str):
    """**img2img**, by diffusers' modular Anima blocks (`AnimaImg2ImgVaeEncoderStep`, `AnimaImg2ImgSetTimestepsStep`),
    each piece their code:

        z0        = encode_vae_image(picture, sample_mode="argmax")   the posterior mean, (z - mean) / std
        timesteps = get_timesteps(scheduler, N, strength)             the full schedule, then truncated
        x_s       = scheduler.scale_noise(z0, timesteps[:1], eps)     eps: the txt2img noise of the seed
        then the trajectory's loop from x_s, steps t_start ... N - 1

    The noise is the txt2img one, so that strength 1 is txt2img bit for bit, as in the engine."""
    from diffusers import AutoencoderKLQwenImage
    from diffusers.modular_pipelines.anima.before_denoise import get_timesteps
    from diffusers.modular_pipelines.anima.encoders import encode_vae_image
    resolution = int(size)
    embeds = read("goldens-anima-text", "anima.py text")["prompt_embeds"][None]
    picture = common.start_picture(resolution, resolution)
    vae = AutoencoderKLQwenImage.from_pretrained(base() / "vae", torch_dtype=torch.float32).eval()
    z0 = encode_vae_image(picture[None], vae, None, torch.device("cpu"), torch.float32, sample_mode="argmax")
    del vae
    scheduler = schedule()
    timesteps, t_start, sigma_s = common.img2img_truncation(get_timesteps, scheduler, STEPS, common.IMG2IMG_STRENGTH)
    eps = initial_latent(resolution, resolution)
    latent = scheduler.scale_noise(z0, timesteps[:1], eps)
    model = load_transformer()
    padding = torch.zeros(1, 1, resolution, resolution)
    payload = {"image": picture, "z0": z0[0, :, 0].clone(), "eps": eps[0, :, 0].clone(), "x_s": latent[0, :, 0].clone(),
               "sigmas": scheduler.sigmas.float(), "t_start": torch.tensor([float(t_start)]), "sigma_s": torch.tensor([sigma_s])}
    print(f"  strength {common.IMG2IMG_STRENGTH}: t_start {t_start}, sigma_s {sigma_s:.6f}, {len(timesteps)} evaluations")
    for k, t in enumerate(timesteps):
        i = t_start + k
        timestep = (t / scheduler.config.num_train_timesteps).expand(1).float()
        velocity = model(hidden_states=latent, timestep=timestep, encoder_hidden_states=embeds,
                         padding_mask=padding, return_dict=False)[0]
        latent = scheduler.step(velocity, t, latent, return_dict=False)[0]
        payload[f"velocity_{i}"] = velocity[0, :, 0].clone()
        payload[f"latent_{i + 1}"] = latent[0, :, 0].clone()
        print(f"  step {i}: sigma = {float(scheduler.sigmas[i]):.4f}", flush=True)
    payload["latent_final"] = latent[0, :, 0].clone()
    write(f"goldens-anima-img2img-{resolution}", payload, {
        "strength": common.IMG2IMG_STRENGTH, "t_start": t_start, "sigma_s": sigma_s, "evaluations": len(timesteps),
        "picture": f"inputs/{common.START_PICTURE}, covered and center-cropped, LANCZOS",
        "recipe": "modular Anima: encode_vae_image (argmax), get_timesteps, scale_noise"})


def stage_vae(size: str):
    """Decoding the final latent. For a rectangle (`768x512`), the latent is the **centered crop** of the
    final 1024 latent: a real latent, and h != w."""
    from diffusers import AutoencoderKLQwenImage
    height, width, name = common.parse_size(size)
    source = name if height == width else "1024"
    trajectory = read(f"goldens-anima-trajectory-{source}", f"anima.py trajectory --size {source}")
    if height != width:
        trajectory["latent_final"] = common.centered_crop(trajectory["latent_final"], height, width)
    vae = AutoencoderKLQwenImage.from_pretrained(base() / "vae", torch_dtype=torch.float32).eval()
    latent = trajectory["latent_final"][None, :, None]
    mean = torch.tensor(vae.config.latents_mean).view(1, 16, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std).view(1, 16, 1, 1, 1)
    image = vae.decode(latent * std + mean, return_dict=False)[0][:, :, 0]
    write(f"goldens-anima-vae-{name}", {"latent_final": trajectory["latent_final"], "image": image[0]})
    common.save_preview(image[0], OUT / f"reference-anima-{name}.png")


# ------------------------------------------------------------------------- the float64 witnesses

def stage_witness(size: str, step: int):
    """**The float64 reference** of a guided evaluation: same input as the oracle at step `step` (its
    latent, its sigma), computed in double end to end. It arbitrates an engine/oracle deviation: if the
    engine is as close to the witness as the fp32 oracle is, the deviation is fp32 rounding.

    The model would weigh 15.6 GB in double: the blocks are loaded one by one (before/after hooks), like
    the other streamed oracles. And diffusers forces RoPE to float32: both places are replaced by their
    transcription in double."""
    import diffusers.models.embeddings as embeddings
    from diffusers import CosmosTransformer3DModel
    if not size.isdigit():
        raise SystemExit(f"witness --size {size}: squares only, the starting latent is the trajectory's")
    resolution = int(size)

    def rotary64(x, freqs_cis, use_real=True, use_real_unbind_dim=-2, sequence_dim=2):
        cos, sin = freqs_cis
        cos, sin = cos[None, None].to(x.device), sin[None, None].to(x.device)
        real, imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)
        rotated = torch.cat([-imag, real], dim=-1)
        return x * cos + rotated * sin
    embeddings.apply_rotary_emb = rotary64

    trajectory = read(f"goldens-anima-trajectory-{resolution}", f"anima.py trajectory --size {resolution}")
    text = read("goldens-anima-text", "anima.py text")
    config = CosmosTransformer3DModel.load_config(base() / "transformer")
    with torch.device("meta"):
        model = CosmosTransformer3DModel.from_config(config)
    dit, _ = turbo_state_dicts()
    top = {k: v.double() for k, v in dit.items() if not k.startswith("transformer_blocks.")}
    model.load_state_dict(top, strict=False, assign=True)

    def rope64(hidden_states, fps=None):
        r = model.rope
        _, _, frames, height, width = hidden_states.shape
        size = [frames // r.patch_size[0], height // r.patch_size[1], width // r.patch_size[2]]
        f64 = torch.float64

        def freqs(dim, factor):
            theta = 10000.0 * factor
            return 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=f64)[: dim // 2] / dim))
        fh, fw, ft = freqs(r.dim_h, r.h_ntk_factor), freqs(r.dim_w, r.w_ntk_factor), freqs(r.dim_t, r.t_ntk_factor)
        seq = torch.arange(max(r.max_size), dtype=f64)
        eh = torch.outer(seq[: size[1]], fh)[None, :, None, :].repeat(size[0], 1, size[2], 1)
        ew = torch.outer(seq[: size[2]], fw)[None, None, :, :].repeat(size[0], size[1], 1, 1)
        et = torch.outer(seq[: size[0]], ft)[:, None, None, :].repeat(1, size[1], size[2], 1)
        angles = torch.cat([et, eh, ew] * 2, dim=-1).flatten(0, 2)
        return torch.cos(angles), torch.sin(angles)
    model.rope.forward = rope64

    # The same boundaries as `dit`, to localize a deviation and not only measure it.
    taps = {}

    def keep(name):
        def hook(_m, _a, o):        # nothing returned: a hook that returns something replaces the output
            taps[name] = (o[0] if isinstance(o, tuple) else o)[0].clone()
        return hook
    model.patch_embed.register_forward_hook(common.tap(taps, "patch_embed_out", lambda o: o[0].flatten(0, 2)))

    def time_hook(_m, _a, o):
        taps["temb"], taps["embedded_timestep"] = o[0][0].clone(), o[1][0].clone()
    model.time_embed.register_forward_hook(time_hook)
    for i in (0, 1, 13, 27):
        model.transformer_blocks[i].register_forward_hook(keep(f"block{i}_out"))
    block0 = model.transformer_blocks[0]
    block0.attn1.register_forward_hook(keep("block0_attn1_out"))
    block0.attn2.register_forward_hook(keep("block0_attn2_out"))
    block0.ff.register_forward_hook(keep("block0_ff_out"))

    def streamed(block, weights):
        def load(module, _args):
            module.load_state_dict({k: v.double() for k, v in weights.items()}, strict=True, assign=True)

        def drop(module, _args, _out):
            module.to_empty(device="meta")
        block.register_forward_pre_hook(load)
        block.register_forward_hook(drop)
    for i, block in enumerate(model.transformer_blocks):
        prefix = f"transformer_blocks.{i}."
        streamed(block, {k[len(prefix):]: v for k, v in dit.items() if k.startswith(prefix)})
    del dit

    latent = (trajectory["latent_init"] if step == 0 else trajectory[f"latent_{step}"])[None, :, None].double()
    sigma = trajectory["sigmas"][step].double()
    padding = torch.zeros(1, 1, resolution, resolution, dtype=torch.float64)
    velocity = model(hidden_states=latent, timestep=sigma.expand(1),
                     encoder_hidden_states=text["prompt_embeds"][None].double(),
                     padding_mask=padding, return_dict=False)[0]
    v64 = velocity[0, :, 0]
    fp32 = trajectory[f"velocity_{step}"].double()
    print(f"  step {step}: |oracle fp32 - fp64| / |fp64| = {float((fp32 - v64).norm() / v64.norm()):.3e}")
    # Stored in fp32 (the golden format), computed in double.
    taps = {k: v.float().contiguous() for k, v in taps.items()}
    path = OUT / f"witness-anima-f64-{resolution}-step{step}.safetensors"
    save_file({"velocity": v64.float().contiguous(), "model_out": v64.float().contiguous(), **taps}, str(path))
    print(f"-> {path}")


def stage_vae_witness(size: str):
    """**The float64 witness of the Qwen-Image VAE's decoder** on `goldens-anima-vae-<size>`'s latent: the
    same diffusers module, its weights widened to double, through the real `vae.decode` (see
    `common.decoder_in_double` for the four replacements that make it fit in 16 GB without changing the
    arithmetic). Stored in fp32 (about 6e-8 away, far below anything judged)."""
    from diffusers import AutoencoderKLQwenImage
    _, _, name = common.parse_size(size)
    common.decoder_in_double()
    f64 = torch.float64
    vae = AutoencoderKLQwenImage.from_pretrained(base() / "vae", torch_dtype=torch.float32).eval().double()
    mean = torch.tensor(vae.config.latents_mean, dtype=f64).view(1, 16, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, dtype=f64).view(1, 16, 1, 1, 1)
    decoded = read(f"goldens-anima-vae-{name}", f"anima.py vae --size {name}")
    latent = decoded["latent_final"].double()[None, :, None]
    image = vae.decode(latent * std + mean, return_dict=False)[0][0, :, 0]
    rmse = (decoded["image"].double() - image).pow(2).mean().sqrt()
    worst, median, plane = common.per_plane(decoded["image"], image)
    psnr = float(20 * torch.log10(2 / rmse))
    print(f"  decoder {name}: oracle fp32 vs fp64 - worst plane {worst:.3e} (plane {plane}), "
          f"median {median:.3e}, PSNR {psnr:.2f} dB")
    path = OUT / f"witness-anima-vae-f64-{name}.safetensors"
    save_file({"image": image.float().contiguous()}, str(path))
    (OUT / f"witness-anima-vae-f64-{name}.json").write_text(json.dumps(
        {"inputs": [f"goldens-anima-vae-{name}"], "oracle_fp32_vs_witness": {
            "decoder_image": {"worst_plane": worst, "median_plane": median, "plane": plane, "psnr_db": psnr}}},
        indent=1))
    print(f"-> {path}")


def everything(goldens: Path, sizes: list[str]) -> None:
    """Text first, then per size what the check reads: the DiT evaluation, the trajectory and the DiT's
    witness (squares), the decoder and its witness. A rectangle's decoder crops the 1024 trajectory's final
    latent, so that trajectory is written too if `sizes` leaves it out."""
    squares = [s for s in sizes if common.parse_size(s)[0] == common.parse_size(s)[1]]
    needed = squares + (["1024"] if len(squares) < len(sizes) and "1024" not in squares else [])
    stages = [["tokenizer"], ["text"]]
    stages += [["dit", "--size", s] for s in sizes]
    stages += [["trajectory", "--size", s] for s in needed]
    stages += [["witness", "--size", s, "--step", "0"] for s in squares]
    stages += [["vae", "--size", s] for s in sizes]
    stages += [["vae-witness", "--size", s] for s in sizes]
    stages += [["img2img", "--size", "512"]] if "512" in squares else []
    common.run_stages(__file__, goldens, stages)


def main() -> None:
    global OUT
    common.unbuffered()
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("stage", choices=["tokenizer", "text", "dit", "bf16", "trajectory", "vae", "witness", "vae-witness", "img2img", "all"])
    p.add_argument("--size", nargs="+", default=None,
                   help="512, 1024 or WIDTHxHEIGHT (dit, vae: a rectangle too; default 512, all: every size)")
    p.add_argument("--step", type=int, default=0, help="witness only: the step of the trajectory (default 0, the one the check reads)")
    p.add_argument("--long", action="store_true", help="text only: the long prompt, past 512 tokens -> goldens-anima-text-long")
    p.add_argument("--variant", choices=quantized.VARIANTS,
                   help="dit and bf16: the Compact's or the Light's published DiT instead of the Standard's")
    p.add_argument("--goldens", help="output folder (default: $SILICONED_GOLDENS, else ./goldens)")
    args = p.parse_args()
    if args.variant and args.stage not in ("dit", "bf16"):
        p.error("--variant goes with dit or bf16")
    OUT = common.goldens_directory(args.goldens) / "anima"
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "all":
        everything(common.goldens_directory(args.goldens), args.size or ["512", "1024", "768x512"])
    elif args.stage == "tokenizer":
        stage_tokenizer()
    elif args.stage == "text":
        stage_text(common.LONG_PROMPT, common.LONG_SEQUENCE_LENGTH, "goldens-anima-text-long") if args.long else stage_text()
    else:
        for size in args.size or ["512"]:
            if args.variant:
                {"dit": stage_dit_variant, "bf16": stage_bf16}[args.stage](size, args.variant)
            elif args.stage == "witness":
                stage_witness(size, args.step)
            else:
                {"dit": stage_dit, "bf16": stage_bf16, "trajectory": stage_trajectory, "vae": stage_vae,
                 "vae-witness": stage_vae_witness, "img2img": stage_img2img}[args.stage](size)


if __name__ == "__main__":
    main()
