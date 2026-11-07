#!/usr/bin/env python3
"""Qwen-Image-2.1 Turbo's goldens: what `siliconed check qwen-image-2.1 <stage>` judges the engine against.

    qwen-image-2.1.py text [--edit N]       Qwen3-VL-8B reads the prompt (and N pictures) -> qwen21/goldens-qwen21-text[-image-N]
    qwen-image-2.1.py text --long           the same on the long prompt, past 512 tokens  -> qwen21/goldens-qwen21-text-long
    qwen-image-2.1.py vision                the vision tower, picture by picture          -> goldens-qwen21-vision
    qwen-image-2.1.py references [--side S] a picture with transparency, prepared          -> goldens-qwen21-preprocess-alpha-S
    qwen-image-2.1.py dit [--edit N]        one DiT evaluation, three ways                 -> goldens-qwen21-dit[-editN]-512
    qwen-image-2.1.py bf16 [--edit N]       that evaluation in bfloat16                    -> witness-bf16-qwen21-dit[-editN]-512
    qwen-image-2.1.py dit|bf16 --variant compact|light   generation on the Compact's (Comfy-Org int8 convrot) or the
                                            Light's (our SQ file) published DiT -> goldens-qwen21-dit-<variant>-512
    qwen-image-2.1.py w8a8 --variant compact   `bf16 --variant compact` with ComfyUI's default int8 products in the
                                            int8 layers (the W8A8 bar)                    -> witness-w8a8-qwen21-dit-compact-512
    qwen-image-2.1.py witness               that evaluation in float64                     -> witness-qwen21-f64-512
    qwen-image-2.1.py trajectory [--edit N] the 8-step sampler                             -> goldens-qwen21-trajectory[-editN]-512
    qwen-image-2.1.py vae [--edit N]        the decoder on the trajectory's final latent   -> goldens-qwen21-vae[-editN]-512
    qwen-image-2.1.py vae-encode            the encoder on both pictures                   -> goldens-qwen21-vae-encode-512
    qwen-image-2.1.py vae-witness           the VAE in float64                             -> witness-qwen21-vae-f64-512
    qwen-image-2.1.py vae-bf16              the decoder in bfloat16, as the pipeline runs it -> witness-bf16-qwen21-vae-512
    qwen-image-2.1.py all                   every stage above, in an order that satisfies their dependencies

`bf16` is the bar of a half-precision path (the ANE's fp16 products): the `dit` golden's evaluation (with
`--edit N`, the edit's) with the reference in bfloat16, on that golden's inputs. It is not part of `all` (costly, and only
that path reads it).

Goldens go in `<goldens>/qwen21/`. The pictures they read ship in `tools/oracle/inputs/` and are copied to
`<goldens>/inputs/`, where the engine reads the very same files: two made with Siliconed and Z-Image (their
prompt and seed are in the PNG), and `rgba-512x384.png`, a synthetic picture with transparency. Each stage rereads what an
earlier one wrote, so the order above is the order to run them in. `--goldens DIR`, else `SILICONED_GOLDENS`,
else `./goldens`. Nothing fits in 16 GB (the DiT weighs 28 GB in float32, the encoder 33 GB): both are built
on `meta` and their layers streamed, which takes a few minutes per stage.

The reference is diffusers `QwenImage21Pipeline` / `QwenImage21Transformer2DModel` /
`AutoencoderKLQwenImage21` and transformers' `Qwen3VLForConditionalGeneration` / `Qwen3VLProcessor`, in float32
on the CPU. **The reference code does the computation**: this script builds its modules and calls the
pipeline's `__call__` and methods; it transcribes only the loading, the last lines of `__call__` (decoding) and
the float64 witnesses. The pipeline is stopped where a stage ends by raising `Captured` from a wrapper
(`encode_prompt`, the encoder's forward, the transformer's forward), so a DiT stage gets the exact
`hidden_states`, `timestep`, `img_mask` and `img_shapes` the pipeline builds; stages after the encoder replay
its goldens in place of `encode_prompt`.

Prompts: generation `PROMPT`, seed 0; editing with N pictures `EDIT_PROMPTS[N]` on `EDIT_IMAGES[N]`. The output
resolution is 512. Without height/width the output takes the aspect ratio of the LAST picture (the pipeline's
rule). The schedule is the pipeline's own: `sample_sigmas` of the Turbo's `model_index.json`, handed to the
constructor as `from_pretrained` does; `__call__` gets neither `sigmas` nor `num_inference_steps`.

What the reference does that a port by analogy would miss (read in the code, checked by the stages):
  turbo    - the schedule lives in `model_index.json` (`sample_sigmas`), not in `scheduler_config.json`: 8 nodes
             used as they are, `float32(sample_sigmas)` then a 0 appended; the last step jumps from 0.414568 to 0.
             `use_dynamic_shifting: false` and `shift: 1.0` make the scheduler's shift the identity, bit for bit;
             mu is still computed by `__call__` and passed, then ignored: the schedule depends neither on the size
             nor on the pictures.
           - `num_inference_steps` is ignored when `sample_sigmas` exists (`retrieve_timesteps` drops it beside
             `sigmas`); only an explicit `sigmas=` at the call would override the grid.
           - t = sigma * 1000 (fp32), then `timestep / 1000`, then * 1000 inside: for these 8 nodes the round
             trip gives sigma back to the bit.
           - the DiT is a full checkpoint, not a base plus a turbo adapter. `true_cfg_scale` 1, no negative
             prompt. The VAE is published in bf16: the module computes in fp32 on bf16-representable weights.
  encoder  - the template is a raw string (not `apply_chat_template`); `<image1>` is literal text before
             `<|vision_start|>`, the 2nd picture is preceded by a space; `_drop_idx` = 14 (system message); the
             kept tokens still contain `<|im_start|>user\\n` and the trailing `<|im_start|>assistant\\n`.
           - `prompt_embeds` = output of LLM layer 35 BEFORE the final RMSNorm (a hook of the pipeline; rms 19
             against 3.3 normalized).
           - deepstack: ViT blocks 8/16/24 -> added after LLM layers 0, 1, 2 (not 8/16/24), on image tokens only.
           - mRoPE interleaved: frequency i of 64 goes to H if i%3 == 1 and i < 60, W if i%3 == 2 and i < 60,
             else T. Image tokens: T = start, H = start + row, W = start + col; the cursor then advances by
             max(gh, gw)/2, not by the token count.
           - ViT: frame duplicated (temporal patch 2), patches ordered by 2x2 merge block, learned 48x48
             positions resampled bilinearly with align_corners=True, MLP GELU-tanh but merger GELU-erf, merger
             LayerNorm on 1152 then regroup x4; deepstack mergers LayerNorm on the regrouped 4608.
           - the encoder's outputs at image positions never reach the DiT: they are overwritten by VAE latents.
             Pictures reach the DiT text only through the LLM's attention.
  inputs   - pictures: RGB -> RGBA (alpha 255), about R^2 at the ratio, sides rounded to 32 by `round`, PIL
             Lanczos; the vision copy is composited over white, the VAE reads 4 channels. A transparent picture
             keeps its alpha: Pillow premultiplies around the resize (RGBA -> RGBa -> RGBA, truncating division
             back), skips it at the same size (transparent pixels keep their RGB); the VAE reads that RGBA, the
             vision copy is `white.paste(img, mask=A)`. Output size follows the LAST picture.
           - noise drawn as randn(1, 1, 64, h, w) (frame axis before channels), packed raster [h*w, 64].
  DiT      - one sequence: text tokens, each VL picture slot expanded x4 and filled with the latents in RASTER
             order, the target block appended last (`target // 4` slots x4). `img_in`, everything: no bias.
           - temb for TWO timesteps [t, 0]: text and references use the t = 0 row (blocks and `norm_out`). One
             `modulation` shared by all 32 blocks: [scale1, gate1, scale2, gate2], no shift, x * (1 + scale),
             residual * tanh(gate). Sinusoid cos FIRST, then sin (256 dims).
           - `txt_in`: zero-centred RMSNorm (weight + 1, eps 1e-6) -> linear -> GELU-tanh -> linear.
           - q/k RMSNorm per head before RoPE; RoPE on complex PAIRS (2i, 2i+1), axes 8/28/28 complex
             frequencies, theta = 1e4; frame = running text position, frozen over a picture block, then
             += max(h, w) (latent tokens); h/w centred: range(-(h - h//2), h//2); negative positions read a
             table by negative indexing.
           - block-causal attention: text causal, a picture block sees everything before it and itself, the
             target sees all; prefill + K/V cache is bitwise equal to the full step (checked).
  VAE      - 2D in effect (frame squeezed), 4 channels in and out; `latent_dist.mode()`; per-channel
             (z - mean)/std from the config; decode clamps to [-1, 1]; the pipeline returns RGBA (alpha not
             exactly 1) and never converts to RGB.

Setup:  in a fresh virtual environment, pip install -r tools/oracle/requirements.txt
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

sys.dont_write_bytecode = True      # the repository carries no caches
import common  # noqa: E402
from common import Checkpoint, Streamed
import quantized  # noqa: E402

REPOSITORY = "Qwen/Qwen-Image-2.1-Turbo"
PROMPT = "a 30 year old woman posing in a library"
EDIT_PROMPTS = {
    1: "Replace the cloudy sky with a clear blue sky, and adapt the light on the women accordingly.",
    2: "Remove the second woman, on the right in the yellow raincoat, from <image1>, and put the dog from <image2> "
       "in her place, lit like the rest of <image1>.",
}
EDIT_IMAGES = {1: ["surfer-raincoat-cloudy-beach.png"], 2: ["surfer-raincoat-cloudy-beach.png", "border-collie-beach.png"]}
ALPHA_PICTURE = "rgba-512x384.png"
SEED = 0
RESOLUTION = 512
#: The scheduler's sigma for `sample_sigmas`, as float32 bit patterns: `float32(sample_sigmas)` then the
#: appended 0. Every trajectory checks it gets these.
SIGMAS_BITS = [0x3f800000, 0x3f7a7be5, 0x3f744524, 0x3f6d375d, 0x3f6523f6, 0x3f585b9f, 0x3f345c57, 0x3ed44242, 0x00000000]

torch.set_grad_enabled(False)

#: Set by `main`: the model's folder in the goldens, and the folder of the pictures.
OUT: Path
INPUTS: Path
_SNAP: Path | None = None


CONFIG_ONLY = False


def snap() -> Path:
    """Everything comes from the Turbo's publication: DiT, encoder, `processor/`, scheduler, `model_index.json`, VAE."""
    global _SNAP
    if _SNAP is None:
        if CONFIG_ONLY:     # --variant: the weights come from the variant's file, the repository gives the configuration
            _SNAP = common.snapshot(REPOSITORY, [], files=["transformer/config.json"])
        else:
            _SNAP = common.snapshot(REPOSITORY, ["transformer", "text_encoder", "processor", "scheduler", "vae"],
                                    files=["model_index.json"])
    return _SNAP


def sample_sigmas() -> list:
    return json.loads((snap() / "model_index.json").read_text())["sample_sigmas"]


def versions() -> dict:
    import importlib.metadata as metadata
    return {"diffusers": metadata.version("diffusers"), "transformers": metadata.version("transformers")}


class Captured(Exception):
    """Raised by a wrapper to stop `__call__` once the wanted boundary has been recorded."""


# ------------------------------------------------------------------ weights, in pieces

def check_keys(module, checkpoint):
    wanted, published = set(module.state_dict()), set(checkpoint.keys)
    if published != wanted:
        missing, extra = sorted(wanted - published), sorted(published - wanted)
        raise SystemExit(f"keys: {len(missing)} missing {missing[:4]}, {len(extra)} extra {extra[:4]}")


def check_resident(module, streamed: tuple[str, ...]):
    """Every parameter AND buffer outside the streamed modules must have left `meta`: a non-persistent
    buffer (RoPE tables, sinusoid frequencies) is not in the file and silently stays there."""
    for name, t in list(module.named_parameters()) + list(module.named_buffers()):
        if t.is_meta and not name.startswith(streamed):
            raise SystemExit(f"still on meta: {name}")


# ------------------------------------------------------------------ goldens

def write(stem: str, payload: dict, extra: dict | None = None):
    common.write_golden(OUT, stem, payload,
                        {"model": f"{REPOSITORY}@{common.REVISIONS[REPOSITORY]}", "seed": SEED, **versions(),
                         **(extra or {})})


def channel_error(got: torch.Tensor, expected: torch.Tensor):
    """The engine's `channelError`: per channel (last axis), norm(got - expected) / norm(expected), the
    denominator floored at 1 % of the median channel norm; -> (worst, channel, median)."""
    got, expected = got.double().reshape(-1, got.shape[-1]), expected.double().reshape(-1, expected.shape[-1])
    num = (got - expected).pow(2).sum(0).sqrt()
    den = expected.pow(2).sum(0).sqrt()
    live = den > 0
    floor = 0.01 * den[live].median()
    r = num[live] / den[live].clamp(min=float(floor))
    r = torch.where(torch.isnan(r), torch.full_like(r, math.inf), r)
    worst = int(r.argmax())
    return float(r[worst]), int(live.nonzero()[worst]), float(r.median())


def gap(name, got, expected):
    worst, channel, median = channel_error(got, expected)
    absmax = float((got.double() - expected.double()).abs().max())
    print(f"  {name}: worst channel {worst:.3e} (#{channel}), median {median:.3e}, max |d| {absmax:.3e}")
    return {"worst": worst, "channel": channel, "median": median, "max_abs": absmax}


def read(stem: str, producer: str) -> dict:
    return common.read_golden(OUT, stem, producer)


# ------------------------------------------------------------------ the pipeline

def load_processor():
    """The Turbo's `Qwen3VLProcessor` as published. Its image processor is the torchvision one: without
    torchvision, transformers falls back to the Pillow one (`…Pil`), whose rescale and normalization are
    two roundings where the torchvision one fuses them, and every pixel value moves by one ulp. Refused
    rather than written silently."""
    from transformers import Qwen3VLProcessor
    from transformers.image_processing_backends import TorchvisionBackend
    processor = Qwen3VLProcessor.from_pretrained(snap() / "processor")
    if not isinstance(processor.image_processor, TorchvisionBackend):
        sys.exit(f"the image processor is {type(processor.image_processor).__name__}, not the torchvision one: "
                 "install torchvision (tools/oracle/requirements.txt)")
    return processor


def make_pipeline(**modules):
    """The real `QwenImage21Pipeline`, to call ITS `__call__` and methods. `_execution_device` is pinned to
    the CPU: with modules half on `meta`, the pipeline's own guess could land on `meta`. Scheduler, processor
    and `sample_sigmas` are the Turbo's, as `from_pretrained` would hand them."""
    from diffusers import FlowMatchEulerDiscreteScheduler, QwenImage21Pipeline

    class Pipeline(QwenImage21Pipeline):
        @property
        def _execution_device(self):
            return torch.device("cpu")

    processor = load_processor()
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(snap() / "scheduler")
    return Pipeline(scheduler=scheduler, vae=modules.get("vae"), text_encoder=modules.get("text_encoder"),
                    processor=processor, transformer=modules.get("transformer"), sample_sigmas=sample_sigmas())


def images_for(n: int):
    from PIL import Image
    return [Image.open(INPUTS / name) for name in EDIT_IMAGES[n]]


def run_pipeline(pipe, n: int, resolution: int, **extra):
    """`__call__` as the Turbo's card writes it: no `sigmas` and no step count (the pipeline's
    `sample_sigmas`), `true_cfg_scale=1`, seed 0 on a CPU generator; editing through `image=[...]` and
    `output_resolution`; latents out."""
    kwargs = dict(prompt=EDIT_PROMPTS[n] if n else PROMPT, true_cfg_scale=1.0,
                  generator=torch.Generator("cpu").manual_seed(SEED),
                  output_resolution=resolution, output_type="latent")
    if n:
        kwargs["image"] = images_for(n)
    kwargs.update(extra)
    try:
        return pipe(**kwargs)
    except Captured:
        return None


def text_golden(n: int) -> str:
    return "goldens-qwen21-text" if n == 0 else f"goldens-qwen21-text-image-{n}"


def replay_encoder(pipe, n: int):
    """Replace `encode_prompt` by the saved encoder goldens (`text` / `text --edit N`). With a single
    unpadded prompt the pipeline's mask is `None` (`encode_prompt` drops an all-true mask)."""
    producer = f"qwen-image-2.1.py text" + (f" --edit {n}" if n else "")
    golden = read(text_golden(n), producer)
    info = json.loads((OUT / f"{text_golden(n)}.json").read_text())

    def encode_prompt(prompt=None, image=None, device=None, num_images_per_prompt=1, **_):
        sizes = [list(i.size) for i in image] if image is not None else []
        assert sizes == info.get("condition_sizes", []), f"condition sizes {sizes} != {info.get('condition_sizes')}"
        return golden["prompt_embeds"][None], None, golden["image_pad_mask"][None].bool()
    pipe.encode_prompt = encode_prompt


# ------------------------------------------------------------------ the encoder (Qwen3-VL-8B)

def load_text_encoder():
    """`Qwen3VLForConditionalGeneration` on `meta` with SDPA (what `from_pretrained` picks): embedding table,
    final norm and the whole vision tower (0.4 B, 2.3 GB in fp32) resident; the 36 LLM layers and `lm_head`
    streamed. The published names match the module's (`model.language_model.*`, `model.visual.*`,
    `lm_head.weight`)."""
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration
    config = AutoConfig.from_pretrained(snap() / "text_encoder")
    with torch.device("meta"):
        model = Qwen3VLForConditionalGeneration._from_config(config, attn_implementation="sdpa",
                                                              dtype=torch.float32)
    model.eval()
    checkpoint = Checkpoint(snap() / "text_encoder", "model.safetensors")
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
    print(f"  attention: {model.config._attn_implementation} / text {config.text_config._attn_implementation}"
          f" / vision {config.vision_config._attn_implementation}")
    return model


def llm_taps(encoder, inputs_of: tuple, outputs_of: tuple):
    layers = encoder.model.language_model.layers
    taps = {}

    def pre(i):
        def hook(_m, args, kwargs):
            taps[f"layer{i}_in"] = (args[0] if args else kwargs["hidden_states"])[0].clone()
            if i == 0:
                cos, sin = kwargs["position_embeddings"]
                taps["rope_cos"], taps["rope_sin"] = cos[0].clone(), sin[0].clone()
        return hook

    def post(i):
        def hook(_m, _args, _kwargs, output):
            taps[f"layer{i}_out"] = (output[0] if isinstance(output, tuple) else output)[0].clone()
        return hook
    for i in inputs_of:
        layers[i].register_forward_pre_hook(pre(i), with_kwargs=True)
    for i in outputs_of:
        layers[i].register_forward_hook(post(i), with_kwargs=True)
    return taps


def capture_encoder_call(encoder, record: dict, stop: bool):
    """Records the kwargs that `_get_qwen_prompt_embeds` hands the encoder (ids, mask, pixels, grids)."""
    def hook(_m, _args, kwargs):
        record.update({k: v for k, v in kwargs.items() if isinstance(v, torch.Tensor)})
        if stop:
            raise Captured
    return encoder.register_forward_pre_hook(hook, with_kwargs=True)


def stage_text(n: int, long: bool = False):
    """`n = 0`: the generation prompt; `n >= 1`: the edit prompt with its N pictures (`text --edit N`);
    `long`: `common.LONG_PROMPT`, generation only. The pipeline cuts no prompt (no `max_sequence_length`)."""
    encoder = load_text_encoder()
    n_layers = len(encoder.model.language_model.layers)
    if n == 0:
        taps = llm_taps(encoder, (0,), (0, n_layers // 2 - 1, n_layers - 1))
    else:
        taps = llm_taps(encoder, (0, 1, 2, 3), (0, 1, 2, n_layers // 2 - 1, n_layers - 1))
    pipe = make_pipeline(text_encoder=encoder)
    call = {}
    capture_encoder_call(encoder, call, stop=False)
    result = {}
    real = pipe.encode_prompt

    def encode_prompt(*args, **kwargs):
        result["out"] = real(*args, **kwargs)
        result["image"] = kwargs.get("image")
        raise Captured
    pipe.encode_prompt = encode_prompt
    prompt = common.LONG_PROMPT if long else EDIT_PROMPTS[n] if n else PROMPT
    run_pipeline(pipe, n, RESOLUTION, prompt=prompt)

    prompt_embeds, mask, image_pad_mask = result["out"]
    assert mask is None, "single prompt: the pipeline drops an all-true mask"
    drop = pipe._drop_idx
    ids = call["input_ids"][0]
    # The pipeline hooks the final RMSNorm to return its input: `hidden_states[-1]` must be the output of
    # the last decoder layer, NOT normalized. Verified here, bit for bit.
    last = taps[f"layer{n_layers - 1}_out"]
    gap_last = float((last[drop:] - prompt_embeds[0]).abs().max())
    normed = encoder.model.language_model.norm(last)            # what transformers would have returned
    print(f"  {len(ids)} tokens, _drop_idx = {drop}, prompt_embeds {tuple(prompt_embeds.shape[1:])}; "
          f"|prompt_embeds - layer {n_layers - 1} output| = {gap_last:.1e}; "
          f"rms un-normalized {float(last.pow(2).mean().sqrt()):.2f} vs normalized {float(normed.pow(2).mean().sqrt()):.2f}")
    assert gap_last == 0, "prompt_embeds is not the un-normalized last layer"
    system = pipe.processor.apply_chat_template(
        [{"role": "system", "content": [{"type": "text", "text": pipe.sys_prompt}]}], tokenize=True, return_dict=False)[0]
    assert ids[:drop].tolist() == list(system), "the dropped prefix is not the system message"

    rope = encoder.model.language_model.rotary_emb
    payload = {"ids": ids.to(torch.int32), "prompt_embeds": prompt_embeds[0],
               "image_pad_mask": image_pad_mask[0].to(torch.int32),
               "inv_freq": rope.inv_freq.float(), **taps}
    extra = {"prompt": prompt, "tokens": len(ids), "drop_idx": drop,
             "embeds": prompt_embeds.shape[1], "last_layer": n_layers - 1,
             "mrope_section": list(rope.mrope_section), "mrope_interleaved": True,
             "rope_theta": encoder.config.text_config.rope_parameters["rope_theta"],
             "attention": encoder.config._attn_implementation,
             "probes": "layerI_in = input of decoder layer I (for I >= 1: AFTER the deepstack addition of "
                       "layer I-1); layerI_out = output of decoder layer I, BEFORE any deepstack addition"}
    if n:
        payload.update({"pixel_values": call["pixel_values"], "image_grid_thw": call["image_grid_thw"].to(torch.int32),
                        "mm_token_type_ids": call["mm_token_type_ids"][0].to(torch.int32)})
        position_ids, _ = encoder.model.get_rope_index(call["input_ids"], call["mm_token_type_ids"],
                                                       image_grid_thw=call["image_grid_thw"],
                                                       attention_mask=call["attention_mask"])
        payload["position_ids"] = position_ids[:, 0].to(torch.int32)
        extra["condition_sizes"] = [list(i.size) for i in result["image"]]
        extra["images"] = EDIT_IMAGES[n]
        extra["image_slots"] = int(image_pad_mask.sum())
    write(text_golden(n) + ("-long" if long else ""), payload, extra)


def stage_vision():
    """The vision tower alone, on the pictures exactly as the pipeline hands them to the processor (resized
    to about R^2, RGBA composited over white). The LLM is never run: `__call__` stops at the encoder's
    forward. Each picture is run alone (attention is per picture anyway: `cu_seqlens`) and the packed run of
    both is compared to it."""
    encoder = load_text_encoder()
    pipe = make_pipeline(text_encoder=encoder)
    visual = encoder.model.visual
    calls = {}
    for n in (1, 2):
        call = {}
        handle = capture_encoder_call(encoder, call, stop=True)
        run_pipeline(pipe, n, RESOLUTION)
        handle.remove()
        calls[n] = call
    payload, extra = {}, {"images": EDIT_IMAGES[2], "deepstack_visual_indexes": list(visual.deepstack_visual_indexes)}
    call = calls[2]
    grids = call["image_grid_thw"]
    sizes = (grids.prod(-1)).tolist()
    pixels = torch.split(call["pixel_values"], sizes)

    block0 = {}

    def keep_block0(_m, _a, o):
        block0["out"] = o.clone()
    visual.blocks[0].register_forward_hook(keep_block0)
    for i, (pv, grid) in enumerate(zip(pixels, grids)):
        out = visual(pv, grid_thw=grid[None], return_dict=True)
        payload[f"pixel_values_{i}"] = pv
        payload[f"image_grid_thw_{i}"] = grid.to(torch.int32)
        payload[f"vit_block0_out_{i}"] = block0["out"]
        payload[f"vit_last_hidden_{i}"] = out.last_hidden_state
        payload[f"vit_out_{i}"] = out.pooler_output
        for j, feature in enumerate(out.deepstack_features):
            payload[f"deepstack{j}_{i}"] = feature
    packed = visual(call["pixel_values"], grid_thw=grids, return_dict=True)
    alone = torch.cat([payload[f"vit_out_{i}"] for i in range(len(sizes))])
    extra["packed_vs_alone_max_abs"] = float((packed.pooler_output - alone).abs().max())
    print(f"  grids {grids.tolist()}; packed vs alone: {extra['packed_vs_alone_max_abs']:.1e}")

    # Is the processor's own resize an identity on these sizes (multiples of 32)? Compare with
    # `do_resize=False`: the processor's code both times.
    from PIL import Image
    for i, name in enumerate(EDIT_IMAGES[2]):
        h, w = (int(v) * 16 for v in grids[i, 1:])
        plain = pipe.processor.image_processor(images=[_white(pipe, Image.open(INPUTS / name), w, h)],
                                               do_resize=False, return_tensors="pt")["pixel_values"]
        extra[f"processor_resize_identity_{i}"] = float((plain - pixels[i]).abs().max())
        print(f"  picture {i} ({w}x{h}): |pixel_values - no-resize| = {extra[f'processor_resize_identity_{i}']:.1e}")

    for n in (1, 2):
        c = calls[n]
        position_ids, delta = encoder.model.get_rope_index(c["input_ids"], c["mm_token_type_ids"],
                                                           image_grid_thw=c["image_grid_thw"],
                                                           attention_mask=c["attention_mask"])
        payload[f"ids_{n}"] = c["input_ids"][0].to(torch.int32)
        payload[f"position_ids_{n}"] = position_ids[:, 0].to(torch.int32)
        extra[f"rope_delta_{n}"] = int(delta[0, 0])
    write("goldens-qwen21-vision", payload, extra)


def _white(pipe, image, width, height):
    """What `__call__` + `_get_qwen_prompt_embeds` hand the processor, rebuilt only to test the processor's
    resize: RGBA, Lanczos resize, alpha composited over white."""
    from PIL import Image
    image = pipe.image_processor.resize(image.convert("RGBA"), width=width, height=height)
    white = Image.new("RGB", image.size, (255, 255, 255))
    white.paste(image, mask=image.getchannel("A"))
    return white


def stage_references(side: int):
    """**No weights**: a picture WITH transparency (`rgba-512x384.png`: an alpha ramp, a fully transparent
    band whose RGB is not zero, discs at alpha 128 and 7) through `__call__` itself, up to the encoder's
    forward: what `image_processor.preprocess` hands the VAE (RGBA, 4 channels, the real alpha) and the
    `pixel_values` the processor makes of the vision copy (composited over white by
    `white.paste(img, mask=A)`). Side 512 resizes it (576x448: Pillow premultiplies around its Lanczos,
    `RGBA -> RGBa -> RGBA`); side 443 lands on 512x384 itself, where Pillow's resize is a copy and the
    transparent pixels keep their RGB."""
    from PIL import Image

    class Stub(torch.nn.Module):
        dtype = torch.float32

        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.norm = torch.nn.Identity()

        def forward(self, **kwargs):
            record.update(kwargs)
            raise Captured
    pipe = make_pipeline(text_encoder=Stub())
    record, vae_inputs = {}, []
    real = pipe.image_processor.preprocess

    def preprocess(*args, **kwargs):
        out = real(*args, **kwargs)
        vae_inputs.append(out.clone())
        return out
    pipe.image_processor.preprocess = preprocess
    image = Image.open(INPUTS / ALPHA_PICTURE)
    assert image.mode == "RGBA", image.mode
    try:
        pipe(prompt=EDIT_PROMPTS[1], image=[image], true_cfg_scale=1.0, output_resolution=side,
             generator=torch.Generator("cpu").manual_seed(SEED), output_type="latent")
    except Captured:
        pass
    vae = vae_inputs[0][0]
    payload = {"vae_image": vae, "pixel_values": record["pixel_values"],
               "image_grid_thw": record["image_grid_thw"].to(torch.int32)}
    alpha = vae[3]
    print(f"  VAE input {tuple(vae.shape)}, alpha in [{float(alpha.min()):.3f}, {float(alpha.max()):.3f}]; "
          f"pixel_values {tuple(record['pixel_values'].shape)}, grid {record['image_grid_thw'].tolist()}")
    write(f"goldens-qwen21-preprocess-alpha-{side}", payload, {"image": ALPHA_PICTURE, "output_resolution": side})


# ------------------------------------------------------------------ the DiT

def load_transformer(dtype=torch.float32, variant: str | None = None):
    """`QwenImage21Transformer2DModel` on `meta`: embedders, the shared modulation and the output resident,
    the 32 blocks streamed. `pos_embed.freqs` (a plain list) and `time_proj.freqs` (a non-persistent buffer)
    are built at construction: on `meta`, so they are redone on the CPU."""
    import diffusers.models.transformers.transformer_qwenimage21 as te
    config = te.QwenImage21Transformer2DModel.load_config(snap() / "transformer")
    with torch.device("meta"):
        model = te.QwenImage21Transformer2DModel.from_config(config)
    model.eval()
    checkpoint = (quantized.Variant("qwen-image-2.1", variant) if variant
                  else Checkpoint(snap() / "transformer", "diffusion_pytorch_model.safetensors.index.json"))
    model.published = checkpoint
    check_keys(model, checkpoint)
    resident = {k: checkpoint.tensor(k, dtype) for k in checkpoint.keys if not k.startswith("transformer_blocks.")}
    model.load_state_dict(resident, strict=False, assign=True)
    model.pos_embed = te.QwenImage21Rope(theta=10000, axes_dim=list(config["axes_dims_rope"]))
    model.time_text_embed.time_proj = te.QwenImage21TemporalTimesteps(timestep_dim=256)
    for i, block in enumerate(model.transformer_blocks):
        Streamed(block, f"transformer_blocks.{i}.", checkpoint, dtype)
    check_resident(model, ("transformer_blocks.",))
    return model, config


def load_vae():
    from diffusers import AutoencoderKLQwenImage21
    return AutoencoderKLQwenImage21.from_pretrained(snap() / "vae", torch_dtype=torch.float32).eval()


def capture_transformer_inputs(pipe, model, n, resolution):
    """`__call__` up to its first transformer call: the exact kwargs it builds (latents with the references
    prepended, `timestep / 1000`, `img_shapes`, `img_mask` with the target slots, a fresh
    `QwenImage21KVCache` and `kv_cache_mode="extract"`)."""
    record = {}

    def grab(**kwargs):
        record.update(kwargs)
        raise Captured
    model.forward = grab
    try:
        run_pipeline(pipe, n, resolution)
    finally:
        del model.forward
    return record


def dit_taps(model):
    taps = {}

    def joint_in(_m, _a, kwargs):
        taps["joint_in"] = kwargs["hidden_states"][0].clone()
    handles = [
        model.img_in.register_forward_hook(common.tap(taps, "img_in_out")),
        model.txt_in.register_forward_hook(common.tap(taps, "txt_in_out")),
        model.time_text_embed.register_forward_hook(common.tap(taps, "temb", lambda o: o)),
        model.modulation.register_forward_hook(common.tap(taps, "modulation", lambda o: o)),
        model.norm_out.register_forward_hook(common.tap(taps, "norm_out")),
        model.transformer_blocks[0].register_forward_pre_hook(joint_in, with_kwargs=True),
        model.transformer_blocks[0].register_forward_hook(common.tap(taps, "layer0_out")),
        model.transformer_blocks[1].register_forward_hook(common.tap(taps, "layer1_out")),
        model.transformer_blocks[-1].register_forward_hook(common.tap(taps, "layer_last_out")),
    ]

    def rope(_m, args, out):
        img_shapes, image_pad_mask = args[0], args[1]
        taps["rope_cos"], taps["rope_sin"] = out.real.float().clone(), out.imag.float().clone()
        image_ids, target = model.build_token_metadata(image_pad_mask, img_shapes)
        taps["joint_image_mask"] = image_pad_mask.to(torch.int32)
        taps["image_ids"] = image_ids.to(torch.int32)
        taps["target_token_mask"] = target.to(torch.int32)
    handles.append(model.pos_embed.register_forward_hook(rope))
    return taps, handles


def stage_dit(n: int = 0):
    import diffusers.models.transformers.transformer_qwenimage21 as te
    t0 = time.time()
    model, config = load_transformer()
    pipe = make_pipeline(transformer=model, vae=load_vae() if n else None)
    replay_encoder(pipe, n)
    inputs = capture_transformer_inputs(pipe, model, n, RESOLUTION)
    target = math.prod(inputs["img_shapes"][0][-1])
    sigma0 = float(pipe.scheduler.sigmas[0])
    taps, handles = dit_taps(model)

    # 1. Prefill: the whole joint sequence (per-segment SDPA), the K/V of the conditions stored.
    out = model(**inputs)[0][0]
    for h in handles:
        h.remove()
    prefix = int((taps["target_token_mask"] == 0).sum())
    print(f"  sequence {out.shape[0]} = prefix {prefix} + target {target}; sigma0 = {sigma0:.6f}, "
          f"timestep {float(inputs['timestep'][0]):.6f}  ({time.time() - t0:.0f} s)", flush=True)
    payload = {"hidden_states": inputs["hidden_states"][0], "timestep": inputs["timestep"],
               "img_mask": inputs["img_mask"][0].to(torch.int32), "model_out_full": out,
               "model_out": out[-target:].clone(), **taps}
    info = {"prompt": EDIT_PROMPTS[n] if n else PROMPT, "references": n, "output_resolution": RESOLUTION,
            "img_shapes": [list(s) for s in inputs["img_shapes"][0]], "prefix": prefix, "target": target,
            "sigma": sigma0, "segments": te._qwenimage21_prefix_segments(taps["image_ids"].long(), prefix),
            "temb_rows": "row 0 = sampled timestep (target tokens), row 1 = t = 0 (text and references)"}

    # 2. The same step from the cache: only the target's queries, K/V of the prefix reread.
    inputs["kv_cache_mode"] = "cached"
    cached = model(**inputs)[0][0]
    print(f"  cached step ({time.time() - t0:.0f} s)", flush=True)
    payload["model_out_cached"] = cached
    info["cached_vs_prefill"] = gap("cached vs prefill", cached, out[-target:])
    # 3. The full pass under the block-causal mask itself: the flex processor (uncompiled -> dense fp32
    # scores) with `build_qwenimage21_block_causal_mask`, no cache.
    for block in model.transformer_blocks:
        block.attn.set_processor(te.QwenImage21FlexAttnProcessor())
    inputs.update(kv_cache=None, kv_cache_mode=None)
    flex = model(**inputs)[0][0]
    for block in model.transformer_blocks:
        block.attn.set_processor(te.QwenImage21AttnProcessor())
    print(f"  flex pass ({time.time() - t0:.0f} s)", flush=True)
    info["prefill_vs_flex_mask"] = gap("segments vs flex mask (target)", out[-target:], flex[-target:])
    info["prefill_vs_flex_mask_all"] = gap("segments vs flex mask (whole sequence)", out, flex)
    info["cached_vs_flex_mask"] = gap("cached vs flex mask", cached, flex[-target:])
    tag = f"-edit{n}" if n else ""
    write(f"goldens-qwen21-dit{tag}-{RESOLUTION}", payload, info)
    print(f"  {time.time() - t0:.0f} s")


def stage_dit_variant(variant: str):
    """Generation's `dit` on the Compact's or the Light's published weights, from the Standard's golden inputs
    (same latent, same text, same timestep): the prefill of the whole joint sequence, whose probes are what
    the check judges at its three ways (prefill, step from the cache, full pass: the Standard's golden holds
    the same rows). Writes `goldens-qwen21-dit-<variant>-512`."""
    t0 = time.time()
    name = f"goldens-qwen21-dit-{RESOLUTION}"
    golden = read(name, "qwen-image-2.1.py dit")
    text = read(text_golden(0), "qwen-image-2.1.py text")
    info = json.loads((OUT / f"{name}.json").read_text())
    model, _ = load_transformer(torch.float32, variant)
    taps, _ = dit_taps(model)
    out = model(hidden_states=golden["hidden_states"][None], encoder_hidden_states=text["prompt_embeds"][None],
                timestep=golden["timestep"], img_shapes=[[tuple(s) for s in info["img_shapes"]]],
                img_mask=golden["img_mask"][None].bool(), return_dict=False)[0][0]
    target = info["target"]
    payload = {"hidden_states": golden["hidden_states"], "timestep": golden["timestep"], "img_mask": golden["img_mask"],
               "model_out_full": out, "model_out": out[-target:].clone(), **taps}
    info = {k: info[k] for k in ("prompt", "references", "output_resolution", "img_shapes", "prefix", "target",
                                 "sigma", "segments", "temb_rows")}
    write(quantized.stem(name, variant), payload,
          {**info, "inputs": name, "weights": model.published.describe(),
           "note": "prefill only: the cached step and the full pass are judged against these rows"})
    print(f"  {time.time() - t0:.0f} s")


def stage_bf16(n: int = 0, variant: str | None = None):
    """`dit [--edit N]` (prefill only) with diffusers in bfloat16, as `from_pretrained(torch_dtype=torch.bfloat16)`
    and `__call__` run it: weights as published, `hidden_states` and `prompt_embeds` in bf16, `timestep` in the
    latents' dtype; the module keeps its own float32 islands (sinusoid `freqs`, RoPE, the RMSNorm variance,
    SDPA's accumulation). The inputs are the dit golden's own; the boundaries the check judges are widened to
    float32 (exact) under the golden's keys. Without a cache the prefill still runs per segment, as the golden's,
    and with references the encoder's mask is `None` there too (`replay_encoder`)."""
    t0 = time.time()
    half = torch.bfloat16
    tag = f"-edit{n}" if n else ""
    name = f"goldens-qwen21-dit{tag}-{RESOLUTION}"
    golden = read(name, "qwen-image-2.1.py dit" + (f" --edit {n}" if n else ""))
    text = read(text_golden(n), "qwen-image-2.1.py text" + (f" --edit {n}" if n else ""))
    info = json.loads((OUT / f"{name}.json").read_text())
    model, _ = load_transformer(half, variant)
    taps, _ = dit_taps(model)
    out = model(hidden_states=golden["hidden_states"][None].to(half),
                encoder_hidden_states=text["prompt_embeds"][None].to(half),
                timestep=golden["timestep"].to(half), img_shapes=[[tuple(s) for s in info["img_shapes"]]],
                img_mask=golden["img_mask"][None].bool(), return_dict=False)[0][0]
    kept = ("img_in_out", "txt_in_out", "temb", "modulation", "norm_out", "joint_in",
            "layer0_out", "layer1_out", "layer_last_out")
    payload = {"model_out": out[-info["target"]:].float(), **{k: taps[k].float() for k in kept}}
    report = {"bf16_vs_fp32_model_out": gap("bf16 vs fp32, model_out", payload["model_out"], golden["model_out"])}
    write(quantized.stem(f"witness-bf16-qwen21-dit{tag}-{RESOLUTION}", variant), payload,
          {"dtype": "bfloat16", "inputs": name, **report, **({"weights": model.published.describe()} if variant else {})})
    print(f"  {time.time() - t0:.0f} s")


def stage_w8a8(variant: str):
    """`bf16 --variant compact` with every linear the Compact publishes in int8 convrot computed by comfy_kitchen's own
    `int8_linear` on the published integers (`quantized.Int8Products`, the recipe cited there): the bar of the engine's
    W8A8 route. ComfyUI's QwenImage21 (comfy/ldm/qwen_image21/model.py at the frozen commit) calls its linears plainly,
    except the MLP's down projection, `linear_input_act(self.out, self.gate_up(x), "swiglu")`: on the eager backend
    the swiglu runs in bf16 before the quantization (`_apply_input_act`), the arithmetic diffusers does in bf16 here, then
    the same `int8_linear`. Writes `witness-w8a8-qwen21-dit-compact-512`, the bf16 bar's keys."""
    if variant != "compact":
        raise SystemExit("w8a8: the Compact only (--variant compact)")
    t0 = time.time()
    half = torch.bfloat16
    name = f"goldens-qwen21-dit-{RESOLUTION}"
    golden = read(name, "qwen-image-2.1.py dit")
    text = read(text_golden(0), "qwen-image-2.1.py text")
    info = json.loads((OUT / f"{name}.json").read_text())
    model, _ = load_transformer(half, variant)
    products = quantized.Int8Products(model, model.published)
    taps, _ = dit_taps(model)
    out = model(hidden_states=golden["hidden_states"][None].to(half),
                encoder_hidden_states=text["prompt_embeds"][None].to(half),
                timestep=golden["timestep"].to(half), img_shapes=[[tuple(s) for s in info["img_shapes"]]],
                img_mask=golden["img_mask"][None].bool(), return_dict=False)[0][0]
    assert products.calls >= len(products.layers), (products.calls, len(products.layers))
    kept = ("img_in_out", "txt_in_out", "temb", "modulation", "norm_out", "joint_in",
            "layer0_out", "layer1_out", "layer_last_out")
    payload = {"model_out": out[-info["target"]:].float(), **{k: taps[k].float() for k in kept}}
    report = {"w8a8_vs_fp32_model_out": gap("w8a8 vs fp32, model_out", payload["model_out"], golden["model_out"])}
    write(quantized.stem(f"witness-w8a8-qwen21-dit-{RESOLUTION}", variant), payload,
          {"dtype": "bfloat16", "inputs": name, **report, "int8": products.describe(), "weights": model.published.describe()})
    print(f"  {time.time() - t0:.0f} s")


# ------------------------------------------------------------------ the trajectory

def stage_trajectory(n: int = 0):
    """The Turbo's loop as `__call__` runs it: 8 steps on `sample_sigmas`, no CFG; sigma checked to the bit
    against `SIGMAS_BITS`."""
    import numpy as np
    t0 = time.time()
    model, _ = load_transformer()
    pipe = make_pipeline(transformer=model, vae=load_vae() if n else None)
    replay_encoder(pipe, n)
    record, velocities, latents = {}, [], []
    real_prepare = pipe.prepare_latents

    def prepare_latents(*args, **kwargs):
        out = real_prepare(*args, **kwargs)
        record["latent_init"] = out[0][0].clone()
        record["references"] = None if out[1] is None else out[1][0].clone()
        return out
    pipe.prepare_latents = prepare_latents

    def keep_velocity(_m, _a, o):
        velocities.append(o[0][0, -record["latent_init"].shape[0]:].clone())
    model.register_forward_hook(keep_velocity)
    modes = []
    forward = model.forward

    def transformer_forward(*args, kv_cache_mode=None, **kwargs):
        modes.append(kv_cache_mode)
        return forward(*args, kv_cache_mode=kv_cache_mode, **kwargs)
    model.forward = transformer_forward

    def on_step(_pipe, i, t, kwargs):
        latents.append(kwargs["latents"][0].clone())
        print(f"  step {i}: sigma = {float(_pipe.scheduler.sigmas[i]):.6f}, t = {float(t):.4f}  "
              f"({time.time() - t0:.0f} s)", flush=True)
        return {}
    final = run_pipeline(pipe, n, RESOLUTION, callback_on_step_end=on_step).images
    scheduler = pipe.scheduler
    bits = scheduler.sigmas.float().numpy().view(np.uint32).tolist()
    print(f"  sigma bits {[hex(b) for b in bits]}; kv_cache_mode per evaluation {modes}")
    assert scheduler.sigmas.dtype == torch.float32 and bits == SIGMAS_BITS, f"sigma moved: {[hex(b) for b in bits]}"
    assert modes == ["extract"] + ["cached"] * (len(SIGMAS_BITS) - 2), modes
    payload = {"latent_init": record["latent_init"], "sigmas": scheduler.sigmas.float(),
               "timesteps": scheduler.timesteps.float(), "latent_final": final[0]}
    if record["references"] is not None:
        payload["reference_latents"] = record["references"]
    for i, (v, x) in enumerate(zip(velocities, latents)):
        payload[f"velocity_{i}"] = v
        payload[f"latent_{i + 1}"] = x
    seq = record["latent_init"].shape[0]
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift
    c = scheduler.config
    mu = calculate_shift(seq, c.get("base_image_seq_len", 256), c.get("max_image_seq_len", 4096),
                         c.get("base_shift", 0.5), c.get("max_shift", 1.15))
    images = images_for(n) if n else []
    width, height = _output_size(images, RESOLUTION)
    tag = f"-edit{n}" if n else ""
    write(f"goldens-qwen21-trajectory{tag}-{RESOLUTION}", payload,
          {"prompt": EDIT_PROMPTS[n] if n else PROMPT, "references": n, "output_resolution": RESOLUTION,
           "height": height, "width": width, "steps": len(velocities), "sample_sigmas": sample_sigmas(),
           "sigma_bits": [f"0x{b:08x}" for b in bits], "kv_cache_modes": modes,
           "mu": mu, "mu_note": "computed and passed by __call__, ignored (use_dynamic_shifting: false)",
           "scheduler": dict(scheduler.config)})
    print(f"  {time.time() - t0:.0f} s")


def _output_size(images, resolution):
    """`__call__`'s rule: the LAST picture's aspect ratio at about R^2, else R x R; multiples of 32."""
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions
    if images:
        width, height, _ = calculate_dimensions(resolution * resolution, images[-1].size[0] / images[-1].size[1])
    else:
        width = height = resolution
    return width // 32 * 32, height // 32 * 32


# ------------------------------------------------------------------ the VAE

def stage_vae(n: int = 0):
    """Decoding the final latent through the last lines of `__call__` (transcribed: they are not a method):
    `_unpack_latents`, the per-channel de-normalization `z * std + mean`, `vae.decode` (which clamps to
    [-1, 1]), frame 0, `postprocess` -> **an RGBA PIL image** (4 channels). The pipeline does no RGBA -> RGB
    conversion: the preview PNG drops the alpha with PIL's `convert("RGB")` (no compositing), the alpha
    statistics are recorded."""
    tag = f"-edit{n}" if n else ""
    producer = "qwen-image-2.1.py trajectory" + (f" --edit {n}" if n else "")
    trajectory = read(f"goldens-qwen21-trajectory{tag}-{RESOLUTION}", producer)
    info = json.loads((OUT / f"goldens-qwen21-trajectory{tag}-{RESOLUTION}.json").read_text())
    height, width = info["height"], info["width"]
    vae = load_vae()
    pipe = make_pipeline(vae=vae)
    latents = pipe._unpack_latents(trajectory["latent_final"][None], height, width, pipe.vae_scale_factor)
    mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1)
    vae_in = latents * std + mean
    image = vae.decode(vae_in, return_dict=False)[0][:, :, 0]
    pixels = pipe.image_processor.postprocess(image, output_type="pt")[0]
    pil = pipe.image_processor.postprocess(image, output_type="pil")[0]
    alpha = pixels[3]
    write(f"goldens-qwen21-vae{tag}-{RESOLUTION}",
          {"latent_final": trajectory["latent_final"], "vae_in": vae_in[0, :, 0], "image": image[0],
           "pixels": pixels},
          {"height": height, "width": width, "pil_mode": pil.mode,
           "alpha_min": float(alpha.min()), "alpha_max": float(alpha.max()), "alpha_mean": float(alpha.mean())})
    target = OUT / f"reference-qwen21{tag}-{RESOLUTION}.png"
    pil.convert("RGB").save(target)
    print(f"  {pil.mode} {pil.size}, alpha in [{float(alpha.min()):.4f}, {float(alpha.max()):.4f}] -> {target}")


def stage_vae_encode():
    """Both pictures through `__call__` -> `prepare_latents` -> `_encode_vae_image` (recorded), for real:
    RGB -> RGBA (alpha 255), Lanczos resize to about R^2, `preprocess` to [-1, 1] (4 channels, alpha
    included), `latent_dist.mode()` (no sampling), then `(z - mean) / std`. The text encoder is not needed:
    `encode_prompt` returns a placeholder (it only feeds the dtype before the transformer call, where
    `__call__` is stopped)."""
    vae = load_vae()
    model, _ = load_transformer()          # only its configuration and the capture are used: no layer runs
    pipe = make_pipeline(vae=vae, transformer=model)

    def placeholder(*_args, **_kwargs):
        return torch.zeros(1, 1, 4096), None, torch.zeros(1, 1, dtype=torch.bool)
    pipe.encode_prompt = placeholder
    records = []
    real = pipe._encode_vae_image

    def encode(image, generator):
        out = real(image, generator)
        records.append((image.clone(), out.clone()))
        return out
    pipe._encode_vae_image = encode
    raw = []

    def keep_moments(_m, _a, o):
        raw.append(o.clone())
    vae.quant_conv.register_forward_hook(keep_moments)
    capture_transformer_inputs(pipe, model, 2, RESOLUTION)
    payload, extra = {}, {"images": EDIT_IMAGES[2], "output_resolution": RESOLUTION}
    for i, ((image, latent), moments) in enumerate(zip(records, raw)):
        payload[f"image_{i}"] = image[0, :, 0]
        payload[f"latent_mean_raw_{i}"] = moments[0, :vae.config.z_dim, 0]
        payload[f"latent_{i}"] = latent[0, :, 0]
        extra[f"size_{i}"] = list(image.shape[-2:])
        print(f"  picture {i}: {tuple(image.shape)} -> {tuple(latent.shape)}, "
              f"alpha in [{float(image[0, 3].min()):.3f}, {float(image[0, 3].max()):.3f}]")
    write(f"goldens-qwen21-vae-encode-{RESOLUTION}", payload, extra)


# ------------------------------------------------------------------ the fp64 witnesses

def stage_witness():
    """**The float64 witness of a WHOLE evaluation** (32 blocks in double, streamed) on the inputs of `dit`.
    The reference forces fp32 in five places, replaced here by the same formulas in double: the sinusoid
    (`QwenImage21TemporalTimesteps`: fp32 `freqs` buffer and `timestep.float()`), the RoPE table
    (`QwenImage21Rope.rope_params`: fp32 angles, `torch.polar` -> complex64), its application
    (`apply_rotary_emb_qwen`: `x.float()` -> complex64), the zero-centred RMSNorm of `txt_in` (`.float()`),
    and the q/k `RMSNorm` (variance `.to(float32)`)."""
    import diffusers.models.normalization as normalization
    import diffusers.models.transformers.transformer_qwenimage21 as te
    t0 = time.time()
    f64 = torch.float64
    dit = read(f"goldens-qwen21-dit-{RESOLUTION}", "qwen-image-2.1.py dit")
    info = json.loads((OUT / f"goldens-qwen21-dit-{RESOLUTION}.json").read_text())
    text = read("goldens-qwen21-text", "qwen-image-2.1.py text")

    def sinusoid(self, timestep):
        half = self.timestep_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=f64) / half)
        args = (self.time_factor * timestep.double())[:, None] * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    te.QwenImage21TemporalTimesteps.forward = sinusoid

    def rope_params(self, index, dim, theta=10000):
        freqs = torch.outer(index.double(), 1.0 / torch.pow(float(theta), torch.arange(0, dim, 2, dtype=f64) / dim))
        return torch.polar(torch.ones_like(freqs), freqs)
    te.QwenImage21Rope.rope_params = rope_params

    def rotary(x, freqs_cis, use_real=False, use_real_unbind_dim=-1):
        x_complex = torch.view_as_complex(x.double().reshape(*x.shape[:-1], -1, 2))
        return torch.view_as_real(x_complex * freqs_cis.unsqueeze(1)).flatten(3).type_as(x)
    te.apply_rotary_emb_qwen = rotary

    def zero_centred(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * (self.weight + 1)
    te.QwenImage21ZeroCenterRMSNorm.forward = zero_centred

    def rms(self, x):
        y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return y * self.weight if self.weight is not None else y
    normalization.RMSNorm.forward = rms

    model, _ = load_transformer(dtype=f64)
    taps = {}

    def keep(name):
        def hook(_m, _a, o):
            taps[name] = o[0].float()
        return hook
    model.transformer_blocks[0].register_forward_hook(keep("layer0_out"))
    model.transformer_blocks[-1].register_forward_hook(keep("layer_last_out"))

    def keep_whole(name):
        def hook(_m, _a, o):
            taps[name] = o.float()
        return hook
    model.time_text_embed.register_forward_hook(keep_whole("temb"))
    model.modulation.register_forward_hook(keep_whole("modulation"))
    img_shapes = [[tuple(s) for s in info["img_shapes"]]]
    out = model(hidden_states=dit["hidden_states"][None].double(), encoder_hidden_states=text["prompt_embeds"][None].double(),
                timestep=dit["timestep"].double(), img_shapes=img_shapes, img_mask=dit["img_mask"][None].bool(),
                return_dict=False)[0][0]
    target = info["target"]
    exact = out[-target:]
    payload = {"model_out": exact.float(), **taps}
    report = {"oracle_vs_witness_model_out": gap("oracle fp32 vs fp64 witness, model_out", dit["model_out"], exact),
              "oracle_vs_witness_temb": gap("temb", dit["temb"], taps["temb"]),
              "oracle_vs_witness_modulation": gap("modulation", dit["modulation"], taps["modulation"]),
              "oracle_vs_witness_layer0": gap("layer 0", dit["layer0_out"], taps["layer0_out"]),
              "oracle_vs_witness_layer_last": gap("last layer", dit["layer_last_out"], taps["layer_last_out"])}
    report["cached_vs_witness_model_out"] = gap("cached fp32 vs fp64 witness", dit["model_out_cached"], exact)
    write(f"witness-qwen21-f64-{RESOLUTION}", payload, {"inputs": f"goldens-qwen21-dit-{RESOLUTION}", **report})
    print(f"  {time.time() - t0:.0f} s")


def stage_vae_witness():
    """**The float64 witness of the VAE** on the inputs of `vae-encode` (both pictures, encoded to `mode()`,
    raw and normalized) and `vae` (`vae_in` decoded), the Turbo's VAE widened to double. About 15 GB
    at 512: the heaviest stage of this script."""
    from diffusers import AutoencoderKLQwenImage21
    vae = AutoencoderKLQwenImage21.from_pretrained(snap() / "vae", torch_dtype=torch.float32).eval().double()
    mean = torch.tensor(vae.config.latents_mean, dtype=torch.float64).view(1, 64, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, dtype=torch.float64).view(1, 64, 1, 1, 1)
    encoded = read(f"goldens-qwen21-vae-encode-{RESOLUTION}", "qwen-image-2.1.py vae-encode")
    payload = {}
    for i in range(2):
        z = vae.encode(encoded[f"image_{i}"].double()[None, :, None]).latent_dist.mode()
        payload[f"latent_mean_raw_{i}"] = z[0, :, 0].clone()
        payload[f"latent_{i}"] = ((z - mean) / std)[0, :, 0].clone()
        print(f"  oracle fp32 latent_{i} vs fp64:", channel_error(encoded[f"latent_{i}"].reshape(64, -1).T,
                                                               payload[f"latent_{i}"].reshape(64, -1).T))
    decoded = read(f"goldens-qwen21-vae-{RESOLUTION}", "qwen-image-2.1.py vae")
    image = vae.decode(decoded["vae_in"].double()[None, :, None], return_dict=False)[0][:, :, 0]
    payload["image"] = image[0].clone()
    rmse = (decoded["image"].double() - payload["image"]).pow(2).mean().sqrt()
    print(f"  oracle fp32 image vs fp64: {float(20 * torch.log10(2 / rmse)):.2f} dB")
    save_file({k: v.float().contiguous() for k, v in payload.items()},
              str(OUT / f"witness-qwen21-vae-f64-{RESOLUTION}.safetensors"))
    print(f"-> {OUT / f'witness-qwen21-vae-f64-{RESOLUTION}.safetensors'}")


def stage_vae_bf16():
    """**The decoder in bfloat16, as the pipeline runs it with `torch_dtype=torch.bfloat16`** on `goldens-qwen21-vae-512`'s final
    latent: `_unpack_latents`, `latents.to(vae.dtype)`, `* std + mean` in bf16, `vae.decode` (`pipeline_qwenimage21.py`). The bar of
    the engine's bf16 decoder (invariant 3): no further from the fp64 witness than this, + 25 %. Its convolutions on bf16 operands
    with fp32 sums (`common.bf16_convolutions_in_fp32`): call it in a process that does nothing else."""
    from diffusers import AutoencoderKLQwenImage21, QwenImage21Pipeline
    common.bf16_convolutions_in_fp32()
    bf16 = torch.bfloat16
    decoded = read(f"goldens-qwen21-vae-{RESOLUTION}", "qwen-image-2.1.py vae")
    exact = read(f"witness-qwen21-vae-f64-{RESOLUTION}", "qwen-image-2.1.py vae-witness")["image"]
    info = json.loads((OUT / f"goldens-qwen21-vae-{RESOLUTION}.json").read_text())
    vae = AutoencoderKLQwenImage21.from_pretrained(snap() / "vae", torch_dtype=bf16).eval()
    # `_unpack_latents` is a static method: the pipeline itself (its processor, its scheduler) is not needed here.
    latents = QwenImage21Pipeline._unpack_latents(decoded["latent_final"][None], info["height"], info["width"], 16).to(bf16)
    mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(bf16)
    std = torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(bf16)
    with torch.no_grad():
        image = vae.decode(latents * std + mean, return_dict=False)[0][0, :, 0].float()
    worst, median, plane = common.per_plane(image, exact)
    print(f"  decoder: bf16 vs fp64 - worst plane {worst:.3e} (plane {plane}), median {median:.3e}")
    write(f"witness-bf16-qwen21-vae-{RESOLUTION}", {"image": image})


#: `all`: the order satisfies the dependencies (a trajectory is decoded by `vae`, both witnesses reread goldens),
#: and the two float64 stages come last. Roughly half an hour on an M1 Pro, the editing trajectories 5 min each.
ALL = [["text"], ["text", "--edit", "1"], ["text", "--edit", "2"], ["vision"],
       ["references"], ["references", "--side", "443"],
       ["dit"], ["dit", "--edit", "1"], ["dit", "--edit", "2"], ["witness"],
       ["vae-encode"],
       ["trajectory"], ["trajectory", "--edit", "1"], ["trajectory", "--edit", "2"],
       ["vae"], ["vae", "--edit", "1"], ["vae", "--edit", "2"], ["vae-witness"], ["vae-bf16"]]


def main() -> None:
    global OUT, INPUTS
    common.unbuffered()
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("stage", choices=["text", "vision", "references", "dit", "bf16", "w8a8", "witness", "trajectory", "vae",
                                     "vae-encode", "vae-witness", "vae-bf16", "all"])
    p.add_argument("--edit", type=int, choices=[0, 1, 2], default=0,
                   help="editing with N pictures (text, dit, bf16, trajectory, vae); 0 is generation")
    p.add_argument("--long", action="store_true", help="text only, generation: the long prompt, past 512 tokens")
    p.add_argument("--side", type=int, default=512, help="references only: the prepared side (512, 443)")
    p.add_argument("--variant", choices=quantized.VARIANTS,
                   help="dit and bf16 (generation): the Compact's or the Light's published DiT instead of the Standard's;"
                        " w8a8: compact, required")
    p.add_argument("--goldens", help="output folder (default: $SILICONED_GOLDENS, else ./goldens)")
    args = p.parse_args()
    if args.variant and (args.stage not in ("dit", "bf16", "w8a8") or args.edit):
        p.error("--variant goes with dit, bf16 or w8a8, generation (no --edit)")
    if args.stage == "w8a8" and args.variant != "compact":
        p.error("w8a8 takes --variant compact: the Compact's int8 convrot is what ComfyUI computes in W8A8")
    global CONFIG_ONLY
    CONFIG_ONLY = bool(args.variant)
    goldens = common.goldens_directory(args.goldens)
    OUT, INPUTS = goldens / "qwen21", common.provide_inputs(goldens)
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    n = args.edit
    if args.stage == "all":
        common.run_stages(__file__, goldens, ALL)
    elif args.stage == "text":
        if args.long and n:
            p.error("--long is generation's: no --edit")
        stage_text(n, args.long)
    elif args.stage == "vision":
        stage_vision()
    elif args.stage == "references":
        stage_references(args.side)
    elif args.stage == "dit":
        stage_dit_variant(args.variant) if args.variant else stage_dit(n)
    elif args.stage == "bf16":
        stage_bf16(n, args.variant)
    elif args.stage == "w8a8":
        stage_w8a8(args.variant)
    elif args.stage == "witness":
        stage_witness()
    elif args.stage == "trajectory":
        stage_trajectory(n)
    elif args.stage == "vae":
        stage_vae(n)
    elif args.stage == "vae-encode":
        stage_vae_encode()
    elif args.stage == "vae-bf16":
        stage_vae_bf16()
    else:
        stage_vae_witness()
    print(f"  total {time.time() - started:.0f} s")


if __name__ == "__main__":
    main()
