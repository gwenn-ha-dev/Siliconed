#!/usr/bin/env python3
"""Krea 2 Turbo's goldens: what `siliconed check krea2 <stage>` judges the engine against.

    krea2.py text [--long]                     template, tokenizer, Qwen3-VL; the 12 taps    -> goldens-krea2-text[-long]
    krea2.py dit --size 512 [1024 768x512]     one DiT evaluation at the first sigma         -> goldens-krea2-dit-<size>
    krea2.py bf16 --size 512 [1024 768x512]    the same evaluation in bfloat16               -> witness-bf16-krea2-dit-<size>
    krea2.py dit|bf16|witness|witness-step --size 512 --variant compact|light   the same, for the Compact's (Comfy-Org int8 convrot) or the
                                               Light's (our SQ file) published DiT -> goldens-krea2-dit-<variant>-<size>
    krea2.py w8a8 --size 512 --variant compact   `bf16 --variant compact` with ComfyUI's default int8 products in the
                                               int8 layers (the W8A8 bar)    -> witness-w8a8-krea2-dit-compact-<size>
    krea2.py trajectory --size 512 [1024]      the 8-step Euler loop                         -> goldens-krea2-trajectory-<size>
    krea2.py trajectory --size 512 --permute 0 N [N...] [--permute-kind group|full]   the free 8-step loop under each
                                               drawn summation order (0: the identity, same calls) -> draw-krea2-<size>-trajectory-perm<kind>N
    krea2.py vae --size 512 [1024 768x512]     the decoder on the final latent               -> goldens-krea2-vae-<size>
    krea2.py witness --size 512                time, img_in, blocks 0-1 in float64           -> witness-krea2-f64-<size>
    krea2.py witness-step --size 512 --step 1  a WHOLE evaluation in float64, at step i      -> witness-krea2-f64-<size>-step<i>
    krea2.py draw --size 512 --step 1 --jitter N   another fp32 draw of step i               -> draw-krea2-<size>-step<i>-jitterN
                                               (--variant compact|light: step 0 only        -> draw-krea2-<variant>-<size>-step0-jitterN)
    krea2.py draw --size 512 --step 0 --permute N [N...] [--permute-kind group|full]   fp32 draws whose summation
                                               order is drawn (the residual stream permuted) -> draw-krea2-<size>-step<i>-perm<kind>N
    krea2.py vae-witness --size 512            the decoder in float64                        -> witness-krea2-vae-f64-<size>
    krea2.py vae-bf16 --size 512               the decoder in bfloat16, as the pipeline runs it -> witness-bf16-krea2-vae-<size>
    krea2.py img2img --size 512 [--variant compact]   img2img at strength 0.6: the picture encoded, the start, the
                                               truncated loop (steps 3-7)                    -> goldens-krea2-img2img[-<variant>]-<size>
    krea2.py img2img-witness --size 512 --step 3 4 5 6 7 [--variant compact]   an img2img step in float64
                                               -> witness-krea2-f64-img2img[-<variant>]-<size>-step<i>
    krea2.py img2img-draw --size 512 --step 3 --permute N [N...] [--variant compact]   the same step's summation
                                               orders drawn, like draw --permute -> draw-krea2-img2img[-<variant>]-<size>-step<i>-permgroupN
    krea2.py all [--size 512 1024 768x512] [--draws N]   all of the above (see below), dependencies in order

Goldens go in `<goldens>/krea2/`; `--goldens DIR`, else `SILICONED_GOLDENS`, else `./goldens`. Each stage
rereads what an earlier one wrote, so run them in the order above. The DiT weighs 48.6 GB in float32 and the
encoder 17.6 GB: both are built on `meta`, their resident parts loaded, and each block is loaded (WIDENED to
fp32) just before its forward pass and dropped just after. Expect minutes per evaluation, most of an hour for a
1024 trajectory, and a machine that swaps; the float64 evaluations (`witness-step`) stream 3.5 GB blocks.

`bf16` is the bar of a half-precision path (the ANE's fp16 products): the `dit` evaluation with the
reference in bfloat16, on the `dit` golden's inputs. It is not part of `all` (costly, and only that path reads it).
`w8a8` is the bar of the engine's W8A8 route (`check krea2 dit --variant compact --int8-activations`): the same bf16
evaluation with every int8 layer computed as ComfyUI does by default (`quantized.Int8Products`); not part of `all` either.

`all` writes what `siliconed check krea2` reads at the sizes asked (default 512): the text, the DiT
evaluation and its head witness, the trajectory and the float64 witness of every one of its eight steps
(`--witness-steps` restricts them), the decoder and its witness. About 25 minutes at 512 on an M1 Pro. The
draws are read only when a verdict fails at the witness, and cost a minute each (eight steps times
`--draws N` of them): they are opt-in, `--draws 8` being what the check looks for.

The reference is diffusers' `Krea2Pipeline`, `Krea2Transformer2DModel` and `AutoencoderKLQwenImage`, with
transformers' `Qwen3VLModel`, in float32 on the CPU. **The reference code does the computation**: this script
builds its modules and calls their `forward`; it transcribes nothing except the loading.

What the reference does that a port by analogy would miss:
  - the text is not one encoder's output but TWELVE taps of Qwen3-VL (`text_encoder_select_layers`, the input
    of layers 2, 5, ... 35), fused by refiners and `txt_in`. The reference pads the prompt in the middle and
    masks; the engine removes the padding, so every `*_valid` tensor here keeps only the valid rows.
  - the VAE is Qwen-Image's, and the denormalisation copied from `Krea2Pipeline.__call__` is to the letter:
    its `latents_std` is the INVERSE of the standard deviation, and the latent is divided by it.
  - the schedule is `linspace(1, 1/N, N)` through an exponential shift at `mu = 1.15` (the distilled model).
  - some boundaries are ill-conditioned in fp32: block 0's text rows amplify a rounding about a thousandfold,
    so one fp32 computation against one says which summation order the engine took, not how close it is.
    `draw` writes more fp32 computations of the same step, each as honest as the oracle: the latent moved by one
    ulp, up or down by a hash of (seed, index), is a fresh realisation of the rounding of the whole forward.
    The float64 witness of the unmoved inputs still judges them (the exact value moves by about 1e-7).

Setup:  in a fresh virtual environment, pip install -r tools/oracle/requirements.txt
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

sys.dont_write_bytecode = True      # the repository carries no caches
import common  # noqa: E402
import quantized  # noqa: E402
from common import Checkpoint, Streamed

REPOSITORY = "krea/Krea-2-Turbo"
PROMPT = "a 30 year old woman posing in a library"      # the reference prompt
SEED = 42
STEPS = 8            # the card: `--steps 8 --cfg 0.0 --mu 1.15`
MAX_SEQUENCE_LENGTH = 512
DIT_INDEX = "diffusion_pytorch_model.safetensors.index.json"

torch.set_grad_enabled(False)

OUT: Path
_SNAP: Path | None = None


VARIANT: str | None = None      # --variant: the Compact's or the Light's published DiT, for the stages that take it


def dit_checkpoint():
    """The DiT's weights: the Standard's shards, or the variant's published file dequantized (`quantized.py`)."""
    return quantized.Variant("krea2", VARIANT) if VARIANT else Checkpoint(snap() / "transformer", DIT_INDEX)


CONFIG_ONLY = False     # --variant: the weights come from the variant's file, the repository gives configurations


def snap() -> Path:
    global _SNAP
    if _SNAP is None:
        if CONFIG_ONLY:
            _SNAP = common.snapshot(REPOSITORY, ["tokenizer", "scheduler"],
                                    files=["model_index.json", "transformer/config.json"])
        else:
            _SNAP = common.snapshot(REPOSITORY, ["text_encoder", "tokenizer", "transformer", "scheduler", "vae"],
                                    files=["model_index.json"])
    return _SNAP


def write(stem: str, payload: dict, extra: dict | None = None):
    common.write_golden(OUT, stem, payload,
                        {"prompt": PROMPT, "seed": SEED, "steps": STEPS,
                         "snapshot": common.REVISIONS[REPOSITORY], **(extra or {})})


def read(stem: str, producer: str) -> dict:
    return common.read_golden(OUT, stem, producer)


# ------------------------------------------------------------------ the text encoder

def load_text_encoder():
    """`Qwen3VLModel` on `meta`, the embedding table and the norm resident, the layers streamed. The vision
    part stays on `meta`: a prompt alone never enters it (and if it were called, the computation would raise
    instead of returning a wrong number)."""
    from transformers import AutoConfig, Qwen3VLModel
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding
    config = AutoConfig.from_pretrained(snap() / "text_encoder")
    with torch.device("meta"):
        model = Qwen3VLModel(config)
    model.eval()
    lm = model.language_model
    checkpoint = Checkpoint(snap() / "text_encoder", "model.safetensors")
    module_keys, published = set(model.state_dict()), set(checkpoint.keys)
    if published != module_keys:
        missing, extra = sorted(module_keys - published), sorted(published - module_keys)
        raise SystemExit(f"keys: {len(missing)} missing {missing[:4]}, {len(extra)} extra {extra[:4]}")
    lm.embed_tokens.load_state_dict({"weight": checkpoint.tensor("language_model.embed_tokens.weight", torch.float32)},
                                    assign=True)
    lm.norm.load_state_dict({"weight": checkpoint.tensor("language_model.norm.weight", torch.float32)}, assign=True)
    # The `inv_freq` buffer is not in the file: built on `meta`, it has to be redone on the CPU.
    lm.rotary_emb = Qwen3VLTextRotaryEmbedding(config=config.text_config)
    for i, layer in enumerate(lm.layers):
        Streamed(layer, f"language_model.layers.{i}.", checkpoint, torch.float32)
    return model


def make_pipeline(**modules):
    """The real `Krea2Pipeline`, to call ITS methods (template, padding, packing, positions, schedule)
    rather than copy them."""
    from diffusers import Krea2Pipeline
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(snap() / "tokenizer")
    index = json.loads((snap() / "model_index.json").read_text())
    pipe = Krea2Pipeline.__new__(Krea2Pipeline)
    Krea2Pipeline.__init__(
        pipe, scheduler=modules.get("scheduler"), vae=modules.get("vae"),
        text_encoder=modules.get("text_encoder"), tokenizer=tokenizer,
        transformer=modules.get("transformer"),
        text_encoder_select_layers=index["text_encoder_select_layers"],
        is_distilled=index["is_distilled"], patch_size=index["patch_size"])
    return pipe


def stage_text(prompt: str = PROMPT, length: int = MAX_SEQUENCE_LENGTH, stem: str = "goldens-krea2-text"):
    """`--long`: `common.LONG_PROMPT`, the pipeline's `max_sequence_length` raised to keep all of it."""
    encoder = load_text_encoder()
    layers = encoder.language_model.layers
    inputs_of = {}

    def capture(i):
        def hook(_m, args, kwargs):
            inputs_of[i] = (args[0] if args else kwargs["hidden_states"])[0].clone()
        return hook
    for i in range(len(layers)):
        layers[i].register_forward_pre_hook(capture(i), with_kwargs=True)

    pipe = make_pipeline(text_encoder=encoder)
    # The identifiers, recomputed here with the same call as the reference, to record them.
    prefix_idx = pipe.prompt_template_encode_start_idx
    text_tokens = pipe.tokenizer([pipe.prompt_template_encode_prefix + prompt], truncation=True,
                                 padding="max_length",
                                 max_length=length + prefix_idx - pipe.prompt_template_encode_num_suffix_tokens,
                                 return_tensors="pt")
    suffix = pipe.tokenizer([pipe.prompt_template_encode_suffix], return_tensors="pt")
    ids = torch.cat([text_tokens.input_ids, suffix.input_ids], dim=1)[0]
    mask = torch.cat([text_tokens.attention_mask, suffix.attention_mask], dim=1)[0].bool()
    prefix_alone = pipe.tokenizer([pipe.prompt_template_encode_prefix], return_tensors="pt").input_ids[0]
    assert prefix_alone.shape[0] == prefix_idx, f"template: {prefix_alone.shape[0]} tokens, {prefix_idx} expected"
    assert torch.equal(ids[:prefix_idx], prefix_alone), "the template does not tokenize the same alone and followed by the prompt"

    hidden, hidden_mask = pipe.get_text_hidden_states(prompt, length, device=torch.device("cpu"))
    hidden, hidden_mask = hidden[0], hidden_mask[0]
    assert torch.equal(hidden_mask, mask[prefix_idx:])

    # Is `hidden_states[i]` the INPUT of layer i? Verified here, not deduced from the code.
    select = pipe.text_encoder_select_layers
    for slot, i in enumerate(select):
        full = inputs_of[i][prefix_idx:]
        gap = float((full - hidden[:, slot]).abs().max())
        assert gap == 0, f"hidden_states[{i}] is not the input of layer {i} (gap {gap})"
    print(f"  hidden_states[i] = input of layer i, for i in {list(select)}: verified")

    valid = hidden_mask.nonzero().flatten()
    real = int(mask.sum())
    print(f"  {real} real tokens out of {ids.shape[0]}; {valid.numel()} after the template")
    payload = {"ids": ids.to(torch.int32), "mask": mask.to(torch.int32),
               "hidden_padded": hidden, "hidden_mask": hidden_mask.to(torch.int32),
               "hidden_valid": hidden[valid],
               # two whole layers, to locate a deviation: the first tapped and the last
               "layer2_in_valid": inputs_of[2][mask][prefix_idx:],
               "layer35_in_valid": inputs_of[35][mask][prefix_idx:]}
    assert real < ids.shape[0], f"{real} tokens: the prompt was cut at {length}"
    write(stem, payload, {
        "prompt": prompt, "max_sequence_length": length, "prefix_tokens": prefix_idx, "real_tokens": real, "valid_after_prefix": int(valid.numel()),
        "select_layers": list(select)})


# ------------------------------------------------------------------ the DiT

def load_transformer(dtype: torch.dtype = torch.float32, variant: str | None = None, streamed: bool = True):
    """The DiT, its blocks streamed. With a `variant`, from the Compact's or the Light's published file,
    as `quantized.Variant` reads it (the Light's our SQ file, decoded by `quantized.sq_decode`; the two tensors Krea hid in its checkpoint left out).
    `streamed=False`: the blocks stay on `meta`, for a caller that loads them itself (`model.published`)."""
    from diffusers import Krea2Transformer2DModel
    config = Krea2Transformer2DModel.load_config(snap() / "transformer")
    with torch.device("meta"):
        model = Krea2Transformer2DModel.from_config(config)
    model.eval()
    checkpoint = quantized.Variant("krea2", variant) if variant else Checkpoint(snap() / "transformer", DIT_INDEX)
    model.published = checkpoint
    module_keys = set(model.state_dict())
    if set(checkpoint.keys) != module_keys:
        missing, extra = sorted(module_keys - set(checkpoint.keys)), sorted(set(checkpoint.keys) - module_keys)
        raise SystemExit(f"keys: {len(missing)} missing {missing[:4]}, {len(extra)} extra {extra[:4]}")
    resident = {k: checkpoint.tensor(k, dtype) for k in checkpoint.keys if not k.startswith("transformer_blocks.")}
    model.load_state_dict(resident, strict=False, assign=True)
    for i, block in enumerate(model.transformer_blocks):
        if streamed:
            Streamed(block, f"transformer_blocks.{i}.", checkpoint, dtype)
    return model


def scheduler_for(pipe, steps=STEPS):
    """The reference schedule: `sigmas = linspace(1, 1/N, N)`, exponential shift at `mu = 1.15`
    (distilled), through the real scheduler and `retrieve_timesteps`."""
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.krea2.pipeline_krea2 import retrieve_timesteps
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(snap() / "scheduler")
    assert pipe.config.is_distilled
    retrieve_timesteps(scheduler, steps, "cpu", sigmas=np.linspace(1.0, 1 / steps, steps), mu=1.15)
    return scheduler


def initial(pipe, height, width):
    """The reference noise (`SEED`), packed by `_pack_latents`, and its [16, h, w] shape."""
    generator = torch.Generator("cpu").manual_seed(SEED)
    packed = pipe.prepare_latents(1, 16, height, width, torch.float32, torch.device("cpu"), generator)
    return packed, unpack(pipe, packed, height, width)


def unpack(pipe, packed, height, width=None):
    return pipe._unpack_latents(packed, height, width or height)[0, :, 0]


def text_inputs():
    text = read("goldens-krea2-text", "krea2.py text")
    return text["hidden_padded"][None], text["hidden_mask"][None].bool()


def valid_rows(t, mask):
    """A `[padded text, image]` sequence -> `[valid text, image]`: what the engine computes."""
    text_len = mask.shape[-1]
    return torch.cat([t[:text_len][mask[0]], t[text_len:]], dim=0)


def probe_dit(model, mask) -> dict:
    """The boundaries the engine is judged at: the text fusion, block 0 inside out, blocks 1, 13, 27.
    Returns the dict the hooks fill during the forward pass."""
    taps = {}
    model.time_embed.register_forward_hook(common.tap(taps, "temb", lambda o: o[0, 0]))
    model.time_mod_proj.register_forward_hook(common.tap(taps, "temb_mod", lambda o: o[0, 0]))
    model.text_fusion.register_forward_hook(common.tap(taps, "text_fused", lambda o: o[0][mask[0]]))
    model.txt_in.register_forward_hook(common.tap(taps, "txt_in_out", lambda o: o[0][mask[0]]))
    model.img_in.register_forward_hook(common.tap(taps, "img_in_out"))
    b0 = model.transformer_blocks[0]
    b0.attn.register_forward_hook(common.tap(taps, "block0_attn_out", lambda o: valid_rows(o[0], mask)))
    b0.ff.register_forward_hook(common.tap(taps, "block0_ff_out", lambda o: valid_rows(o[0], mask)))
    for i in (0, 1, 13, 27):
        model.transformer_blocks[i].register_forward_hook(common.tap(taps, f"block{i}_out", lambda o: valid_rows(o[0], mask)))
    return taps


def stage_dit(size: str):
    """One evaluation at the first sigma, with probes: the text fusion, block 0 inside out, blocks 1, 13, 27."""
    height, width, name = common.parse_size(size)
    embeds, mask = text_inputs()
    model = load_transformer()
    pipe = make_pipeline(transformer=model)
    scheduler = scheduler_for(pipe)
    packed, latent = initial(pipe, height, width)
    positions = pipe.prepare_position_ids(embeds.shape[1], height // 16, width // 16, torch.device("cpu"))
    t = scheduler.timesteps[0]
    timestep = (t / scheduler.config.num_train_timesteps).expand(1).float()

    taps = probe_dit(model, mask)

    out = model(hidden_states=packed, encoder_hidden_states=embeds, timestep=timestep,
                position_ids=positions, encoder_attention_mask=mask, return_dict=False)[0]
    payload = {"latent_in": latent, "timestep": timestep, "model_out": unpack(pipe, out, height, width),
               "model_out_packed": out[0], **taps}
    write(f"goldens-krea2-dit-{name}", payload, {"sigma": float(scheduler.sigmas[0])})


def stage_dit_variant(size: str, variant: str):
    """`dit` on the Compact's or the Light's published weights, from the Standard's golden inputs (same latent,
    same text, same timestep): writes `goldens-krea2-dit-<variant>-<size>`."""
    height, width, name = common.parse_size(size)
    standard = read(f"goldens-krea2-dit-{name}", f"krea2.py dit --size {name}")
    embeds, mask = text_inputs()
    model = load_transformer(variant=variant)
    pipe = make_pipeline(transformer=model)
    packed = pipe._pack_latents(standard["latent_in"][None], 1, 16, height // 8, width // 8)
    positions = pipe.prepare_position_ids(embeds.shape[1], height // 16, width // 16, torch.device("cpu"))
    taps = probe_dit(model, mask)
    out = model(hidden_states=packed, encoder_hidden_states=embeds, timestep=standard["timestep"],
                position_ids=positions, encoder_attention_mask=mask, return_dict=False)[0]
    payload = {"latent_in": standard["latent_in"], "timestep": standard["timestep"],
               "model_out": unpack(pipe, out, height, width), "model_out_packed": out[0], **taps}
    write(quantized.stem(f"goldens-krea2-dit-{name}", variant), payload,
          {"inputs": f"goldens-krea2-dit-{name}", "weights": model.published.describe()})


def stage_bf16(size: str, variant: str | None = None):
    """`dit` with diffusers in bfloat16, as `from_pretrained(torch_dtype=torch.bfloat16)` runs it: weights,
    latent, text and timestep in bf16. The inputs are the dit golden's own, so only the arithmetic differs;
    the boundaries are widened to float32 (exact) under the golden's keys, minus its inputs."""
    height, width, name = common.parse_size(size)
    golden = read(f"goldens-krea2-dit-{name}", f"krea2.py dit --size {name}")
    embeds, mask = text_inputs()
    model = load_transformer(torch.bfloat16, variant)
    pipe = make_pipeline(transformer=model)
    positions = pipe.prepare_position_ids(embeds.shape[1], height // 16, width // 16, torch.device("cpu"))
    packed = pipe._pack_latents(golden["latent_in"][None], 1, 16, height // 8, width // 8)
    taps = probe_dit(model, mask)
    half = torch.bfloat16
    out = model(hidden_states=packed.to(half), encoder_hidden_states=embeds.to(half),
                timestep=golden["timestep"].to(half), position_ids=positions, encoder_attention_mask=mask,
                return_dict=False)[0].float()
    payload = {"model_out": unpack(pipe, out, height, width), "model_out_packed": out[0],
               **{k: v.float() for k, v in taps.items()}}
    write(quantized.stem(f"witness-bf16-krea2-dit-{name}", variant), payload,
          {"dtype": "bfloat16", "inputs": f"goldens-krea2-dit-{name}",
           **({"weights": model.published.describe()} if variant else {})})


def stage_w8a8(size: str, variant: str):
    """`bf16 --variant compact` with every linear the Compact publishes in int8 convrot computed by comfy_kitchen's own
    `int8_linear` on the published integers (`quantized.Int8Products`, the recipe cited there): the bar of the engine's
    W8A8 route. ComfyUI's Krea 2 (comfy/ldm/krea2/model.py at the frozen commit of `quant_fixtures.COMFY`) calls its
    linears plainly: the attention's `wq`, `wk`, `wv` and `gate` one `Linear` each on the same x (the same integers four
    times), `wo(out * sigmoid(gate))`, the MLP `down(silu(gate(x)) * up(x))`, no input activation or residual folded into a
    product. Inputs: the Standard's dit golden, as `bf16` reads them. Writes `witness-w8a8-krea2-dit-compact-<size>`, the
    bf16 bar's keys."""
    if variant != "compact":
        raise SystemExit("w8a8: the Compact only (--variant compact): its int8 convrot is what ComfyUI computes in W8A8")
    height, width, name = common.parse_size(size)
    golden = read(f"goldens-krea2-dit-{name}", f"krea2.py dit --size {name}")
    embeds, mask = text_inputs()
    model = load_transformer(torch.bfloat16, variant)
    products = quantized.Int8Products(model, model.published)
    pipe = make_pipeline(transformer=model)
    positions = pipe.prepare_position_ids(embeds.shape[1], height // 16, width // 16, torch.device("cpu"))
    packed = pipe._pack_latents(golden["latent_in"][None], 1, 16, height // 8, width // 8)
    taps = probe_dit(model, mask)
    half = torch.bfloat16
    out = model(hidden_states=packed.to(half), encoder_hidden_states=embeds.to(half),
                timestep=golden["timestep"].to(half), position_ids=positions, encoder_attention_mask=mask,
                return_dict=False)[0].float()
    assert products.calls == len(products.layers), (products.calls, len(products.layers))
    payload = {"model_out": unpack(pipe, out, height, width), "model_out_packed": out[0],
               **{k: v.float() for k, v in taps.items()}}
    write(quantized.stem(f"witness-w8a8-krea2-dit-{name}", variant), payload,
          {"dtype": "bfloat16", "inputs": f"goldens-krea2-dit-{name}", "int8": products.describe(),
           "weights": model.published.describe()})


def stage_trajectory(size: str):
    resolution = int(size)
    embeds, mask = text_inputs()
    model = load_transformer()
    pipe = make_pipeline(transformer=model)
    scheduler = scheduler_for(pipe)
    scheduler.set_begin_index(0)
    packed, latent = initial(pipe, resolution, resolution)
    grid = resolution // 16
    positions = pipe.prepare_position_ids(embeds.shape[1], grid, grid, torch.device("cpu"))
    payload = {"latent_init": latent.clone(), "sigmas": scheduler.sigmas.float()}
    for i, t in enumerate(scheduler.timesteps):
        timestep = (t / scheduler.config.num_train_timesteps).expand(1).float()
        velocity = model(hidden_states=packed, encoder_hidden_states=embeds, timestep=timestep,
                         position_ids=positions, encoder_attention_mask=mask, return_dict=False)[0]
        packed = scheduler.step(velocity, t, packed, return_dict=False)[0]
        payload[f"velocity_{i}"] = unpack(pipe, velocity, resolution).clone()
        payload[f"latent_{i + 1}"] = unpack(pipe, packed, resolution).clone()
        print(f"  step {i}: sigma = {float(scheduler.sigmas[i]):.4f}", flush=True)
    payload["latent_final"] = unpack(pipe, packed, resolution).clone()
    write(f"goldens-krea2-trajectory-{resolution}", payload)


def stage_vae(size: str):
    """Decoding the final latent. For a rectangle (`768x512`), the latent is the **centered crop** of the
    final 1024 latent: a real latent, and h != w."""
    from diffusers import AutoencoderKLQwenImage
    height, width, name = common.parse_size(size)
    source = name if height == width else "1024"
    trajectory = read(f"goldens-krea2-trajectory-{source}", f"krea2.py trajectory --size {source}")
    if height != width:
        trajectory["latent_final"] = common.centered_crop(trajectory["latent_final"], height, width)
    vae = AutoencoderKLQwenImage.from_pretrained(snap() / "vae", torch_dtype=torch.float32).eval()
    latent = trajectory["latent_final"][None, :, None]
    # The denormalisation of `Krea2Pipeline.__call__`, to the letter: `latents_std` there is the INVERSE
    # of the standard deviation, and we divide by it.
    mean = torch.tensor(vae.config.latents_mean).view(1, 16, 1, 1, 1)
    inv_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, 16, 1, 1, 1)
    image = vae.decode(latent / inv_std + mean, return_dict=False)[0][:, :, 0]
    write(f"goldens-krea2-vae-{name}", {"latent_final": trajectory["latent_final"], "image": image[0]})
    common.save_preview(image[0], OUT / f"reference-krea2-{name}.png")


# ------------------------------------------------------------------ img2img

def stage_img2img(size: str):
    """**img2img.** diffusers 0.41 has no img2img pipeline for Krea 2: the recipe is diffusers' own img2img
    construction (`QwenImageImg2ImgPipeline`: the same VAE, the same scheduler class) on `Krea2Pipeline`'s noise,
    schedule and loop, each piece the reference's code:

        z0        = encode_vae_image(picture, sample_mode="argmax")   the posterior mean, (z - mean) / std
                    (Anima's modular block: Krea's VAE is Qwen-Image's, the same weights; the engine takes the mean too)
        timesteps = get_timesteps(N, strength)                           the full schedule, then truncated
        x_s       = scheduler.scale_noise(z0, timesteps[:1], eps)        eps: Krea2Pipeline's own noise for the seed
        then the trajectory's loop from x_s, steps t_start ... N - 1

    The noise is the txt2img one (`prepare_latents`), so that strength 1 is txt2img bit for bit, as in the engine.
    With `--variant`, the Standard's start (the same picture, z0, noise and x_s) through the variant's DiT."""
    import types
    from diffusers.modular_pipelines.anima.encoders import encode_vae_image
    from diffusers.pipelines.qwenimage.pipeline_qwenimage_img2img import QwenImageImg2ImgPipeline
    resolution = int(size)
    grid = resolution // 16
    embeds, mask = text_inputs()
    standard = f"goldens-krea2-img2img-{resolution}"
    if VARIANT:
        start = read(standard, f"krea2.py img2img --size {resolution}")
        picture, z0, eps = start["image"], start["z0"], start["eps"]
    else:
        from diffusers import AutoencoderKLQwenImage
        picture = common.start_picture(resolution, resolution)
        vae = AutoencoderKLQwenImage.from_pretrained(snap() / "vae", torch_dtype=torch.float32).eval()
        z0 = encode_vae_image(picture[None], vae, None, torch.device("cpu"), torch.float32, sample_mode="argmax")[0, :, 0]
        del vae
    model = load_transformer(variant=VARIANT)
    pipe = make_pipeline(transformer=model)
    scheduler = scheduler_for(pipe)
    timesteps, t_start, sigma_s = common.img2img_truncation(
        lambda s, n, strength: QwenImageImg2ImgPipeline.get_timesteps(types.SimpleNamespace(scheduler=s), n, strength, "cpu"),
        scheduler, STEPS, common.IMG2IMG_STRENGTH)
    if not VARIANT:
        _, eps = initial(pipe, resolution, resolution)
    packed_noise = pipe._pack_latents(eps[None], 1, 16, resolution // 8, resolution // 8)
    packed = scheduler.scale_noise(pipe._pack_latents(z0[None], 1, 16, resolution // 8, resolution // 8), timesteps[:1], packed_noise)
    if VARIANT:
        assert torch.equal(unpack(pipe, packed, resolution), start["x_s"]), "the Standard's x_s is not recomputed bit for bit"
    positions = pipe.prepare_position_ids(embeds.shape[1], grid, grid, torch.device("cpu"))
    payload = {"image": picture, "z0": z0.clone(), "eps": eps.clone(), "x_s": unpack(pipe, packed, resolution).clone(),
               "sigmas": scheduler.sigmas.float(), "t_start": torch.tensor([float(t_start)]), "sigma_s": torch.tensor([sigma_s])}
    print(f"  strength {common.IMG2IMG_STRENGTH}: t_start {t_start}, sigma_s {sigma_s:.6f}, {len(timesteps)} evaluations")
    for k, t in enumerate(timesteps):
        i = t_start + k
        timestep = (t / scheduler.config.num_train_timesteps).expand(1).float()
        velocity = model(hidden_states=packed, encoder_hidden_states=embeds, timestep=timestep,
                         position_ids=positions, encoder_attention_mask=mask, return_dict=False)[0]
        packed = scheduler.step(velocity, t, packed, return_dict=False)[0]
        payload[f"velocity_{i}"] = unpack(pipe, velocity, resolution).clone()
        payload[f"latent_{i + 1}"] = unpack(pipe, packed, resolution).clone()
        print(f"  step {i}: sigma = {float(scheduler.sigmas[i]):.4f}", flush=True)
    payload["latent_final"] = unpack(pipe, packed, resolution).clone()
    write(quantized.stem(standard, VARIANT), payload, {
        "strength": common.IMG2IMG_STRENGTH, "t_start": t_start, "sigma_s": sigma_s, "evaluations": len(timesteps),
        "picture": f"inputs/{common.START_PICTURE}, covered and center-cropped, LANCZOS",
        "recipe": "QwenImageImg2ImgPipeline.get_timesteps, scale_noise; Anima's encode_vae_image (argmax)",
        **({"inputs": standard, "weights": model.published.describe()} if VARIANT else {})})


def img2img_step_inputs(resolution: int, step: int):
    """The inputs of step `step` of the img2img (`step_inputs`' for the img2img golden): its latent at that step (`x_s`
    at `t_start`), the timestep as the reference receives it, the version's fused text — `txt_in` is a DiT layer, the
    variant's own."""
    golden = read(quantized.stem(f"goldens-krea2-img2img-{resolution}", VARIANT),
                  f"krea2.py img2img --size {resolution}{' --variant ' + VARIANT if VARIANT else ''}")
    t_start = int(golden["t_start"][0])
    if not t_start <= step < STEPS:
        raise SystemExit(f"img2img --step {step}: the img2img runs steps {t_start} to {STEPS - 1}")
    latent = golden["x_s" if step == t_start else f"latent_{step}"]
    timestep = ((golden["sigmas"][step] * 1000).float() / 1000).reshape(1)
    text = read(quantized.stem(f"goldens-krea2-dit-{resolution}", VARIANT), f"krea2.py dit --size {resolution}")["txt_in_out"]
    return latent, timestep, text


def stage_img2img_witness(size: str, step: int):
    """**The float64 reference of an img2img step**: the whole evaluation in double (`evaluate64`) on the img2img
    golden's latent at step `step` (its `x_s` at `t_start`), the timestep as the reference receives it, the fp32
    `txt_in` output. With `--variant`, the variant's weights on the variant's golden latents."""
    resolution = int(size)
    latent, timestep, text = img2img_step_inputs(resolution, step)
    velocity, _ = evaluate64(latent, timestep, text, resolution, resolution)
    path = OUT / f"{quantized.stem(f'witness-krea2-f64-img2img-{resolution}', VARIANT)}-step{step}.safetensors"
    save_file({"velocity": velocity}, str(path))
    print(f"-> {path}")


# ------------------------------------------------------------------ the float64 witnesses

def patch64():
    """The three places where diffusers forces fp32, replaced by their transcription in double:
    `Krea2RMSNorm` (`x.float()`), `Krea2TimestepEmbedding` (`timestep.float()`) and RoPE
    (`apply_rotary_emb` in `x.float()`). Otherwise the reference would be no more precise than what it
    judges. Returns the module and the dtype."""
    import math
    import diffusers.models.transformers.transformer_krea2 as k2
    f64 = torch.float64

    def rms64(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * (self.weight + 1.0)
    k2.Krea2RMSNorm.forward = rms64

    def time64(self, timestep, dtype):
        half = self.embed_dim // 2
        # The TRUE sinusoid, in double: the reference computes it in fp32, where the angle rounded near
        # 1000 rad and an fp32 `exp` at one ulp skew it by 2e-5. The timestep, for its part, is the one
        # the reference receives (fp32).
        freqs = torch.exp(-math.log(1e4) * torch.arange(half, dtype=f64) / half)
        args = (timestep.float().double() * 1e3)[:, None, None] * freqs
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.linear_2(torch.nn.functional.gelu(self.linear_1(emb), approximate="tanh"))
    k2.Krea2TimestepEmbedding.forward = time64

    def rotary64(x, freqs_cis, use_real=True, use_real_unbind_dim=-1, sequence_dim=2):
        cos, sin = freqs_cis                       # [S, D], adjacent pairs
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        real, imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
        rotated = torch.stack([-imag, real], dim=-1).flatten(3)
        return x * cos + rotated * sin
    k2.apply_rotary_emb = rotary64

    return k2, f64


def rope64(ids, config):
    f64 = torch.float64
    cos, sin = [], []
    for i, dim in enumerate(config["axes_dims_rope"]):
        freqs = 1.0 / (config["rope_theta"] ** (torch.arange(0, dim, 2, dtype=f64) / dim))
        angles = torch.outer(ids[:, i].double(), freqs)
        cos.append(angles.cos().repeat_interleave(2, dim=-1))
        sin.append(angles.sin().repeat_interleave(2, dim=-1))
    return torch.cat(cos, dim=-1), torch.cat(sin, dim=-1)


def evaluate64(latent, timestep, text, height: int, width: int):
    """**One whole evaluation in float64** on the engine's inputs: `latent [16, h, w]`, `timestep` as the
    reference receives it, `text` the fp32 output of `txt_in` (valid rows). The 28 blocks in double, loaded
    one by one (3.5 GB each). Returns the velocity `[16, h, w]` and the probes of blocks 0, 1, 13 and 27."""
    k2, f64 = patch64()
    config = k2.Krea2Transformer2DModel.load_config(snap() / "transformer")
    with torch.device("meta"):
        model = k2.Krea2Transformer2DModel.from_config(config)
    checkpoint = dit_checkpoint()
    top = [k for k in checkpoint.keys if k.startswith(("time_embed.", "time_mod_proj.", "img_in.", "final_layer."))]
    model.load_state_dict({k: checkpoint.tensor(k, f64) for k in top}, strict=False, assign=True)
    model.eval()
    pipe = make_pipeline(transformer=model)
    packed = pipe._pack_latents(latent[None].double(), 1, 16, height // 8, width // 8)
    text = text.double()[None]
    t = text.shape[1]
    temb = model.time_embed(timestep, dtype=f64)
    temb_mod = model.time_mod_proj(torch.nn.functional.gelu(temb, approximate="tanh"))
    x = torch.cat([text, model.img_in(packed)], dim=1)
    rope = rope64(pipe.prepare_position_ids(t, height // 16, width // 16, torch.device("cpu")), config)
    probes = {}
    for i, block in enumerate(model.transformer_blocks):
        block.load_state_dict(checkpoint.under(f"transformer_blocks.{i}.", f64), strict=True, assign=True)
        x = block(x, temb_mod, rope, None)
        block.to_empty(device="meta")
        if i in (0, 1, 13, 27):     # the boundaries the dit stage judges
            probes[f"block{i}_out"] = x[0].float().contiguous()
        print(f"  block {i}", end="\r", flush=True)
    out = model.final_layer(x[:, t:], temb)
    velocity = pipe._unpack_latents(out, height, width)[0, :, 0]
    return velocity.float().contiguous(), probes


def step_inputs(resolution: int, step: int):
    """The inputs of step `step` of the trajectory: the oracle's latent at that step, its timestep as the
    reference receives it (`t = sigma * 1000` in fp32, then `t / 1000` in fp32), the fp32 output of `txt_in`
    (valid rows): what the engine's pass receives."""
    trajectory = read(f"goldens-krea2-trajectory-{resolution}", f"krea2.py trajectory --size {resolution}")
    dit = read(quantized.stem(f"goldens-krea2-dit-{resolution}", VARIANT), f"krea2.py dit --size {resolution}")
    latent = trajectory["latent_init" if step == 0 else f"latent_{step}"]
    timestep = ((trajectory["sigmas"][step] * 1000).float() / 1000).reshape(1)
    return latent, timestep, dit["txt_in_out"]


def stage_witness_step(size: str, step: int):
    """**The float64 reference of a WHOLE evaluation**, at step `step` of the trajectory (`step_inputs`):
    it arbitrates the steps where the engine and the oracle part by more than the threshold."""
    if not size.isdigit():
        raise SystemExit(f"witness-step --size {size}: squares only, the starting latents are the trajectory's")
    resolution = int(size)
    if VARIANT and step != 0:
        raise SystemExit("witness-step --variant: step 0 only (the later latents belong to the Standard's trajectory)")
    latent, timestep, text = step_inputs(resolution, step)
    velocity, probes = evaluate64(latent, timestep, text, resolution, resolution)
    path = OUT / f"{quantized.stem(f'witness-krea2-f64-{resolution}', VARIANT)}-step{step}.safetensors"
    save_file({"velocity": velocity, "model_out": velocity.clone(), **probes}, str(path))
    print(f"-> {path}")


def moved_by_one_ulp(x: torch.Tensor, seed: int) -> torch.Tensor:
    """Every value moved by one ulp, up or down by a hash of (seed, index): the engine's own jitter
    (`moveByOneULP`), so that both sides draw the same neighbours of the same inputs."""
    up = torch.tensor([((i * 2654435761 + seed * 40503) % 2**64 >> 7) & 1 for i in range(x.numel())],
                      dtype=torch.bool).reshape(x.shape)
    return torch.where(up, torch.nextafter(x, torch.tensor(float("inf"), dtype=x.dtype)),
                       torch.nextafter(x, torch.tensor(float("-inf"), dtype=x.dtype)))


def stage_draw(size: str, step: int, jitter: int):
    """**One fp32 draw of step `step`**: the reference's blocks and final layer called one by one on
    `step_inputs` (valid rows, no padding, no mask: as the engine and the witness), the latent moved by one
    ulp under `--jitter N`. Its `model_out` against `witness-step`'s is the oracle's error on that draw: several
    draws say whether the single one of `dit` was typical. The check reads them as the oracle's cloud."""
    resolution = int(size)
    if VARIANT and step != 0:
        raise SystemExit("draw --variant: step 0 only (the later latents belong to the Standard's trajectory)")
    latent, timestep, text = step_inputs(resolution, step)
    if jitter:
        latent = moved_by_one_ulp(latent, jitter)
    model = load_transformer(variant=VARIANT)
    pipe = make_pipeline(transformer=model)
    packed = pipe._pack_latents(latent[None], 1, 16, resolution // 8, resolution // 8)
    text = text[None]
    t = text.shape[1]
    temb = model.time_embed(timestep, dtype=torch.float32)
    temb_mod = model.time_mod_proj(torch.nn.functional.gelu(temb, approximate="tanh"))
    x = torch.cat([text, model.img_in(packed)], dim=1)
    rope = model.rotary_emb(pipe.prepare_position_ids(t, resolution // 16, resolution // 16, torch.device("cpu")))
    payload = {}
    for i, block in enumerate(model.transformer_blocks):
        x = block(x, temb_mod, rope, None)
        if i in (0, 1, 13, 27):
            payload[f"block{i}_out"] = x[0].clone()
        print(f"  block {i}", end="\r", flush=True)
    payload["model_out"] = unpack(pipe, model.final_layer(x[:, t:], temb), resolution)
    path = OUT / f"{quantized.stem(f'draw-krea2-{resolution}', VARIANT)}-step{step}{f'-jitter{jitter}' if jitter else ''}.safetensors"
    save_file({k: v.contiguous() for k, v in payload.items()}, str(path))
    print(f"-> {path}")


# ------------------------------------------------------------------ the summation-order draws
#
# A one-ulp jitter of the latent moves the inputs, not the arithmetic: every product still sums its terms in
# the order torch's BLAS chose once. Here the arithmetic itself is drawn. The residual stream's channels are
# reordered everywhere at once - the rows of every linear that writes it, the columns of every linear that
# reads it, every per-channel vector (norm weights, modulation tables, the time embedding the final layer adds)
# - so that the model computes the SAME function, its sums over the hidden dimension (q, k, v, gate, ff.gate,
# ff.up, time_mod_proj, final_layer.linear, the RMS norms) taken in another order. Untouched: the head space
# (to_q/k/v outputs, norm_q/k, RoPE, the gate's product, to_out's input), the SwiGLU's hidden space, text_fusion
# and txt_in's first linear. The boundaries are put back in the reference's order before they are written.
#
# Two families, both defined here to the bit so that the engine can draw the same ones:
#   group  whole groups of 256 contiguous channels permuted: the engine's int8 route rotates each group of 256
#          (convrot), and a group permutation commutes with that rotation, so the engine can reproduce it;
#   full   any permutation of the channels.
# A permutation is Fisher-Yates driven by SplitMix64 (`fisher_yates`); `perm[c]` is the reference channel that
# permuted channel `c` carries: x_permuted = x[..., perm].

MASK64 = (1 << 64) - 1
PERMUTATION_GROUP = 256


def splitmix64(state: int) -> tuple[int, int]:
    """SplitMix64 (Steele, Lea & Flood, 2014): the next state and its output, everything modulo 2^64."""
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


def fisher_yates(n: int, seed: int) -> torch.Tensor:
    """[0, n) shuffled: the generator's state starts at `seed`; for i = n-1 down to 1, j = (next output) mod (i + 1),
    a[i] and a[j] swapped. (The modulo's bias, below 2^-50 here, is part of the definition.)"""
    a, state = list(range(n)), seed & MASK64
    for i in range(n - 1, 0, -1):
        state, r = splitmix64(state)
        j = r % (i + 1)
        a[i], a[j] = a[j], a[i]
    return torch.tensor(a, dtype=torch.long)


def residual_permutation(dim: int, kind: str, seed: int, group: int = PERMUTATION_GROUP) -> torch.Tensor:
    """The permutation of `seed` in its family: `full` shuffles the `dim` channels, `group` the `dim / group` groups
    (each group keeps its inner order). Seed 0 is the identity in both: the draw loop with nothing moved."""
    if seed == 0:
        return torch.arange(dim)
    if kind == "full":
        return fisher_yates(dim, seed)
    assert dim % group == 0, f"{dim} channels are not whole groups of {group}"
    return (fisher_yates(dim // group, seed)[:, None] * group + torch.arange(group)).flatten()


# What each tensor of the model does with the residual stream, by name (a block's below `transformer_blocks.<i>.`).
# Anything not listed raises: a tensor forgotten would make the draw a different model, not another rounding.
_ROWS = {"img_in.weight", "img_in.bias", "time_embed.linear_2.weight", "time_embed.linear_2.bias",
         "txt_in.linear_2.weight", "txt_in.linear_2.bias", "final_layer.norm.weight",
         "block:norm1.weight", "block:norm2.weight", "block:attn.to_out.0.weight", "block:ff.down.weight"}
_COLUMNS = {"final_layer.linear.weight", "block:attn.to_q.weight", "block:attn.to_k.weight", "block:attn.to_v.weight",
            "block:attn.to_gate.weight", "block:ff.gate.weight", "block:ff.up.weight"}
_TABLES = {"final_layer.scale_shift_table", "block:scale_shift_table"}      # [n, d]: n per-channel vectors
_UNTOUCHED = {"time_embed.linear_1.weight", "time_embed.linear_1.bias", "txt_in.norm.weight", "txt_in.linear_1.weight",
              "txt_in.linear_1.bias", "final_layer.linear.bias", "block:attn.norm_q.weight", "block:attn.norm_k.weight"}


def permuted(key: str, t: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """`t`, the model's tensor `key`, as the permuted model holds it (a fresh tensor, or `t` itself when untouched)."""
    name = "block:" + key.split(".", 2)[2] if key.startswith("transformer_blocks.") else key
    d = perm.numel()
    if name in _ROWS:
        return t[perm].contiguous()
    if name in _COLUMNS:
        return t[:, perm].contiguous()
    if name in _TABLES:
        return t[:, perm].contiguous()
    if name == "time_mod_proj.weight":     # [6d, d]: reads temb (permuted), writes six per-channel vectors
        return t.view(6, d, d)[:, perm][:, :, perm].reshape(6 * d, d).contiguous()
    if name == "time_mod_proj.bias":
        return t.view(6, d)[:, perm].reshape(6 * d).contiguous()
    if name in _UNTOUCHED or name.startswith("text_fusion."):
        return t
    raise KeyError(f"{key}: no rule for the residual permutation")


def permuted_state(state: dict, perm: torch.Tensor, prefix: str = "") -> dict:
    return {k: permuted(prefix + k, v, perm) for k, v in state.items()}


def stage_permuted_draws(size: str, step: int, seeds: list[int], kind: str, img2img: bool = False):
    """**fp32 draws of step `step` with the summation order drawn** (the residual permutation of each seed, see
    above), on `step_inputs` like `stage_draw`, which they otherwise follow call for call. The model is read once:
    each block is loaded, then run under every seed's permutation, so N draws cost one load. Writes, per seed,
    `draw-krea2[-<variant>]-<size>-step<i>-perm<kind><seed>` with `stage_draw`'s boundaries and those of `dit`'s
    head (temb, temb_mod, img_in_out, block 0's attention and feed-forward), in the reference's channel order."""
    resolution = int(size)
    if VARIANT and step != 0 and not img2img:
        raise SystemExit("draw --variant: step 0 only (the later latents belong to the Standard's trajectory)")
    # `img2img-draw`: the img2img golden's step, whose latents are the version's own.
    latent, timestep, text = (img2img_step_inputs if img2img else step_inputs)(resolution, step)
    model = load_transformer(variant=VARIANT, streamed=False)
    pipe = make_pipeline(transformer=model)
    d = model.hidden_size
    perms = {s: residual_permutation(d, kind, s) for s in seeds}
    inverse = {s: torch.argsort(p) for s, p in perms.items()}
    top = {k: v for k, v in model.state_dict().items()
           if k.startswith(("time_embed.", "time_mod_proj.", "img_in.", "final_layer."))}
    packed = pipe._pack_latents(latent[None], 1, 16, resolution // 8, resolution // 8)
    t = text.shape[0]
    rope = model.rotary_emb(pipe.prepare_position_ids(t, resolution // 16, resolution // 16, torch.device("cpu")))
    x, temb, temb_mod, payload = {}, {}, {}, {s: {} for s in seeds}
    for s in seeds:
        p, q = perms[s], inverse[s]
        model.load_state_dict(permuted_state(top, p), strict=False, assign=True)
        temb[s] = model.time_embed(timestep, dtype=torch.float32)
        temb_mod[s] = model.time_mod_proj(torch.nn.functional.gelu(temb[s], approximate="tanh"))
        img = model.img_in(packed)
        x[s] = torch.cat([text[:, p][None], img], dim=1)
        payload[s].update(temb=temb[s][0, 0][q], temb_mod=temb_mod[s][0, 0].view(6, d)[:, q].flatten(),
                          img_in_out=img[0][:, q])
    taps = {}
    b0 = model.transformer_blocks[0]
    b0.attn.register_forward_hook(common.tap(taps, "block0_attn_out"))
    b0.ff.register_forward_hook(common.tap(taps, "block0_ff_out"))
    empty = {k: torch.empty(v.shape, dtype=v.dtype, device="meta") for k, v in b0.state_dict().items()}
    for i, block in enumerate(model.transformer_blocks):
        weights = model.published.under(f"transformer_blocks.{i}.", torch.float32)
        for s in seeds:
            p, q = perms[s], inverse[s]
            block.load_state_dict(permuted_state(weights, p, f"transformer_blocks.{i}."), assign=True)
            x[s] = block(x[s], temb_mod[s], rope, None)
            block.load_state_dict(empty, assign=True)
            if i == 0:
                payload[s].update({name: taps.pop(name)[:, q] for name in ("block0_attn_out", "block0_ff_out")})
            if i in (0, 1, 13, 27):
                payload[s][f"block{i}_out"] = x[s][0][:, q]
        del weights
        print(f"  block {i}", end="\r", flush=True)
    for s in seeds:
        model.load_state_dict(permuted_state(top, perms[s]), strict=False, assign=True)
        payload[s]["model_out"] = unpack(pipe, model.final_layer(x[s][:, t:], temb[s]), resolution)
        name = f"draw-krea2-{'img2img-' if img2img else ''}{resolution}"
        path = OUT / f"{quantized.stem(name, VARIANT)}-step{step}-perm{kind}{s}.safetensors"
        save_file({k: v.contiguous() for k, v in payload[s].items()}, str(path))
        print(f"-> {path}")


def stage_permuted_trajectory(size: str, seeds: list[int], kind: str):
    """**The free trajectory under drawn summation orders**: the 8 Euler steps from the seed's noise, each velocity the
    permuted model's (`stage_permuted_draws`' call for call: valid text rows, no mask), each step `x += (σ[i+1] − σ[i])·v`
    in fp32 as the scheduler does — the reference's own chaos, which a draw restarted from the golden's latent at each
    step cannot show. Seed 0 is the identity: the nominal pass along the same calls. Each block is loaded once per step
    and run under every seed. Writes, per seed, `draw-krea2-<size>-trajectory-perm<kind><seed>` (latent_1 … latent_8)."""
    resolution = int(size)
    trajectory = read(f"goldens-krea2-trajectory-{resolution}", f"krea2.py trajectory --size {resolution}")
    text = read(f"goldens-krea2-dit-{resolution}", f"krea2.py dit --size {resolution}")["txt_in_out"]
    sigmas = trajectory["sigmas"].float()
    model = load_transformer(streamed=False)
    pipe = make_pipeline(transformer=model)
    d = model.hidden_size
    perms = {s: residual_permutation(d, kind, s) for s in seeds}
    top = {k: v for k, v in model.state_dict().items()
           if k.startswith(("time_embed.", "time_mod_proj.", "img_in.", "final_layer."))}
    t = text.shape[0]
    rope = model.rotary_emb(pipe.prepare_position_ids(t, resolution // 16, resolution // 16, torch.device("cpu")))
    empty = {k: torch.empty(v.shape, dtype=v.dtype, device="meta") for k, v in model.transformer_blocks[0].state_dict().items()}
    latent = {s: trajectory["latent_init"].clone() for s in seeds}
    payload = {s: {} for s in seeds}
    for step in range(len(sigmas) - 1):
        timestep = ((sigmas[step] * 1000).float() / 1000).reshape(1)
        x, temb, temb_mod = {}, {}, {}
        for s in seeds:
            p = perms[s]
            model.load_state_dict(permuted_state(top, p), strict=False, assign=True)
            temb[s] = model.time_embed(timestep, dtype=torch.float32)
            temb_mod[s] = model.time_mod_proj(torch.nn.functional.gelu(temb[s], approximate="tanh"))
            packed = pipe._pack_latents(latent[s][None], 1, 16, resolution // 8, resolution // 8)
            x[s] = torch.cat([text[:, p][None], model.img_in(packed)], dim=1)
        for i, block in enumerate(model.transformer_blocks):
            weights = model.published.under(f"transformer_blocks.{i}.", torch.float32)
            for s in seeds:
                block.load_state_dict(permuted_state(weights, perms[s], f"transformer_blocks.{i}."), assign=True)
                x[s] = block(x[s], temb_mod[s], rope, None)
                block.load_state_dict(empty, assign=True)
            del weights
            print(f"  step {step}, block {i}", end="\r", flush=True)
        for s in seeds:
            model.load_state_dict(permuted_state(top, perms[s]), strict=False, assign=True)
            velocity = unpack(pipe, model.final_layer(x[s][:, t:], temb[s]), resolution)
            latent[s] = latent[s] + (sigmas[step + 1] - sigmas[step]) * velocity
            payload[s][f"latent_{step + 1}"] = latent[s].clone()
        print(f"  step {step}: sigma = {float(sigmas[step]):.4f}          ", flush=True)
    for s in seeds:
        path = OUT / f"draw-krea2-{resolution}-trajectory-perm{kind}{s}.safetensors"
        save_file({k: v.contiguous() for k, v in payload[s].items()}, str(path))
        print(f"-> {path}")


def stage_witness(size: str):
    """**The float64 reference of the head**: the engine's inputs (the oracle's latent and ITS `txt_in`
    output: fp32, valid rows, no padding), then time, `img_in` and the first two blocks in double."""
    height, width, name = common.parse_size(size)
    k2, f64 = patch64()
    dit = read(quantized.stem(f"goldens-krea2-dit-{name}", VARIANT), f"krea2.py dit --size {name}")
    config = k2.Krea2Transformer2DModel.load_config(snap() / "transformer")
    with torch.device("meta"):
        model = k2.Krea2Transformer2DModel.from_config(config)
    checkpoint = dit_checkpoint()
    wanted = [k for k in checkpoint.keys if k.startswith(("time_embed.", "time_mod_proj.", "img_in.",
                                                           "transformer_blocks.0.", "transformer_blocks.1."))]
    model.load_state_dict({k: checkpoint.tensor(k, torch.float32).double() for k in wanted}, strict=False, assign=True)
    model.eval()

    pipe = make_pipeline(transformer=model)
    text = dit["txt_in_out"].double()[None]
    t = text.shape[1]
    packed, _ = initial(pipe, height, width)
    timestep = dit["timestep"].double()
    temb = model.time_embed(timestep, dtype=f64)
    temb_mod = model.time_mod_proj(torch.nn.functional.gelu(temb, approximate="tanh"))
    img = model.img_in(packed.double())
    x = torch.cat([text, img], dim=1)
    # the positions of the sequence WITHOUT padding: text at the origin, image on its grid
    ids = pipe.prepare_position_ids(t, height // 16, width // 16, torch.device("cpu"))
    rope = rope64(ids, config)

    taps = {}

    def keep(name):
        def hook(_m, _a, o):
            taps[name] = o[0].float().clone()
        return hook
    b0 = model.transformer_blocks[0]
    b0.attn.register_forward_hook(keep("block0_attn_out"))
    b0.ff.register_forward_hook(keep("block0_ff_out"))
    x = b0(x, temb_mod, rope, None)
    taps["block0_out"] = x[0].float().clone()
    x = model.transformer_blocks[1](x, temb_mod, rope, None)
    taps["block1_out"] = x[0].float().clone()
    payload = {"temb": temb[0, 0].float(), "temb_mod": temb_mod[0, 0].float(), "img_in_out": img[0].float(), **taps}
    # The outputs are stored in fp32 (the golden tensor format), computed in double.
    path = OUT / f"{quantized.stem(f'witness-krea2-f64-{name}', VARIANT)}.safetensors"
    save_file({k: v.contiguous() for k, v in payload.items()}, str(path))
    print(f"-> {path} ({len(payload)} tensors)")


def stage_vae_witness(size: str):
    """**The float64 witness of the decoder** on `goldens-krea2-vae-<size>`'s latent: the real `vae.decode`,
    weights widened to double (`common.decoder_in_double` makes it fit in 16 GB without changing the
    arithmetic). Stored in fp32."""
    from diffusers import AutoencoderKLQwenImage
    _, _, name = common.parse_size(size)
    common.decoder_in_double()
    f64 = torch.float64
    vae = AutoencoderKLQwenImage.from_pretrained(snap() / "vae", torch_dtype=torch.float32).eval().double()
    mean = torch.tensor(vae.config.latents_mean, dtype=f64).view(1, 16, 1, 1, 1)
    inv_std = 1.0 / torch.tensor(vae.config.latents_std, dtype=f64).view(1, 16, 1, 1, 1)
    decoded = read(f"goldens-krea2-vae-{name}", f"krea2.py vae --size {name}")
    latent = decoded["latent_final"].double()[None, :, None]
    image = vae.decode(latent / inv_std + mean, return_dict=False)[0][0, :, 0]
    rmse = (decoded["image"].double() - image).pow(2).mean().sqrt()
    worst, median, plane = common.per_plane(decoded["image"], image)
    psnr = float(20 * torch.log10(2 / rmse))
    print(f"  decoder {name}: oracle fp32 vs fp64 - worst plane {worst:.3e} (plane {plane}), "
          f"median {median:.3e}, PSNR {psnr:.2f} dB")
    path = OUT / f"witness-krea2-vae-f64-{name}.safetensors"
    save_file({"image": image.float().contiguous()}, str(path))
    (OUT / f"witness-krea2-vae-f64-{name}.json").write_text(json.dumps(
        {"inputs": [f"goldens-krea2-vae-{name}"], "oracle_fp32_vs_witness": {
            "decoder_image": {"worst_plane": worst, "median_plane": median, "plane": plane, "psnr_db": psnr}}},
        indent=1))
    print(f"-> {path}")


def stage_vae_bf16(size: str):
    """**The decoder in bfloat16, as the pipeline runs it with `torch_dtype=torch.bfloat16`** on `goldens-krea2-vae-<size>`'s
    latent: `latents.to(vae.dtype)`, `/ (1 / std) + mean` in bf16, then `vae.decode` (`pipeline_qwenimage.py`'s decoding, which
    Krea's follows). The bar of the engine's bf16 decoder (invariant 3): no further from the fp64 witness than this, + 25 %. Its
    convolutions on bf16 operands with fp32 sums (`common.bf16_convolutions_in_fp32`): call it in a process that does nothing else."""
    from diffusers import AutoencoderKLQwenImage
    _, _, name = common.parse_size(size)
    common.bf16_convolutions_in_fp32()
    bf16 = torch.bfloat16
    decoded = read(f"goldens-krea2-vae-{name}", f"krea2.py vae --size {name}")
    exact = read(f"witness-krea2-vae-f64-{name}", f"krea2.py vae-witness --size {name}")["image"]
    vae = AutoencoderKLQwenImage.from_pretrained(snap() / "vae", torch_dtype=bf16).eval()
    latent = decoded["latent_final"][None, :, None].to(bf16)
    mean = torch.tensor(vae.config.latents_mean).view(1, 16, 1, 1, 1).to(bf16)
    inv_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, 16, 1, 1, 1).to(bf16)
    with torch.no_grad():
        image = vae.decode(latent / inv_std + mean, return_dict=False)[0][0, :, 0].float()
    worst, median, plane = common.per_plane(image, exact)
    print(f"  decoder {name}: bf16 vs fp64 - worst plane {worst:.3e} (plane {plane}), median {median:.3e}")
    write(f"witness-bf16-krea2-vae-{name}", {"image": image})


def everything(goldens: Path, sizes: list[str], witness_steps: list[int], draws: int) -> None:
    """Per size, in dependency order: the text, the evaluation and its head witness, the trajectory and the
    whole-evaluation witnesses of its steps (squares: they start from the trajectory's latents), then the
    decoder and its witness. A rectangle's decoder crops the 1024 trajectory's final latent, so that
    trajectory is written too if `sizes` leaves it out."""
    squares = [s for s in sizes if common.parse_size(s)[0] == common.parse_size(s)[1]]
    needed = squares + (["1024"] if len(squares) < len(sizes) and "1024" not in squares else [])
    stages = [["text"]]
    stages += [["dit", "--size", s] for s in sizes]
    stages += [["witness", "--size", s] for s in sizes]
    stages += [["trajectory", "--size", s] for s in needed]
    stages += [["witness-step", "--size", s, "--step", str(i)] for s in squares for i in witness_steps]
    stages += [["draw", "--size", s, "--step", str(i), "--jitter", str(j)]
               for s in squares for i in range(8) for j in range(1, draws + 1)]
    stages += [["vae", "--size", s] for s in sizes]
    stages += [["vae-witness", "--size", s] for s in sizes]
    stages += [["vae-bf16", "--size", s] for s in sizes]
    if "512" in squares:
        stages += [["img2img", "--size", "512"]] + [["img2img-witness", "--size", "512", "--step", str(i)] for i in range(3, STEPS)]
    common.run_stages(__file__, goldens, stages)


def main() -> None:
    global OUT
    common.unbuffered()
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("stage", choices=["text", "dit", "bf16", "w8a8", "trajectory", "vae", "witness", "witness-step", "draw",
                                     "vae-witness", "vae-bf16", "img2img", "img2img-witness", "img2img-draw", "all"])
    p.add_argument("--size", nargs="+", default=None,
                   help="512, 1024 or WIDTHxHEIGHT (dit, witness, vae: a rectangle too; default 512, all: every size)")
    p.add_argument("--step", type=int, nargs="+", default=[0],
                   help="witness-step, img2img-witness and draw: the step(s) of the trajectory, each in its own process")
    p.add_argument("--jitter", type=int, nargs="+", default=[0],
                   help="draw only: seed(s) of the one-ulp move of the latent (0 = the unmoved inputs)")
    p.add_argument("--permute", type=int, nargs="+", default=None,
                   help="trajectory, draw and img2img-draw: seed(s) of the residual permutation (the summation order drawn; 0 the identity), all in one process")
    p.add_argument("--permute-kind", choices=["group", "full"], default="group",
                   help="draw --permute: groups of 256 channels (what the int8 engine can draw too) or any permutation")
    p.add_argument("--witness-steps", type=int, nargs="+", default=list(range(8)),
                   help="all only: the steps whose whole evaluation is taken in float64 (default: all eight)")
    p.add_argument("--draws", type=int, default=0, help="all only: jitters 1..N of every step's draw (default 0: none)")
    p.add_argument("--long", action="store_true", help="text only: the long prompt, past 512 tokens -> goldens-krea2-text-long")
    p.add_argument("--variant", choices=quantized.VARIANTS,
                   help="dit and bf16: the Compact's or the Light's published DiT instead of the Standard's; w8a8: compact, required")
    p.add_argument("--goldens", help="output folder (default: $SILICONED_GOLDENS, else ./goldens)")
    args = p.parse_args()
    if args.stage == "w8a8" and args.variant != "compact":
        p.error("w8a8 takes --variant compact: the Compact's int8 convrot is what ComfyUI computes in W8A8")
    if args.variant and args.stage not in ("dit", "bf16", "w8a8", "witness", "witness-step", "draw", "img2img", "img2img-witness",
                                           "img2img-draw"):
        p.error("--variant goes with dit, bf16, w8a8, witness, witness-step, draw, img2img, img2img-witness or img2img-draw")
    global CONFIG_ONLY, VARIANT
    CONFIG_ONLY, VARIANT = bool(args.variant), args.variant
    OUT = common.goldens_directory(args.goldens) / "krea2"
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "all":
        everything(common.goldens_directory(args.goldens), args.size or ["512", "1024", "768x512"], args.witness_steps, args.draws)
    elif args.stage == "text":
        stage_text(common.LONG_PROMPT, common.LONG_SEQUENCE_LENGTH, "goldens-krea2-text-long") if args.long else stage_text()
        return
    sizes = args.size or ["512"]
    if args.stage == "img2img-draw" and args.permute is None:
        p.error("img2img-draw takes --permute")
    if args.permute is not None and args.stage == "trajectory":
        for size in sizes:
            stage_permuted_trajectory(size, args.permute, args.permute_kind)
        return
    if args.permute is not None:
        if args.stage not in ("draw", "img2img-draw") or len(args.step) > 1 or args.jitter != [0]:
            p.error("--permute goes with trajectory, draw or img2img-draw (one --step, no --jitter)")
        for size in sizes:
            stage_permuted_draws(size, args.step[0], args.permute, args.permute_kind, img2img=args.stage == "img2img-draw")
        return
    jitters = args.jitter if args.stage == "draw" else [0]
    if args.stage in ("witness-step", "img2img-witness", "draw") and len(args.step) * len(jitters) > 1:
        common.run_stages(__file__, common.goldens_directory(args.goldens),
                          [[args.stage, "--size", s, "--step", str(i)] + (["--jitter", str(j)] if args.stage == "draw" else [])
                          + (["--variant", args.variant] if args.variant else [])
                           for s in sizes for i in args.step for j in jitters])
        return
    for size in sizes:
        if args.stage == "witness-step":
            stage_witness_step(size, args.step[0])
        elif args.stage == "img2img-witness":
            stage_img2img_witness(size, args.step[0])
        elif args.stage == "draw":
            stage_draw(size, args.step[0], args.jitter[0])
        elif args.variant and args.stage in ("dit", "bf16", "w8a8"):
            {"dit": stage_dit_variant, "bf16": stage_bf16, "w8a8": stage_w8a8}[args.stage](size, args.variant)
        else:
            {"dit": stage_dit, "bf16": stage_bf16, "trajectory": stage_trajectory, "vae": stage_vae, "witness": stage_witness,
             "vae-witness": stage_vae_witness, "vae-bf16": stage_vae_bf16, "img2img": stage_img2img}[args.stage](size)


if __name__ == "__main__":
    main()
