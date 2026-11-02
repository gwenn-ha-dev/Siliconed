#!/usr/bin/env python3
"""Z-Image Turbo's goldens: what `siliconed check z-image <stage>` judges the engine against.

    z-image.py text [--long]              the Qwen3-4B encoder             -> goldens-text[-long]
    z-image.py dit --size 512 [...]       one DiT evaluation, 41 boundaries -> goldens-dit-<size>
    z-image.py witness --size 512         the same evaluation in float64    -> witness-dit-f64-<size>
    z-image.py bf16 --size 1024           the same evaluation in bfloat16   -> witness-bf16-dit-<size>
    z-image.py trajectory --size 512      the 8-step sampler and its image  -> goldens-trajectory-<size>
    z-image.py vae --size 768x512         the decoder on a rectangle        -> goldens-vae-<size>
    z-image.py vae-witness --size 768x512 the same decoding in float64      -> witness-vae-f64-<size>
    z-image.py img2img --size 512         img2img at strength 0.6: the picture encoded, the start, the truncated
                                          sampler (steps 3-7)              -> goldens-img2img-<size>
    z-image.py all                        every stage at every size the check reads (not `bf16`)
    z-image.py dit|bf16|witness --variant compact|light   the same, for the Compact's or the Light's published
                                          DiT, dequantized by its publisher's code (`quantized.py`)

`bf16` is the bar of a half-precision path (the ANE's fp16 products); `all` leaves it out
because it is costly and only that path reads it.

With `--variant`, `dit` reads the inputs of the Standard's `goldens-dit-<size>` (same latent, same conditioning, same
sigma) and runs the same reference DiT with the variant's published weights, dequantized to float32; it writes
`goldens-dit-<variant>-<size>`. `bf16` and `witness` take the variant the same way. `siliconed check z-image dit
--variant X` judges the engine on the variant's map against them.

`text --long` reads `common.LONG_PROMPT` (past 512 tokens) with the sequence length raised to match; no
other stage reads it.

Each stage consumes the one before it (`dit` reads the conditioning `text` wrote, `vae` crops the
1024 trajectory's final latent), so they are written in that order and from the same prompt:
`all` does it and ends by proving the DiT goldens carry the very conditioning the text golden does.

`--size` is the engine's: `512`, or WIDTHxHEIGHT (`768x512` is 768 wide, 512 high). The folder is
`--goldens DIR`, else `SILICONED_GOLDENS`, else `./goldens`. The DiT weights are 24.6 GB in float32
and are streamed block by block, so a 16 GB machine runs this; a 512 evaluation takes about a minute
and a half, a 1024 trajectory most of an hour.

Setup:  in a fresh virtual environment, pip install -r tools/oracle/requirements.txt
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.dont_write_bytecode = True      # the repository carries no caches
import common  # noqa: E402
import quantized  # noqa: E402
from common import Checkpoint, Recorder, Streamed

REPOSITORY = "Tongyi-MAI/Z-Image-Turbo"
PROMPT = "a 30 year old woman posing in a library"      # the reference prompt; every stage descends from it
MAX_SEQUENCE_LENGTH = 512
DIT_SEED = 0
TRAJECTORY_SEED = 42
SIGMA = 0.8            # step 3 of the 8-step schedule: a mid-trajectory evaluation, neither pure noise nor image
PATCH_KEY = "2-1"      # the only published patch size: {patch_size}-{f_patch_size}
ROUTED_PREFIXES = ("layers.", "noise_refiner.", "context_refiner.")

torch.set_grad_enabled(False)


# ======================================================================================= text

def text(goldens: Path, prompt: str, length: int = MAX_SEQUENCE_LENGTH, stem: str = "goldens-text") -> None:
    """Reproduces `ZImagePipeline._encode_prompt`:

        chat template (add_generation_prompt, enable_thinking) -> tokenize, pad to 512 (`--long`: 1024) ->
        Qwen3 trunk with output_hidden_states -> hidden_states[-2] -> rows of the real tokens

    The DiT receives the real token count, not 512: the padding is cut by the attention mask."""
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen3 import Qwen3Model
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

    root = common.snapshot(REPOSITORY, ["text_encoder", "tokenizer"])
    tokenizer = AutoTokenizer.from_pretrained(str(root / "tokenizer"))
    templated = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                              add_generation_prompt=True, enable_thinking=True)
    encoded = tokenizer([templated], padding="max_length", max_length=length,
                        truncation=True, return_tensors="pt")
    ids, mask = encoded.input_ids, encoded.attention_mask.bool()
    real = int(mask.sum())
    print(repr(templated))
    print(f"real tokens: {real} of {ids.shape[1]}")

    # 16.1 GB in float32: built on `meta`, each layer streamed in and widened.
    config = AutoConfig.from_pretrained(str(root / "text_encoder"))
    with torch.device("meta"):
        model = Qwen3Model(config)
    model.eval()
    checkpoint = Checkpoint(root / "text_encoder", "model.safetensors.index.json")
    Streamed(model.embed_tokens, "model.embed_tokens.", checkpoint, torch.float32)
    for i, layer in enumerate(model.layers):
        Streamed(layer, f"model.layers.{i}.", checkpoint, torch.float32)
    model.norm.load_state_dict({"weight": checkpoint.tensor("model.norm.weight", torch.float32)}, assign=True)

    # `to_empty()` initialises nothing: it hands back whatever the allocator gives (zeros, in
    # practice), so `inv_freq` would be 0, every RoPE angle 0, cos 1 and sin 0: the encoder would
    # run with no position information, a bag of words, while the whole DiT chain stayed consistent
    # (both sides get the same conditioning) and only an independent implementation could tell. The
    # buffer is therefore rebuilt by the class that defines it, never copied from a formula.
    model.rotary_emb = Qwen3RotaryEmbedding(config)
    assert float(model.rotary_emb.inv_freq.abs().max()) > 0, "inv_freq is zero: the goldens would be a bag of words"

    # `hidden_states[-2]` is what the pipeline reads. It is the output of layer 34, before the final norm
    # (hidden_states[-1] is the normed last layer); the probes below prove it rather than assert it.
    probes: dict[str, torch.Tensor] = {}

    def probe(module, name):
        def hook(_m, _inputs, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            probes[name] = t.detach().to(torch.float32).clone()
        module.register_forward_hook(hook)

    probe(model.layers[33], "layer33_out")
    probe(model.layers[34], "layer34_out")
    probe(model.layers[35], "layer35_out")
    probe(model.norm, "final_norm_out")

    out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
    hidden = out.hidden_states
    for name in ("layer33_out", "layer34_out", "layer35_out", "final_norm_out"):
        print(f"  |hidden_states[-2] - {name:15s}| max = {float((hidden[-2] - probes[name]).abs().max()):.3e}")
    assert torch.equal(hidden[-2], probes["layer34_out"]), "hidden_states[-2] is not layer 34's output"

    padded = hidden[-2]
    payload = {
        "input_ids": ids.to(torch.int32),
        "attention_mask": mask.to(torch.int32),
        "hidden_states_-2_padded": padded,
        "cap_feats": padded[0][mask[0]],
        "layer33_out": probes["layer33_out"],
        "layer34_out": probes["layer34_out"],
        "layer35_out": probes["layer35_out"],
        "final_norm_out": probes["final_norm_out"],
    }
    assert real < length, f"{real} tokens: the prompt was cut at {length}"
    common.write_golden(goldens, stem, payload, {
        "prompt": prompt, "templated": templated, "real_tokens": real,
        "max_sequence_length": length, "dtype": "float32", "device": "cpu"})


def conditioning(goldens: Path) -> torch.Tensor:
    return common.read_golden(goldens, "goldens-text", "z-image.py text")["cap_feats"].to(torch.float32)


# ======================================================================================== dit

def build_dit(root: Path, dtype: torch.dtype = torch.float32, variant: str | None = None):
    """The reference `ZImageTransformer2DModel`, on `meta`, with the blocks streamed.

    The 11.6 M parameters outside the blocks (embedders, final layer) stay resident. With a `variant`, the
    weights are the Compact's or the Light's published file, dequantized by its publisher's code
    (`quantized.Variant`); only `transformer/config.json` is read from `root`."""
    from diffusers import ZImageTransformer2DModel
    directory = root / "transformer"
    config = {k: v for k, v in json.loads((directory / "config.json").read_text()).items()
              if not k.startswith("_")}
    with torch.device("meta"):
        model = ZImageTransformer2DModel.from_config(config)
    model.eval()
    checkpoint = (quantized.Variant("z-image", variant) if variant
                  else Checkpoint(directory, "diffusion_pytorch_model.safetensors.index.json"))
    resident = {k: checkpoint.tensor(k, dtype) for k in checkpoint.keys if not k.startswith(ROUTED_PREFIXES)}
    model.load_state_dict(resident, strict=False, assign=True)
    for group in ("noise_refiner", "context_refiner", "layers"):
        for i, block in enumerate(getattr(model, group)):
            Streamed(block, f"{group}.{i}.", checkpoint, dtype)
    model.published = checkpoint
    return model, config


def configuration() -> Path:
    """The published `transformer/config.json` alone: a variant's weights come from another repository."""
    return common.snapshot(REPOSITORY, [], files=["transformer/config.json"])


def probe_dit(model, rec: Recorder) -> None:
    """The boundaries the engine is judged at: embedders, refiners, block 0 inside out, a few
    block outputs, the head. Block 0's internals are there because a deviation at a block's output
    cannot be located otherwise: seven GEMMs, four norms, QK-norm, RoPE, SDPA and two residuals sit
    between its input and its output."""
    rec.output(model.t_embedder, "t_embedder_out")
    rec.output(model.all_x_embedder[PATCH_KEY], "x_embedder_out")
    rec.output(model.cap_embedder, "cap_embedder_out")
    rec.input(model.noise_refiner[0], "noise_refiner_in")
    rec.output(model.noise_refiner[0], "noise_refiner_0_out")
    rec.output(model.noise_refiner[1], "noise_refiner_out")
    rec.input(model.context_refiner[0], "context_refiner_in")
    rec.output(model.context_refiner[1], "context_refiner_out")
    # Block 0's four inputs. The RoPE table and the modulation input are the output of no module, so
    # a pre-hook on the block is the only place to see them. Since diffusers 0.41 the attention mask is
    # None at batch 1 (every sequence has full length) and is simply not recorded.
    rec.inputs(model.layers[0], ["unified_in", "block_attn_mask", "block_freqs_cis", "block_adaln_input"])

    b = model.layers[0]
    rec.output(b.adaLN_modulation, "layer0_adaln_raw")
    rec.output(b.attention_norm1, "layer0_attn_norm1_out")
    rec.input(b.attention.to_q, "layer0_attn_in")            # norm1(x) * scale_msa
    rec.output(b.attention.to_q, "layer0_q")
    rec.output(b.attention.to_k, "layer0_k")
    rec.output(b.attention.to_v, "layer0_v")
    if b.attention.norm_q is not None:
        rec.output(b.attention.norm_q, "layer0_q_normed")    # per head, before RoPE
        rec.output(b.attention.norm_k, "layer0_k_normed")
    rec.input(b.attention.to_out[0], "layer0_sdpa_out")      # after RoPE + SDPA + reassembly
    rec.output(b.attention.to_out[0], "layer0_attn_proj_out")
    rec.output(b.attention, "layer0_attention_out")
    rec.output(b.attention_norm2, "layer0_attn_norm2_out")
    rec.output(b.ffn_norm1, "layer0_ffn_norm1_out")
    rec.input(b.feed_forward.w1, "layer0_ffn_in")            # ffn_norm1(x) * scale_mlp
    rec.output(b.feed_forward.w1, "layer0_w1_out")
    rec.output(b.feed_forward.w3, "layer0_w3_out")
    rec.input(b.feed_forward.w2, "layer0_swiglu_out")        # silu(w1) * w3
    rec.output(b.feed_forward.w2, "layer0_w2_out")
    rec.output(b.feed_forward, "layer0_feed_forward_out")
    rec.output(b.ffn_norm2, "layer0_ffn_norm2_out")
    for i in (0, 1, 15, 29):
        rec.output(model.layers[i], f"layers_{i}_out")
    rec.input(model.all_final_layer[PATCH_KEY], "final_layer_in")
    rec.output(model.all_final_layer[PATCH_KEY], "final_layer_out")


def dit(goldens: Path, size: str, variant: str | None = None) -> None:
    height, width, name = common.parse_size(size)
    root = configuration() if variant else common.snapshot(REPOSITORY, ["transformer"])
    model, config = build_dit(root, variant=variant)
    rec = Recorder()
    probe_dit(model, rec)

    if variant:
        # The Standard's inputs, bit for bit: the variant differs by its weights alone.
        standard = common.read_golden(goldens, f"goldens-dit-{name}", f"z-image.py dit --size {name}")
        latent, caps, sigma = (standard[k] for k in ("in_latent", "in_caps", "in_sigma"))
    else:
        torch.manual_seed(DIT_SEED)
        latent = torch.randn(16, 1, height // 8, width // 8, dtype=torch.float32)
        # The real conditioning, not a `randn`: a Qwen3 hidden state has neither its scale nor its outliers.
        caps = conditioning(goldens)
        sigma = torch.tensor([SIGMA], dtype=torch.float32)
    rec.tensors["in_latent"], rec.tensors["in_caps"], rec.tensors["in_sigma"] = latent.clone(), caps.clone(), sigma.clone()

    out = model(x=[latent], t=sigma, cap_feats=[caps], return_dict=False)
    rec.keep("model_out", out[0] if isinstance(out, (tuple, list)) else out)

    common.write_golden(goldens, quantized.stem(f"goldens-dit-{name}", variant), rec.tensors, {
        "resolution": name, "height": height, "width": width, "seed": DIT_SEED, "sigma": float(sigma),
        "cap_len": int(caps.shape[0]), "dtype": "float32", "device": "cpu", "config": config,
        **({"weights": model.published.describe(), "inputs": f"goldens-dit-{name}"} if variant else {})},
        summary_extra=True)


def bf16(goldens: Path, size: str, variant: str | None = None) -> None:
    """The same evaluation as `dit` with diffusers in bfloat16, as `from_pretrained(torch_dtype=torch.bfloat16)`
    runs it: weights, latent, text and timestep all bf16. The inputs are the dit golden's own, so the
    only difference with it is the arithmetic; its boundaries are widened to float32 (exact) and keep
    the dit golden's keys. A half-precision engine path is judged against this, not against the float32
    golden. Writes `witness-bf16-dit-<size>`; with a `variant`, the variant's weights rounded to bf16 once
    (`quantized.py`) and `witness-bf16-dit-<variant>-<size>`."""
    _, _, name = common.parse_size(size)
    golden = common.read_golden(goldens, f"goldens-dit-{name}", f"z-image.py dit --size {name}")
    half = torch.bfloat16
    model, config = build_dit(configuration() if variant else common.snapshot(REPOSITORY, ["transformer"]), half,
                              variant)
    rec = Recorder()
    probe_dit(model, rec)
    latent, caps, sigma = (golden[k].to(half) for k in ("in_latent", "in_caps", "in_sigma"))
    rec.tensors["in_latent"], rec.tensors["in_caps"], rec.tensors["in_sigma"] = (
        t.to(torch.float32) for t in (latent, caps, sigma))
    out = model(x=[latent], t=sigma, cap_feats=[caps], return_dict=False)
    rec.keep("model_out", out[0] if isinstance(out, (tuple, list)) else out)
    common.write_golden(goldens, quantized.stem(f"witness-bf16-dit-{name}", variant), rec.tensors, {
        "resolution": name, "dtype": "bfloat16", "device": "cpu", "inputs": f"goldens-dit-{name}",
        "config": config, **({"weights": model.published.describe()} if variant else {})}, summary_extra=True)


# ===================================================================================== witness

def channel_error(got: torch.Tensor, exact: torch.Tensor) -> tuple[float, float]:
    """The engine's `channelError(got, exact, columns: h*w)` on a planar [C, h, w]: one column per
    position, relative L2 over channels, floored at 1 % of the median column norm."""
    g, e = got.double().reshape(got.shape[0], -1), exact.double().reshape(exact.shape[0], -1)
    num, den = (g - e).pow(2).sum(dim=0).sqrt(), e.pow(2).sum(dim=0).sqrt()
    live = den > 0
    floor = 0.01 * den[live].sort().values[live.sum() // 2]
    ratio = (num[live] / torch.clamp(den[live], min=floor)).sort().values
    return float(ratio[-1]), float(ratio[len(ratio) // 2])


def witness(goldens: Path, size: str, variant: str | None = None) -> None:
    """The same evaluation in float64: the exact value against which the float32 oracle and the
    engine are both judged. Where a result sits at the threshold, the engine passes if it is no
    further from this witness than the oracle is, plus 25 %.

    The inputs are the dit golden's own, read as they are; only the weights are widened (bf16 to
    float64 is exact). Writes `witness-dit-f64-<size>`; with a `variant`, the dequantized weights (float32 to
    float64 is exact too), judged against the variant's own float32 golden, `witness-dit-f64-<variant>-<size>`."""
    _, _, name = common.parse_size(size)
    golden = common.read_golden(goldens, quantized.stem(f"goldens-dit-{name}", variant),
                                f"z-image.py dit --size {name}" + (f" --variant {variant}" if variant else ""))
    latent, caps, sigma = golden["in_latent"], golden["in_caps"], golden["in_sigma"]
    f64 = torch.float64
    model, _ = build_dit(configuration() if variant else common.snapshot(REPOSITORY, ["transformer"]), f64, variant)
    # Tables built at construction (RoPE...) follow the model's dtype where they are floating point.
    for buffer_name, buffer in list(model.named_buffers()):
        if buffer.is_floating_point() and buffer.device.type != "meta":
            parent = model.get_submodule(buffer_name.rsplit(".", 1)[0]) if "." in buffer_name else model
            setattr(parent, buffer_name.rsplit(".", 1)[-1], buffer.to(f64))
    exact = model([latent.to(f64)], sigma.to(f64), [caps.to(f64)], return_dict=False)[0][0].to(f64)
    oracle = golden["model_out"]
    worst, median = channel_error(oracle.squeeze(1), exact.squeeze(1))
    print(f"fp32 oracle against the fp64 witness: worst {worst:.3e}  median {median:.3e}")
    print(f"the engine passes if it is within {1.25 * worst:.3e} of the witness (oracle + 25 %)")
    path = goldens / f"{quantized.stem(f'witness-dit-f64-{name}', variant)}.safetensors"
    from safetensors.torch import save_file
    save_file({"in_latent": latent.contiguous(), "in_caps": caps.contiguous(), "in_sigma": sigma.contiguous(),
               "model_out": exact.to(torch.float32).contiguous(), "oracle_model_out": oracle.contiguous()},
              str(path))
    print(f"-> {path}")


# ================================================================================== trajectory

def trajectory(goldens: Path, size: str, steps: int = 8) -> None:
    """The loop of `ZImagePipeline.__call__`, through the pipeline's own schedule functions rather
    than a copy of them (diffusers 0.41, which fixed the redundant terminal timestep: N steps are
    N evaluations, each of which moves the latent):

        sigmas    = linspace(1, 1/N, N); static shift 3; then a final 0
        timestep  = (1000 - t) / 1000        time is reversed
        noise_pred = -transformer(...)       the output is negated
        latents   = scheduler.step(noise_pred, t, latents)
        image     = vae.decode(latents / 0.3611 + 0.1159)   scaled AND shifted
    """
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
    from diffusers import __version__ as diffusers_version
    from diffusers.pipelines.z_image.pipeline_z_image import (calculate_shift, get_default_z_image_sigmas,
                                                              retrieve_timesteps)
    from diffusers.utils.torch_utils import randn_tensor

    resolution = int(size)
    root = common.snapshot(REPOSITORY, ["transformer", "vae", "scheduler"])
    caps = conditioning(goldens)
    model, _ = build_dit(root)

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(str(root / "scheduler"))
    latent_size = resolution // 8
    # `mu` is computed and passed as the pipeline does, and ignored by the published config
    # (`use_dynamic_shifting` is false: a static shift of 3).
    mu = calculate_shift((latent_size // 2) ** 2,
                         scheduler.config.get("base_image_seq_len", 256),
                         scheduler.config.get("max_image_seq_len", 4096),
                         scheduler.config.get("base_shift", 0.5),
                         scheduler.config.get("max_shift", 1.15))
    timesteps, _ = retrieve_timesteps(scheduler, steps, torch.device("cpu"),
                                      sigmas=get_default_z_image_sigmas(steps), mu=mu)
    scheduler.set_begin_index(0)
    sigmas = scheduler.sigmas.tolist()
    assert len(timesteps) == steps and sigmas[-1] == 0 and all(a > b for a, b in zip(sigmas, sigmas[1:]))
    print(f"sigma = {[round(s, 6) for s in sigmas]}  ({steps} evaluations)")

    generator = torch.Generator().manual_seed(TRAJECTORY_SEED)
    latents = randn_tensor((1, 16, latent_size, latent_size), generator=generator,
                           device=torch.device("cpu"), dtype=torch.float32)
    out = {"latent_init": latents.clone()}
    for i, t in enumerate(timesteps):
        timestep = ((1000 - t.expand(latents.shape[0])) / 1000).to(torch.float32)
        v = model(list(latents.unsqueeze(2).unbind(dim=0)), timestep, [caps], return_dict=False)[0]
        noise_pred = -torch.stack([x.float() for x in v], dim=0).squeeze(2)
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        out[f"v_{i}"] = noise_pred.clone()
        out[f"latent_{i + 1}"] = latents.clone()
        print(f"step {i}: t {float(t):8.2f}  sigma {sigmas[i]:.6f}  |v|max {float(noise_pred.abs().max()):.4f}")

    vae = AutoencoderKL.from_pretrained(str(root / "vae"), torch_dtype=torch.float32).eval()
    scaled = latents / vae.config.scaling_factor + vae.config.shift_factor
    out["latent_scaled"] = scaled.clone()
    image = vae.decode(scaled, return_dict=False)[0]
    out["image_raw"] = image.clone()

    # The image is for looking at: tensors are compared and never opened, so a reference nobody has
    # seen can be wrong in a way no norm shows.
    try:
        from PIL import Image
        pixels = (image / 2 + 0.5).clamp(0, 1)[0].permute(1, 2, 0).mul(255).round().to(torch.uint8).numpy()
        Image.fromarray(pixels).save(goldens / f"reference-{resolution}.png")
    except ImportError:
        print("(Pillow missing: no PNG written)")

    common.write_golden(goldens, f"goldens-trajectory-{resolution}", out, {
        "resolution": resolution, "steps": steps, "evaluations": len(timesteps), "seed": TRAJECTORY_SEED,
        "sigmas": sigmas, "diffusers": diffusers_version, "dtype": "float32", "device": "cpu"},
        summary_extra=True)


# ======================================================================================== img2img

def img2img(goldens: Path, size: str, steps: int = 8) -> None:
    """**img2img**, by `ZImageImg2ImgPipeline`'s own pieces (diffusers 0.41):

        z0        = (mean - shift_factor) * scaling_factor            the posterior MEAN: the pipeline draws a sample
                    (`retrieve_latents`, sample_mode "sample"); the engine takes the mean, deterministic, and so does this
        timesteps = ZImageImg2ImgPipeline.get_timesteps(N, strength)  the full schedule, then truncated
        x_s       = scheduler.scale_noise(z0, timesteps[:1], eps)     eps: the txt2img noise of the seed (no sample
                                                                      drawn before it, so strength 1 is txt2img bit for bit)
        then the pipeline's loop from x_s: reversed time, negated output, `scheduler.step`

    `velocity_i` is the negated output (`noise_pred`), what the engine's sampler records."""
    import types
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.z_image.pipeline_z_image import calculate_shift
    from diffusers.pipelines.z_image.pipeline_z_image_img2img import (ZImageImg2ImgPipeline, get_default_z_image_sigmas,
                                                                      retrieve_latents, retrieve_timesteps)
    from diffusers.utils.torch_utils import randn_tensor

    resolution = int(size)
    root = common.snapshot(REPOSITORY, ["transformer", "vae", "scheduler"])
    caps = conditioning(goldens)
    picture = common.start_picture(resolution, resolution)
    vae = AutoencoderKL.from_pretrained(str(root / "vae"), torch_dtype=torch.float32).eval()
    mean = retrieve_latents(vae.encode(picture[None]), sample_mode="argmax")
    z0 = (mean - vae.config.shift_factor) * vae.config.scaling_factor
    del vae

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(str(root / "scheduler"))
    latent_size = resolution // 8
    mu = calculate_shift((latent_size // 2) ** 2,
                         scheduler.config.get("base_image_seq_len", 256),
                         scheduler.config.get("max_image_seq_len", 4096),
                         scheduler.config.get("base_shift", 0.5),
                         scheduler.config.get("max_shift", 1.15))
    retrieve_timesteps(scheduler, steps, torch.device("cpu"), sigmas=get_default_z_image_sigmas(steps), mu=mu)
    timesteps, t_start, sigma_s = common.img2img_truncation(
        lambda s, n, strength: ZImageImg2ImgPipeline.get_timesteps(types.SimpleNamespace(scheduler=s), n, strength, "cpu"),
        scheduler, steps, common.IMG2IMG_STRENGTH)
    generator = torch.Generator().manual_seed(TRAJECTORY_SEED)
    eps = randn_tensor((1, 16, latent_size, latent_size), generator=generator, device=torch.device("cpu"), dtype=torch.float32)
    latents = scheduler.scale_noise(z0, timesteps[:1], eps)
    model, _ = build_dit(root)
    out = {"image": picture, "z0": z0[0].clone(), "eps": eps[0].clone(), "x_s": latents[0].clone(),
           "sigmas": scheduler.sigmas.float(), "t_start": torch.tensor([float(t_start)]), "sigma_s": torch.tensor([sigma_s])}
    print(f"strength {common.IMG2IMG_STRENGTH}: t_start {t_start}, sigma_s {sigma_s:.6f}, {len(timesteps)} evaluations")
    for k, t in enumerate(timesteps):
        i = t_start + k
        timestep = ((1000 - t.expand(latents.shape[0])) / 1000).to(torch.float32)
        v = model(list(latents.unsqueeze(2).unbind(dim=0)), timestep, [caps], return_dict=False)[0]
        noise_pred = -torch.stack([x.float() for x in v], dim=0).squeeze(2)
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        out[f"velocity_{i}"] = noise_pred[0].clone()
        out[f"latent_{i + 1}"] = latents[0].clone()
        print(f"step {i}: t {float(t):8.2f}  sigma {float(scheduler.sigmas[i]):.6f}", flush=True)
    out["latent_final"] = latents[0].clone()
    common.write_golden(goldens, f"goldens-img2img-{resolution}", out, {
        "resolution": resolution, "steps": steps, "seed": TRAJECTORY_SEED, "strength": common.IMG2IMG_STRENGTH,
        "t_start": t_start, "sigma_s": sigma_s, "evaluations": len(timesteps), "prompt": PROMPT,
        "picture": f"inputs/{common.START_PICTURE}, covered and center-cropped, LANCZOS",
        "recipe": "ZImageImg2ImgPipeline.get_timesteps, scale_noise; the posterior mean", "dtype": "float32", "device": "cpu"})


# ======================================================================================== vae

def vae(goldens: Path, size: str) -> None:
    """Flux's decoder on a RECTANGULAR latent: a centered crop of the 1024 trajectory's final
    latent. A real denoised latent (so a real image) with h != w, which is what makes a
    height/width swap visible; any square hides it. Same denormalisation as the trajectory."""
    from diffusers import AutoencoderKL
    height, width, name = common.parse_size(size)
    final = common.read_golden(goldens, "goldens-trajectory-1024",
                               "z-image.py trajectory --size 1024")["latent_8"][0]    # [16, 128, 128]
    h, w = height // 8, width // 8
    y0, x0 = (final.shape[1] - h) // 2, (final.shape[2] - w) // 2
    latent = final[:, y0:y0 + h, x0:x0 + w].contiguous()
    decoder = AutoencoderKL.from_pretrained(str(common.snapshot(REPOSITORY, ["vae"]) / "vae"),
                                            torch_dtype=torch.float32).eval()
    image = decoder.decode(latent[None] / decoder.config.scaling_factor + decoder.config.shift_factor,
                           return_dict=False)[0][0]
    common.write_golden(goldens, f"goldens-vae-{name}", {"latent": latent, "image": image}, {
        "source": "goldens-trajectory-1024 latent_8", "crop": [y0, x0, h, w], "height": height, "width": width})


def vae_witness(goldens: Path, size: str) -> None:
    """**The float64 witness of Flux's decoder** on `goldens-vae-<size>`'s latent: the same diffusers module,
    its weights widened to double (float32 to float64 is exact), its convolutions and its bottleneck attention
    cut into bands (`common.decoder_in_double`, which leaves each output's sum untouched). `siliconed check
    z-image vae` judges the engine per plane at it: no further from it than the fp32 oracle, plus 25 %.
    Stored in fp32, far below anything judged. Process-wide patches: call it in a process that does nothing else."""
    from diffusers import AutoencoderKL
    _, _, name = common.parse_size(size)
    common.decoder_in_double()
    decoded = common.read_golden(goldens, f"goldens-vae-{name}", f"z-image.py vae --size {name}")
    decoder = AutoencoderKL.from_pretrained(str(common.snapshot(REPOSITORY, ["vae"]) / "vae"),
                                            torch_dtype=torch.float32).eval().double()
    latent = decoded["latent"].double()[None]
    with torch.no_grad():
        image = decoder.decode(latent / decoder.config.scaling_factor + decoder.config.shift_factor,
                               return_dict=False)[0][0]
    worst, median, plane = common.per_plane(decoded["image"], image)
    rmse = (decoded["image"].double() - image).pow(2).mean().sqrt()
    psnr = float(20 * torch.log10(2 / rmse))
    print(f"  decoder {name}: oracle fp32 vs fp64 - worst plane {worst:.3e} (plane {plane}), "
          f"median {median:.3e}, PSNR {psnr:.2f} dB")
    common.write_golden(goldens, f"witness-vae-f64-{name}", {"image": image.float()}, {
        "inputs": [f"goldens-vae-{name}"], "dtype": "float64", "device": "cpu",
        "oracle_fp32_vs_witness": {"worst_plane": worst, "median_plane": median, "plane": plane, "psnr_db": psnr}})


# ======================================================================================== all

def everything(goldens: Path, prompt: str, sizes: list[str]) -> None:
    text(goldens, prompt)
    for size in sizes:
        dit(goldens, size)
    for size in sizes:
        witness(goldens, size)          # the float64 evaluation `siliconed check z-image dit` judges at, when it exists
    for size in sizes:
        if common.parse_size(size)[0] == common.parse_size(size)[1]:
            trajectory(goldens, size)   # the check's trajectories are square
    if "1024" in sizes:
        vae(goldens, "768x512")
    if "512" in sizes:
        img2img(goldens, "512")         # `siliconed check z-image img2img`
    # A stage regenerated without its successors leaves the goldens straddling two prompts, and the
    # deviation then measured has nothing to do with the engine.
    caps = conditioning(goldens)
    for size in sizes:
        golden = common.read_golden(goldens, f"goldens-dit-{size}", f"z-image.py dit --size {size}")["in_caps"]
        if not torch.equal(golden.to(torch.float32), caps):
            raise SystemExit(f"goldens-dit-{size} was written from another conditioning than goldens-text")
        print(f"dit {size}: in_caps == cap_feats, bit for bit")
    if "1024" in sizes:
        # Last, in its own process: the float64 decoder patches the reference's classes for good.
        common.run_stages(__file__, goldens, [["vae-witness", "--size", "768x512"]])


def main() -> None:
    common.unbuffered()
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("stage", choices=["text", "dit", "witness", "trajectory", "bf16", "vae", "vae-witness", "img2img", "all"])
    p.add_argument("--size", nargs="+", default=None,
                   help="512, 1024 or WIDTHxHEIGHT (default: 512; vae: 768x512; all: 512 1024)")
    p.add_argument("--goldens", help="output folder (default: $SILICONED_GOLDENS, else ./goldens)")
    p.add_argument("--prompt", default=PROMPT, help="text and all only; the checks read the prompt from the golden")
    p.add_argument("--variant", choices=quantized.VARIANTS,
                   help="dit, bf16 and witness: the Compact's or the Light's published DiT instead of the Standard's")
    p.add_argument("--long", action="store_true", help="text only: the long prompt, past 512 tokens -> goldens-text-long")
    args = p.parse_args()
    goldens = common.goldens_directory(args.goldens)
    if args.variant and args.stage not in ("dit", "bf16", "witness"):
        p.error("--variant goes with dit, bf16 or witness")
    sizes = args.size or {"vae": ["768x512"], "vae-witness": ["768x512"], "all": ["512", "1024", "768x512", "832x1216"]}.get(args.stage, ["512"])
    if args.stage == "text" and args.long:
        text(goldens, common.LONG_PROMPT, common.LONG_SEQUENCE_LENGTH, "goldens-text-long")
    elif args.stage == "text":
        text(goldens, args.prompt)
    elif args.stage == "all":
        everything(goldens, args.prompt, sizes)
    else:
        for size in sizes:
            if args.variant:
                {"dit": dit, "witness": witness, "bf16": bf16}[args.stage](goldens, size, args.variant)
            else:
                {"dit": dit, "witness": witness, "bf16": bf16, "trajectory": trajectory, "vae": vae,
                 "vae-witness": vae_witness, "img2img": img2img}[args.stage](goldens, size)


if __name__ == "__main__":
    main()
