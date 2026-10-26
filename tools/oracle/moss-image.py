#!/usr/bin/env python3
"""MOSS-Image's goldens: what `siliconed check moss <stage>` judges the engine against.

The recipe is the community turbo: MOSS-Image-SFT + the LoRA `lodestones/MOSS-Image-SFT-8-steps-dmd`
(`lora.safetensors`, strength 1), 8 Euler steps, CFG 1 (one forward per step), 512 x 512, seed 42.

    moss-image.py schedule                     the sigmas of 2..10, 20 and 50 steps          -> goldens-moss-schedule
    moss-image.py tokenizer                    chat template and ids, six prompts            -> goldens-moss-tokenizer
    moss-image.py text                         Qwen3.5-9B on the reference prompt (taps)     -> goldens-moss-text
    moss-image.py text-witness                 the same, in float64                          -> witness-moss-text-f64
    moss-image.py vae                          a picture encoded, then decoded               -> goldens-moss-vae-512
    moss-image.py vae-witness                  the decoder in float64                        -> witness-moss-vae-f64-512
    moss-image.py dit [--lora]                 one DiT evaluation at step 0 (taps)           -> goldens-moss-dit[-lora]-512
    moss-image.py witness [--lora]             the same evaluation in float64                -> witness-moss-dit[-lora]-f64-512
    moss-image.py dit|witness --variant compact   the same, on the Compact's published DiT (Comfy-Org's int8 convrot,
                                               dequantized by comfy_kitchen: `quantized.py`) -> goldens-moss-dit-compact-512,
                                               witness-moss-dit-compact-f64-512
    moss-image.py dit-bf16 [--variant compact] the dit evaluation in bfloat16, on its golden's inputs (taps)
                                               -> witness-bf16-moss-dit[-compact]-512 (the bar of --neural-engine)
    moss-image.py bf16                         LoRA merged into bf16 weights vs kept apart   -> witness-moss-bf16-512
    moss-image.py trajectory                   the 8 steps with the LoRA                     -> goldens-moss-trajectory-512
    moss-image.py vae-final                    the final latent decoded, the image's md5     -> goldens-moss-vae-final-512
    moss-image.py all                          all of the above, in dependency order

Goldens go in `<goldens>/moss/`; `--goldens DIR`, else `SILICONED_GOLDENS`, else `./goldens`. Each stage rereads what an
earlier one wrote, so run them in the order above. The DiT weighs 20 GB and the encoder 36 GB in float32: both are built
on `meta`, their resident parts loaded, and each layer is loaded (WIDENED to fp32) just before its forward pass and
dropped just after (`common.Streamed`).

The reference is NOT in diffusers 0.41: `MOSSImageTransformer`, `MOSSImageScheduler` and `MOSSImagePipeline` are one
file of the model repository (`pipeline_moss_image.py`), loaded here by path; the encoder is transformers'
`Qwen3_5Model` and the VAE diffusers' `AutoencoderKLFlux2`. **The reference code does the computation**: this script builds
its modules and calls their `forward`; the float64 witnesses are the only transcriptions (RoPE, the delta rule).

What the reference does that a port by analogy would miss:
  - the DiT is given `sigma * 1000` as its timestep (1000 at the first step), not sigma; its sinusoid is cos THEN sin,
    in float64, and the time MLP ends on a Linear: the shared modulation takes the time vector WITHOUT an activation;
  - ONE modulation (6 x 3072) serves the 42 blocks; the head has its own (shift, scale). `(1 + scale)` everywhere;
  - the DiT's RMSNorm multiplies by `weight` (initialised to 1); Qwen3.5's by `1 + weight` (zero-centred), and the
    gated norm of the delta-net layers by `weight` again;
  - RoPE: four axes (r, f, h, w) of 16 bands each, all with the SAME 16 frequencies; the table is `[sin(64) | cos(64)]`
    and the rotation pairs element i with element i + 64 of the head (not adjacent pairs); the image's h and w are centred,
    its f is the text length, the text's f is its rank; the table is float32, never rounded to bf16;
  - the text is the encoder's LAST hidden state AFTER its final norm, every token of the chat template kept (no system
    prompt, `enable_thinking=True` ends the prompt on `<think>\\n`), then a Linear and two bias-carrying transformer blocks;
  - the sequence is image THEN text, joint attention without a mask; the text blocks see only the text;
  - the feed-forward is SITU, not SwiGLU, with `ffn_up_gate` = [hidden | gate] in that order;
  - the latent is already 128 channels (32 x 2 x 2): the pipeline denormalises with the VAE's batch-norm statistics,
    THEN unpatchifies (channel = c * 4 + py * 2 + px), and only then decodes;
  - the LoRA is rank 128 / alpha 128 on the 252 block linears (q, k, v, o, ffn_up_gate, ffn_down), fp32. Merged into
    bf16 weights it would vanish in the rounding (see `bf16`): merge in fp32 or keep it apart.

Setup:  in a fresh virtual environment, pip install -r tools/oracle/requirements.txt
"""
import argparse
import hashlib
import importlib.util
import inspect
import json
import sys
import textwrap
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.dont_write_bytecode = True      # the repository carries no caches
import common  # noqa: E402
import quantized  # noqa: E402
from common import Checkpoint, Streamed

REPOSITORY = "OpenMOSS-Team/MOSS-Image-SFT"
LORA_REPOSITORY = "lodestones/MOSS-Image-SFT-8-steps-dmd"
PROMPT = "a 30 year old woman posing in a library"      # the reference prompt
SEED = 42
STEPS = 8
MAX_SEQUENCE_LENGTH = 4096
DIT_INDEX = "diffusion_pytorch_model.safetensors.index.json"
TEXT_INDEX = "model.safetensors.index.json"
EXTRA_PROMPTS = [       # tokenizer cases beyond the reference prompt
    "Une femme de trente ans pose dans une bibliothèque, éclairage doux — café crème ☕",
    "line one\nline two\n\n  indented, tab\there",
    '{"scene_graph":{"objects":[{"id":"o1","label":"woman","bbox":[0.1,0.2,0.6,0.9]}]},"style":"photo"}',
    "35 mm f/1.4, 1,000 px: 2026-10-25 (3.5%) don't we're",
    common.LONG_PROMPT,
]

torch.set_grad_enabled(False)

OUT: Path
_SNAP: Path | None = None
_LORA: Path | None = None
_REF = None
VARIANT: str | None = None      # --variant: the Compact's published DiT, for `dit` and `witness`


def snap() -> Path:
    if VARIANT:
        # The weights come from the variant's file: the repository gives the reference's code and the DiT's config.
        return common.snapshot(REPOSITORY, [], files=["pipeline_moss_image.py", "transformer/config.json"])
    return full_snap()


def full_snap() -> Path:
    global _SNAP
    if _SNAP is None:
        _SNAP = common.snapshot(REPOSITORY, ["text_encoder", "tokenizer", "transformer", "scheduler", "vae"],
                                files=["model_index.json", "pipeline_moss_image.py"])
    return _SNAP


def lora_path() -> Path:
    global _LORA
    if _LORA is None:
        _LORA = common.snapshot(LORA_REPOSITORY, [], files=["lora.safetensors"]) / "lora.safetensors"
    return _LORA


def reference():
    """The repository's `pipeline_moss_image.py`, loaded by path (nothing is put on `sys.path`)."""
    global _REF
    if _REF is None:
        spec = importlib.util.spec_from_file_location("pipeline_moss_image", snap() / "pipeline_moss_image.py")
        _REF = importlib.util.module_from_spec(spec)
        sys.modules["pipeline_moss_image"] = _REF
        spec.loader.exec_module(_REF)
    return _REF


def write(stem: str, payload: dict, extra: dict | None = None):
    common.write_golden(OUT, stem, payload,
                        {"prompt": PROMPT, "seed": SEED, "steps": STEPS,
                         "snapshot": common.REVISIONS[REPOSITORY], **(extra or {})})


def read(stem: str, producer: str) -> dict:
    return common.read_golden(OUT, stem, producer)


def tap_input(store: dict, name: str, pick=lambda inputs: inputs[0][0]):
    """A forward PRE-hook keeping `pick(inputs)`; it returns None (a returned value would replace the arguments)."""
    def hook(_module, inputs):
        store[name] = pick(inputs).clone()
    return hook


# ------------------------------------------------------------------ the schedule

def make_scheduler(steps: int):
    ref = reference()
    scheduler = ref.MOSSImageScheduler(num_train_timesteps=1000, shift=5.0, training_shift=1.0)
    scheduler.set_timesteps(steps)
    return scheduler


def stage_schedule():
    """The real `MOSSImageScheduler.set_timesteps`: shift 5 on `linspace(1, 5/1004, n)`, snapped to the nearest
    value of the 1000-point training grid, deduplicated; a terminal 0 is appended by the Euler scheduler."""
    payload = {}
    counts = {}
    training = torch.linspace(1, 1 / 1000, 1000)        # training_shift 1: the grid is the linspace itself
    payload["training_grid"] = training.flip(0).contiguous()     # ascending, as the reference searches it
    for n in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50]:
        scheduler = make_scheduler(n)
        sigmas = scheduler.sigmas.float()
        payload[f"sigmas_{n}"] = sigmas
        payload[f"timesteps_{n}"] = scheduler.timesteps.float()
        ascending = payload["training_grid"]
        index = torch.stack([(ascending == s).nonzero().flatten()[0] for s in sigmas[:-1]])
        payload[f"grid_index_{n}"] = index.to(torch.int32)      # position in the ascending training grid
        counts[n] = int(len(sigmas) - 1)
        print(f"  {n:2d} steps -> {counts[n]:2d} sigmas: {[round(float(s), 4) for s in sigmas]}")
    write("goldens-moss-schedule", payload, {"distinct_sigmas": counts, "note":
          "sigmas_n has a terminal 0; timesteps_n = sigmas_n[:-1] * 1000 is what the DiT receives"})


# ------------------------------------------------------------------ the tokenizer

def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(snap() / "tokenizer")


def encode_ids(tokenizer, prompt: str):
    """`MOSSImagePipeline.encode_prompt`, up to the encoder: the chat template, then the tokenizer."""
    formatted = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                              add_generation_prompt=True, enable_thinking=True)
    inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=MAX_SEQUENCE_LENGTH)
    return formatted, inputs


def stage_tokenizer():
    tokenizer = load_tokenizer()
    payload, templates = {}, {}
    for i, prompt in enumerate([PROMPT, *EXTRA_PROMPTS]):
        formatted, inputs = encode_ids(tokenizer, prompt)
        name = "ids" if i == 0 else f"ids_{i}"
        payload[name] = inputs.input_ids[0].to(torch.int32)
        templates[name] = {"prompt": prompt if i < len(EXTRA_PROMPTS) else "<common.LONG_PROMPT>", "formatted": formatted
                           if i < len(EXTRA_PROMPTS) else formatted[:200] + "…", "tokens": int(inputs.input_ids.shape[1])}
        print(f"  {name}: {inputs.input_ids.shape[1]} tokens")
    write("goldens-moss-tokenizer", payload, {"templates": templates})


# ------------------------------------------------------------------ the text encoder

def load_text_encoder(dtype=torch.float32):
    """`Qwen3_5Model` on `meta`, the final norm resident, the 32 layers streamed. The embedding table (4 GB in fp32)
    is never loaded: `embed_rows` reads the rows of the prompt. The vision tower stays on `meta`: a prompt alone
    never enters it (and if it did, the computation would raise instead of returning a wrong number)."""
    from transformers import AutoConfig, Qwen3_5Model
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
    config = AutoConfig.from_pretrained(snap() / "text_encoder")
    with torch.device("meta"):
        model = Qwen3_5Model(config)
    model.eval()
    lm = model.language_model
    checkpoint = Checkpoint(snap() / "text_encoder", TEXT_INDEX)
    module_keys, published = set(model.state_dict()), set(checkpoint.keys)
    if published != module_keys:
        missing, extra = sorted(module_keys - published), sorted(published - module_keys)
        raise SystemExit(f"keys: {len(missing)} missing {missing[:4]}, {len(extra)} extra {extra[:4]}")
    lm.norm.load_state_dict({"weight": checkpoint.tensor("language_model.norm.weight", dtype)}, assign=True)
    lm.rotary_emb = Qwen3_5TextRotaryEmbedding(config=config.text_config)    # `inv_freq` is no tensor of the file
    for i, layer in enumerate(lm.layers):
        Streamed(layer, f"language_model.layers.{i}.", checkpoint, dtype)
    model.published = checkpoint
    return model


def embed_rows(checkpoint: Checkpoint, ids: list[int], dtype) -> torch.Tensor:
    key = "language_model.embed_tokens.weight"
    table = checkpoint._open[checkpoint._shard[key]].get_slice(key)
    return torch.cat([table[i:i + 1] for i in ids]).to(dtype)[None]


def text_taps(model) -> dict:
    """The boundaries the engine is judged at: every layer's output, the rope tables, the final norm and, inside
    layer 0 (delta-net) and layer 3 (full attention), their parts."""
    taps = {}
    lm = model.language_model

    def rope_hook(_module, _inputs, output):
        taps["rope_cos"], taps["rope_sin"] = output[0][0].clone(), output[1][0].clone()
    lm.rotary_emb.register_forward_hook(rope_hook)
    for i, layer in enumerate(lm.layers):
        layer.register_forward_hook(common.tap(taps, f"layer{i}_out", lambda o: o[0]))
    lm.norm.register_forward_hook(common.tap(taps, "final", lambda o: o[0]))
    l0, l3 = lm.layers[0], lm.layers[3]
    l0.input_layernorm.register_forward_hook(common.tap(taps, "layer0_input_norm", lambda o: o[0]))
    l0.linear_attn.in_proj_qkv.register_forward_hook(common.tap(taps, "layer0_in_proj_qkv", lambda o: o[0]))
    l0.linear_attn.norm.register_forward_pre_hook(tap_input(taps, "layer0_delta_core", lambda i: i[0]))
    l0.linear_attn.norm.register_forward_pre_hook(tap_input(taps, "layer0_delta_z", lambda i: i[1]))
    l0.linear_attn.norm.register_forward_hook(common.tap(taps, "layer0_delta_normed", lambda o: o))
    l0.linear_attn.register_forward_hook(common.tap(taps, "layer0_attn_out", lambda o: o[0]))
    l0.mlp.register_forward_hook(common.tap(taps, "layer0_mlp_out", lambda o: o[0]))
    l3.input_layernorm.register_forward_hook(common.tap(taps, "layer3_input_norm", lambda o: o[0]))
    l3.self_attn.q_proj.register_forward_hook(common.tap(taps, "layer3_q_proj", lambda o: o[0]))
    l3.self_attn.o_proj.register_forward_pre_hook(tap_input(taps, "layer3_attn_gated"))
    l3.self_attn.register_forward_hook(common.tap(taps, "layer3_attn_out", lambda o: o[0][0]))
    l3.mlp.register_forward_hook(common.tap(taps, "layer3_mlp_out", lambda o: o[0]))
    return taps


def run_text(model, dtype, ids):
    inputs_embeds = embed_rows(model.published, ids, dtype)
    taps = text_taps(model)
    attention_mask = torch.ones(1, len(ids), dtype=torch.long)
    out = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=False, return_dict=True)
    assert torch.equal(out.last_hidden_state[0], taps["final"])
    return inputs_embeds[0], taps


def stage_text():
    ids = encode_ids(load_tokenizer(), PROMPT)[1].input_ids[0].tolist()
    tokens = read("goldens-moss-tokenizer", "moss-image.py tokenizer")["ids"]
    assert tokens.tolist() == ids
    model = load_text_encoder()
    embed, taps = run_text(model, torch.float32, ids)
    write("goldens-moss-text", {"ids": torch.tensor(ids, dtype=torch.int32), "embed": embed, **taps},
          {"tokens": len(ids), "layer_types": model.config.text_config.layer_types})


def patch_text64():
    """Qwen3.5's float32 places in double: the norms, the rotary table, the delta-net's `a.float()` and `A_log.float()`,
    and its delta rule, replaced by the plain token-by-token recurrence (a different algorithm from the reference's
    64-token chunks, hence an independent witness)."""
    import torch.nn.functional as F
    import transformers.models.qwen3_5.modeling_qwen3_5 as q
    f64 = torch.float64

    def rms(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * (1.0 + self.weight)
    q.Qwen3_5RMSNorm.forward = rms

    def gated(self, hidden, gate):
        hidden = hidden * torch.rsqrt(hidden.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * hidden * F.silu(gate)
    q.Qwen3_5RMSNormGated.forward = gated

    def rotary(self, x, position_ids):
        position_ids = position_ids.expand(3, -1, -1)
        dim = int(self.config.head_dim * self.config.rope_parameters.get("partial_rotary_factor", 1.0))
        inv_freq = 1.0 / (self.config.rope_parameters["rope_theta"] ** (torch.arange(0, dim, 2, dtype=f64) / dim))
        freqs = position_ids[..., None].to(f64) * inv_freq
        return self.recomposition_frequencies(freqs.cos()).to(x.dtype), self.recomposition_frequencies(freqs.sin()).to(x.dtype)
    q.Qwen3_5TextRotaryEmbedding.forward = rotary

    def delta(query, key, value, g, beta, chunk_size=64, initial_state=None, output_final_state=False,
              use_qk_l2norm_in_kernel=False, **kwargs):
        assert initial_state is None
        query, key, value, beta, g = (x.to(f64) for x in (query, key, value, beta, g))
        if use_qk_l2norm_in_kernel:
            query = query * torch.rsqrt((query * query).sum(-1, keepdim=True) + 1e-6)
            key = key * torch.rsqrt((key * key).sum(-1, keepdim=True) + 1e-6)
        query = query * query.shape[-1] ** -0.5
        batch, length, heads, dk = key.shape
        state = torch.zeros(batch, heads, dk, value.shape[-1], dtype=f64)
        out = torch.zeros_like(value)
        for t in range(length):
            state = state * g[:, t].exp()[..., None, None]
            memory = (state * key[:, t][..., None]).sum(-2)
            update = (value[:, t] - memory) * beta[:, t][..., None]
            state = state + key[:, t][..., None] * update[:, :, None, :]
            out[:, t] = (state * query[:, t][..., None]).sum(-2)
        return out, None
    q.torch_chunk_gated_delta_rule = delta

    source = textwrap.dedent(inspect.getsource(q.Qwen3_5GatedDeltaNet.forward))
    source = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("@"))
    patched = source.replace("a.float()", "a").replace("self.A_log.float()", "self.A_log")
    assert patched != source and "float()" not in patched.split("g = ")[1].split("\n")[0]
    namespace = dict(vars(q))
    exec(patched, namespace)
    q.Qwen3_5GatedDeltaNet.forward = namespace["forward"]


def stage_text_witness():
    """The encoder in float64 on the fp32 golden's tokens: every layer's output and the final state, stored in fp32."""
    golden = read("goldens-moss-text", "moss-image.py text")
    patch_text64()
    ids = golden["ids"].tolist()
    model = load_text_encoder(torch.float64)
    inputs_embeds = embed_rows(model.published, ids, torch.float64)
    taps = {}
    for i, layer in enumerate(model.language_model.layers):
        layer.register_forward_hook(common.tap(taps, f"layer{i}_out", lambda o: o[0]))
    model.language_model.norm.register_forward_hook(common.tap(taps, "final", lambda o: o[0]))
    model(inputs_embeds=inputs_embeds, attention_mask=torch.ones(1, len(ids), dtype=torch.long),
          use_cache=False, return_dict=True)
    payload = {k: v.float().contiguous() for k, v in taps.items()}
    report = {}
    for name in ("layer0_out", "layer3_out", "layer15_out", "layer31_out", "final"):
        report[name] = relative(golden[name], taps[name])
        print(f"  {name}: fp32 against fp64, {report[name]}")
    save_file(payload, str(OUT / "witness-moss-text-f64.safetensors"))
    (OUT / "witness-moss-text-f64.json").write_text(json.dumps(
        {"inputs": ["goldens-moss-text"], "oracle_fp32_vs_witness": report}, indent=1))
    print(f"-> {OUT / 'witness-moss-text-f64.safetensors'}")


def relative(got, exact, axis: int = -1) -> dict:
    """The relative L2 error per channel (`axis`, the last by default; the first of a `[C, H, W]` map): worst and
    median, and the global one."""
    got, exact = got.double().movedim(axis, -1), exact.double().movedim(axis, -1)
    channels = exact.reshape(-1, exact.shape[-1]) if exact.ndim > 1 else exact.reshape(-1, 1)
    diff = (got.reshape(channels.shape) - channels)
    per = diff.norm(dim=0) / channels.norm(dim=0).clamp(min=0.01 * float(channels.norm(dim=0).median()))
    return {"worst_channel": float(per.max()), "median_channel": float(per.median()),
            "global": float(diff.norm() / channels.norm())}


# ------------------------------------------------------------------ the DiT

class LoraCheckpoint:
    """The DiT's checkpoint with the LoRA merged on read: `W' = W + strength * (alpha / rank) * up @ down`, computed in the
    requested dtype from the fp32 factors, the published bf16 weight widened first (exact). The 252 targets are checked."""

    def __init__(self, base: Checkpoint, path: Path, strength: float = 1.0):
        self.base, self.strength, self.keys = base, strength, base.keys
        self.file = safe_open(str(path), framework="pt")
        names = set(self.file.keys())
        self.targets = sorted(k[len("diffusion_model."):-len(".lora_down.weight")] for k in names
                              if k.endswith(".lora_down.weight"))
        missing = [t for t in self.targets if t + ".weight" not in set(base.keys)]
        assert len(self.targets) == 252 and not missing, (len(self.targets), missing[:3])
        self.target_keys = {t + ".weight": t for t in self.targets}

    def delta(self, name: str, dtype) -> torch.Tensor:
        prefix = f"diffusion_model.{name}"
        down = self.file.get_tensor(prefix + ".lora_down.weight").to(dtype)
        up = self.file.get_tensor(prefix + ".lora_up.weight").to(dtype)
        alpha = float(self.file.get_tensor(prefix + ".alpha"))
        return (up @ down) * (self.strength * alpha / down.shape[0])

    def tensor(self, key: str, dtype) -> torch.Tensor:
        """The merge is made in fp32 (fp64 for a fp64 request) and ROUNDED to `dtype` once, at the end."""
        wide = torch.float64 if dtype == torch.float64 else torch.float32
        t = self.base.tensor(key, wide)
        return (t + self.delta(self.target_keys[key], wide)).to(dtype) if key in self.target_keys else t.to(dtype)

    def under(self, prefix: str, dtype) -> dict:
        n = len(prefix)
        return {k[n:]: self.tensor(k, dtype) for k in self.keys if k.startswith(prefix)}


def dit_checkpoint(lora: bool):
    base = quantized.Variant("moss-image", VARIANT) if VARIANT else Checkpoint(snap() / "transformer", DIT_INDEX)
    return LoraCheckpoint(base, lora_path()) if lora else base


def build_dit(dtype, lora: bool):
    ref = reference()
    config = ref.MOSSImageTransformer.load_config(snap() / "transformer")
    with torch.device("meta"):
        model = ref.MOSSImageTransformer.from_config(config)
    model.eval()
    checkpoint = dit_checkpoint(lora)
    module_keys = set(model.state_dict())
    if set(checkpoint.keys) != module_keys:
        missing, extra = sorted(module_keys - set(checkpoint.keys)), sorted(set(checkpoint.keys) - module_keys)
        raise SystemExit(f"keys: {len(missing)} missing {missing[:4]}, {len(extra)} extra {extra[:4]}")
    resident = {k: checkpoint.tensor(k, dtype) for k in checkpoint.keys if not k.startswith("layers.")}
    model.load_state_dict(resident, strict=False, assign=True)
    model.published = checkpoint
    return model, checkpoint


def load_dit(dtype=torch.float32, lora=False):
    model, checkpoint = build_dit(dtype, lora)
    for i, block in enumerate(model.layers):
        Streamed(block, f"layers.{i}.", checkpoint, dtype)
    return model


def initial_latent(height=512, width=512) -> torch.Tensor:
    from diffusers.utils.torch_utils import randn_tensor
    generator = torch.Generator("cpu").manual_seed(SEED)
    return randn_tensor((1, 128, height // 16, width // 16), generator=generator, device=torch.device("cpu"),
                        dtype=torch.float32)


def probe_dit(model) -> dict:
    """Time, modulations, the embedded image, the rope table, the text path (projection and both text blocks), block 0
    inside out, blocks 1, 13, 27 and 41 (the last), the head."""
    taps = {}
    model.time_embedding[0].register_forward_pre_hook(tap_input(taps, "time_sinusoid"))
    model.time_embedding.register_forward_hook(common.tap(taps, "time_embedding_out", lambda o: o[0]))
    model.shared_block_modulation.register_forward_hook(common.tap(taps, "shared_modulation", lambda o: o[0]))
    model.video_head_modulation.register_forward_hook(common.tap(taps, "head_modulation", lambda o: o[0]))
    model.video_embedder.register_forward_hook(common.tap(taps, "video_embed", lambda o: o[0]))
    model.rope.register_forward_hook(common.tap(taps, "rope", lambda o: o[0]))
    model.text_embedder.proj.register_forward_hook(common.tap(taps, "text_proj_out", lambda o: o[0]))
    model.text_embedder.blocks[0].register_forward_hook(common.tap(taps, "text_block0_out", lambda o: o[0]))
    model.text_embedder.blocks[1].register_forward_hook(common.tap(taps, "text_block1_out", lambda o: o[0]))
    b0 = model.layers[0]
    for name in ("q", "k", "v", "o", "ffn_up_gate", "ffn_down"):
        b0.proj[name].register_forward_hook(common.tap(taps, f"block0_{name}_out", lambda o: o[0]))
    b0.q_norm.register_forward_hook(common.tap(taps, "block0_q_norm_out", lambda o: o[0]))
    for i in (0, 1, 13, 27, 41):
        model.layers[i].register_forward_hook(common.tap(taps, f"block{i}_out", lambda o: o[0]))
    model.final_norm_video.register_forward_hook(common.tap(taps, "final_norm_out", lambda o: o[0]))
    return taps


def dit_inputs(text: torch.Tensor):
    scheduler = make_scheduler(STEPS)
    return initial_latent(), scheduler.timesteps[0].reshape(1).float(), text[None], scheduler


def stage_dit(size: str, lora: bool):
    height, width, name = common.parse_size(size)
    assert height == width == 512
    text = read("goldens-moss-text", "moss-image.py text")["final"]
    latent, timestep, embeds, scheduler = dit_inputs(text)
    model = load_dit(lora=lora)
    taps = probe_dit(model)
    out = model(latent, timestep, embeds).sample
    payload = {"latent_in": latent[0], "timestep": timestep, "text_in": embeds[0], "model_out": out[0], **taps}
    stem = quantized.stem(f"goldens-moss-dit-lora-{name}" if lora else f"goldens-moss-dit-{name}", VARIANT)
    write(stem, payload, {"sigma": float(scheduler.sigmas[0]), "lora": lora and "lora.safetensors, strength 1",
                          "lora_revision": common.REVISIONS[LORA_REPOSITORY] if lora else None,
                          **({"weights": model.published.describe()} if VARIANT else {})})


def patch_dit64():
    ref = reference()
    f64 = torch.float64

    def rms(self, x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * self.weight
    ref._RMSNorm.forward = rms


def rope64(height, width, text_length, base=10000.0, bands=16):
    """The table of `_RoPE.forward`, transcribed in double (coordinates (r, f, h, w); image then text)."""
    f64 = torch.float64
    h, w = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    image = torch.stack((torch.zeros_like(h), torch.full_like(h, text_length), h, w), -1).flatten(0, 1).to(f64)
    image[:, 2] -= (height - 1) / 2
    image[:, 3] -= (width - 1) / 2
    text = torch.zeros(text_length, 4, dtype=f64)
    text[:, 1] = torch.arange(text_length)
    frequencies = 1 / (base ** (torch.arange(bands, dtype=f64) / bands))
    phase = torch.cat((image, text), 0).unsqueeze(-1) * frequencies
    return torch.cat((phase.sin(), phase.cos()), -2).flatten(-2).unsqueeze(0)


def stage_witness(size: str, lora: bool):
    """One whole evaluation in float64 on the fp32 golden's inputs (latent, timestep, the encoder's fp32 output)."""
    _, _, name = common.parse_size(size)
    stem = quantized.stem(f"goldens-moss-dit-lora-{name}" if lora else f"goldens-moss-dit-{name}", VARIANT)
    golden = read(stem, f"moss-image.py dit{' --lora' if lora else ''}{' --variant ' + VARIANT if VARIANT else ''}")
    patch_dit64()
    f64 = torch.float64
    ref = reference()
    config = ref.MOSSImageTransformer.load_config(snap() / "transformer")
    with torch.device("meta"):
        model = ref.MOSSImageTransformer.from_config(config)
    model.eval()
    checkpoint = dit_checkpoint(lora)
    model.load_state_dict({k: checkpoint.tensor(k, f64) for k in checkpoint.keys if not k.startswith("layers.")},
                          strict=False, assign=True)
    height = width = 32
    text = golden["text_in"].double()[None]
    t = text.shape[1]
    # the sinusoid in double, NOT rounded to float32 as the reference does before its MLP: the exact value
    half = config["freq_dim"] // 2
    freq = 10000 ** (-torch.arange(half, dtype=f64) / half)
    phase = golden["timestep"].double()[:, None] * freq
    sinusoid = torch.cat((phase.cos(), phase.sin()), -1)
    time = model.time_embedding(sinusoid)
    mod = model.shared_block_modulation(time).view(1, 6, -1)
    head = model.video_head_modulation(time).view(1, 2, -1)
    rope = rope64(height, width, t, config["rope_base"])
    video = model.video_embedder(golden["latent_in"].double().flatten(1).t()[None])
    probes = {"time_embedding_out": time[0], "shared_modulation": mod[0].flatten(), "head_modulation": head[0].flatten(),
              "video_embed": video[0], "rope": rope[0]}
    x = model.text_embedder.proj(text)
    for i, block in enumerate(model.text_embedder.blocks):
        x = block(x, rope[:, height * width:])
        probes[f"text_block{i}_out"] = x[0]
    probes["text_proj_out"] = model.text_embedder.proj(text)[0]
    x = torch.cat((video, x), 1)
    for i, block in enumerate(model.layers):
        block.load_state_dict(checkpoint.under(f"layers.{i}.", f64), strict=True, assign=True)
        x = block(x, rope, mod)
        block.to_empty(device="meta")
        if i in (0, 1, 13, 27, 41):
            probes[f"block{i}_out"] = x[0]
        print(f"  block {i}", end="\r", flush=True)
    y = model.final_norm_video(x[:, :height * width])
    probes["final_norm_out"] = y[0]
    y = y * (1 + head[:, 1, None]) + head[:, 0, None]
    out = model.final_linear_video(y).transpose(1, 2).reshape(1, -1, height, width)[0]
    payload = {"model_out": out.float(), **{k: v.float() for k, v in probes.items()}}
    report = {k: relative(golden[k], payload[k], 0 if golden[k].ndim == 3 else -1) for k in payload if k in golden}
    for k in ("model_out", "block0_out", "block41_out", "text_block1_out", "shared_modulation"):
        print(f"  {k}: fp32 against fp64, {report[k]}")
    path = OUT / f"witness-moss-dit{'-lora' if lora else ''}{'-' + VARIANT if VARIANT else ''}-f64-{name}.safetensors"
    save_file({k: v.contiguous() for k, v in payload.items()}, str(path))
    path.with_suffix(".json").write_text(json.dumps({"inputs": [stem], "oracle_fp32_vs_witness": report}, indent=1))
    print(f"-> {path}")


def apart_hook(lora: "LoraCheckpoint", name: str):
    """The LoRA kept apart, as the model card's demo applies it: `y = base(x) + (alpha / rank) * up(down(x))`, factors cast to
    the activation dtype. A forward hook that RETURNS a value on purpose: it replaces the linear's output."""
    def hook(_module, inputs, output):
        x = inputs[0]
        prefix = f"diffusion_model.{name}"
        down = lora.file.get_tensor(prefix + ".lora_down.weight").to(x.dtype)
        up = lora.file.get_tensor(prefix + ".lora_up.weight").to(x.dtype)
        scale = float(lora.file.get_tensor(prefix + ".alpha")) / down.shape[0]
        return output + torch.nn.functional.linear(torch.nn.functional.linear(x, down), up) * scale
    return hook


def stage_bf16(size: str):
    """The same evaluation with bfloat16 weights and activations, two ways of carrying the LoRA: MERGED into the bf16
    weights (the delta is added in fp32, the sum rounded to bf16) and KEPT APART (the demo's way: bf16 base, bf16 low-rank
    path). The inputs are the lora golden's. Written under `model_out_merged` and `model_out_apart`; the verdict (error against
    the fp32 `model_out`, and the share of the delta that survives the rounding) goes in the manifest."""
    _, _, name = common.parse_size(size)
    golden = read(f"goldens-moss-dit-lora-{name}", "moss-image.py dit --lora")
    half = torch.bfloat16
    latent, timestep, embeds = golden["latent_in"][None].to(half), golden["timestep"], golden["text_in"][None].to(half)
    payload, report = {}, {}
    # merged: LoraCheckpoint widens to fp32, merges, and the streamed block receives bf16 (the cast rounds the sum)
    model = load_dit(half, lora=True)
    payload["model_out_merged"] = model(latent, timestep, embeds).sample[0].float()
    del model
    # apart: base bf16 weights; the LoRA factors stay fp32 in the file and are applied in bf16 like the demo
    model, checkpoint = build_dit(half, False)
    lora = LoraCheckpoint(checkpoint, lora_path())      # the factors only; the merge is not used here
    for i, block in enumerate(model.layers):
        Streamed(block, f"layers.{i}.", checkpoint, half)
        for key in ("q", "k", "v", "o", "ffn_up_gate", "ffn_down"):
            block.proj[key].register_forward_hook(apart_hook(lora, f"layers.{i}.proj.{key}"))
    payload["model_out_apart"] = model(latent, timestep, embeds).sample[0].float()
    for key, value in payload.items():
        report[key] = relative(value, golden["model_out"], 0)
        print(f"  {key} against the fp32 model_out: {report[key]}")
    # how much of a block-0 delta survives the rounding into bf16
    w = checkpoint.tensor("layers.0.proj.q.weight", torch.float32)
    d = lora.delta("layers.0.proj.q", torch.float32)
    merged = (w + d).to(half).float() - w
    report["layer0_q_delta"] = {"delta_norm_over_weight_norm": float(d.norm() / w.norm()),
                                "survived_delta_error": float((merged - d).norm() / d.norm())}
    print(f"  layers.0.proj.q: {report['layer0_q_delta']}")
    path = OUT / f"witness-moss-bf16-{name}.safetensors"
    save_file({k: v.contiguous() for k, v in payload.items()}, str(path))
    path.with_suffix(".json").write_text(json.dumps({"inputs": [f"goldens-moss-dit-lora-{name}"], "report": report}, indent=1))
    print(f"-> {path}")


def stage_dit_bf16(size: str):
    """`dit` with the reference in bfloat16, as the pipeline runs it under `torch_dtype=torch.bfloat16`: the weights
    in bf16 (the Compact's dequantized, then rounded), the module casting its fp32 inputs itself (`forward` casts the
    latent, the text and the time vector to its weights' dtype). The inputs are the dit golden's, so only the
    arithmetic differs; the taps are widened to float32 (exact) under the golden's keys. It is the bar of the engine's
    half-precision paths (`--neural-engine`, `--bf16-activations`): `witness-bf16-…`, the name `siliconed check` looks for.
    No LoRA: `bf16` is the stage that weighs it."""
    _, _, name = common.parse_size(size)
    stem = quantized.stem(f"goldens-moss-dit-{name}", VARIANT)
    golden = read(stem, f"moss-image.py dit{' --variant ' + VARIANT if VARIANT else ''}")
    model = load_dit(torch.bfloat16)
    taps = probe_dit(model)
    out = model(golden["latent_in"][None], golden["timestep"], golden["text_in"][None]).sample
    payload = {"model_out": out[0].float(), **{k: v.float() for k, v in taps.items()}}
    for k in ("model_out", "block0_out", "block41_out"):
        print(f"  {k}: bf16 against fp32, {relative(golden[k], payload[k], 0 if golden[k].ndim == 3 else -1)}")
    write(stem.replace("goldens-", "witness-bf16-"), payload,
          {"dtype": "bfloat16", "inputs": stem, **({"weights": model.published.describe()} if VARIANT else {})})


# ------------------------------------------------------------------ the trajectory

def stage_trajectory(size: str):
    height, width, name = common.parse_size(size)
    assert height == width == 512
    text = read("goldens-moss-text", "moss-image.py text")["final"]
    latents, _, embeds, scheduler = dit_inputs(text)
    model = load_dit(lora=True)
    payload = {"latent_init": latents[0].clone(), "sigmas": scheduler.sigmas.float(),
               "timesteps": scheduler.timesteps.float()}
    for i, t in enumerate(scheduler.timesteps):
        velocity = model(latents, t.reshape(1), embeds).sample.float()      # CFG 1: one forward, no negative branch
        latents = scheduler.step(velocity, t, latents, return_dict=False)[0]
        payload[f"velocity_{i}"] = velocity[0].clone()
        payload[f"latent_{i + 1}"] = latents[0].clone()
        print(f"  step {i}: sigma = {float(scheduler.sigmas[i]):.4f}, t = {float(t):.1f}", flush=True)
    payload["latent_final"] = latents[0].clone()
    write(f"goldens-moss-trajectory-{name}", payload, {"lora": "lora.safetensors, strength 1", "cfg": 1.0,
                                                       "steps_run": len(scheduler.timesteps)})


# ------------------------------------------------------------------ the VAE

def load_vae(dtype=torch.float32):
    from diffusers import AutoencoderKLFlux2
    return AutoencoderKLFlux2.from_pretrained(snap() / "vae", torch_dtype=dtype).eval()


def denormalize_unpatchify(vae, latent: torch.Tensor) -> torch.Tensor:
    """`MOSSImagePipeline.__call__` after its loop, to the letter: batch-norm statistics, then the 2x2 unpatchify."""
    z = latent.to(dtype=vae.dtype)
    eps = vae.config.batch_norm_eps
    z = z * (vae.bn.running_var + eps).sqrt().to(z).view(1, -1, 1, 1) + vae.bn.running_mean.to(z).view(1, -1, 1, 1)
    b, c, h, w = z.shape
    return z.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, 2 * h, 2 * w)


def normalize_patchify(vae, z: torch.Tensor) -> torch.Tensor:
    """The inverse (the pipeline never encodes): patchify with the same channel order, then (z - mean) / sqrt(var + eps)."""
    b, c, h, w = z.shape
    p = z.reshape(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)
    eps = vae.config.batch_norm_eps
    return (p - vae.bn.running_mean.view(1, -1, 1, 1)) / (vae.bn.running_var + eps).sqrt().view(1, -1, 1, 1)


def decode_taps(vae) -> dict:
    taps = {}
    vae.post_quant_conv.register_forward_hook(common.tap(taps, "post_quant_conv_out", lambda o: o[0]))
    d = vae.decoder
    d.conv_in.register_forward_hook(common.tap(taps, "decoder_conv_in", lambda o: o[0]))
    d.mid_block.register_forward_hook(common.tap(taps, "decoder_mid", lambda o: o[0]))
    d.up_blocks[0].register_forward_hook(common.tap(taps, "decoder_up0", lambda o: o[0]))
    d.up_blocks[1].register_forward_hook(common.tap(taps, "decoder_up1", lambda o: o[0]))
    return taps


def stage_vae(size: str):
    """A realistic latent without the DiT: a test picture resized to 512, encoded (posterior mean), patchified and
    normalised. Decoded through the pipeline's own path. The keys the engine reads: `latent` (the DiT's space),
    `latent_z` (the decoder's input), `image` (after the pipeline's clamp)."""
    from PIL import Image
    height, width, name = common.parse_size(size)
    assert height == width == 512
    vae = load_vae()
    picture = Image.open(Path(__file__).resolve().parent / "inputs" / "border-collie-beach.png").convert("RGB")
    picture = picture.resize((512, 512), Image.LANCZOS)
    pixels = torch.from_numpy(__import__("numpy").asarray(picture).copy()).permute(2, 0, 1).float() / 127.5 - 1
    moments = vae._encode(pixels[None])
    z_mode = moments[:, :32]
    latent = normalize_patchify(vae, z_mode)
    taps = decode_taps(vae)
    z = denormalize_unpatchify(vae, latent)
    image = vae.decode(z, return_dict=False)[0].clamp(-1, 1)
    print(f"  denormalised z against the posterior mean: {relative(z[0], z_mode[0], 0)}")
    write(f"goldens-moss-vae-{name}", {"image_in": pixels, "moments": moments[0], "latent": latent[0], "latent_z": z[0],
                                       "image": image[0], **taps},
          {"picture": "inputs/border-collie-beach.png, LANCZOS to 512x512", "decode": "bn denormalise, unpatchify, decode, clamp"})
    common.save_preview(image[0], OUT / f"reference-moss-vae-{name}.png")


def stage_vae_witness(size: str):
    _, _, name = common.parse_size(size)
    common.decoder_in_double()
    f64 = torch.float64
    vae = load_vae(f64)
    golden = read(f"goldens-moss-vae-{name}", f"moss-image.py vae --size {name}")
    z = denormalize_unpatchify(vae, golden["latent"].double()[None])
    image = vae.decode(z, return_dict=False)[0].clamp(-1, 1)[0]
    worst, median, plane = common.per_plane(golden["image"], image)
    rmse = (golden["image"].double() - image).pow(2).mean().sqrt()
    psnr = float(20 * torch.log10(2 / rmse))
    print(f"  decoder {name}: fp32 against fp64 - worst plane {worst:.3e} (plane {plane}), median {median:.3e}, PSNR {psnr:.2f} dB")
    save_file({"image": image.float().contiguous()}, str(OUT / f"witness-moss-vae-f64-{name}.safetensors"))
    (OUT / f"witness-moss-vae-f64-{name}.json").write_text(json.dumps(
        {"inputs": [f"goldens-moss-vae-{name}"], "oracle_fp32_vs_witness": {
            "decoder_image": {"worst_plane": worst, "median_plane": median, "plane": plane, "psnr_db": psnr}}}, indent=1))
    print(f"-> {OUT / f'witness-moss-vae-f64-{name}.safetensors'}")


def stage_vae_final(size: str):
    """The trajectory's final latent through the same path; the image as the pipeline's 8-bit output, its md5 in the manifest."""
    _, _, name = common.parse_size(size)
    trajectory = read(f"goldens-moss-trajectory-{name}", f"moss-image.py trajectory --size {name}")
    vae = load_vae()
    z = denormalize_unpatchify(vae, trajectory["latent_final"][None])
    image = vae.decode(z, return_dict=False)[0].clamp(-1, 1)
    pil = reference().VaeImageProcessor(vae_scale_factor=16).postprocess(image, output_type="pil")[0]
    md5 = hashlib.md5(pil.tobytes()).hexdigest()
    write(f"goldens-moss-vae-final-{name}", {"latent": trajectory["latent_final"], "latent_z": z[0], "image": image[0]},
          {"image_md5_rgb8": md5, "note": "md5 of the pipeline's PIL image bytes (RGB, 8 bits); informative, never bit-exact elsewhere"})
    pil.save(OUT / f"reference-moss-{name}.png")
    print(f"  md5 of the RGB8 image: {md5}")


def everything(goldens: Path):
    stages = [["schedule"], ["tokenizer"], ["text"], ["vae", "--size", "512"], ["dit", "--size", "512"],
              ["dit", "--size", "512", "--lora"], ["text-witness"], ["vae-witness", "--size", "512"],
              ["witness", "--size", "512", "--lora"], ["witness", "--size", "512"], ["dit-bf16", "--size", "512"], ["bf16", "--size", "512"],
              ["trajectory", "--size", "512"], ["vae-final", "--size", "512"]]
    common.run_stages(__file__, goldens, stages)


def main() -> None:
    global OUT
    common.unbuffered()
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("stage", choices=["schedule", "tokenizer", "text", "text-witness", "vae", "vae-witness", "dit", "witness",
                                     "dit-bf16", "bf16", "trajectory", "vae-final", "all"])
    p.add_argument("--size", default="512", help="512 (the only size)")
    p.add_argument("--lora", action="store_true", help="dit, witness: the 8-step LoRA merged at strength 1")
    p.add_argument("--variant", choices=list(quantized.SOURCES["moss-image"]),
                   help="dit, witness, dit-bf16: the Compact's published DiT (Comfy-Org's int8 convrot) instead of the publisher's")
    p.add_argument("--goldens", help="output folder (default: $SILICONED_GOLDENS, else ./goldens)")
    args = p.parse_args()
    global VARIANT
    if args.variant and (args.stage not in ("dit", "witness", "dit-bf16") or args.lora):
        p.error("--variant goes with dit, witness or dit-bf16, without --lora (the Compact takes the Standard's LoRA)")
    VARIANT = args.variant
    OUT = common.goldens_directory(args.goldens) / "moss"
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "all":
        everything(common.goldens_directory(args.goldens))
    elif args.stage == "dit":
        stage_dit(args.size, args.lora)
    elif args.stage == "witness":
        stage_witness(args.size, args.lora)
    else:
        {"schedule": stage_schedule, "tokenizer": stage_tokenizer, "text": stage_text, "text-witness": stage_text_witness,
         "vae": lambda: stage_vae(args.size), "vae-witness": lambda: stage_vae_witness(args.size),
         "bf16": lambda: stage_bf16(args.size), "dit-bf16": lambda: stage_dit_bf16(args.size), "trajectory": lambda: stage_trajectory(args.size),
         "vae-final": lambda: stage_vae_final(args.size)}[args.stage]()


if __name__ == "__main__":
    main()
