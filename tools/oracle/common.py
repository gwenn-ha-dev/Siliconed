"""What the oracle scripts share: where goldens go, how they are written, which reference they read.

A golden is the reference implementation's answer (diffusers / transformers, float32, on the CPU)
frozen in a safetensors file, with a `.json` manifest beside it. `siliconed check <model> <stage>`
judges the engine against it. The reference runs on the CPU in float32 on purpose: an oracle must
not depend on a device's arithmetic, and must not be a reduced-precision oracle without saying so.

Everything here is import-light on purpose (torch and safetensors only); the scripts import
diffusers or transformers themselves.
"""
import json
import os
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# Revisions are pinned: a repository that moves under the same name would change the goldens
# without any code changing. Bump one only to adopt the new weights, and regenerate everything.
REVISIONS = {
    "Tongyi-MAI/Z-Image-Turbo": "f332072aa78be7aecdf3ee76d5c247082da564a6",
    "Qwen/Qwen-Image-2.1-Turbo": "d65dbc9a7e8f6b5479e33dee6030eaab2a906509",
    # Anima is published in two repositories: the diffusers layout of the base model carries the
    # tokenizers, the encoder, the conditioner's and the transformer's configuration and the VAE;
    # the Turbo weights (DiT and conditioner in one ComfyUI-format file) live in the other.
    "circlestone-labs/Anima-Base-v1.0-Diffusers": "073c3a9db359c31ad0e8aa268d15775473c2176c",
    "circlestone-labs/Anima": "f973fc41ec7545364ac9776c2440285f43ff2a30",
    "krea/Krea-2-Turbo": "98e0fe118d17c9e3547fbb2e25acdbae2cadf7c7",
    # MOSS-Image: the reference code is a file of the repository (`pipeline_moss_image.py`), not diffusers';
    # the 8-step turbo is a community LoRA published in a second repository.
    "OpenMOSS-Team/MOSS-Image-SFT": "f972bd4b4b9e3852133cc5da954d87bc32cd8fb9",
    "lodestones/MOSS-Image-SFT-8-steps-dmd": "f95df83b5e90ae91ea412434b3f9b7f1c8c6314f",
}

# The long prompt of `text --long`: 844 Qwen tokens, 954 T5 tokens, past the 512 the reference pipelines
# cut at by default and inside the 1,024 the engine reads (template excluded). The pipelines are given
# `max_sequence_length=LONG_SEQUENCE_LENGTH`, so nothing of it is cut on either side.
LONG_PROMPT = (Path(__file__).resolve().parent / "inputs" / "long-prompt.txt").read_text().strip()
LONG_SEQUENCE_LENGTH = 1024

# ------------------------------------------------------------------------------------ the folder


def goldens_directory(argument: str | None = None) -> Path:
    """`--goldens DIR`, else `SILICONED_GOLDENS`, else `./goldens`: the engine's own rule, so a
    folder written here is the folder `siliconed check` reads without being told."""
    directory = Path(argument or os.environ.get("SILICONED_GOLDENS") or "goldens")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def provide_inputs(goldens: Path) -> Path:
    """The test pictures (`tools/oracle/inputs/`), copied beside the goldens, where `siliconed check`
    reads them: the oracle and the engine read the very same files. A file already there is left alone."""
    import shutil
    inputs = goldens / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    for picture in sorted((Path(__file__).resolve().parent / "inputs").glob("*.png")):
        if not (inputs / picture.name).exists():
            shutil.copyfile(picture, inputs / picture.name)
    return inputs


def tensor_summary(tensors: dict, *, extra: bool = False) -> dict:
    """Shape and norm of each tensor: enough to recognise a golden at a glance, never judged."""
    out = {}
    for name, t in tensors.items():
        entry = {"shape": list(t.shape), "dtype": str(t.dtype)}
        if t.is_floating_point():
            entry["l2"] = float(t.double().pow(2).sum().sqrt())
            if extra:
                entry["mean"] = float(t.double().mean())
                entry["absmax"] = float(t.abs().max())
        out[name] = entry
    return out


def write_golden(directory: Path, stem: str, tensors: dict, manifest: dict, *,
                 metadata: dict | None = None, summary_extra: bool = False) -> Path:
    """`<stem>.safetensors` and `<stem>.json`. Tensors are made contiguous: safetensors refuses
    a strided view, and a view would also tie the file to its source's storage."""
    tensors = {k: v.contiguous() for k, v in tensors.items()}
    path = directory / f"{stem}.safetensors"
    save_file(tensors, str(path), metadata=metadata)
    manifest = {**manifest, "torch": torch.__version__,
                "tensors": tensor_summary(tensors, extra=summary_extra)}
    (directory / f"{stem}.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"-> {path} ({len(tensors)} tensors)")
    return path


def read_golden(directory: Path, stem: str, producer: str) -> dict:
    """A golden an earlier stage wrote, or an exit that says which command writes it."""
    path = directory / f"{stem}.safetensors"
    if not path.exists():
        raise SystemExit(f"{path} is missing: write it first with `{producer}`")
    from safetensors.torch import load_file
    return load_file(str(path))


# --------------------------------------------------------------------------------- the reference


def snapshot(repository: str, components: list[str], files: list[str] = ()) -> Path:
    """The pinned revision of `repository`, restricted to the given sub-folders (and, beside them, single
    files such as `model_index.json`), in the standard Hugging Face cache (HF_HOME / HF_HUB_CACHE apply).
    Returns the snapshot's root.

    Some repositories are gated: `hf auth login` first. Nothing is downloaded that a stage does not
    name, since the full repositories weigh tens of gigabytes: a repository that also publishes weights the
    stage does not read (Anima's base model carries a 4 GB transformer) is asked for its files by name."""
    from huggingface_hub import hf_hub_download, snapshot_download
    revision = REVISIONS[repository]
    root = None
    if components:
        root = Path(snapshot_download(repository, revision=revision, allow_patterns=[f"{c}/*" for c in components]))
    for name in files:
        path = Path(hf_hub_download(repository, name, revision=revision))
        root = root or path.parents[len(Path(name).parts) - 1]
    return root


def parse_size(text: str) -> tuple[int, int, str]:
    """`512` -> (height 512, width 512); `768x512` -> (height 512, width 768), written as the engine
    writes it: WIDTHxHEIGHT. A rectangle is what catches a height/width swap; at h = w every
    convention is right."""
    if "x" in text:
        width, height = (int(v) for v in text.split("x"))
        return height, width, text
    return int(text), int(text), text


def banner(*parts) -> None:
    print(*parts, flush=True)


# ------------------------------------------------------------------------------ streamed weights


class Checkpoint:
    """A sharded safetensors checkpoint (`*.safetensors.index.json`) or a single `.safetensors` file,
    read tensor by tensor.

    The published weights are bfloat16 and a Z-Image DiT is 24.6 GB once widened to float32 on a
    machine with 16: nothing is ever held whole. Widening happens on read, into a fresh tensor,
    because assigning bfloat16 tensors into a float32 module silently makes that module bfloat16
    and the oracle would be a half-precision oracle without saying so."""

    def __init__(self, directory: Path, index_name: str):
        if index_name.endswith(".safetensors"):             # one file: its own key order
            with safe_open(str(directory / index_name), framework="pt") as f:
                weight_map = {k: index_name for k in f.keys()}
        else:
            weight_map = json.loads((directory / index_name).read_text())["weight_map"]
        self.keys = list(weight_map)                        # the index's order, which allocation follows
        self._shard = weight_map
        self._open = {shard: safe_open(str(directory / shard), framework="pt")
                      for shard in sorted(set(self._shard.values()))}

    def tensor(self, key: str, dtype: torch.dtype) -> torch.Tensor:
        t = self._open[self._shard[key]].get_tensor(key)
        return t.to(dtype) if t.is_floating_point() else t

    def under(self, prefix: str, dtype: torch.dtype) -> dict:
        """Every tensor below `prefix`, keyed relative to it, each freshly allocated."""
        n = len(prefix)
        return {k[n:]: self.tensor(k, dtype) for k in self.keys if k.startswith(prefix)}


class Streamed:
    """Brings a module's weights in just before its forward pass and drops them just after.

    The module is built on the `meta` device (no memory); `load_state_dict(assign=True)` swaps real
    tensors in, and swapping `meta` ones back frees them. Peak memory is one block, not the model.

    Allocation here is deliberate and not to be optimised: each load allocates fresh tensors. Reusing
    buffers across blocks is ~30 % faster and changes the goldens in the last bits (torch's float32
    CPU products depend on where their operands sit in memory), so a golden written with reuse is
    not the golden everyone else regenerates.

    The hooks are methods that return None, never lambdas: a forward pre-hook that returns a value
    REPLACES the module's positional arguments, and `load_state_dict` returns a named tuple that a
    lambda would hand to the block as its input."""

    def __init__(self, module: torch.nn.Module, prefix: str, checkpoint: Checkpoint, dtype: torch.dtype):
        self.module, self.prefix, self.checkpoint, self.dtype = module, prefix, checkpoint, dtype
        self.empty = {k: torch.empty(v.shape, dtype=v.dtype, device="meta")
                      for k, v in module.state_dict().items()}
        module.register_forward_pre_hook(self.load)
        module.register_forward_hook(self.drop)

    def load(self, _module, _inputs):
        self.module.load_state_dict(self.checkpoint.under(self.prefix, self.dtype), assign=True)

    def drop(self, _module, _inputs, _output):
        self.module.load_state_dict(self.empty, assign=True)


# ------------------------------------------------------------------------------------- recording


class Recorder:
    """Captures tensors at module boundaries through forward hooks.

    Complex tensors (RoPE's `freqs_cis`) are stored as `[..., 2]` real/imaginary, the layout
    `torch.view_as_real` gives and the engine reads; safetensors carries no complex type."""

    def __init__(self):
        self.tensors: dict[str, torch.Tensor] = {}

    def keep(self, name: str, value) -> None:
        if isinstance(value, (tuple, list)):
            value = value[0]
        if not torch.is_tensor(value):
            return                      # e.g. the attention mask, None at batch 1
        t = value.detach()
        if t.is_complex():
            t = torch.view_as_real(t)
        self.tensors[name] = t.to(torch.float32).contiguous().clone()

    def output(self, module, name: str) -> None:
        def hook(_m, _inputs, out):
            self.keep(name, out)
        module.register_forward_hook(hook)

    def input(self, module, name: str) -> None:
        def hook(_m, inputs):
            self.keep(name, inputs[0])
        module.register_forward_pre_hook(hook)

    def inputs(self, module, names: list[str]) -> None:
        """Several positional arguments of one module, for inputs that are the output of no module."""
        def hook(_m, inputs):
            for i, name in enumerate(names):
                if i < len(inputs):
                    self.keep(name, inputs[i])
        module.register_forward_pre_hook(hook)


def tap(store: dict, name: str, pick=lambda output: output[0]):
    """A forward hook that keeps `pick(output)` in `store[name]`. A function that returns None, never a
    lambda over an assignment: a hook that returns a value REPLACES the module's output."""
    def hook(_module, _inputs, output):
        store[name] = pick(output).clone()
    return hook


def centered_crop(latent: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """The centered `[C, height/8, width/8]` crop of a latent `[C, H, W]`: a real denoised latent of the
    shape under test, which is how a height/width swap becomes visible (any square hides it)."""
    h, w = height // 8, width // 8
    y0, x0 = (latent.shape[1] - h) // 2, (latent.shape[2] - w) // 2
    return latent[:, y0:y0 + h, x0:x0 + w].contiguous()


def per_plane(got: torch.Tensor, exact: torch.Tensor) -> tuple[float, float, int]:
    """The engine's `errorPerPlan`: relative L2 error per plane (first axis), the denominator floored
    at 1 % of the median plane norm. Returns (worst, median, index of the worst plane)."""
    planes = exact.shape[0]
    d = (got.double() - exact.double()).reshape(planes, -1).norm(dim=1)
    n = exact.double().reshape(planes, -1).norm(dim=1)
    r = d / n.clamp(min=0.01 * float(n.median()))
    return float(r.max()), float(r.sort().values[planes // 2]), int(r.argmax())


def save_preview(image: torch.Tensor, path: Path) -> None:
    """A `[3, H, W]` image in [-1, 1] as a PNG, to look at: tensors are compared and never opened, so a
    reference nobody has seen can be wrong in a way no norm shows."""
    from PIL import Image
    pixels = ((image.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 0).numpy()
    Image.fromarray(pixels).save(path)
    print(f"-> {path}")


def decoder_in_double() -> None:
    """Makes the Qwen-Image VAE (`AutoencoderKLQwenImage`, used by Anima and Krea 2) and Flux's
    (`AutoencoderKL`, Z-Image's: only the banded convolutions and attention concern it) computable in
    float64 on a 16 GB machine, for the decoder's witness. Four replacements, each the same arithmetic:

      - `QwenImageUpsample` casts to fp32 around a nearest-exact copy: a witness through it would be
        no better than fp32. The copy is made in double instead;
      - the frame cache keeps a clone of every convolution's input until the end of the pass (20 GB
        at 1024 in fp32): on ONE frame no clone is ever read again, so the cache keeps only its
        `"Rep"` markers (the upsampler's skip of `time_conv` stays the real one);
      - a causal k x k x k convolution on one frame preceded by its two frames of zeros is the 2D
        convolution of its LAST slice. Using it here does not assume it: the fp32 oracle runs the real
        3D convolution, and oracle-against-witness would be O(1) if the reduction were wrong;
      - convolutions in double go through im2col (up to 14 GB at 1024) and the bottleneck attention
        through a 16 384 x 16 384 matrix: both are cut into bands of output rows / query rows, which
        leaves each output's own sum untouched.

    Process-wide and not undoable: call it in a stage that does nothing else."""
    import torch.nn.functional as F
    import diffusers.models.autoencoders.autoencoder_kl_qwenimage as q
    f64 = torch.float64
    budget = 256 * 2**20     # bytes of im2col per band

    def conv2d_bands(x, w, b, stride, padding):
        (sh, sw), (ph, pw) = stride, padding
        x = F.pad(x, (pw, pw, ph, ph))
        kh, kw = w.shape[-2:]
        out_h, out_w = (x.shape[-2] - kh) // sh + 1, (x.shape[-1] - kw) // sw + 1
        rows = max(1, budget // (8 * x.shape[1] * kh * kw * out_w))
        bands = []
        for r0 in range(0, out_h, rows):
            r1 = min(out_h, r0 + rows)
            bands.append(F.conv2d(x[..., r0 * sh: (r1 - 1) * sh + kh, :], w, b, (sh, sw)))
        return torch.cat(bands, dim=-2)

    conv2d_forward = torch.nn.Conv2d._conv_forward

    def conv2d(self, x, w, b):
        if x.dtype != f64:
            return conv2d_forward(self, x, w, b)
        assert self.padding_mode == "zeros" and self.dilation == (1, 1) and self.groups == 1
        return conv2d_bands(x, w, b, self.stride, self.padding)
    torch.nn.Conv2d._conv_forward = conv2d

    causal_forward = q.QwenImageCausalConv3d.forward

    def causal(self, x, cache_x=None):
        if x.dtype != f64 or cache_x is not None or x.shape[2] != 1:
            return causal_forward(self, x, cache_x)
        pw, _, ph, _, pt, _ = self._padding
        assert pt == self.kernel_size[0] - 1 and self.stride[1:] == (1, 1)
        y = conv2d_bands(x[:, :, 0], self.weight[:, :, -1], self.bias, (1, 1), (ph, pw))
        return y[:, :, None]
    q.QwenImageCausalConv3d.forward = causal

    def upsample(self, x):
        return torch.nn.Upsample.forward(self, x)
    q.QwenImageUpsample.forward = upsample

    sdpa = F.scaled_dot_product_attention

    def attention(qq, k, v, *args, **kwargs):
        if qq.dtype != f64:
            return sdpa(qq, k, v, *args, **kwargs)
        # diffusers' `AttnProcessor2_0` (Flux's VAE) passes the defaults by name.
        assert not args and kwargs.get("attn_mask") is None and not kwargs.get("dropout_p", 0.0) \
            and not kwargs.get("is_causal", False) and kwargs.get("scale") is None
        scale = qq.shape[-1] ** -0.5
        return torch.cat([torch.softmax((qq[..., i: i + 2048, :] @ k.transpose(-1, -2)) * scale, dim=-1) @ v
                          for i in range(0, qq.shape[-2], 2048)], dim=-2)
    F.scaled_dot_product_attention = attention

    class Markers(list):
        """The frame cache of ONE frame: its markers (`"Rep"`), never its clones."""
        def __setitem__(self, i, value):
            super().__setitem__(i, value if isinstance(value, str) else None)
    clear = q.AutoencoderKLQwenImage.clear_cache

    def clear_cache(self):
        clear(self)
        self._feat_map, self._enc_feat_map = Markers(self._feat_map), Markers(self._enc_feat_map)
    q.AutoencoderKLQwenImage.clear_cache = clear_cache


def run_stages(script: str, goldens: Path, stages: list[list[str]]) -> None:
    """`all`: every stage of `script`, one process each, in order, stopping at the first failure.

    A process per stage is deliberate, not a convenience: the float64 witnesses patch the reference's
    classes for the rest of the process (RoPE, norms, the VAE's convolutions), memory goes back to the
    system between stages, and each golden is then written by exactly the command a reader would type
    to rebuild it alone, the one `siliconed check list` prints."""
    import subprocess
    for arguments in stages:
        command = [sys.executable, "-u", script, *arguments, "--goldens", str(goldens)]
        print(f"\n=== {Path(script).name} {' '.join(arguments)}", flush=True)
        subprocess.run(command, check=True)


def unbuffered() -> None:
    """Progress lines reach a redirected log as they happen: a twenty-minute run is not silent."""
    sys.stdout.reconfigure(line_buffering=True)
