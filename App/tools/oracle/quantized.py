"""The Compact DiTs as the oracle reads them: the oracle's weights for `--variant compact`.

A Compact is a third party's file, ComfyUI's int8 `convrot` (Comfy-Org's Z-Image, Qwen-Image-2.1, Krea 2 and
MOSS-Image, Bedovyy's Anima), pinned below as the app pins it (the family's `…Modules.swift`: repository,
revision, sha256). Its oracle is the reference DiT of diffusers, in float32 on the CPU, with **those published
weights, decoded by code that is not the engine's**: `dequantize_int8_convrot_weight` of comfy_kitchen at the
frozen version `fetch_comfy` downloads and hashes (`(q * scale)` rotated back by the Hadamard blocks, in float32).

Tensors the file does not quantize (norms, embedders, the F32/F16/BF16 ones) are read as stored. Names go from
the publisher's to diffusers' by `to_map_names` (diffusers' own converters where it has them). Nothing is held
whole: `Variant` has `Checkpoint`'s interface (`keys`, `tensor`, `under`), so `common.Streamed` brings one block
in, dequantized, and drops it, as for the published bf16 weights.

The bf16 bar (`bf16 --variant compact`) is the float32 weights rounded to bf16 once, the nearest bf16 can be to
the published weights: diffusers in bf16 on those weights.

The W8A8 bar (`w8a8 --variant compact`) is the publisher's own forward, never a dequantization: diffusers in bf16
as for the bf16 bar, every linear the file quantizes computed by ComfyUI's default int8 product (`Int8Products`,
below, where the recipe is read line by line).
"""
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

import torch

sys.dont_write_bytecode = True

# family -> variant -> where the app reads the DiT: the pins of the family's `…Modules.swift`.
SOURCES = {
    "z-image": {
        "compact": dict(repo="Comfy-Org/Z-Image-Turbo", revision="6fc90a3b1b653e935a0d175e260736de25b84df5",
                        file="split_files/diffusion_models/z_image_turbo_int8_convrot.safetensors", format="convrot",
                        sha256="be517ebd47c912a5626a588e1aeea43e6be4a43c0cdcd2b48a2a780d9f358635"),
    },
    "qwen-image-2.1": {
        "compact": dict(repo="Comfy-Org/Qwen-Image-2.1", revision="df94239739eef7205973a145ffa6f441f7b64e84",
                        file="diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors", format="convrot",
                        sha256="8144a0182964aa8403ad6dbc011db837304167d235161b9aabaa8532e967314c"),
    },
    "anima": {
        "compact": dict(repo="Bedovyy/Anima-INT8", revision="d7ba6686f161e9a829373106eb4b881327247ba9",
                        file="anima-turbo-v1.1-int8convrot.safetensors", format="convrot",
                        sha256="c462bea054de87ea03000e83f2b60e6d58fbcb6b925d9f607f8a470eeaad52b4"),
    },
    "krea2": {
        "compact": dict(repo="Comfy-Org/Krea-2", revision="1171f4dc7e325bf3734586942e737872506c73e1",
                        file="diffusion_models/krea2_turbo_int8_convrot.safetensors", format="convrot",
                        sha256="8e4eeda70dd5037ab1ba2bef6b417f9f901e26093117cf397f741fc1fdaaf3f1"),
    },
    "moss-image": {
        "compact": dict(repo="Comfy-Org/MOSS-Image", revision="530a4c94f92c35ef6818b1b035a983fe9dcf5f54",
                        file="diffusion_models/moss_image_sft_int8_convrot.safetensors", format="convrot",
                        sha256="b4d1475bc7eeffd7b809e82ea13a1fba1fb85c4e18365af3813722faa5c0fe4f"),
    },
}
VARIANTS = ("compact",)

# Tensors of a published file that belong to no diffusers DiT: Anima's text adapter (the conditioner, not the
# DiT) and the two tensors Krea left in its checkpoint, an easter egg the model never reads (`last.down` and
# `last.up`, which the app does not read either). Dropped, and listed in the golden's manifest.
OUTSIDE = ("llm_adapter.", "model.diffusion_model.llm_adapter.", "last.down.weight", "last.up.weight")


def stem(base: str, variant: str | None) -> str:
    """The golden's file stem for a variant: `goldens-dit-512` -> `goldens-dit-compact-512`. The variant goes
    before the size, the way `siliconed check` names it; the Standard (None) keeps the name."""
    if variant is None:
        return base
    head, _, size = base.rpartition("-")
    return f"{head}-{variant}-{size}"


def pinned(source: dict) -> Path:
    """A pinned file (`repo`, `revision`, `file`, `sha256`) in the Hugging Face cache, downloaded on first use, its
    sha256 checked."""
    from huggingface_hub import hf_hub_download
    path = Path(hf_hub_download(source["repo"], source["file"], revision=source["revision"]))
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 24):
            digest.update(block)
    if digest.hexdigest() != source["sha256"]:
        raise SystemExit(f"{path}: sha256 {digest.hexdigest()}, the app pins {source['sha256']}")
    return path


def to_map_names(family: str, tensors: dict) -> dict:
    """Published names -> diffusers' (the DiT's state dict), by diffusers' own converter where it has one. Anima's
    go by the oracle's own table."""
    if family == "anima":
        import anima
        return {anima.diffusers_name(k.removeprefix("model.diffusion_model.")): v for k, v in tensors.items()}
    if family == "moss-image":
        # The reference's own names; Comfy-Org fuses each block's `proj.qkv` = [q; k; v] on the output axis.
        out = {}
        for k, v in tensors.items():
            if k.startswith("layers.") and k.endswith(".proj.qkv.weight"):
                for part, x in zip(torch.chunk(v, 3, dim=0), "qkv"):
                    out[k.replace(".proj.qkv.", f".proj.{x}.")] = part
            else:
                out[k] = v
        return out
    if family == "z-image":
        from diffusers.loaders.single_file_utils import convert_z_image_transformer_checkpoint_to_diffusers
        return convert_z_image_transformer_checkpoint_to_diffusers(dict(tensors))
    if family == "krea2":
        from diffusers.loaders.single_file_utils import convert_krea2_transformer_checkpoint_to_diffusers
        return convert_krea2_transformer_checkpoint_to_diffusers(dict(tensors))
    # Qwen-Image-2.1: diffusers' names under a prefix, each block's `gate_up` = [gate; proj] on the output axis.
    out = {}
    for k, v in tensors.items():
        k = k.removeprefix("model.diffusion_model.")
        if k.endswith(".img_mlp.gate_up.weight"):
            gate, proj = torch.chunk(v, 2, dim=0)
            out[k.replace(".gate_up.", ".gate_layer.")] = gate
            out[k.replace(".gate_up.", ".proj.")] = proj
        else:
            out[k] = v
    return out


# ================================================================================== ComfyUI's frozen files
#
# `comfy_quant` = {"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}: the weight stored is
# `W·R`, quantized per output row (`weight` I8 [N, K], `weight_scale` F32 [N, 1]). What the publisher's code does,
# read at the frozen revisions of `COMFY` (downloaded by `fetch_comfy`, sha256 in its manifest):
#
# * ComfyUI `comfy/ops.py` (`_load_quantized_module`): reads `convrot` and `convrot_groupsize` (default 256) from the
#   layer's JSON into `TensorWiseINT8Layout.Params`; `comfy/quant_ops.py` aliases that layout to comfy_kitchen's.
# * comfy_kitchen `tensor/int8_utils.py`: `_build_hadamard(G)` = the Kronecker power of
#   H4 = [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]] (the "regular" Hadamard, not Sylvester's), divided by sqrt(G);
#   G must be a power of 4. Symmetric, orthogonal. `_rotate_weight(W, h, G)` = W grouped [N, K/G, G] @ h.T (raises
#   when G does not divide K: no partial group exists).
# * comfy_kitchen `backends/eager/quantization.py`: `quantize_int8_convrot_weight` = rowwise quantization of
#   `_rotate_weight(W)`; `dequantize_int8_convrot_weight(q, s, G)` = `_rotate_weight(q.float() * s, h)`, in fp32 —
#   what `Variant` reads; `tensor/int8.py` `dequantize` then casts to the layer's `orig_dtype` (the compute dtype,
#   bf16 usually).
# * The default forward (`int8_linear`, ops.py `_int8_linear` and the `aten.linear` handler) never dequantizes: x
#   rotated in its own dtype (`_rotate_activation`: x grouped @ h), then quantized to int8 per row, int8 × int8 →
#   int32 GEMM, times s_x · s_w. A LoRA patched on the fly makes the cast hand back the dequantized weight instead.
#
# So W = (q·s)·Rᵀ = (q·s)·R (R symmetric), y = x·Wᵀ = (x·R)·(q·s)ᵀ.

COMFY = dict(
    comfyui=dict(repo="Comfy-Org/ComfyUI", commit="b0b743566f65daafc423b4fea8a2fbda94b3384a",
                 files=["comfy/quant_ops.py", "comfy/ops.py", "comfy/supported_models.py", "comfy/ldm/lumina/model.py"]),
    kitchen=dict(version="0.2.37", wheel="comfy_kitchen-0.2.37-py3-none-any.whl",
                 url="https://files.pythonhosted.org/packages/9e/08/0a904ae1a1b82f57e7990eee8d93d48466a185934aa585cc625cdb27761e/"
                     "comfy_kitchen-0.2.37-py3-none-any.whl",
                 sha256="b626bd51ff45937f1446accf078ccea9d11c3686ffa0c89b0c1305d66d765afe",
                 files=["comfy_kitchen/tensor/int8.py", "comfy_kitchen/tensor/int8_utils.py",
                        "comfy_kitchen/backends/eager/quantization.py",
                        # The W8A8 bar's (`Int8Products`): what int8_linear calls, and what it cites.
                        "comfy_kitchen/backends/_activations.py", "comfy_kitchen/tensor/base.py",
                        "comfy_kitchen/backends/triton/quantization.py", "comfy_kitchen/backends/triton/__init__.py",
                        "comfy_kitchen/backends/eager/__init__.py"]),
)
#: Where the frozen files are kept: beside the oracle's environment, which git ignores (ComfyUI is GPL-3,
#: comfy_kitchen Apache-2.0: neither is versioned here). `SILICONED_COMFY_DIR` names another place.
COMFY_DIR = os.environ.get("SILICONED_COMFY_DIR") or str(Path(__file__).resolve().parents[2] / "oracle-env" / "comfy")


def fetch_comfy() -> dict:
    """The publisher's files at their frozen revisions, into `COMFY_DIR`; their sha256 go into its manifest.json."""
    import io
    import zipfile
    os.makedirs(COMFY_DIR, exist_ok=True)
    files = {}

    def fetch(url):
        with urllib.request.urlopen(urllib.request.Request(url), timeout=120) as r:
            return r.read()

    def keep(path, data, origin):
        out = os.path.join(COMFY_DIR, path)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "wb") as f:
            f.write(data)
        files[path] = dict(origin=origin, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())

    c = COMFY["comfyui"]
    for p in c["files"]:
        url = f"https://raw.githubusercontent.com/{c['repo']}/{c['commit']}/{p}"
        keep(f"comfyui/{p}", fetch(url), url)
    k = COMFY["kitchen"]
    wheel = fetch(k["url"])
    assert hashlib.sha256(wheel).hexdigest() == k["sha256"], "comfy_kitchen wheel: sha256 differs"
    with zipfile.ZipFile(io.BytesIO(wheel)) as z:
        for p in k["files"]:
            keep(f"kitchen/{p}", z.read(p), f"{k['wheel']}!{p}")
    manifest = dict(COMFY, downloaded=files)
    with open(os.path.join(COMFY_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
        f.write("\n")
    return manifest


def comfy_functions():
    """Comfy's own functions, loaded from the downloaded files (fetched on first use): `_build_hadamard`,
    `_rotate_weight` (int8_utils.py, imported whole: it needs torch only) and `dequantize_int8_convrot_weight`,
    `dequantize_int8_simple` (their source taken verbatim from eager/quantization.py with `ast` — the module itself
    imports half of comfy_kitchen). Returns (the utils module, `dequantize_int8_convrot_weight`)."""
    import ast
    import importlib.util
    base = os.path.join(COMFY_DIR, "kitchen", "comfy_kitchen")
    if not os.path.exists(os.path.join(base, "tensor", "int8_utils.py")):
        fetch_comfy()       # the pinned files, their sha256 checked
    spec = importlib.util.spec_from_file_location("comfy_int8_utils", os.path.join(base, "tensor", "int8_utils.py"))
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)
    src = open(os.path.join(base, "backends", "eager", "quantization.py")).read()
    wanted = {"dequantize_int8_simple", "dequantize_int8_convrot_weight"}
    nodes = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    assert {n.name for n in nodes} == wanted
    ns = dict(torch=torch, _build_hadamard=utils._build_hadamard, _rotate_weight=utils._rotate_weight)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "quantization.py", "exec"), ns)
    return utils, ns["dequantize_int8_convrot_weight"]


class Variant:
    """A published quantized DiT read like a `common.Checkpoint`: map (diffusers) names, one tensor at a time."""

    def __init__(self, family: str, variant: str):
        from safetensors import safe_open
        self.family, self.variant = family, variant
        self.source = SOURCES[family][variant]
        self.path = pinned(self.source)
        self.dropped: list[str] = []
        self._file = safe_open(str(self.path), framework="pt")
        keys = list(self._file.keys())
        self._quantized = {k for k in keys if k.endswith(".weight") and k[:-len(".weight")] + ".comfy_quant" in keys}
        # Published name -> the map names it yields, found by converting `meta` tensors of its shape.
        self._where: dict[str, str] = {}
        for name in keys:
            if name.endswith((".weight_scale", ".comfy_quant", ".comfy_attention.config")):
                continue        # a scale or a JSON text beside a weight, ComfyUI's attention preference: no tensor
            if name.startswith(OUTSIDE):
                self.dropped.append(name)
                continue
            shape = list(self._file.get_slice(name).get_shape())
            for mapped in to_map_names(self.family, {name: torch.empty(shape, device="meta")}):
                self._where[mapped] = name
        _, self._comfy_dequantize = comfy_functions()
        self.keys = list(self._where)
        self._cache: tuple[str, dict] | None = None
        self._int8_cache: tuple[str, dict, dict, int] | None = None

    # ----------------------------------------------------------------------------- one tensor

    def _convrot(self, name: str) -> torch.Tensor:
        if name not in self._quantized:
            return self._file.get_tensor(name).float()
        layer = name[:-len(".weight")]
        config = json.loads(bytes(self._file.get_tensor(layer + ".comfy_quant").tolist()).decode())
        assert config.get("format") == "int8_tensorwise" and config.get("convrot") is True, config
        group = int(config.get("convrot_groupsize", 256))
        return self._comfy_dequantize(self._file.get_tensor(name), self._file.get_tensor(layer + ".weight_scale"), group)

    def published(self, name: str) -> dict:
        """The map tensors of one published tensor (a fused one yields several), float32, the last one kept."""
        if self._cache and self._cache[0] == name:
            return self._cache[1]
        converted = to_map_names(self.family, {name: self._convrot(name)})
        self._cache = (name, converted)
        return converted

    def is_int8(self, key: str) -> bool:
        """The map tensor `key` (a `….weight`) comes from a weight the file publishes in int8 convrot."""
        return self._where.get(key) in self._quantized

    def int8(self, key: str) -> tuple[torch.Tensor, torch.Tensor, int]:
        """The published integers of the map weight `key`, as stored: `weight` I8 [n, K], `weight_scale` F32 [n, 1] and the
        rotation's group, the rows of a fused tensor (`qkv`) split as the weight is (by the same converter: a split by
        output rows leaves each row's integers and scale whole). The last published tensor is kept: q, k, v read it once."""
        name = self._where[key]
        if not (self._int8_cache and self._int8_cache[0] == name):
            layer = name[:-len(".weight")]
            config = json.loads(bytes(self._file.get_tensor(layer + ".comfy_quant").tolist()).decode())
            assert config.get("format") == "int8_tensorwise" and config.get("convrot") is True, config
            q, s = self._file.get_tensor(name), self._file.get_tensor(layer + ".weight_scale")
            assert q.dtype == torch.int8 and s.dtype == torch.float32 and s.shape == (q.shape[0], 1), (name, q.dtype, s.shape)
            self._int8_cache = (name, to_map_names(self.family, {name: q}), to_map_names(self.family, {name: s}),
                                int(config.get("convrot_groupsize", 256)))
        _, qs, ss, group = self._int8_cache
        return qs[key], ss[key], group

    def tensor(self, key: str, dtype: torch.dtype) -> torch.Tensor:
        t = self.published(self._where[key])[key]
        # Rounded once from the float32 dequantization (the bf16 bar), or widened: a fresh tensor either way.
        return t.to(dtype).clone() if t.is_floating_point() else t.clone()

    def under(self, prefix: str, dtype: torch.dtype) -> dict:
        n = len(prefix)
        return {k[n:]: self.tensor(k, dtype) for k in self.keys if k.startswith(prefix)}

    def describe(self) -> dict:
        return {"variant": self.variant, "repository": self.source["repo"], "revision": self.source["revision"],
                "file": self.source["file"], "sha256": self.source["sha256"], "format": self.source["format"],
                "dequantized_by": "comfy_kitchen dequantize_int8_convrot_weight", "dropped": self.dropped}


# ================================================================================== the W8A8 bar
#
# What ComfyUI computes for a layer of a Compact (`comfy_quant` = {"format": "int8_tensorwise", "convrot": true,
# "convrot_groupsize": 256}), read at the frozen revisions of `COMFY` (ComfyUI b0b74356, comfy_kitchen
# 0.2.37), the files `fetch_comfy` downloads and hashes:
#
#   comfy/quant_ops.py:238    QUANT_ALGOS["int8_tensorwise"] has "quantize_input": False: the input is not turned into a
#                             QuantizedTensor before the call (the fp8 formats do that, not this one).
#   comfy/ops.py:1360         mixed_precision_ops(compute_dtype=torch.bfloat16); a layer without `comfy_quant` is cast to
#                             it at load (ops.py:1200) — every plain F32 tensor of a Compact runs in bf16. The model's
#                             dtype is bf16 first (comfy/supported_models.py, ZImage: [bfloat16, float32]).
#   comfy/ops.py:1485, 1418–1430  `weight_only_quant`: the bias cast to the input's dtype, the weight kept quantized
#                             (`.to(input.dtype)` only relabels its `orig_dtype`, comfy_kitchen tensor/base.py `_handle_to`),
#                             then `F.linear(input, weight, bias)`.
#   comfy_kitchen tensor/int8.py:265–296  the `aten.linear` handler: `int8_linear(x.contiguous(), q, s, bias,
#                             out_dtype = x.dtype, convrot, convrot_groupsize)`. No dequantization on this path.
#   comfy_kitchen backends/eager/quantization.py:985 `int8_linear` — the backend of the CPU and of MPS (a Mac's
#                             ComfyUI runs this one; Triton's, on CUDA and XPU, is discussed below):
#     :1041  h = _build_hadamard(256, dtype=x.dtype) — the regular Hadamard (Kronecker powers of H4, tensor/int8_utils.py:24)
#            / 16, its entries ±1/16 exact in bf16;
#     :1042  x = _rotate_activation(x, h, 256) — int8_utils.py:70–72, `x.reshape(-1, K/256, 256) @ h` IN x's DTYPE: a bf16
#            matmul (fp32 sums, the result rounded to bf16 once);
#     :1048  quantize_int8_rowwise(x) — :873 `abs_max = x.abs().amax(-1)` (bf16, exact), :874 `scale = (abs_max.float()
#            / 127).clamp(min=1e-30)` (fp32), :875 `x / _int8_scale_for_math(scale, x)`: :817 the scale CAST TO bf16 for the
#            division, the quotient rounded to bf16, then :827 `.round_()` (half to even) `.clamp_(-128, 127)` — so −128
#            occurs (a quotient of −127.5 rounds to −128) and the row's scale used to multiply back (fp32) is not the one
#            divided by (bf16): a per-row relative bias up to 2⁻⁹, part of the recipe;
#     :1052  _int8_matmul_accumulate(x_8, q.T) — torch._int_mm, int32 sums, exact (|Σ| ≤ 128·127·K < 2³¹ for K < 132 000);
#     :1065–1072  per chunk of rows: `result.float()` (int32 → fp32, round to nearest even past 2²⁴), times
#            `(s_x · s_w)` computed in fp32, the product rounded to fp32, then `.to(out_dtype)`: bf16;
#     :1078  `result + bias.to(bf16)`: a bf16 addition; ends in `_apply_residual` (none here).
#   The ZImage graph (comfy/ldm/lumina/model.py, JointAttention, FeedForward, JointTransformerBlock) calls these layers as
#   plain `Linear`s: no input activation or residual folded in (`linear_input_act` is not used there), `qkv` fused
#   (one quantization of x for the three: the same integers as three products on the same x).
#
# Which layers: the 202 layers carrying `comfy_quant` (every block's qkv, out, w1, w2, w3 and adaLN_modulation, the
# refiners' likewise); the embedders, the final layer, the norms and the pad tokens are F32 in the file and run in bf16.
#
# The Triton backend (backends/triton/quantization.py:1020, backends/triton/__init__.py:239) differs in rounding only: its quantizer (:801–812) divides
# by the bf16 scale too and rounds with `rint` (half to even), but its epilogue (:1007–1012) keeps `acc · s_x · s_w +
# bias` in fp32 and rounds once to bf16 — one bf16 rounding fewer than the eager path. This bar is the eager path.
#
# Everything else is diffusers in bf16, the bf16 bar's arithmetic (`--variant compact bf16`): the bar measures the
# int8 activations, not ComfyUI's attention or norms.

def comfy_int8_linear():
    """comfy_kitchen's eager `int8_linear` and the helpers it calls, their source taken verbatim from the downloaded
    `backends/eager/quantization.py` (`ast`: the module imports half of comfy_kitchen), the rotation from
    `tensor/int8_utils.py` and the input activation from `backends/_activations.py` (both imported whole: torch only).
    Returns (`int8_linear`, Comfy's own `_int8_matmul_accumulate`); the integer product inside `int8_linear` is
    `exact_int8_product`, equal bit for bit (checked on the first layer by `Int8Products`)."""
    import ast
    import importlib.util
    base = Path(COMFY_DIR) / "kitchen" / "comfy_kitchen"
    if not (base / "backends" / "_activations.py").exists():
        fetch_comfy()

    def module(name, path):
        spec = importlib.util.spec_from_file_location(name, str(path))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m
    utils = module("comfy_int8_utils", base / "tensor" / "int8_utils.py")
    activations = module("comfy_activations", base / "backends" / "_activations.py")
    source = (base / "backends" / "eager" / "quantization.py").read_text()
    wanted = {"int8_linear", "quantize_int8_rowwise", "_round_int8", "_int8_scale_for_math", "_int8_stochastic_rng",
              "_int8_matmul_accumulate", "_round_up", "_int8_mm_n_alignment", "_cuda_device_is_turing"}
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    assert {n.name for n in nodes} == wanted, wanted - {n.name for n in nodes}
    ns = dict(torch=torch, _build_hadamard=utils._build_hadamard, _rotate_activation=utils._rotate_activation,
              _apply_input_act=activations.apply_input_act, _apply_residual=activations.apply_residual,
              _turing_device_cache={})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "quantization.py", "exec"), ns)
    comfy_product = ns["_int8_matmul_accumulate"]
    ns["_int8_matmul_accumulate"] = exact_int8_product
    return ns["int8_linear"], comfy_product


def exact_int8_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """int8 [M, K] × int8 [K, N] → int32 [M, N], in fp64: every product (|·| ≤ 2¹⁴) and every partial sum is an integer
    below 2⁵³, so the result is exact in any order, and below 2³¹ it is the int32 GEMM's to the bit."""
    assert a.dtype == torch.int8 and b.dtype == torch.int8
    out = a.double() @ b.double()
    assert out.abs().max() < 2 ** 31
    return out.to(torch.int32)


class Int8Forward:
    """One linear's forward replaced by ComfyUI's: `int8_linear` on the published integers of its weight (read at the
    call, dropped after), its bias as the module holds it (bf16, as ComfyUI casts it). A callable object, not a lambda."""

    def __init__(self, products: "Int8Products", module: torch.nn.Linear, key: str):
        self.products, self.module, self.key = products, module, key

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dtype == torch.bfloat16, (self.key, x.dtype)       # the recipe's precondition in this bar
        q, s, group = self.products.variant.int8(self.key)
        bias = self.module.bias
        self.products.check_product(x, q, group)
        out = self.products.linear(x.contiguous(), q.contiguous(), s, None if bias is None else bias.to(x.dtype),
                                   x.dtype, convrot=True, convrot_groupsize=group)
        self.products.calls += 1
        return out


class Int8Products:
    """Every `nn.Linear` of `model` whose weight the Variant publishes in int8 convrot computes as ComfyUI does by default
    (`comfy_int8_linear`); the others are untouched. `model` is built in bf16 on the Variant (`Streamed` still brings in
    the dequantized weights of these layers; they are never read)."""

    def __init__(self, model: torch.nn.Module, variant: Variant):
        self.variant = variant
        self.linear, self._comfy_product = comfy_int8_linear()
        self.calls, self.checked = 0, False
        self.layers = []
        for name, m in model.named_modules():
            if isinstance(m, torch.nn.Linear) and variant.is_int8(f"{name}.weight"):
                m.forward = Int8Forward(self, m, f"{name}.weight")
                self.layers.append(name)
        published = {variant._where[k] for k in variant.keys if variant.is_int8(k)}
        mapped = {variant._where[f"{n}.weight"] for n in self.layers}
        assert mapped == published, f"int8 layers without a module: {sorted(published - mapped)[:4]}"

    def check_product(self, x: torch.Tensor, q: torch.Tensor, group: int) -> None:
        """Once: the exact fp64 product against Comfy's own (`torch._int_mm`), on the integers of the first layer."""
        if self.checked:
            return
        h = self.linear.__globals__["_build_hadamard"](group, device=x.device, dtype=x.dtype)
        rotated = self.linear.__globals__["_rotate_activation"](x.reshape(-1, x.shape[-1]), h, group)
        x8, _ = self.linear.__globals__["quantize_int8_rowwise"](rotated)
        assert torch.equal(exact_int8_product(x8, q.T.contiguous()), self._comfy_product(x8, q.T.contiguous())), \
            "the exact integer product differs from torch._int_mm"
        self.checked = True

    def describe(self) -> dict:
        return {"products": "comfy_kitchen 0.2.37 eager int8_linear (activation rotated by the regular Hadamard 256 in "
                            "bf16, quantized to int8 per row, int8 x int8 -> int32 exact, times s_x * s_w, bf16 out)",
                "layers": len(self.layers), "calls": self.calls, "rest": "diffusers in bf16"}
