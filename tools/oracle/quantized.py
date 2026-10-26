"""The Compact and Light DiTs as their publishers' own code reads them: the oracle's weights for `--variant`.

The Compact (8 bits) and the Light (4 to 6 bits) are third parties' files, pinned below as the app pins
them (`Forge/Installation.swift`: repository, revision, sha256). A variant's oracle is the reference
DiT of diffusers, in float32 on the CPU, with **those published weights, dequantized by the publisher's
library**, never by ours:

* GGUF (unsloth's Z-Image and Qwen-Image-2.1, vanes430's Anima, vantagewithai's Krea 2): `gguf-py`, llama.cpp's
  own reader, `gguf.quants.dequantize` on the stored blocks. diffusers' GGUF loader, asked for float32,
  gives the same bits on every tensor (`quant_fixtures.py` counts the differing bits and stops at one).
* ComfyUI's int8 convrot (Comfy-Org's Qwen-Image-2.1, Krea 2 and MOSS-Image): `dequantize_int8_convrot_weight` of
  comfy_kitchen at the frozen version `quant_fixtures.py` downloads and hashes (`(q * scale)` rotated back
  by the Hadamard blocks, in float32).

Tensors the file does not quantize (norms, embedders, the F32/F16/BF16 ones) are read as stored. Names go from
the publisher's to diffusers' by the tables of `quant_fixtures.py` (diffusers' own converters where it
has them). Nothing is held whole: `Variant` has `Checkpoint`'s interface (`keys`, `tensor`, `under`), so
`common.Streamed` brings one block in, dequantized, and drops it, as for the published bf16 weights.

The bf16 bar (`--variant X bf16`) is this same dequantization rounded to bf16 once, the nearest bf16 can be to
the published weights: diffusers in bf16 on those weights.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.dont_write_bytecode = True
import quant_fixtures as qf  # noqa: E402

# family -> variant -> where the app reads the DiT. Same pins as `Forge/Installation.swift`.
SOURCES = {
    "z-image": {
        "compact": dict(repo="unsloth/Z-Image-Turbo-GGUF", revision="6c80814333b7b6a70a2e5b469a7c6437ce65de0f",
                        file="z-image-turbo-Q8_0.gguf", format="gguf",
                        sha256="f163d60b0eb427469510b8226243d196574a18139a2e40c017409cfbda95ecfe"),
        "light": dict(repo="unsloth/Z-Image-Turbo-GGUF", revision="6c80814333b7b6a70a2e5b469a7c6437ce65de0f",
                      file="z-image-turbo-Q4_K_M.gguf", format="gguf",
                      sha256="e6494f87de6abaf6a561924f50317a5f271fc34bb4222aabbd801197df8f7daa"),
    },
    "qwen-image-2.1": {
        "compact": dict(repo="Comfy-Org/Qwen-Image-2.1", revision="df94239739eef7205973a145ffa6f441f7b64e84",
                        file="diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors", format="convrot",
                        sha256="8144a0182964aa8403ad6dbc011db837304167d235161b9aabaa8532e967314c"),
        "light": dict(repo="unsloth/Qwen-Image-2.1-Turbo-GGUF", revision="5d1f9f759ea877ae556cda88dbec958b62ad3809",
                      file="qwen-image-2.1-turbo-Q4_K_M.gguf", format="gguf",
                      sha256="8940e8b522af886e0c0f5afddf720e82dd223cb9a43580b5f61c4d740452ac20"),
    },
    "anima": {
        "compact": dict(repo="vanes430/Anima-Turbo-V1.1-GGUF", revision="efdf070de980a52fc03027d2f6ed543cc1d87a8f",
                        file="anima-turbo-v1.1-Q8_0.gguf", format="gguf",
                        sha256="74d6c9251f34bc4152fda460dd7befaa9cec79946c270a378fee3d33048153bd"),
        "light": dict(repo="vanes430/Anima-Turbo-V1.1-GGUF", revision="efdf070de980a52fc03027d2f6ed543cc1d87a8f",
                      file="anima-turbo-v1.1-Q4_K_M.gguf", format="gguf",
                      sha256="709fceeec6b4f0eb155330c83fd2539406e86c01e431407276a1f31095484782"),
    },
    "krea2": {
        "compact": dict(repo="Comfy-Org/Krea-2", revision="1171f4dc7e325bf3734586942e737872506c73e1",
                        file="diffusion_models/krea2_turbo_int8_convrot.safetensors", format="convrot",
                        sha256="8e4eeda70dd5037ab1ba2bef6b417f9f901e26093117cf397f741fc1fdaaf3f1"),
        "light": dict(repo="vantagewithai/Krea-2-Turbo-GGUF", revision="390847e2835c16bdf82303731e3d35b874760d89",
                      file="krea2_turbo-Q4_K_M.gguf", format="gguf",
                      sha256="bc12f539de7a7a6ddf9bac17ec6e5bfecd42c3517190f491131d8464a1034b40"),
    },
    "moss-image": {
        "compact": dict(repo="Comfy-Org/MOSS-Image", revision="530a4c94f92c35ef6818b1b035a983fe9dcf5f54",
                        file="diffusion_models/moss_image_sft_int8_convrot.safetensors", format="convrot",
                        sha256="b4d1475bc7eeffd7b809e82ea13a1fba1fb85c4e18365af3813722faa5c0fe4f"),
    },
}
VARIANTS = ("compact", "light")

# Tensors of a published file that belong to no diffusers DiT: Anima's text adapter (the conditioner, not the
# DiT) and the two tensors Krea left in its checkpoint, an easter egg the model never reads (`last.down` and
# `last.up`, which the engine's forge ignores too). Dropped, and listed in the golden's manifest.
OUTSIDE = ("llm_adapter.", "model.diffusion_model.llm_adapter.", "last.down.weight", "last.up.weight")


def stem(base: str, variant: str | None) -> str:
    """The golden's file stem for a variant: `goldens-dit-512` -> `goldens-dit-compact-512`. The variant goes
    before the size, the way `siliconed check` names it; the Standard (None) keeps the name."""
    if variant is None:
        return base
    head, _, size = base.rpartition("-")
    return f"{head}-{variant}-{size}"


def published_file(family: str, variant: str) -> tuple[Path, dict]:
    """The pinned file in the Hugging Face cache (downloaded on first use), its sha256 checked."""
    from huggingface_hub import hf_hub_download
    source = SOURCES[family][variant]
    path = Path(hf_hub_download(source["repo"], source["file"], revision=source["revision"]))
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 24):
            digest.update(block)
    if digest.hexdigest() != source["sha256"]:
        raise SystemExit(f"{path}: sha256 {digest.hexdigest()}, the app pins {source['sha256']}")
    return path, source


def to_map_names(family: str, tensors: dict) -> dict:
    """Published names -> diffusers' (the DiT's state dict). Anima's go by the oracle's own table, which unlike the
    fixtures' keeps the query and key projections: here every tensor is read, not a row slice."""
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
    return qf.to_map_names(family, tensors)


class Variant:
    """A published quantized DiT read like a `common.Checkpoint`: map (diffusers) names, one tensor at a time."""

    def __init__(self, family: str, variant: str):
        self.family, self.variant = family, variant
        self.path, self.source = published_file(family, variant)
        self.dropped: list[str] = []
        if self.source["format"] == "gguf":
            self._open_gguf()
        else:
            self._open_convrot()
        self.keys = list(self._where)
        self._cache: tuple[str, dict] | None = None

    # ------------------------------------------------------------------------------ the index

    def _index(self, shapes: dict[str, list[int]]):
        """Published name -> the map names it yields, found by converting `meta` tensors of its shape."""
        self._where: dict[str, str] = {}
        for name, shape in shapes.items():
            if name.startswith(OUTSIDE):
                self.dropped.append(name)
                continue
            for mapped in to_map_names(self.family, {name: torch.empty(shape, device="meta")}):
                self._where[mapped] = name

    def _open_gguf(self):
        from gguf import GGUFReader
        self._reader = GGUFReader(str(self.path))
        self._tensors = {t.name: t for t in self._reader.tensors}
        # `shape` is ggml's `ne`, fastest axis first: torch's shape is its reverse.
        self._index({n: [int(x) for x in reversed(t.shape)] for n, t in self._tensors.items()})

    def _open_convrot(self):
        from safetensors import safe_open
        self._file = safe_open(str(self.path), framework="pt")
        keys = list(self._file.keys())
        self._quantized = {k for k in keys if k.endswith(".weight") and k[:-len(".weight")] + ".comfy_quant" in keys}
        shapes = {}
        for k in keys:
            slice_ = self._file.get_slice(k)
            if k.endswith((".weight_scale", ".comfy_quant", ".comfy_attention.config")):
                continue        # a scale or a JSON text beside a weight, ComfyUI's attention preference: no tensor
            shapes[k] = list(slice_.get_shape())
        self._index(shapes)
        if not (Path(qf.COMFY_DIR) / "kitchen" / "comfy_kitchen" / "tensor" / "int8_utils.py").exists():
            qf.fetch_comfy()        # the pinned files, their sha256 checked
        _, self._comfy_dequantize = qf.comfy_functions()

    # ----------------------------------------------------------------------------- one tensor

    def _gguf(self, name: str) -> torch.Tensor:
        from gguf import GGMLQuantizationType as T
        from gguf.quants import dequantize
        t = self._tensors[name]
        shape = [int(x) for x in reversed(t.shape)]
        if t.tensor_type == T.F32:
            return torch.from_numpy(np.array(t.data, dtype=np.float32)).reshape(shape)
        if t.tensor_type == T.F16:
            return torch.from_numpy(np.array(t.data, dtype=np.float16)).reshape(shape).float()
        if t.tensor_type == T.BF16:
            raw = np.array(t.data).view(np.int16)
            return torch.from_numpy(raw).view(torch.bfloat16).reshape(shape).float()
        return torch.from_numpy(np.ascontiguousarray(dequantize(t.data, t.tensor_type))).reshape(shape)

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
        value = self._gguf(name) if self.source["format"] == "gguf" else self._convrot(name)
        converted = to_map_names(self.family, {name: value})
        self._cache = (name, converted)
        return converted

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
                "dequantized_by": "gguf-py (gguf.quants.dequantize)" if self.source["format"] == "gguf"
                else "comfy_kitchen dequantize_int8_convrot_weight",
                "dropped": self.dropped}
