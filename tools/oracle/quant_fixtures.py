"""Pieces of the published 8-bit checkpoints, and their dequantization by the publisher's own code.

For each quantized source the engine installs (its Compact and Light versions), reads a handful of tensors by HTTP `Range`
at the frozen revision -- never the whole file -- and writes, under Tests/SiliconedTests/Fixtures/quant/<slug>/:

  pieces.safetensors   the chosen tensors under their published names and dtypes, bytes identical to
                       the published ones, plus `reference.<w>` (F32, torch shape [rows, K]) for every
                       quantized weight <w>: its dequantization by the publisher's library (torchao for
                       unsloth, SDNQ for Disty0), asked for float32 output;
  manifest.json        repository, revision, file, byte ranges, library versions, sha256 of the pieces,
                       and how the reference compares with `q.float() * s.float()` and with the exact
                       product in fp64.

A quantized weight is a row slice [0, rows) of the published [N, K] tensor (rows = --rows, default
256), with the matching rows of its scale (and zero point). Every scheme here is per output row, so a
row slice is self-contained, and a few hundred rows keep each source to a few tens of MB once the F32
reference is added. `--rows 0` takes whole tensors (hundreds of MB). Non-quantized tensors are whole.

What the publishers' code does, read in the installed packages (versions in the manifest):

* torchao `Int8Tensor.dequantize` (quantization/quantize_/workflows/int8/int8_tensor.py) calls
  `dequantize_affine` -> `_dequantize_affine_no_dtype_check` (quantization/quant_primitives.py):
  `(q.to(out) - zero_point.to(out)) * scale`, then `.to(out)`. `out` defaults to the tensor's nominal
  dtype, which the unsloth metadata sets to bfloat16: the default dequantization rounds to bf16.
* torchao `Float8Tensor.dequantize` (quantization/quantize_/workflows/float8/float8_tensor.py) calls
  `_dequantize_affine_float8`: `q.to(float32) * scale`, then `.to(out)`; same bf16 default.
* SDNQ `dequantize_symmetric` (sdnq/dequantizer.py): `q.to(scale.dtype).mul_(scale)`, then
  `.to(result_dtype)`. The scale keeps the loading dtype (Disty0's config has dequantize_fp32 = false):
  loaded in bf16 the product is rounded to bf16; loaded in float32 (our oracles' convention) the
  scale is widened first and the product is exact.

The reference stored is the float32 one (torchao `dequantize(torch.float32)`, SDNQ with a float32
scale and result dtype); the manifest also counts how far the publisher's default bf16 path lands.

GGUF sources (`format='gguf'`) go through `build_gguf`, see the comment above it: the
header and row slices read by `Range`, a reduced `pieces.gguf` read back by diffusers' own loader,
references from diffusers' Q8_0 dequantization asked for in fp32, under the map's (diffusers) names.
The published K-quants and the older 4- and 5-bit blocks (Q4_K, Q5_K, Q6_K, Q4_0,
Q4_1, Q5_0, Q5_1) take the same path; their reference is also computed by gguf-py, llama.cpp's own
Python package (`gguf.quants.dequantize`), and the two must agree at the bit on every value.

ComfyUI int8 convrot sources (`format='convrot'`) go through `build_convrot`, see the comment
above it: row slices by `Range`, Comfy's own `dequantize_int8_convrot_weight` (comfy_kitchen, frozen
version, downloaded into comfy/ by `fetch_comfy`) in fp32, and the exact `(q·s)·R` in fp64.

    python tools/oracle/quant_fixtures.py [--rows N] [slug ...]
"""
import argparse, hashlib, json, os, re, struct, sys, urllib.request
from importlib.metadata import version

import torch
from safetensors import safe_open

# The repository's root is two folders above this script (tools/oracle/); `swift test` reads the fixtures there.
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    'Tests', 'SiliconedTests', 'Fixtures', 'quant')

SOURCES = {
    'z-image-sdnq-int8': dict(
        repo='Disty0/Z-Image-Turbo-SDNQ-int8',
        revision='43d0947f91c14b93da75704be27565ee30694f21',
        file='transformer/diffusion_pytorch_model.safetensors',
        config='transformer/config.json',
        scheme='sdnq'),
    'qwen21-fp8': dict(
        repo='unsloth/Qwen-Image-2.1-FP8',
        revision='bb21a8ba7f0371c19ad1f72c82a5e8fc10f39939',
        file='Qwen-Image-2.1-FP8.safetensors',
        scheme='torchao'),
    'qwen21-int8': dict(
        repo='unsloth/Qwen-Image-2.1-FP8',
        revision='bb21a8ba7f0371c19ad1f72c82a5e8fc10f39939',
        file='Qwen-Image-2.1-INT8.safetensors',
        scheme='torchao'),
    'z-image-gguf-q8': dict(
        repo='unsloth/Z-Image-Turbo-GGUF',
        revision='6c80814333b7b6a70a2e5b469a7c6437ce65de0f',
        file='z-image-turbo-Q8_0.gguf',
        format='gguf', family='z-image'),
    'qwen21-gguf-q8': dict(
        repo='unsloth/Qwen-Image-2.1-GGUF',
        revision='2c31ccd392b367a6637841a143813320a02dff55',
        file='qwen-image-2.1-Q8_0.gguf',
        format='gguf', family='qwen-image-2.1'),
    # Below 8 bits: the same two unsloth repositories at the same revisions. The
    # types of each file are in its manifest. Z-Image Q4_K_M mixes Q4_K, Q5_K and Q6_K; Q6_K is all
    # Q6_K; Q4_0 carries Q4_1 on the feed-forward w2, Q5_0 carries Q5_1 there; Qwen's Q4_K_M mixes
    # Q4_K, Q5_K, Q6_K and Q8_0.
    'z-image-gguf-q4km': dict(
        repo='unsloth/Z-Image-Turbo-GGUF',
        revision='6c80814333b7b6a70a2e5b469a7c6437ce65de0f',
        file='z-image-turbo-Q4_K_M.gguf',
        format='gguf', family='z-image'),
    'z-image-gguf-q6k': dict(
        repo='unsloth/Z-Image-Turbo-GGUF',
        revision='6c80814333b7b6a70a2e5b469a7c6437ce65de0f',
        file='z-image-turbo-Q6_K.gguf',
        format='gguf', family='z-image'),
    'z-image-gguf-q4-0': dict(
        repo='unsloth/Z-Image-Turbo-GGUF',
        revision='6c80814333b7b6a70a2e5b469a7c6437ce65de0f',
        file='z-image-turbo-Q4_0.gguf',
        format='gguf', family='z-image'),
    'z-image-gguf-q5-0': dict(
        repo='unsloth/Z-Image-Turbo-GGUF',
        revision='6c80814333b7b6a70a2e5b469a7c6437ce65de0f',
        file='z-image-turbo-Q5_0.gguf',
        format='gguf', family='z-image'),
    'qwen21-gguf-q4km': dict(
        repo='unsloth/Qwen-Image-2.1-GGUF',
        revision='2c31ccd392b367a6637841a143813320a02dff55',
        file='qwen-image-2.1-Q4_K_M.gguf',
        format='gguf', family='qwen-image-2.1'),
    # Anima Turbo v1.1's Compact (Q8_0) and Light (Q4_K_M), ComfyUI-GGUF (`cosmos`), original names.
    'anima-gguf-q8': dict(
        repo='vanes430/Anima-Turbo-V1.1-GGUF',
        revision='efdf070de980a52fc03027d2f6ed543cc1d87a8f',
        file='anima-turbo-v1.1-Q8_0.gguf',
        format='gguf', family='anima'),
    'anima-gguf-q4km': dict(
        repo='vanes430/Anima-Turbo-V1.1-GGUF',
        revision='efdf070de980a52fc03027d2f6ed543cc1d87a8f',
        file='anima-turbo-v1.1-Q4_K_M.gguf',
        format='gguf', family='anima'),
    # Krea 2 Turbo's Compact DiT (Comfy-Org's int8 convrot) and Light DiT (vantagewithai's GGUF
    # Q4_K_M), original names, diffusers' converter.
    'krea2-turbo-convrot': dict(
        repo='Comfy-Org/Krea-2',
        revision='1171f4dc7e325bf3734586942e737872506c73e1',
        file='diffusion_models/krea2_turbo_int8_convrot.safetensors',
        format='convrot', family='krea2'),
    'krea2-turbo-gguf-q4km': dict(
        repo='vantagewithai/Krea-2-Turbo-GGUF',
        revision='390847e2835c16bdf82303731e3d35b874760d89',
        file='krea2_turbo-Q4_K_M.gguf',
        format='gguf', family='krea2'),
}

# Krea's easter egg in the GGUFs of Krea 2 Turbo (`Recipes.kreaEasterEgg`): outside the model, never a piece.
KREA_EGG = ('last.down.weight', 'last.up.weight')

ITEMSIZE = {'I8': 1, 'F8_E4M3': 1, 'BF16': 2, 'F16': 2, 'F32': 4}
ST_DTYPE = {torch.int8: 'I8', torch.float8_e4m3fn: 'F8_E4M3', torch.bfloat16: 'BF16',
            torch.float16: 'F16', torch.float32: 'F32', torch.float64: 'F64', torch.uint8: 'U8'}
TORCH_DTYPE = {'I8': torch.int8, 'F8_E4M3': torch.float8_e4m3fn, 'BF16': torch.bfloat16,
               'F16': torch.float16, 'F32': torch.float32}
QUANT = ('I8', 'F8_E4M3')


# ---------------------------------------------------------------- HTTP

def url_of(src, path):
    return f"https://huggingface.co/{src['repo']}/resolve/{src['revision']}/{path}"


def fetch(url, first=None, last=None):
    """Bytes [first, last] (inclusive, as in the Range header), or the whole small file."""
    req = urllib.request.Request(url)
    if first is not None:
        req.add_header('Range', f'bytes={first}-{last}')
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
        if first is not None:
            assert r.status == 206, f'{url}: HTTP {r.status}, range ignored'
            assert len(data) == last - first + 1, (len(data), first, last)
    return data


def write_safetensors(path, tensors, metadata):
    """safetensors writer with a sorted header: `safetensors.torch.save_file` serializes the metadata
    from a hash map whose order changes from run to run, and the pieces must be reproducible."""
    header, blobs, offset = {'__metadata__': dict(sorted(metadata.items()))}, [], 0
    for k in sorted(tensors):
        t = tensors[k].contiguous()
        b = t.view(torch.uint8).numpy().tobytes() if t.dim() else t.reshape(1).view(torch.uint8).numpy().tobytes()
        header[k] = dict(dtype=ST_DTYPE[t.dtype], shape=list(t.shape), data_offsets=[offset, offset + len(b)])
        blobs.append(b)
        offset += len(b)
    raw = json.dumps(header, separators=(',', ':')).encode()
    raw += b' ' * (-len(raw) % 8)
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(raw)))
        f.write(raw)
        for b in blobs:
            f.write(b)


# ---------------------------------------------------------------- choosing the pieces

def published_weights(h):
    """{logical weight name: {'q': key, 's': key, 'z': key or None}} for every quantized linear."""
    out = {}
    for k, v in h.items():
        if k == '__metadata__' or v['dtype'] not in QUANT:
            continue
        m = re.fullmatch(r'(.+)\._weight_qdata', k)                 # torchao flattened subclass
        if m:
            base = m.group(1)
            z = f'{base}._weight_zero_point'
            out[f'{base}.weight'] = dict(q=k, s=f'{base}._weight_scale', z=z if z in h else None)
            continue
        m = re.fullmatch(r'(.+)\.weight', k)                         # SDNQ: weight I8 + scale
        if m and f'{m.group(1)}.scale' in h:
            out[k] = dict(q=k, s=f'{m.group(1)}.scale', z=None)
            continue
        if not re.search(r'_weight_zero_point$', k):
            raise SystemExit(f'unrecognized quantized tensor {k}')
    return out


def block_of(name):
    m = re.match(r'(layers|transformer_blocks|blocks)\.(\d+)\.', name)
    return int(m.group(2)) if m else None


def choose(h):
    """Deterministic choice, from the header alone (see the manifest)."""
    qw = published_weights(h)
    shape = lambda w: h[qw[w]['q']]['shape']
    last = max(b for b in map(block_of, qw) if b is not None)
    first_block = sorted(w for w in qw if block_of(w) == 0)
    attn = [w for w in first_block if re.search(r'\.(attn|attention)\.to_q\.weight$', w)]
    mlp = [w for w in first_block if not re.search(r'\.(attn|attention)\.', w)]
    mlp_wide = max(mlp, key=lambda w: (shape(w)[1], w))
    last_out = [w for w in qw if block_of(w) == last and re.search(r'\.to_out\.0\.weight$', w)]
    outside = sorted(w for w in qw if block_of(w) is None)
    quantized = attn[:1] + [mlp_wide] + last_out[:1] + outside[:1]

    plain = [k for k in sorted(h) if k != '__metadata__' and h[k]['dtype'] not in QUANT
             and k not in {v['s'] for v in qw.values()} and k not in {v['z'] for v in qw.values()}]
    # A non-quantized linear of the first block if there is one (Z-Image keeps layers.0's adaLN in
    # bf16 while every other block's is int8: a forge that guesses by name would miss it), else the
    # smallest 2-D non-quantized weight; plus one 1-D norm weight of the first block.
    first_plain_2d = [k for k in plain if block_of(k) == 0 and len(h[k]['shape']) == 2]
    if first_plain_2d:
        plain_2d = first_plain_2d[0]
    else:
        two_d = [k for k in plain if len(h[k]['shape']) == 2 and k.endswith('.weight')]
        plain_2d = min(two_d, key=lambda k: (h[k]['shape'][0] * h[k]['shape'][1], k))
    norm = [k for k in plain if block_of(k) == 0 and len(h[k]['shape']) == 1 and 'norm' in k][0]
    return qw, quantized, [plain_2d, norm]


# ---------------------------------------------------------------- dequantization by the publisher

def dequant_torchao(meta, weight, data):
    """torchao: rebuild the subclass from the metadata the way unsloth's loader does
    (`unflatten_tensor_state_dict`), then `dequantize()`."""
    from torchao.prototype.safetensors.safetensors_support import unflatten_tensor_state_dict
    tensors, leftover = unflatten_tensor_state_dict(data, meta)
    t = tensors[weight]
    assert not leftover, leftover
    return t, t.dequantize(torch.float32), t.dequantize()            # default: nominal dtype


def dequant_sdnq(q, s):
    """SDNQ `dequantize_weight` for weights_dtype int8 (symmetric, no SVD, no group, no hadamard),
    once as loaded in float32, once as loaded in bf16 (the scale keeps the loading dtype)."""
    from sdnq.dequantizer import dequantize_weight
    f32 = dequantize_weight('int8', q.clone(), s.to(torch.float32), dtype=torch.float32)
    bf16 = dequantize_weight('int8', q.clone(), s.clone(), dtype=torch.bfloat16)
    return f32, bf16


# ---------------------------------------------------------------- comparisons

def compare(ref, q, s, z, default):
    simple = q.float() * s.float()                                   # the engine's target formula
    exact = (q.double() - (z.double() if z is not None else 0)) * s.double()
    d_simple = (ref != simple)
    d_exact = (ref.double() != exact)
    out = dict(
        values=ref.numel(),
        vs_q_times_s_fp32=dict(different=int(d_simple.sum()),
                               max_abs=float((ref - simple).abs().max())),
        vs_exact_fp64=dict(different=int(d_exact.sum()),
                           max_abs=float((ref.double() - exact).abs().max())),
    )
    if default is not None:
        dd = default.float()
        rel = ((dd.double() - exact).abs() / exact.abs().clamp_min(1e-300))
        out['publisher_default_path'] = dict(
            dtype=str(default.dtype).removeprefix('torch.'),
            different_from_fp32_reference=int((dd != ref).sum()),
            max_abs=float((dd.double() - exact).abs().max()),
            max_rel=float(rel[exact != 0].max()) if bool((exact != 0).any()) else 0.0)
    if z is not None:
        out['zero_point_all_zero'] = bool((z == 0).all())
    return out


# ---------------------------------------------------------------- one source

def build(slug, rows):
    src = SOURCES[slug]
    url = url_of(src, src['file'])
    expected = json.load(open(os.path.join(ROOT, slug, 'header.json')))

    n = struct.unpack('<Q', fetch(url, 0, 7))[0]
    raw_header = fetch(url, 8, 8 + n - 1)
    h = json.loads(raw_header)
    assert h == expected, f'{slug}: published header differs from header.json'
    base = 8 + n
    meta = h.get('__metadata__', {})

    qw, quantized, plain = choose(h)
    ranges, raw, shapes = {}, {}, {}

    def read(key, nrows=None):
        v = h[key]
        a, b = v['data_offsets']
        shape = list(v['shape'])
        if nrows and shape[0] > nrows:
            row = (b - a) // shape[0]
            b = a + nrows * row
            shape[0] = nrows
        assert (b - a) == ITEMSIZE[v['dtype']] * (1 if not shape else int(torch.tensor(shape).prod()))
        raw[key] = fetch(url, base + a, base + b - 1)
        ranges[key] = [base + a, base + b - 1]
        shapes[key] = shape

    for w in quantized:
        for role in ('q', 's', 'z'):
            if qw[w][role]:
                read(qw[w][role], rows)
    for k in plain:
        read(k)

    tensors = {k: torch.frombuffer(bytearray(raw[k]), dtype=TORCH_DTYPE[h[k]['dtype']]).reshape(shapes[k])
               for k in raw}

    comparisons, out = {}, dict(tensors)
    for w in quantized:
        q, s = tensors[qw[w]['q']], tensors[qw[w]['s']]
        z = tensors[qw[w]['z']] if qw[w]['z'] else None
        if src['scheme'] == 'torchao':
            data = {qw[w][r]: tensors[qw[w][r]] for r in ('q', 's', 'z') if qw[w][r]}
            sub, ref, default = dequant_torchao(meta, w, data)
            kind = type(sub).__name__
        else:
            ref, default = dequant_sdnq(q, s)
            kind = 'SDNQ int8 symmetric'
        assert ref.dtype == torch.float32 and list(ref.shape) == list(q.shape)
        out[f'reference.{w}'] = ref.contiguous()
        comparisons[w] = dict(kind=kind, **compare(ref, q, s, z, default))

    pieces_meta = {}
    if src['scheme'] == 'torchao':
        names = [n_ for n_ in json.loads(meta['tensor_names'])
                 if n_ in quantized or n_ in plain]
        for k in names:
            pieces_meta[k] = meta[k]
        pieces_meta['tensor_names'] = json.dumps(names)
        for k in ('unsloth_format', 'unsloth_metadata'):
            pieces_meta[k] = meta[k]
    else:
        pieces_meta.update(meta)
    pieces_meta['siliconed.pieces'] = json.dumps(
        {k: dict(published_shape=h[k]['shape'], shape=shapes[k]) for k in sorted(raw)})

    path = os.path.join(ROOT, slug, 'pieces.safetensors')
    write_safetensors(path, out, pieces_meta)

    # Re-read: every published tensor must come back byte for byte.
    with open(path, 'rb') as f:
        m = struct.unpack('<Q', f.read(8))[0]
        ph = json.loads(f.read(m))
        blob = f.read()
    for k in raw:
        a, b = ph[k]['data_offsets']
        assert ph[k]['dtype'] == h[k]['dtype'], k
        assert blob[a:b] == raw[k], f'{slug}: {k} not byte-identical'
    with safe_open(path, 'pt') as f:
        for w in quantized:
            assert torch.equal(f.get_tensor(f'reference.{w}'), out[f'reference.{w}'])

    manifest = dict(
        repo=src['repo'], revision=src['revision'], file=src['file'], url=url,
        header=dict(bytes=n, sha256=hashlib.sha256(raw_header).hexdigest()),
        rows=rows or 'all',
        libraries={p: version(p) for p in ('torch', 'torchao', 'sdnq', 'safetensors')},
        quantized=[dict(weight=w, **{r: qw[w][r] for r in ('q', 's', 'z') if qw[w][r]})
                   for w in quantized],
        plain=plain,
        tensors={k: dict(dtype=h[k]['dtype'], published_shape=h[k]['shape'], shape=shapes[k],
                         range=ranges[k]) for k in sorted(raw)},
        comparisons=comparisons,
        pieces=dict(file='pieces.safetensors', bytes=os.path.getsize(path),
                    sha256=hashlib.sha256(open(path, 'rb').read()).hexdigest()),
    )
    if 'config' in src:
        manifest['quantization_config'] = json.loads(fetch(url_of(src, src['config'])))['quantization_config']
    with open(os.path.join(ROOT, slug, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
        f.write('\n')
    return manifest


# ---------------------------------------------------------------- GGUF
#
# What is written under Tests/SiliconedTests/Fixtures/quant/<slug>/ for a GGUF source:
#
#   header.json             the published header, readable: version, alignment, data start, every
#                           key/value with its type, every tensor (ggml type, `ne` as stored, torch
#                           shape = `ne` reversed, offset). Versioned.
#   header.gguf             the published bytes [0, data start), verbatim — what the Swift check lays
#                           over a sparse file of the published size to run the recognition. Rebuilt.
#   pieces.gguf             a GGUF of the same version, the same key/values (bytes copied), and the
#                           chosen tensors only, each a row slice of the published one (same ggml
#                           type, `ne[1]` reduced), bytes identical. A fused weight (Z-Image's
#                           `attention.qkv`, Qwen's `img_mlp.gate_up`) takes `rows` rows from each of
#                           its parts, so that the forge's split still finds its parts. Rebuilt.
#   references.safetensors  `reference.<map name>` F32 `[rows, K]` for every map tensor the pieces
#                           produce: diffusers' `load_gguf_checkpoint` on pieces.gguf, its Q8_0
#                           dequantization asked for in fp32, then the published names converted to
#                           diffusers' (Z-Image: diffusers' own `convert_z_image_transformer_
#                           checkpoint_to_diffusers`; Qwen: the prefix dropped, `gate_up` cut in two
#                           halves `[gate_layer; proj]`). Rebuilt.
#   manifest.json           sources, byte ranges, row ranges, versions, comparisons, sha256. Versioned.
#
# diffusers' default path (`dequantize_gguf_tensor`, then `.to(compute_dtype)` in `GGUFLinear`)
# multiplies the fp16 `d` by the int8 `q` **in fp16**: the product is rounded to fp16. Asked for in
# fp32 (`dtype=torch.float32`, the argument diffusers' block functions take), the product is exact.
# The manifest counts both.

#
# Below 8 bits: a quantized row is a run of blocks along the input, each block self-contained
# (its scales inside it), so a row slice stays whole rows as for Q8_0. Block = (values, bytes):
#   Q4_0 (32, 18)   fp16 d, 16 bytes of nibbles               w = d·(q − 8)
#   Q4_1 (32, 20)   fp16 d, fp16 m, nibbles                   w = d·q + m
#   Q5_0 (32, 22)   fp16 d, 4 bytes of 5th bits, nibbles      w = d·(q − 16)
#   Q5_1 (32, 24)   fp16 d, fp16 m, 5th bits, nibbles         w = d·q + m
#   Q4_K (256, 144) fp16 d, fp16 dmin, 12 bytes of eight 6-bit (sc, m), 128 bytes of nibbles
#                                                             w = (d·sc)·q − dmin·m, 32 values per (sc, m)
#   Q5_K (256, 176) as Q4_K, plus 32 bytes of 5th bits
#   Q6_K (256, 210) 128 bytes low nibbles, 64 bytes high 2 bits, sixteen int8 sc, fp16 d **last**
#                                                             w = (d·sc)·(q − 32), 16 values per sc
# Every product there is exact in fp32 (at most 11 + 8 + 6 = 25 significant bits for Q6_K's d·sc·q,
# which is the one rounding), so diffusers (torch) and gguf-py (numpy) land on the same float whatever
# the order or a fused multiply-add: that is what `build_gguf` checks, value by value, bit by bit.

GGML = {0: 'F32', 1: 'F16', 2: 'Q4_0', 3: 'Q4_1', 6: 'Q5_0', 7: 'Q5_1', 8: 'Q8_0',
        12: 'Q4_K', 13: 'Q5_K', 14: 'Q6_K', 30: 'BF16'}
GGML_BLOCK = {'Q4_0': (32, 18), 'Q4_1': (32, 20), 'Q5_0': (32, 22), 'Q5_1': (32, 24), 'Q8_0': (32, 34),
              'Q4_K': (256, 144), 'Q5_K': (256, 176), 'Q6_K': (256, 210)}
GGML_PLAIN = ('F32', 'F16', 'BF16')
GGUF_VALUE = {0: '<B', 1: '<b', 2: '<H', 3: '<h', 4: '<I', 5: '<i', 6: '<f', 7: '<?', 10: '<Q', 11: '<q', 12: '<d'}
GGUF_TYPE_NAME = {0: 'u8', 1: 'i8', 2: 'u16', 3: 'i16', 4: 'u32', 5: 'i32', 6: 'f32', 7: 'bool', 8: 'string',
                  9: 'array', 10: 'u64', 11: 'i64', 12: 'f64'}


class Truncated(Exception):
    pass


def parse_gguf_header(buf):
    """The header of a GGUF file from its first bytes; `Truncated` when they are not enough."""
    p = 0

    def take(fmt):
        nonlocal p
        n = struct.calcsize(fmt)
        if p + n > len(buf):
            raise Truncated
        v = struct.unpack_from(fmt, buf, p)[0]
        p += n
        return v

    def string():
        nonlocal p
        n = take('<Q')
        if p + n > len(buf):
            raise Truncated
        v = buf[p:p + n].decode('utf-8')
        p += n
        return v

    def value(t):
        if t == 8:
            return string()
        if t == 9:
            it, n = take('<I'), take('<Q')
            return [value(it) for _ in range(n)]
        return take(GGUF_VALUE[t])

    if buf[:4] != b'GGUF':
        raise SystemExit('not a GGUF file')
    p = 4
    h = dict(version=take('<I'))
    n_tensors, n_kv = take('<Q'), take('<Q')
    kv = {}
    for _ in range(n_kv):
        k, t = string(), take('<I')
        kv[k] = dict(type=GGUF_TYPE_NAME[t], value=value(t))
    kv_end = p
    tensors = {}
    for _ in range(n_tensors):
        name = string()
        ne = [take('<Q') for _ in range(take('<I'))]
        t, off = take('<I'), take('<Q')
        tensors[name] = dict(type=GGML.get(t, f'ggml type {t}'), ne=ne, shape=ne[::-1], offset=off)
    alignment = kv.get('general.alignment', {}).get('value', 32)
    h.update(alignment=alignment, kv_bytes=[24, kv_end], tensors_end=p,
             data_start=(p + alignment - 1) // alignment * alignment, metadata=kv, tensors=tensors)
    return h


def gguf_row_bytes(t):
    ne0 = t['ne'][0]
    if t['type'] in GGML_BLOCK:
        n, b = GGML_BLOCK[t['type']]
        assert ne0 % n == 0, f"row of {ne0} values is not a whole number of {t['type']} blocks"
        return ne0 // n * b
    return {'BF16': 2 * ne0, 'F16': 2 * ne0, 'F32': 4 * ne0}[t['type']]


def gguf_tensor_bytes(t):
    rows = 1
    for d in t['ne'][1:]:
        rows *= d
    return rows * gguf_row_bytes(t)


def fused_parts(name):
    """How many parts the family recipe cuts a fused weight into (along the output axis)."""
    if name.endswith('.attention.qkv.weight'):
        return 3
    if name.endswith('.img_mlp.gate_up.weight'):
        return 2
    return 1


def choose_gguf(h):
    """Deterministic, from the header alone: the same kinds of tensors as for the safetensors, then,
    for a file that mixes block types (a K_M), a tensor of each type not yet covered — in the first
    block that has it, its smallest attention linear if any, else its smallest tensor. A file of Q8_0
    alone gets the choice of the 8-bit sources."""
    ts = h['tensors']
    strip = lambda n: n.removeprefix('model.diffusion_model.')
    block = lambda n: block_of(strip(n))
    unknown = sorted({t['type'] for t in ts.values() if t['type'] not in GGML_BLOCK and t['type'] not in GGML_PLAIN})
    assert not unknown, f'block types outside the retained list: {unknown}'
    # Anima's text adapter (`llm_adapter.*`) is not a map tensor: it goes beside the map, judged by
    # `quant-weights --source`. Its self-attention q/k rows are permuted by the forge (RoPE): the pieces
    # take the cross-attention's, whose rows are the published ones.
    q8 = sorted(n for n, t in ts.items() if t['type'] in GGML_BLOCK and not strip(n).startswith('llm_adapter.')
                and strip(n) not in KREA_EGG)
    last = max(b for b in map(block, q8) if b is not None)
    first = [n for n in q8 if block(n) == 0]
    attn = [n for n in first if re.search(r'\.(attention\.qkv|attn\.to_q|cross_attn\.q_proj|attn\.wq)\.weight$', n)]
    mlp = [n for n in first if not re.search(r'\.(attention|attn)\.', n) and 'adaLN' not in n]
    mlp_wide = max(mlp, key=lambda n: (ts[n]['ne'][0], n))
    fused_mlp = [n for n in first if n.endswith('.img_mlp.gate_up.weight')]
    last_out = [n for n in q8 if block(n) == last and re.search(r'\.(attention\.out|attn\.to_out\.0|self_attn\.output_proj|attn\.wo)\.weight$', n)]
    outside = [n for n in q8 if block(n) is None]
    quantized = list(dict.fromkeys(attn[:1] + [mlp_wide] + fused_mlp[:1] + last_out[:1] + outside[:1]))
    size = lambda n: (gguf_tensor_bytes(ts[n]), n)
    for kind in sorted({ts[n]['type'] for n in q8}):
        if not any(ts[n]['type'] == kind for n in quantized):
            of_kind = [n for n in q8 if ts[n]['type'] == kind]
            low = min((b for b in map(block, of_kind) if b is not None), default=None)
            here = [n for n in of_kind if block(n) == low] or of_kind
            quantized.append(min([n for n in here if re.search(r'\.(attention|attn)\.', n)] or here, key=size))

    plain = sorted(n for n, t in ts.items() if t['type'] in GGML_PLAIN and not strip(n).startswith('llm_adapter.'))
    bf16_2d = min((n for n in plain if ts[n]['type'] in ('BF16', 'F16') and len(ts[n]['ne']) == 2
                   and n.endswith('.weight')), key=size)
    f32_2d = [n for n in plain if ts[n]['type'] == 'F32' and len(ts[n]['ne']) == 2 and min(ts[n]['ne']) > 1
              and n.endswith('.weight')]
    norm = [n for n in plain if block(n) == 0 and len(ts[n]['ne']) == 1 and 'norm' in n and ts[n]['type'] == 'F32'][0]
    return quantized, [bf16_2d] + sorted(f32_2d, key=size)[:1] + [norm]


def write_gguf(path, h, raw_header, pieces):
    """A GGUF with the published key/values (bytes copied) and `pieces`: name → (published info,
    piece `ne`, bytes), tensors in name order, each on the published alignment."""
    a = h['alignment']
    infos, blobs, off = b'', [], 0
    for name in sorted(pieces):
        t, ne, data = pieces[name]
        enc = name.encode()
        infos += struct.pack('<Q', len(enc)) + enc + struct.pack('<I', len(ne))
        infos += b''.join(struct.pack('<Q', d) for d in ne)
        infos += struct.pack('<IQ', {v: k for k, v in GGML.items()}[t['type']], off)
        blobs.append((off, data))
        off = (off + len(data) + a - 1) // a * a
    k0, k1 = h['kv_bytes']
    head = b'GGUF' + struct.pack('<IQQ', h['version'], len(pieces), len(h['metadata'])) + raw_header[k0:k1] + infos
    head += b'\0' * (-len(head) % a)
    with open(path, 'wb') as f:
        f.write(head)
        for o, data in blobs:
            f.seek(len(head) + o)
            f.write(data)
        f.truncate(len(head) + off)


def fetch_gguf_header(url):
    n = 1 << 16
    while True:
        buf = fetch(url, 0, n - 1)
        try:
            h = parse_gguf_header(buf)
        except Truncated:
            n *= 4
            continue
        if h['data_start'] <= len(buf):
            return h, buf[:h['data_start']]
        n = h['data_start']


def published_size(url):
    req = urllib.request.Request(url)
    req.add_header('Range', 'bytes=0-0')
    with urllib.request.urlopen(req, timeout=120) as r:
        return int(r.headers['Content-Range'].split('/')[1])


def dequantize_gguf_fp32(param):
    """diffusers' `dequantize_gguf_tensor`, the block function asked for `dtype=torch.float32`."""
    return dequantize_gguf_dtype(param, torch.float32)


def dequantize_gguf_dtype(param, dtype):
    from diffusers.quantizers.gguf.utils import (GGML_QUANT_SIZES, _quant_shape_from_byte_shape,
                                                 dequantize_functions)
    if not hasattr(param, 'quant_type'):
        return param.to(dtype)
    block_size, type_size = GGML_QUANT_SIZES[param.quant_type]
    t = param.as_tensor().view(torch.uint8)
    shape = _quant_shape_from_byte_shape(t.shape, type_size, block_size)
    blocks = t.reshape((t.numel() // type_size, type_size))
    return dequantize_functions[param.quant_type](blocks, block_size, type_size, dtype=dtype).reshape(shape)


def gguf_py_dequantize(data, kind, ne):
    """gguf-py's `gguf.quants.dequantize` on the published bytes of a piece: float32 [rows, ne[0]]."""
    import numpy as np
    from gguf import GGMLQuantizationType
    from gguf.quants import dequantize
    rows = len(data) // gguf_row_bytes(dict(type=kind, ne=ne))
    raw = np.frombuffer(data, dtype=np.uint8).reshape(rows, -1)
    out = dequantize(raw, GGMLQuantizationType[kind])
    assert out.dtype == np.float32 and out.shape == (rows, ne[0]), (out.dtype, out.shape)
    return np.ascontiguousarray(out)


# Anima: ComfyUI (Cosmos) → diffusers, the oracle's tables (`anima.py`:
# `BLOCK_RULES`, `TOP_RULES`); diffusers has no single-file converter for it.
ANIMA_BLOCK = [
    ('adaln_modulation_self_attn.1.', 'norm1.linear_1.'), ('adaln_modulation_self_attn.2.', 'norm1.linear_2.'),
    ('adaln_modulation_cross_attn.1.', 'norm2.linear_1.'), ('adaln_modulation_cross_attn.2.', 'norm2.linear_2.'),
    ('adaln_modulation_mlp.1.', 'norm3.linear_1.'), ('adaln_modulation_mlp.2.', 'norm3.linear_2.'),
    ('self_attn.q_proj.', 'attn1.to_q.'), ('self_attn.k_proj.', 'attn1.to_k.'), ('self_attn.v_proj.', 'attn1.to_v.'),
    ('self_attn.output_proj.', 'attn1.to_out.0.'), ('self_attn.q_norm.', 'attn1.norm_q.'), ('self_attn.k_norm.', 'attn1.norm_k.'),
    ('cross_attn.q_proj.', 'attn2.to_q.'), ('cross_attn.k_proj.', 'attn2.to_k.'), ('cross_attn.v_proj.', 'attn2.to_v.'),
    ('cross_attn.output_proj.', 'attn2.to_out.0.'), ('cross_attn.q_norm.', 'attn2.norm_q.'), ('cross_attn.k_norm.', 'attn2.norm_k.'),
    ('mlp.layer1.', 'ff.net.0.proj.'), ('mlp.layer2.', 'ff.net.2.'),
]
ANIMA_TOP = {
    'x_embedder.proj.1.weight': 'patch_embed.proj.weight',
    't_embedder.1.linear_1.weight': 'time_embed.t_embedder.linear_1.weight',
    't_embedder.1.linear_2.weight': 'time_embed.t_embedder.linear_2.weight',
    't_embedding_norm.weight': 'time_embed.norm.weight',
    'final_layer.adaln_modulation.1.weight': 'norm_out.linear_1.weight',
    'final_layer.adaln_modulation.2.weight': 'norm_out.linear_2.weight',
    'final_layer.linear.weight': 'proj_out.weight',
}


def anima_map_name(name):
    name = name.removeprefix('model.diffusion_model.')
    if name in ANIMA_TOP:
        return ANIMA_TOP[name]
    m = re.fullmatch(r'blocks\.(\d+)\.(.+)', name)
    assert m, f'Anima key without a rule: {name}'
    for a, b in ANIMA_BLOCK:
        if m.group(2).startswith(a):
            out = f'transformer_blocks.{m.group(1)}.{b}{m.group(2)[len(a):]}'
            # The forge permutes these rows (split RoPE → interleaved): a piece of them is never chosen.
            assert not re.search(r'\.attn1\.(to_[qk]|norm_[qk])\.weight$', out), out
            return out
    raise AssertionError(f'Anima block key without a rule: {name}')


def to_map_names(family, tensors):
    """Published names → the map's (diffusers) names, by diffusers' converter where it has one."""
    if family == 'z-image':
        from diffusers.loaders.single_file_utils import convert_z_image_transformer_checkpoint_to_diffusers
        return convert_z_image_transformer_checkpoint_to_diffusers(dict(tensors))
    if family == 'anima':
        return {anima_map_name(k): v for k, v in tensors.items()}
    if family == 'krea2':
        from diffusers.loaders.single_file_utils import convert_krea2_transformer_checkpoint_to_diffusers
        return convert_krea2_transformer_checkpoint_to_diffusers(dict(tensors))
    out = {}
    for k, v in tensors.items():
        k = k.removeprefix('model.diffusion_model.')
        if k.endswith('.img_mlp.gate_up.weight'):
            gate, proj = torch.chunk(v, 2, dim=0)
            out[k.replace('.gate_up.', '.gate_layer.')] = gate
            out[k.replace('.gate_up.', '.proj.')] = proj
        else:
            out[k] = v
    return out


def build_gguf(slug, rows):
    from diffusers.models.model_loading_utils import load_gguf_checkpoint
    from diffusers.quantizers.gguf.utils import dequantize_gguf_tensor
    src = SOURCES[slug]
    url = url_of(src, src['file'])
    folder = os.path.join(ROOT, slug)
    os.makedirs(folder, exist_ok=True)

    h, raw_header = fetch_gguf_header(url)
    size = published_size(url)
    readable = {k: h[k] for k in ('version', 'alignment', 'data_start', 'metadata', 'tensors')}
    path = os.path.join(folder, 'header.json')
    if os.path.exists(path):
        assert json.load(open(path)) == json.loads(json.dumps(readable)), f'{slug}: published header differs from header.json'
    else:
        with open(path, 'w') as f:
            json.dump(readable, f, indent=1, ensure_ascii=False)
            f.write('\n')
    with open(os.path.join(folder, 'header.gguf'), 'wb') as f:
        f.write(raw_header)

    quantized, plain = choose_gguf(h)
    pieces, ranges, row_ranges = {}, {}, {}
    for name in quantized + plain:
        t = h['tensors'][name]
        ne = list(t['ne'])
        n_rows = ne[1] if len(ne) == 2 else 1
        parts = fused_parts(name)
        if rows and len(ne) == 2 and n_rows > rows * parts:
            spans = [(k * n_rows // parts, k * n_rows // parts + rows) for k in range(parts)]
        else:
            spans = [(0, n_rows)]
        rb = gguf_row_bytes(t)
        base = h['data_start'] + t['offset']
        data = b''
        ranges[name] = []
        for a, b in spans:
            first, last = base + a * rb, base + b * rb - 1
            data += fetch(url, first, last)
            ranges[name].append([first, last])
        if len(ne) == 2:
            ne[1] = sum(b - a for a, b in spans)
        pieces[name] = (t, ne, data)
        row_ranges[name] = [list(s) for s in spans]

    pieces_path = os.path.join(folder, 'pieces.gguf')
    write_gguf(pieces_path, h, raw_header, pieces)

    # Read back by the reference reader: every piece byte for byte, its type, its shape.
    from gguf import GGUFReader
    reader = GGUFReader(pieces_path)
    assert sorted(t.name for t in reader.tensors) == sorted(pieces)
    for t in reader.tensors:
        info, ne, data = pieces[t.name]
        assert GGML[int(t.tensor_type)] == info['type'] and [int(x) for x in t.shape] == ne, t.name
        assert bytes(t.data.tobytes()) == data, f'{t.name} not byte-identical'
    assert parse_gguf_header(open(pieces_path, 'rb').read())['metadata'] == h['metadata']

    # diffusers' own loader on the pieces, then its dequantization (fp32, and its default).
    params = load_gguf_checkpoint(pieces_path, return_tensors=True)
    fp32 = {k: dequantize_gguf_fp32(v).contiguous() for k, v in params.items()}
    comparisons = {}
    for name in quantized:
        info, ne, data = pieces[name]
        kind = info['type']
        ref = fp32[name].reshape(-1)
        default = dequantize_gguf_tensor(params[name]).reshape(-1)
        dd = default.float()
        # gguf-py, llama.cpp's Python package: the second reference, compared bit for bit (a value
        # compare would take -0 for +0).
        gp = torch.from_numpy(gguf_py_dequantize(data, kind, ne)).reshape(-1)
        c = dict(kind=f'GGUF {kind}', values=ref.numel(),
                 gguf_py=dict(different_bits=int((gp.view(torch.int32) != ref.view(torch.int32)).sum()),
                              different_values=int((gp != ref).sum())))
        if kind == 'Q8_0':
            raw = torch.frombuffer(bytearray(data), dtype=torch.uint8).reshape(-1, 34)
            d, q = raw[:, :2].contiguous().view(torch.float16), raw[:, 2:].contiguous().view(torch.int8)
            ref, dd = ref.reshape(-1, 32), dd.reshape(-1, 32)
            simple = d.float() * q.float()
            exact = d.double() * q.double()
            dh = d.view(torch.int16) & 0x7FFF
            c['vs_d_times_q_fp32'] = dict(different=int((ref != simple).sum()), max_abs=float((ref - simple).abs().max()))
            c.update(d_subnormal=int(((dh > 0) & (dh < 0x0400)).sum()), d_zero=int((dh == 0).sum()))
        else:
            # The exact value: diffusers' own block function asked for float64, where every product
            # and the one sum are exact. fl32 of it must be the fp32 reference (one rounding).
            exact = dequantize_gguf_dtype(params[name], torch.float64).reshape(-1)
            rounded = exact.float()
            c['fl32_of_exact'] = dict(different_bits=int((rounded.view(torch.int32) != ref.view(torch.int32)).sum()))
        rel = (dd.double() - exact).abs() / exact.abs().clamp_min(1e-300)
        c['vs_exact_fp64'] = dict(different=int((ref.double() != exact).sum()),
                                  max_abs=float((ref.double() - exact).abs().max()))
        c['publisher_default_path'] = dict(
            dtype=str(default.dtype).removeprefix('torch.'),
            different_from_fp32_reference=int((dd != ref).sum()),
            max_abs=float((dd.double() - exact).abs().max()),
            max_rel=float(rel[exact != 0].max()))
        comparisons[name] = c

    mapped = to_map_names(src['family'], fp32)
    produced = {}
    for name in quantized + plain:
        one = to_map_names(src['family'], {name: fp32[name]})
        produced[name] = sorted(one)
    references = {f'reference.{k}': v.contiguous() for k, v in mapped.items()}
    for k, v in references.items():
        assert v.dtype == torch.float32, k
    ref_path = os.path.join(folder, 'references.safetensors')
    write_safetensors(ref_path, references, {'siliconed.references': 'diffusers GGUF dequantization in fp32, map names'})

    sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
    manifest = dict(
        repo=src['repo'], revision=src['revision'], file=src['file'], url=url, format='gguf',
        family=src['family'], file_bytes=size,
        header=dict(bytes=h['data_start'], sha256=hashlib.sha256(raw_header).hexdigest(), version=h['version'],
                    alignment=h['alignment'], tensors=len(h['tensors']), metadata=len(h['metadata'])),
        rows=rows or 'all',
        types={k: sum(t['type'] == k for t in h['tensors'].values())
               for k in sorted({t['type'] for t in h['tensors'].values()})},
        libraries={p_: version(p_) for p_ in ('torch', 'diffusers', 'gguf', 'numpy')},
        quantized=[m for n in quantized for m in produced[n]],
        plain=[m for n in plain for m in produced[n]],
        published={n: dict(type=h['tensors'][n]['type'], ne=h['tensors'][n]['ne'], piece_ne=pieces[n][1],
                           rows=row_ranges[n], range=ranges[n], map=produced[n]) for n in sorted(pieces)},
        comparisons=comparisons,
        pieces=dict(file='pieces.gguf', bytes=os.path.getsize(pieces_path), sha256=sha(pieces_path)),
        references=dict(file='references.safetensors', bytes=os.path.getsize(ref_path), sha256=sha(ref_path)),
        header_file=dict(file='header.gguf', sha256=sha(os.path.join(folder, 'header.gguf'))),
    )
    with open(os.path.join(folder, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
        f.write('\n')
    return manifest


# ---------------------------------------------------------------- ComfyUI int8 convrot
#
# `comfy_quant` = {"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}: the weight
# stored is `W·R`, quantized per output row (`weight` I8 [N, K], `weight_scale` F32 [N, 1]). What the
# publisher's code does, read at the frozen revisions of `COMFY` (downloaded by `fetch_comfy` into
# Tests/SiliconedTests/Fixtures/quant/comfy/, sha256 in its manifest):
#
# * ComfyUI `comfy/ops.py` (`_load_quantized_module`): reads `convrot` and `convrot_groupsize`
#   (default 256) from the layer's JSON into `TensorWiseINT8Layout.Params`; `comfy/quant_ops.py`
#   aliases that layout to comfy_kitchen's.
# * comfy_kitchen `tensor/int8_utils.py`: `_build_hadamard(G)` = the Kronecker power of
#   H4 = [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]] (the "regular" Hadamard, not Sylvester's),
#   divided by sqrt(G); G must be a power of 4. Symmetric, orthogonal. `_rotate_weight(W, h, G)` =
#   W grouped [N, K/G, G] @ h.T (raises when G does not divide K: no partial group exists).
# * comfy_kitchen `backends/eager/quantization.py`: `quantize_int8_convrot_weight` = rowwise
#   quantization of `_rotate_weight(W)`; `dequantize_int8_convrot_weight(q, s, G)` =
#   `_rotate_weight(q.float() * s, h)`, in fp32 — the reference stored here; `tensor/int8.py`
#   `dequantize` then casts to the layer's `orig_dtype` (the compute dtype, bf16 usually).
# * The default forward (`int8_linear`, ops.py `_int8_linear` and the `aten.linear` handler) never
#   dequantizes: x rotated in its own dtype (`_rotate_activation`: x grouped @ h), then quantized to
#   int8 per row, int8 × int8 → int32 GEMM, times s_x · s_w. A LoRA patched on the fly makes the
#   cast hand back the dequantized weight instead.
#
# So W = (q·s)·Rᵀ = (q·s)·R (R symmetric), y = x·Wᵀ = (x·R)·(q·s)ᵀ. stable-diffusion.cpp's
# docs/int8_convrot.md (frozen commit) says the same: `W_rot = W R`, groups of G contiguous input
# features, `X_rot = X R`, G a power of 4 dividing K.
#
# Written under Tests/SiliconedTests/Fixtures/quant/<slug>/: header.json (versioned), pieces.safetensors (published
# names: row slices of the I8 weights — whole rows, so whole groups of K —, their `weight_scale` rows,
# `comfy_quant` as published, a few plain tensors; a fused qkv / gate_up takes `rows` rows of each
# part), references.safetensors under the map's names: `reference.<w>` F32 = Comfy's
# `dequantize_int8_convrot_weight` in fp32, `exact64.<w>` F64 = (q·s)·R in fp64 — exact: every
# partial sum is s·m/16 with |m| < 2¹⁵, checked against an integer computation —, and
# `reference.<plain>` F32; manifest.json (versioned, with every layer's `comfy_quant` text, which the
# recognition writes into its sparse file).

COMFY = dict(
    comfyui=dict(repo='Comfy-Org/ComfyUI', commit='b0b743566f65daafc423b4fea8a2fbda94b3384a',
                 files=['comfy/quant_ops.py', 'comfy/ops.py', 'comfy/supported_models.py', 'comfy/ldm/lumina/model.py']),
    kitchen=dict(version='0.2.37', wheel='comfy_kitchen-0.2.37-py3-none-any.whl',
                 url='https://files.pythonhosted.org/packages/9e/08/0a904ae1a1b82f57e7990eee8d93d48466a185934aa585cc625cdb27761e/'
                     'comfy_kitchen-0.2.37-py3-none-any.whl',
                 sha256='b626bd51ff45937f1446accf078ccea9d11c3686ffa0c89b0c1305d66d765afe',
                 files=['comfy_kitchen/tensor/int8.py', 'comfy_kitchen/tensor/int8_utils.py',
                        'comfy_kitchen/backends/eager/quantization.py',
                        # The W8A8 bar's (`quantized.Int8Products`): what int8_linear calls, and what it cites.
                        'comfy_kitchen/backends/_activations.py', 'comfy_kitchen/tensor/base.py',
                        'comfy_kitchen/backends/triton/quantization.py', 'comfy_kitchen/backends/triton/__init__.py',
                        'comfy_kitchen/backends/eager/__init__.py']),
    sdcpp=dict(repo='leejet/stable-diffusion.cpp', commit='3f8527a46c54ecf4cb4ed6003da8e8982283c73c',
               files=['docs/int8_convrot.md']),
)
COMFY_DIR = os.path.join(ROOT, 'comfy')

SOURCES.update({
    # Z-Image's Compact DiT. The repository answers to `Comfy-Org/z_image_turbo` too (the same commit): the canonical
    # name, the one the app pins.
    'z-image-convrot': dict(
        repo='Comfy-Org/Z-Image-Turbo', revision='6fc90a3b1b653e935a0d175e260736de25b84df5',
        file='split_files/diffusion_models/z_image_turbo_int8_convrot.safetensors',
        format='convrot', family='z-image'),
    # Anima's Compact DiT: Bedovyy's int8 convrot of CircleStone's Turbo v1.1 (every kept tensor the publisher's
    # `anima-turbo-v1.1.safetensors`'s to the bit), names under `model.diffusion_model.`, its first and last blocks'
    # attention and five MLPs left in bf16, the text adapter bf16.
    'anima-turbo-convrot': dict(
        repo='Bedovyy/Anima-INT8', revision='d7ba6686f161e9a829373106eb4b881327247ba9',
        file='anima-turbo-v1.1-int8convrot.safetensors',
        format='convrot', family='anima'),
    'qwen21-convrot': dict(
        repo='Comfy-Org/Qwen-Image-2.1', revision='cb504a4090723e43f17ad01cec0359490e2de613',
        file='diffusion_models/qwen_image_2.1_int8_convrot.safetensors',
        format='convrot', family='qwen-image-2.1'),
})


# The official Turbo (Qwen/Qwen-Image-2.1-Turbo) quantized by the same two third
# parties as the base. Comfy-Org republished its int8 convrot (72a49122) with 31 more tensors,
# `transformer_blocks.{1..31}.attn.comfy_attention.config` (U8 JSON); every other tensor kept its
# dtype, shape and bytes (Xet terms shared, the rest compared byte for byte). `vs_dit` compares every
# dequantized piece with the same rows of the Turbo's bf16 DiT (local Hugging Face cache) and of the base's
# (by `Range`): a Turbo quantization must sit about one quantization step from the first and further
# from the second.
QWEN21_DITS = dict(
    turbo=dict(repo='Qwen/Qwen-Image-2.1-Turbo', revision='d65dbc9a7e8f6b5479e33dee6030eaab2a906509'),
    base=dict(repo='Qwen/Qwen-Image-2.1', revision='d26bb61231c349cf6b7896fa83353113880e1ba3'),
    calibration=dict(convrot='qwen21-convrot', gguf='qwen21-gguf-q4km'))
SOURCES.update({
    'qwen21-turbo-convrot': dict(
        repo='Comfy-Org/Qwen-Image-2.1', revision='df94239739eef7205973a145ffa6f441f7b64e84',
        file='diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors',
        format='convrot', family='qwen-image-2.1', dits=QWEN21_DITS),
    'qwen21-turbo-gguf-q4km': dict(
        repo='unsloth/Qwen-Image-2.1-Turbo-GGUF', revision='5d1f9f759ea877ae556cda88dbec958b62ad3809',
        file='qwen-image-2.1-turbo-Q4_K_M.gguf',
        format='gguf', family='qwen-image-2.1', dits=QWEN21_DITS),
})


def fetch_comfy():
    """The publisher's files at their frozen revisions (not versioned: ComfyUI is GPL-3, comfy_kitchen
    Apache-2.0); their sha256 go into comfy/manifest.json, which is."""
    import io, zipfile
    os.makedirs(COMFY_DIR, exist_ok=True)
    sha = lambda b: hashlib.sha256(b).hexdigest()
    files = {}

    def keep(path, data, origin):
        out = os.path.join(COMFY_DIR, path)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'wb') as f:
            f.write(data)
        files[path] = dict(origin=origin, bytes=len(data), sha256=sha(data))

    for key in ('comfyui', 'sdcpp'):
        c = COMFY[key]
        for p in c['files']:
            url = f"https://raw.githubusercontent.com/{c['repo']}/{c['commit']}/{p}"
            keep(f"{key}/{p}", fetch(url), url)
    k = COMFY['kitchen']
    wheel = fetch(k['url'])
    assert sha(wheel) == k['sha256'], 'comfy_kitchen wheel: sha256 differs'
    with zipfile.ZipFile(io.BytesIO(wheel)) as z:
        for p in k['files']:
            keep(f"kitchen/{p}", z.read(p), f"{k['wheel']}!{p}")
    manifest = dict(COMFY, downloaded=files)
    with open(os.path.join(COMFY_DIR, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
        f.write('\n')
    return manifest


def comfy_functions():
    """Comfy's own functions, loaded from the downloaded files: `_build_hadamard`, `_rotate_weight`
    (int8_utils.py, imported whole: it needs torch only) and `dequantize_int8_convrot_weight`,
    `dequantize_int8_simple` (their source taken verbatim from eager/quantization.py with `ast` —
    the module itself imports half of comfy_kitchen)."""
    import ast, importlib.util
    base = os.path.join(COMFY_DIR, 'kitchen', 'comfy_kitchen')
    spec = importlib.util.spec_from_file_location('comfy_int8_utils', os.path.join(base, 'tensor', 'int8_utils.py'))
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)
    src = open(os.path.join(base, 'backends', 'eager', 'quantization.py')).read()
    wanted = {'dequantize_int8_simple', 'dequantize_int8_convrot_weight'}
    nodes = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    assert {n.name for n in nodes} == wanted
    ns = dict(torch=torch, _build_hadamard=utils._build_hadamard, _rotate_weight=utils._rotate_weight)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'quantization.py', 'exec'), ns)
    return utils, ns['dequantize_int8_convrot_weight']


def channel_error(got, expected):
    """`channelError` of the engine (Diagnostic.swift), channel = output row of a published [N, K]
    weight: ‖got − exp‖ / max(‖exp‖, 1 % of the median norm), per row; → (worst, median)."""
    got, expected = got.double(), expected.double()
    num = (got - expected).pow(2).sum(1).sqrt()
    den = expected.pow(2).sum(1).sqrt()
    live = den > 0
    r = num[live] / den[live].clamp(min=float(0.01 * den[live].median()))
    return float(r.max()), float(r.median())


def choose_convrot(h):
    """Deterministic, from the header alone: the first quantized block's attention input and widest-K MLP weight
    (and a fused gate_up), the last quantized block's output projection, an adaLN of K = one group (Z-Image), a
    refiner's weight; plus a plain 2-D weight and a norm. Anima (Bedovyy's): its names under `model.diffusion_model.`,
    block 0 left in bf16 and the first quantized MLP in block 4, the attention input its `self_attn.v_proj` (the
    forge permutes q's and k's rows: never a piece) — the other families' choice is unchanged."""
    block = lambda k: block_of(k.removeprefix('model.diffusion_model.'))
    attention = lambda w: re.search(r'\.(attention|attn|self_attn|cross_attn)\.', w)
    qw = sorted(k for k, v in h.items() if k != '__metadata__' and v['dtype'] == 'I8')
    last = max(b for b in map(block, qw) if b is not None)
    start = min(b for b in map(block, qw) if b is not None)
    first = [w for w in qw if block(w) == start]
    attn = [w for w in first if re.search(r'\.(attention\.qkv|attn\.to_q|attn\.wq|self_attn\.v_proj)\.weight$', w)]
    mlp = [w for w in first if not attention(w) and 'adaLN' not in w]
    if not mlp:
        # Anima: the MLP of its first quantized blocks stays bf16; the first block whose MLP is int8.
        b = min(block(w) for w in qw if not attention(w) and 'adaLN' not in w)
        mlp = [w for w in qw if block(w) == b and not attention(w) and 'adaLN' not in w]
    wide = max(mlp, key=lambda w: (h[w]['shape'][1], w))
    fused = [w for w in first if w.endswith('.img_mlp.gate_up.weight')]
    small_k = [w for w in first if h[w]['shape'][1] == 256]
    last_out = [w for w in qw if block(w) == last
                and re.search(r'\.(attention\.out|attn\.to_out\.0|attn\.wo|self_attn\.output_proj)\.weight$', w)]
    outside = [w for w in qw if block(w) is None]
    quantized = list(dict.fromkeys(attn[:1] + [wide] + fused[:1] + small_k[:1] + last_out[:1] + outside[:1]))
    plain = sorted(k for k, v in h.items() if k != '__metadata__' and v['dtype'] in ('F32', 'BF16')
                   and '.llm_adapter.' not in k and not k.startswith('llm_adapter.'))
    two_d = [k for k in plain if len(h[k]['shape']) == 2 and k.endswith('.weight') and min(h[k]['shape']) > 1]
    plain_2d = min(two_d, key=lambda k: (h[k]['shape'][0] * h[k]['shape'][1], k))
    norm = [k for k in plain if block(k) == 0 and len(h[k]['shape']) == 1 and 'norm' in k][0]
    return quantized, [plain_2d, norm]


def build_convrot(slug, rows):
    src = SOURCES[slug]
    url = url_of(src, src['file'])
    folder = os.path.join(ROOT, slug)
    os.makedirs(folder, exist_ok=True)
    # The manifest is committed, the files it lists are not (.gitignore): a fresh clone has only the former.
    if not os.path.exists(os.path.join(COMFY_DIR, 'kitchen', 'comfy_kitchen', 'tensor', 'int8_utils.py')):
        fetch_comfy()
    utils, comfy_dequantize = comfy_functions()

    n = struct.unpack('<Q', fetch(url, 0, 7))[0]
    raw_header = fetch(url, 8, 8 + n - 1)
    h = json.loads(raw_header)
    path = os.path.join(folder, 'header.json')
    if os.path.exists(path):
        assert json.load(open(path)) == h, f'{slug}: published header differs from header.json'
    else:
        with open(path, 'w') as f:
            json.dump(h, f, indent=1, ensure_ascii=False)
            f.write('\n')
    base = 8 + n
    size = published_size(url)

    # Every layer's `comfy_quant` (72 bytes each): the recognition needs their text, and every one
    # must be the same convrot description.
    cq_names = sorted(k for k in h if k.endswith('.comfy_quant'))
    comfy_quant = {}
    for k in cq_names:
        a, b = h[k]['data_offsets']
        comfy_quant[k] = fetch(url, base + a, base + b - 1).decode('utf-8')
    configs = {t for t in comfy_quant.values()}
    parsed = [json.loads(t) for t in configs]
    assert all(p.get('format') == 'int8_tensorwise' and p.get('convrot') is True for p in parsed), configs
    group = {int(p.get('convrot_groupsize', 256)) for p in parsed}
    assert len(group) == 1, group
    G = group.pop()
    # Any other U8 tensor is a JSON text too: Comfy-Org's Turbo carries
    # `transformer_blocks.{1..31}.attn.comfy_attention.config` = {"attention": "comfy_kitchen_int8"},
    # a ComfyUI attention-backend choice, not a weight; recorded for the recognition, not a piece.
    other_u8 = sorted(k for k, v in h.items() if k != '__metadata__' and v['dtype'] == 'U8' and k not in comfy_quant)
    comfy_other = {}
    for k in other_u8:
        a, b = h[k]['data_offsets']
        comfy_other[k] = fetch(url, base + a, base + b - 1).decode('utf-8')

    quantized, plain = choose_convrot(h)
    raw, ranges, row_ranges, shapes = {}, {}, {}, {}

    def read_rows(key, spans):
        v = h[key]
        a, b = v['data_offsets']
        row = (b - a) // v['shape'][0]
        data = b''
        ranges[key] = []
        for r0, r1 in spans:
            first, last = base + a + r0 * row, base + a + r1 * row - 1
            data += fetch(url, first, last)
            ranges[key].append([first, last])
        raw[key] = data
        row_ranges[key] = [list(s) for s in spans]
        shapes[key] = [sum(r1 - r0 for r0, r1 in spans)] + v['shape'][1:]

    for w in quantized:
        nrows, parts = h[w]['shape'][0], fused_parts(w)
        spans = ([(k * nrows // parts, k * nrows // parts + rows) for k in range(parts)]
                 if rows and nrows > rows * parts else [(0, nrows)])
        layer = w[:-len('.weight')]
        read_rows(w, spans)
        read_rows(layer + '.weight_scale', spans)
        read_rows(layer + '.comfy_quant', [(0, h[layer + '.comfy_quant']['shape'][0])])
        shapes[layer + '.comfy_quant'] = h[layer + '.comfy_quant']['shape']
    for k in plain:
        read_rows(k, [(0, h[k]['shape'][0])])

    tensors = {k: torch.frombuffer(bytearray(raw[k]), dtype=TORCH_DTYPE[h[k]['dtype']] if h[k]['dtype'] != 'U8'
                                   else torch.uint8).reshape(shapes[k]) for k in raw}
    pieces_path = os.path.join(folder, 'pieces.safetensors')
    write_safetensors(pieces_path, tensors, {'siliconed.pieces': json.dumps(
        {k: dict(published_shape=h[k]['shape'], shape=shapes[k], rows=row_ranges[k]) for k in sorted(raw)})})
    with open(pieces_path, 'rb') as f:
        m = struct.unpack('<Q', f.read(8))[0]
        ph = json.loads(f.read(m))
        blob = f.read()
    for k in raw:
        a, b = ph[k]['data_offsets']
        assert ph[k]['dtype'] == h[k]['dtype'] and blob[a:b] == raw[k], f'{slug}: {k} not byte-identical'

    # The references: Comfy's fp32 dequantization, the exact product in fp64, under published names…
    h64 = utils._build_hadamard(G, dtype=torch.float64)
    h_int = (h64 * G ** 0.5).round().long()
    assert torch.equal(h64, h64.T) and torch.equal(h64 @ h64, torch.eye(G, dtype=torch.float64))
    assert torch.equal(h_int.double() / G ** 0.5, h64)
    comparisons, published_refs, published_exact = {}, {}, {}
    for w in quantized:
        layer = w[:-len('.weight')]
        q, s = tensors[w], tensors[layer + '.weight_scale']
        assert q.shape[1] % G == 0 and s.dtype == torch.float32 and list(s.shape) == [q.shape[0], 1]
        ref = comfy_dequantize(q, s, G)
        assert ref.dtype == torch.float32 and ref.shape == q.shape
        exact = utils._rotate_weight(q.double() * s.double(), h64, G)
        m = (q.long().reshape(q.shape[0], -1, G) @ h_int.T).reshape(q.shape)       # integer, exact
        assert torch.equal(exact, s.double() * m.double() / G ** 0.5), f'{w}: the fp64 witness is not exact'
        rounded = exact.float()
        bf16 = ref.to(torch.bfloat16)
        rel = (bf16.double() - exact).abs() / exact.abs().clamp_min(1e-300)
        ce_ref, ce_round, ce_bf16 = channel_error(ref, exact), channel_error(rounded, exact), channel_error(bf16, exact)
        comparisons[w] = dict(
            kind='ComfyUI int8_tensorwise convrot', group=G, values=ref.numel(),
            comfy_fp32_vs_exact_fp64=dict(different_from_rounded_exact=int((ref != rounded).sum()),
                                          max_abs=float((ref.double() - exact).abs().max()),
                                          channel_error_worst=ce_ref[0], channel_error_median=ce_ref[1]),
            rounded_exact_vs_exact_fp64=dict(channel_error_worst=ce_round[0], channel_error_median=ce_round[1]),
            comfy_default_bf16=dict(different_from_fp32_reference=int((bf16.float() != ref).sum()),
                                    max_rel=float(rel[exact != 0].max()),
                                    channel_error_worst=ce_bf16[0], channel_error_median=ce_bf16[1]),
            smallest_nonzero_abs_qR=float((m[m != 0].abs().min().double() / G ** 0.5)),
            smallest_scale=float(s.min()))
        published_refs[w], published_exact[w] = ref, exact

    # … then under the map's names (diffusers' converter for Z-Image, the prefix and gate_up for Qwen).
    family = src['family']
    references, produced = {}, {}
    for w in quantized:
        one = to_map_names(family, {w: published_refs[w]})
        ex = to_map_names(family, {w: published_exact[w]})
        produced[w] = sorted(one)
        for k in one:
            references[f'reference.{k}'] = one[k].contiguous()
            references[f'exact64.{k}'] = ex[k].contiguous()
    for k in plain:
        one = to_map_names(family, {k: tensors[k].float()})
        produced[k] = sorted(one)
        for name, v in one.items():
            references[f'reference.{name}'] = v.contiguous()
    ref_path = os.path.join(folder, 'references.safetensors')
    write_safetensors(ref_path, references, {'siliconed.references':
        'reference.<w>: ComfyUI dequantize_int8_convrot_weight in fp32; exact64.<w>: (q·s)·R in fp64; map names'})

    sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
    manifest = dict(
        repo=src['repo'], revision=src['revision'], file=src['file'], url=url, format='convrot',
        family=family, file_bytes=size,
        header=dict(bytes=n, sha256=hashlib.sha256(raw_header).hexdigest(), tensors=len(h) - ('__metadata__' in h)),
        rows=rows or 'all', group=G,
        comfy=dict(manifest='../comfy/manifest.json',
                   functions='kitchen/comfy_kitchen/backends/eager/quantization.py: dequantize_int8_convrot_weight; '
                             'kitchen/comfy_kitchen/tensor/int8_utils.py: _build_hadamard, _rotate_weight'),
        libraries={p_: version(p_) for p_ in ('torch', 'diffusers', 'safetensors')},
        quantized=[m_ for w in quantized for m_ in produced[w]],
        plain=[m_ for k in plain for m_ in produced[k]],
        published={k: dict(dtype=h[k]['dtype'], shape=h[k]['shape'], piece_shape=shapes[k], rows=row_ranges[k],
                           range=ranges[k], map=produced.get(k, [])) for k in sorted(raw)},
        comparisons=comparisons,
        comfy_quant=comfy_quant,
        **(dict(comfy_other_u8=comfy_other) if comfy_other else {}),
        pieces=dict(file='pieces.safetensors', bytes=os.path.getsize(pieces_path), sha256=sha(pieces_path)),
        references=dict(file='references.safetensors', bytes=os.path.getsize(ref_path), sha256=sha(ref_path)),
    )
    with open(os.path.join(folder, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
        f.write('\n')
    return manifest


# ---------------------------------------------------------------- text encoders
#
# The two 8-bit encoders of the Compact install (Compact and Light installs), read by `Range` like the DiTs:
#
# * Disty0's SDNQ int8 `text_encoder/` of Z-Image (Qwen3-4B, names `model.*`): `<l>.weight` I8 [N, K]
#   + `<l>.scale` BF16 [N, 1], embed and norms BF16, no `lm_head`. Reference: SDNQ's
#   `dequantize_weight` with the scale widened to fp32 first (exact).
# * unsloth's `Qwen-Image-2.1-text_encoder-INT8-ConvRot` (Qwen3-VL-8B, HF names `model.language_model.*`,
#   `model.visual.*`): the language model's linears ComfyUI `int8_tensorwise` convrot (group 256,
#   `weight_scale` F32 [N, 1], `comfy_quant` per layer), embed, norms and the whole vision tower BF16,
#   `lm_head` dropped. Reference: Comfy's `dequantize_int8_convrot_weight` in fp32 (comfy_kitchen at the
#   frozen version of `COMFY`), plus the exact `(q·s)·R` in fp64.
#
# Written under Tests/SiliconedTests/Fixtures/quant/<slug>/: header.json and config.json (versioned: the published
# header, and the config the forge's plan reads — Disty0's own `text_encoder/config.json`; for unsloth's
# single file, the vendor's `text_encoder/config.json` at the revision `Forge/Installation.swift` pins),
# pieces.safetensors (published names, published bytes: row slices of the 8-bit weights with their
# scale rows and `comfy_quant`, row slices of `embed_tokens`, whole norms and vision tensors),
# references.safetensors under the **published** names (`reference.<w>` F32 [rows, K], `exact64.<w>`
# F64 for convrot): the check undoes nothing in them — it applies the map's transposition and per-head
# permutation itself, written there, not taken from the forge.

SOURCES.update({
    'z-image-encoder-sdnq-int8': dict(
        repo='Disty0/Z-Image-Turbo-SDNQ-int8', revision='43d0947f91c14b93da75704be27565ee30694f21',
        file='text_encoder/model.safetensors', config='text_encoder/config.json',
        format='encoder', scheme='sdnq', family='z-image'),
    'qwen21-encoder-convrot': dict(
        repo='unsloth/Qwen-Image-2.1-FP8', revision='bb21a8ba7f0371c19ad1f72c82a5e8fc10f39939',
        file='Qwen-Image-2.1-text_encoder-INT8-ConvRot.safetensors',
        config_from=dict(repo='Qwen/Qwen-Image-2.1', revision='d26bb61231c349cf6b7896fa83353113880e1ba3',
                         path='text_encoder/config.json'),
        format='encoder', scheme='convrot', family='qwen-image-2.1'),
    # Krea 2's Compact encoder: Comfy-Org's int8 convrot of Qwen3-VL-4B, `embed_tokens` rotated
    # too. Its config is Krea's (gated: read from the local snapshot, the bytes `Krea2Published` holds).
    'krea2-encoder-convrot': dict(
        repo='Comfy-Org/Krea-2', revision='1171f4dc7e325bf3734586942e737872506c73e1',
        file='text_encoders/qwen3vl_4b_int8_convrot.safetensors',
        config_from=dict(repo='krea/Krea-2-Turbo', revision='98e0fe118d17c9e3547fbb2e25acdbae2cadf7c7',
                         path='text_encoder/config.json', local=True),
        format='encoder', scheme='convrot', family='krea2'),
})


def choose_encoder(h, rows, vision=True):
    """Deterministic, from the header alone: the first layer's q_proj (several heads), k_proj, the
    widest-K MLP linear, q_norm (permuted per head) and input_layernorm, rows of embed_tokens; for a VL
    model, the first vision block's fused qkv (weight and bias, whole: its thirds are permuted),
    patch_embed and pos_embed — unless the family's forge drops the vision tower (Krea 2 reads text
    only). → {published name: [row spans]}."""
    lm = 'model.language_model.' if 'model.language_model.embed_tokens.weight' in h else 'model.'
    l0 = lm + 'layers.0.'
    mlp = max((k for k in h if k.startswith(l0 + 'mlp.') and k.endswith('.weight')), key=lambda k: (h[k]['shape'][1], k))
    chosen = {l0 + 'self_attn.q_proj.weight': [(0, rows)], l0 + 'self_attn.k_proj.weight': [(0, rows)],
              mlp: [(0, rows)], l0 + 'self_attn.q_norm.weight': None, l0 + 'input_layernorm.weight': None,
              # Two spans of the table: its first rows and rows of the image tokens' region.
              lm + 'embed_tokens.weight': [(0, rows // 2), (151643, 151643 + rows // 2)]}
    if vision and 'model.visual.blocks.0.attn.qkv.weight' in h:
        for k in ('blocks.0.attn.qkv.weight', 'blocks.0.attn.qkv.bias', 'patch_embed.proj.weight', 'pos_embed.weight'):
            chosen['model.visual.' + k] = None
    return chosen


def build_encoder(slug, rows):
    src = SOURCES[slug]
    url = url_of(src, src['file'])
    folder = os.path.join(ROOT, slug)
    os.makedirs(folder, exist_ok=True)
    convrot = src['scheme'] == 'convrot'
    if convrot:
        if not os.path.exists(os.path.join(COMFY_DIR, 'kitchen', 'comfy_kitchen', 'tensor', 'int8_utils.py')):
            fetch_comfy()
        utils, comfy_dequantize = comfy_functions()

    n = struct.unpack('<Q', fetch(url, 0, 7))[0]
    raw_header = fetch(url, 8, 8 + n - 1)
    h = json.loads(raw_header)
    path = os.path.join(folder, 'header.json')
    if os.path.exists(path):
        assert json.load(open(path)) == h, f'{slug}: published header differs from header.json'
    else:
        with open(path, 'w') as f:
            json.dump(h, f, indent=1, ensure_ascii=False)
            f.write('\n')
    if 'config' in src:
        config_url = url_of(src, src['config'])
    else:
        c = src['config_from']
        config_url = f"https://huggingface.co/{c['repo']}/resolve/{c['revision']}/{c['path']}"
    if src.get('config_from', {}).get('local'):
        # A gated repository: the file of the local snapshot (`hf download`, with a token).
        config_bytes = open(os.path.join(hub_snapshot(c['repo'], c['revision']), c['path']), 'rb').read()
    else:
        config_bytes = fetch(config_url)
    with open(os.path.join(folder, 'config.json'), 'wb') as f:
        f.write(config_bytes)
    base = 8 + n
    size = published_size(url)

    chosen = choose_encoder(h, rows, vision=src['family'] != 'krea2')
    raw, ranges, row_ranges, shapes = {}, {}, {}, {}

    def read_rows(key, spans):
        v = h[key]
        a, b = v['data_offsets']
        nrows = v['shape'][0] if v['shape'] else 1
        spans = spans or [(0, nrows)]
        row = (b - a) // nrows
        data, ranges[key] = b'', []
        for r0, r1 in spans:
            first, last = base + a + r0 * row, base + a + r1 * row - 1
            data += fetch(url, first, last)
            ranges[key].append([first, last])
        raw[key] = data
        row_ranges[key] = [list(s) for s in spans]
        shapes[key] = [sum(r1 - r0 for r0, r1 in spans)] + v['shape'][1:]

    quantized = []
    for k, spans in chosen.items():
        read_rows(k, spans)
        layer = k[:-len('.weight')]
        if h[k]['dtype'] == 'I8':
            quantized.append(k)
            scale = layer + ('.weight_scale' if convrot else '.scale')
            read_rows(scale, spans)
            if convrot:
                read_rows(layer + '.comfy_quant', None)
                shapes[layer + '.comfy_quant'] = h[layer + '.comfy_quant']['shape']

    tensors = {k: torch.frombuffer(bytearray(raw[k]), dtype=TORCH_DTYPE[h[k]['dtype']] if h[k]['dtype'] != 'U8'
                                   else torch.uint8).reshape(shapes[k]) for k in raw}
    pieces_path = os.path.join(folder, 'pieces.safetensors')
    write_safetensors(pieces_path, tensors, {'siliconed.pieces': json.dumps(
        {k: dict(published_shape=h[k]['shape'], shape=shapes[k], rows=row_ranges[k]) for k in sorted(raw)})})
    with open(pieces_path, 'rb') as f:
        m = struct.unpack('<Q', f.read(8))[0]
        ph = json.loads(f.read(m))
        blob = f.read()
    for k in raw:
        a, b = ph[k]['data_offsets']
        assert ph[k]['dtype'] == h[k]['dtype'] and blob[a:b] == raw[k], f'{slug}: {k} not byte-identical'

    references, comparisons, group = {}, {}, None
    for w in quantized:
        layer = w[:-len('.weight')]
        q = tensors[w]
        if convrot:
            cq = json.loads(raw[layer + '.comfy_quant'].decode('utf-8'))
            assert cq.get('format') == 'int8_tensorwise' and cq.get('convrot') is True, cq
            G = int(cq.get('convrot_groupsize', 256))
            group = G
            s = tensors[layer + '.weight_scale']
            assert s.dtype == torch.float32 and list(s.shape) == [q.shape[0], 1] and q.shape[1] % G == 0
            ref = comfy_dequantize(q, s, G)
            h64 = utils._build_hadamard(G, dtype=torch.float64)
            h_int = (h64 * G ** 0.5).round().long()
            assert torch.equal(h_int.double() / G ** 0.5, h64)
            exact = utils._rotate_weight(q.double() * s.double(), h64, G)
            mm = (q.long().reshape(q.shape[0], -1, G) @ h_int.T).reshape(q.shape)
            assert torch.equal(exact, s.double() * mm.double() / G ** 0.5), f'{w}: the fp64 witness is not exact'
            references[f'exact64.{w}'] = exact.contiguous()
            ce_ref, ce_round = channel_error(ref, exact), channel_error(exact.float(), exact)
            comparisons[w] = dict(kind='ComfyUI int8_tensorwise convrot', group=G, values=ref.numel(),
                                  comfy_fp32_vs_exact_fp64=dict(different_from_rounded_exact=int((ref != exact.float()).sum()),
                                                                channel_error_worst=ce_ref[0], channel_error_median=ce_ref[1]),
                                  rounded_exact_vs_exact_fp64=dict(channel_error_worst=ce_round[0],
                                                                   channel_error_median=ce_round[1]))
        else:
            s = tensors[layer + '.scale']
            assert s.dtype == torch.bfloat16 and list(s.shape) == [q.shape[0], 1]
            ref, default = dequant_sdnq(q, s)
            comparisons[w] = dict(kind='SDNQ int8 symmetric', **compare(ref, q, s, None, default))
        assert ref.dtype == torch.float32 and ref.shape == q.shape
        references[f'reference.{w}'] = ref.contiguous()
    ref_path = os.path.join(folder, 'references.safetensors')
    write_safetensors(ref_path, references, {'siliconed.references':
        'reference.<w>: the publisher\'s dequantization in fp32' + ('; exact64.<w>: (q·s)·R in fp64' if convrot else '')
        + '; published names and layout'})

    sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
    manifest = dict(
        repo=src['repo'], revision=src['revision'], file=src['file'], url=url, format='encoder',
        scheme=src['scheme'], family=src['family'], file_bytes=size,
        header=dict(bytes=n, sha256=hashlib.sha256(raw_header).hexdigest(), tensors=len(h) - ('__metadata__' in h)),
        config=dict(url=config_url, sha256=hashlib.sha256(config_bytes).hexdigest()),
        rows=rows, group=group,
        libraries={p_: version(p_) for p_ in ('torch', 'safetensors') + (() if convrot else ('sdnq',))},
        quantized=quantized,
        plain=[k for k in chosen if k not in quantized],
        published={k: dict(dtype=h[k]['dtype'], shape=h[k]['shape'], piece_shape=shapes[k], rows=row_ranges[k],
                           range=ranges[k]) for k in sorted(raw)},
        comparisons=comparisons,
        pieces=dict(file='pieces.safetensors', bytes=os.path.getsize(pieces_path), sha256=sha(pieces_path)),
        references=dict(file='references.safetensors', bytes=os.path.getsize(ref_path), sha256=sha(ref_path)),
    )
    if convrot:
        manifest['comfy'] = dict(manifest='../comfy/manifest.json',
                                 functions='kitchen/comfy_kitchen/backends/eager/quantization.py: dequantize_int8_convrot_weight; '
                                           'kitchen/comfy_kitchen/tensor/int8_utils.py: _build_hadamard, _rotate_weight')
        # Every layer's `comfy_quant` text, for the recognition's sparse file (72 bytes each).
        cq_names = sorted(k for k in h if k.endswith('.comfy_quant'))
        manifest['comfy_quant'] = {k: fetch(url, base + h[k]['data_offsets'][0], base + h[k]['data_offsets'][1] - 1)
                                   .decode('utf-8') for k in cq_names}
    with open(os.path.join(folder, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
        f.write('\n')
    return manifest


# ---------------------------------------------------------------- against the bf16 DiTs

def hub_snapshot(repo, revision):
    from huggingface_hub.constants import HF_HUB_CACHE      # HF_HOME / HF_HUB_CACHE apply
    return os.path.join(HF_HUB_CACHE, 'models--' + repo.replace('/', '--'), 'snapshots', revision)


class DiTRows:
    """Rows [0, n) of a diffusers DiT tensor, as float32: from the local snapshot when it is there,
    else by `Range` (the index, the shard's header, then only the rows' bytes)."""

    def __init__(self, repo, revision):
        self.repo, self.revision, self.local = repo, revision, hub_snapshot(repo, revision)
        self.src = dict(repo=repo, revision=revision)
        index = 'transformer/diffusion_pytorch_model.safetensors.index.json'
        if os.path.isdir(os.path.join(self.local, 'transformer')):
            self.where = 'Hugging Face cache (local)'
            self.index = json.load(open(os.path.join(self.local, index)))['weight_map']
        else:
            self.where = 'Range'
            self.index = json.loads(fetch(url_of(self.src, index)))['weight_map']
        self.headers = {}

    def rows(self, name, n):
        shard = 'transformer/' + self.index[name]
        if self.where != 'Range':
            with safe_open(os.path.join(self.local, shard), 'pt') as f:
                t = f.get_slice(name)
                return (t[:n] if len(t.get_shape()) > 1 else t[:]).float().clone()
        url = url_of(self.src, shard)
        if shard not in self.headers:
            m = struct.unpack('<Q', fetch(url, 0, 7))[0]
            self.headers[shard] = (8 + m, json.loads(fetch(url, 8, 8 + m - 1)))
        base, h = self.headers[shard]
        v = h[name]
        a, b = v['data_offsets']
        shape = v['shape']
        row = (b - a) // shape[0] if len(shape) > 1 else b - a
        n = n if len(shape) > 1 else 1
        data = fetch(url, base + a, base + a + n * row - 1)
        t = torch.frombuffer(bytearray(data), dtype=TORCH_DTYPE[v['dtype']])
        return t.reshape([n] + shape[1:] if len(shape) > 1 else shape).float()


def dit_distance(ref, dit):
    """channelError worst and median (row = channel) and the global relative error."""
    ref, dit = ref.double().reshape(ref.shape[0], -1) if ref.dim() > 1 else ref.double()[None], \
        dit.double().reshape(dit.shape[0], -1) if dit.dim() > 1 else dit.double()[None]
    worst, median = channel_error(ref, dit)
    return dict(channel_error_worst=worst, channel_error_median=median,
                relative=float((ref - dit).norm() / dit.norm()), identical=bool(torch.equal(ref, dit)))


def compare_dits(slug, manifest):
    """`vs_dit` in the manifest: every reference (dequantized piece, or plain tensor widened) of this
    source against the same rows of the Turbo's bf16 DiT and of the base's; and, as calibration, the
    base source of the same third party (its existing fixtures) against both. The stop: a
    Turbo piece closer to the base than to the Turbo."""
    dits = SOURCES[slug]['dits']
    readers = {k: DiTRows(**dits[k]) for k in ('turbo', 'base')}

    def against(folder):
        out = {}
        with safe_open(os.path.join(ROOT, folder, 'references.safetensors'), 'pt') as f:
            for key in sorted(f.keys()):
                if not key.startswith('reference.'):
                    continue
                name = key[len('reference.'):]
                ref = f.get_tensor(key)
                n = ref.shape[0] if ref.dim() > 1 else 0
                rows = {k: r.rows(name, n) for k, r in readers.items()}
                c = {k: dit_distance(ref, v) for k, v in rows.items()}
                # Where the piece sits on the line base → Turbo: β = <ref − base, d> / ‖d‖², d = Turbo − base.
                # A 4-bit step (~7 % of a row) dwarfs d (~0.2–0.5 %), so the two distances barely differ;
                # β does not: the quantization error is uncorrelated with d, β ≈ 1 for a quantized Turbo,
                # ≈ 0 for a quantized base, with a spread of about ‖error‖ / (‖d‖·√n).
                d = (rows['turbo'] - rows['base']).double().reshape(-1)
                e = (ref.double().reshape(-1) - rows['base'].double().reshape(-1))
                dd = float(d @ d)
                c['beta'] = float(e @ d) / dd if dd else None
                c['turbo_minus_base_relative'] = float(d.norm() / rows['turbo'].double().norm()) if dd else 0.0
                if c['beta'] is None:
                    c['closer_to'] = 'both (Turbo = base here)'
                else:
                    c['closer_to'] = 'turbo' if c['beta'] > 0.5 else 'base'
                out[name] = c
        return out

    own = against(slug)
    calib_slug = dits['calibration'][manifest['format']]
    calibration = against(calib_slug)
    manifest['vs_dit'] = dict(
        dits={k: dict(repo=r.repo, revision=r.revision, read=r.where) for k, r in readers.items()},
        note='reference.<w> (fp32) against rows [0, n) of the bf16 DiT widened to fp32; channel = output row',
        pieces=own, calibration=dict(source=calib_slug, pieces=calibration))
    with open(os.path.join(ROOT, slug, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
        f.write('\n')
    wrong = [n for n in manifest['quantized'] + manifest['plain'] if own[n]['closer_to'] == 'base']
    wrong_calib = [n for n in calibration if calibration[n]['closer_to'] == 'turbo']
    for n in manifest['quantized'] + manifest['plain']:
        c, b = own[n], calibration.get(n)
        fmt = lambda x: '   -  ' if x is None else f'{x:6.3f}'
        line = (f"  {n:46s} vs turbo {c['turbo']['channel_error_worst']:.3e}/{c['turbo']['channel_error_median']:.3e}"
                f"  vs base {c['base']['channel_error_worst']:.3e}/{c['base']['channel_error_median']:.3e}"
                f"  |d| {c['turbo_minus_base_relative']:.2e}  beta {fmt(c['beta'])}")
        if b:
            line += (f"   [{calib_slug}: vs base {b['base']['channel_error_median']:.3e}, "
                     f"vs turbo {b['turbo']['channel_error_median']:.3e}, beta {fmt(b['beta'])}]")
        print(line)
    if wrong or wrong_calib:
        raise SystemExit(f'STOP: {slug} pieces closer to the base than to the Turbo: {wrong}; '
                         f'calibration pieces closer to the Turbo: {wrong_calib}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--rows', type=int, default=256, help='rows of each quantized weight (0 = all)')
    ap.add_argument('slugs', nargs='*', default=list(SOURCES))
    args = ap.parse_args()
    disagreements = 0
    for slug in args.slugs:
        if SOURCES[slug].get('format') == 'encoder':
            m = build_encoder(slug, args.rows)
            print(f"{slug}: pieces {m['pieces']['bytes'] / 1e6:.1f} MB sha256 {m['pieces']['sha256'][:16]}, "
                  f"references {m['references']['bytes'] / 1e6:.1f} MB sha256 {m['references']['sha256'][:16]}")
            for w, c in m['comparisons'].items():
                if 'group' in c:
                    r, e = c['comfy_fp32_vs_exact_fp64'], c['rounded_exact_vs_exact_fp64']
                    print(f"  {w:60s} Comfy fp32 vs fp64: {r['channel_error_worst']:.3e}, "
                          f"{r['different_from_rounded_exact']}/{c['values']} ≠ fl32(exact); fl32(exact): {e['channel_error_worst']:.3e}")
                else:
                    print(f"  {w:60s} q*s fp32: {c['vs_q_times_s_fp32']['different']} diff, "
                          f"fp64: {c['vs_exact_fp64']['different']} diff / {c['values']}")
            continue
        if SOURCES[slug].get('format') == 'convrot':
            m = build_convrot(slug, args.rows)
            if 'dits' in SOURCES[slug]:
                compare_dits(slug, m)
            print(f"{slug}: group {m['group']}, pieces {m['pieces']['bytes'] / 1e6:.1f} MB, "
                  f"references {m['references']['bytes'] / 1e6:.1f} MB")
            for w, c in m['comparisons'].items():
                r, e, d = c['comfy_fp32_vs_exact_fp64'], c['rounded_exact_vs_exact_fp64'], c['comfy_default_bf16']
                print(f"  {w:50s} Comfy fp32 vs fp64: {r['channel_error_worst']:.3e} (median "
                      f"{r['channel_error_median']:.3e}), {r['different_from_rounded_exact']}/{c['values']} ≠ fl32(exact)   "
                      f"fl32(exact): {e['channel_error_worst']:.3e}   bf16 default: {d['channel_error_worst']:.3e}")
            continue
        if SOURCES[slug].get('format') == 'gguf':
            m = build_gguf(slug, args.rows)
            if 'dits' in SOURCES[slug]:
                compare_dits(slug, m)
            print(f"{slug}: pieces {m['pieces']['bytes'] / 1e6:.1f} MB, references {m['references']['bytes'] / 1e6:.1f} MB")
            print(f"  types {m['types']}")
            for w, c in m['comparisons'].items():
                d = c['publisher_default_path']
                own = (f"d*q fp32: {c['vs_d_times_q_fp32']['different']:>7d} diff" if 'vs_d_times_q_fp32' in c
                       else f"fl32(fp64) {c['fl32_of_exact']['different_bits']:>4d} bits diff")
                print(f"  {w:62s} {c['kind'][5:]:5s} {c['values']:>8d} values   gguf-py: "
                      f"{c['gguf_py']['different_bits']} bits diff   {own}   "
                      f"fp64: {c['vs_exact_fp64']['different']:>7d} diff   "
                      f"default {d['dtype']}: {d['different_from_fp32_reference']} diff, max rel {d['max_rel']:.2e}")
                disagreements += c['gguf_py']['different_bits']
            continue
        m = build(slug, args.rows)
        print(f"{slug}: {m['pieces']['bytes'] / 1e6:.1f} MB  sha256 {m['pieces']['sha256'][:16]}")
        for w, c in m['comparisons'].items():
            d = c.get('publisher_default_path', {})
            print(f"  {w:55s} q*s fp32: {c['vs_q_times_s_fp32']['different']:>7d} diff   "
                  f"fp64: {c['vs_exact_fp64']['different']:>7d} diff   "
                  f"default {d.get('dtype', '-')}: {d.get('different_from_fp32_reference', 0)} diff, "
                  f"max rel {d.get('max_rel', 0):.2e}")
    # The stop: the two GGUF references (diffusers, gguf-py) must agree at the bit.
    if disagreements:
        raise SystemExit(f'STOP: gguf-py and diffusers differ on {disagreements} values')


if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    main()
