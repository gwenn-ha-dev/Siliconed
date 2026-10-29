English · [Français](VERIFY.fr.md)

# Verifying Siliconed

Siliconed claims that each stage of each model it ships computes what the reference implementation computes. **This page shows how to check that claim yourself**, on your Mac, without trusting our figures.

The reference is the publisher's own: [diffusers](https://github.com/huggingface/diffusers) and [transformers](https://github.com/huggingface/transformers), in float32 on the CPU, with the published weights. Python scripts in this repository run it and freeze its answers into *goldens*. `siliconed check` runs the engine on the same inputs and judges it against them, stage by stage: text encoder, denoiser, sampler, decoder.

What is **not** claimed: the same bits as the reference, or as another Mac. The engine computes in another order on another device; it is judged on how close it comes to the exact value (the [rule](#the-rule) below), not on its bits.

The figures here, and the checks, are those of the render at full size, Faster Rendering off. Faster Rendering produces, by construction, another image from the same seed, and nothing compares that image with the reference ([ENGINE](ENGINE.md#faster-rendering)).

## Setting up

You need the app (0.17 or later), Python 3.12, and this repository at the app's version (Siliconed → About Siliconed shows it). `siliconed` ships inside the app: it is the very engine the app renders with, and it runs without Xcode.

```sh
alias siliconed=/Applications/Siliconed.app/Contents/Helpers/siliconed
git clone --branch v0.17 https://github.com/gwenn-ha-dev/Siliconed && cd Siliconed
python3.12 -m venv oracle-env && oracle-env/bin/pip install -r tools/oracle/requirements.txt
```

The first run of an oracle script downloads the reference weights from the Hugging Face Hub, at pinned revisions (sizes in [section 3](#3--every-model)); `HF_HOME` or `HF_HUB_CACHE` decides where they go. Krea 2's reference, `krea/Krea-2-Turbo`, is gated: accept its license on the Hub, then `oracle-env/bin/hf auth login`, before `krea2.py`; do the same for any other repository that asks for it. The int8 convrot code that two of the Compact versions need is fetched, once, at a pinned commit and checked against its sha256.

To check your own build instead: `swift build -c release --product siliconed` (Xcode 26), then `.build/release/siliconed` in place of the alias.

Each stage prints the map it reads (a `map` line), its measures, then one verdict line: `✓`, `✗`, or `—` (skipped, and why). The exit code says the same: 0 passed, 1 failed, 2 skipped, 64 a usage error. Goldens go to `./goldens`, for the scripts and the checks alike, unless `--goldens DIR` or `SILICONED_GOLDENS` says otherwise.

## 1 · The kernels: five minutes, no weights

The numerical kernels (RoPE tables, resampling, the GPU's element-wise operations, attention, the split of a matrix product between GPU and AMX) are checked without any model.

```sh
oracle-env/bin/python tools/oracle/kernels.py all
siliconed check kernels
```

On our MacBook Pro M1 Pro:

```
✓ kernels widen · 7 measures
✓ kernels arena · 6 measures
✓ kernels elementwise · 7 measures
✓ kernels gelu · 1 measure
✓ kernels sdpa S=4128 · 17 measures
✓ kernels gemm-split · 9 measures
✓ kernels rope · 13 measures
✓ kernels resample · 6 measures
```

The [kernels workflow](https://github.com/gwenn-ha-dev/Siliconed/actions/workflows/kernels.yml) runs these two commands on GitHub's Macs each time the engine or its checks change. Its GPU is virtual, so the GPU stages are skipped there; the CPU ones are judged.

## 2 · One whole model: Anima, ten minutes and 5 GB

Anima is the lightest model to check end to end. Install it in the app first (Standard): `siliconed check` reads the weights the app installed, never the ones the oracle downloads. Then:

```sh
oracle-env/bin/python tools/oracle/anima.py all     # downloads 5 GB of reference weights once, then 7 min
siliconed check anima                               # every stage at every size: 3 min
```

On our MacBook Pro M1 Pro (the measures between the `map` line and the verdict line are left out):

```
✓ anima tokenizer · 1 measure
map ~/Library/Application Support/Siliconed/store/qwen3-0.6b-encoder.v0.silicon · 1.2 GB · bfloat16 (standard weights)
✓ anima text · 9 measures · nearest its threshold: prompt_embeds 1.33e-05 of 5.00e-04
map ~/Library/Application Support/Siliconed/store/anima-turbo-dit.v0.silicon · 3.9 GB · bfloat16 (standard weights)
✓ anima dit 512 · 12 measures · nearest its threshold: model_out per plane 1.93e-06 of 2.51e-06
map ~/Library/Application Support/Siliconed/store/anima-turbo-dit.v0.silicon · 3.9 GB · bfloat16 (standard weights)
✓ anima dit 1024 · 12 measures · nearest its threshold: model_out per plane 3.04e-06 of 4.74e-06
map ~/Library/Application Support/Siliconed/store/anima-turbo-dit.v0.silicon · 3.9 GB · bfloat16 (standard weights)
✓ anima dit 768x512 · 12 measures · nearest its threshold: temb 4.14e-05 of 5.00e-04
map ~/Library/Application Support/Siliconed/store/anima-turbo-dit.v0.silicon · 3.9 GB · bfloat16 (standard weights)
✓ anima trajectory 512 · 9 measures · nearest its threshold: v_3 9.79e-05 of 5.00e-04
map ~/Library/Application Support/Siliconed/store/anima-turbo-dit.v0.silicon · 3.9 GB · bfloat16 (standard weights)
✓ anima trajectory 1024 · 9 measures · nearest its threshold: v_7 2.17e-04 of 5.00e-04
✓ anima vae 512 · 1 measure · nearest its threshold: image RGB 6.33e-07 of 1.28e-06
✓ anima vae 1024 · 1 measure · nearest its threshold: image RGB 3.18e-07 of 1.05e-06
✓ anima vae 768x512 · 2 measures · nearest its threshold: image RGB 3.66e-07 of 1.28e-06
```

Your figures will differ in the last digits: the reference itself rounds differently on another CPU. The verdicts should not. A stage can also be run alone, `siliconed check anima dit --size 1024`; `siliconed check list` names them all.

## 3 · Every model

The same two commands, for each model: `tools/oracle/<script> all`, then `siliconed check <model>`.

| Model | Script | Download | Oracle, every stage | Its heaviest stage | Check |
|---|---|---|---|---|---|
| Z-Image | `z-image.py` | 31 GB | 39 min | the 1024² trajectory: 14 min, 13.3 GB | 7 min |
| Qwen-Image-2.1 | `qwen-image-2.1.py` | 30 GB | 27 min | the fp64 decoder witness: 43 s, 15.5 GB | 10 min |
| Anima | `anima.py` | 5 GB | 7 min | the 1024² decoder: 1 min, 20.8 GB | 3 min |
| Krea 2 | `krea2.py` | 33 GB | 1 h 45 min | the eight fp64 step witnesses at 1024²: 44 min, 11 GB | 16 min |
| MOSS-Image | `moss-image.py` | 28 GB | 11 min | the 8-step trajectory: 3.6 min, 10.6 GB | 3 min |

On a MacBook Pro M1 Pro, 16 GB, wall time. The download comes on top of the app's own weights. The oracle streams the denoiser block by block in float32 (Krea 2's is 48 GB at that precision), so a 16 GB Mac runs every script; a stage whose peak passes the machine's memory (the 1024² decoders) runs in swap and takes longer.

The verdict lines of the other four models on our MacBook Pro M1 Pro, with the `map` line each stage prints (Anima's are in the section above; the measures are left out):

<details>
<summary>Z-Image</summary>

```
✓ z-image tokenizer · 3 measures
map ~/Library/Application Support/Siliconed/store/qwen3-4b-encoder.v0.silicon · 7.8 GB · bfloat16 (standard weights)
✓ z-image text · 3 measures · nearest its threshold: layer34_out 1.45e-05 of 5.00e-04
map ~/Library/Application Support/Siliconed/store/z-image-turbo-dit.v1.silicon · 12.3 GB · bfloat16 ×514, float32 ×7 (standard weights)
✓ z-image block 512 · 19 measures · nearest its threshold: layer0_adaln_raw 6.63e-05 of 5.00e-04
map ~/Library/Application Support/Siliconed/store/z-image-turbo-dit.v1.silicon · 12.3 GB · bfloat16 ×514, float32 ×7 (standard weights)
✓ z-image block 1024 · 19 measures · nearest its threshold: layer0_adaln_raw 6.63e-05 of 5.00e-04
map ~/Library/Application Support/Siliconed/store/z-image-turbo-dit.v1.silicon · 12.3 GB · bfloat16 ×514, float32 ×7 (standard weights)
✓ z-image dit 512 · 11 measures · nearest its threshold: t_embedder_out 3.73e-04 of 5.00e-04
map ~/Library/Application Support/Siliconed/store/z-image-turbo-dit.v1.silicon · 12.3 GB · bfloat16 ×514, float32 ×7 (standard weights)
✓ z-image dit 1024 · 11 measures · nearest its threshold: t_embedder_out 3.73e-04 of 5.00e-04
map ~/Library/Application Support/Siliconed/store/z-image-turbo-dit.v1.silicon · 12.3 GB · bfloat16 ×514, float32 ×7 (standard weights)
✓ z-image dit 768x512 · 12 measures · nearest its threshold: t_embedder_out 3.73e-04 of 5.00e-04
map ~/Library/Application Support/Siliconed/store/z-image-turbo-dit.v1.silicon · 12.3 GB · bfloat16 ×514, float32 ×7 (standard weights)
✓ z-image dit 832x1216 · 12 measures · nearest its threshold: t_embedder_out 3.73e-04 of 5.00e-04
map ~/Library/Application Support/Siliconed/store/z-image-turbo-dit.v1.silicon · 12.3 GB · bfloat16 ×514, float32 ×7 (standard weights)
✓ z-image trajectory 512 · 16 measures · nearest its threshold: v_5 7.41e-05 of 5.00e-04
map ~/Library/Application Support/Siliconed/store/z-image-turbo-dit.v1.silicon · 12.3 GB · bfloat16 ×514, float32 ×7 (standard weights)
✓ z-image trajectory 1024 · 16 measures · nearest its threshold: v_3 1.65e-04 of 5.00e-04
✓ z-image vae 768x512 · 2 measures · nearest its threshold: image RGB 2.52e-07 of 8.50e-07
```

</details>

<details>
<summary>Qwen-Image-2.1</summary>

```
map ~/Library/Application Support/Siliconed/store/qwen3-vl-8b-encoder.v0.silicon · 16.3 GB · bfloat16 (standard weights)
✓ qwen-image-2.1 text · 11 measures · nearest its threshold: prompt_embeds 2.22e-05 of 5.00e-04
map ~/Library/Application Support/Siliconed/store/qwen3-vl-8b-encoder.v0.silicon · 16.3 GB · bfloat16 (standard weights)
✓ qwen-image-2.1 text --edit 1 · 16 measures · nearest its threshold: layer0_in 8.42e-05 of 5.00e-04
map ~/Library/Application Support/Siliconed/store/qwen3-vl-8b-encoder.v0.silicon · 16.3 GB · bfloat16 (standard weights)
✓ qwen-image-2.1 text --edit 2 · 16 measures · nearest its threshold: layer0_in 1.50e-04 of 5.00e-04
map ~/Library/Application Support/Siliconed/store/qwen3-vl-8b-encoder.v0.silicon · 16.3 GB · bfloat16 (standard weights)
✓ qwen-image-2.1 vision · 22 measures · nearest its threshold: vit_out_1 2.13e-04 of 5.00e-04
✓ qwen-image-2.1 references 512 · 2 measures
✓ qwen-image-2.1 references 443 · 2 measures
map ~/Library/Application Support/Siliconed/store/qwen-image-2.1-turbo-dit.v0.silicon · 14.2 GB · bfloat16 ×296, float32 ×1 (standard weights)
✓ qwen-image-2.1 dit 512 · 24 measures · nearest its threshold: step norm_out 1.74e-04 of 5.00e-04
map ~/Library/Application Support/Siliconed/store/qwen-image-2.1-turbo-dit.v0.silicon · 14.2 GB · bfloat16 ×296, float32 ×1 (standard weights)
✓ qwen-image-2.1 dit-edit1 512 · 24 measures · nearest its threshold: temb 2.11e-04 of 5.00e-04
map ~/Library/Application Support/Siliconed/store/qwen-image-2.1-turbo-dit.v0.silicon · 14.2 GB · bfloat16 ×296, float32 ×1 (standard weights)
✓ qwen-image-2.1 dit-edit2 512 · 24 measures · nearest its threshold: temb 2.11e-04 of 5.00e-04
map ~/Library/Application Support/Siliconed/store/qwen-image-2.1-turbo-dit.v0.silicon · 14.2 GB · bfloat16 ×296, float32 ×1 (standard weights)
✓ qwen-image-2.1 trajectory 512 · 11 measures · nearest its threshold: v_3 6.50e-06 of 5.00e-04
map ~/Library/Application Support/Siliconed/store/qwen-image-2.1-turbo-dit.v0.silicon · 14.2 GB · bfloat16 ×296, float32 ×1 (standard weights)
✓ qwen-image-2.1 trajectory-edit1 512 · 13 measures · nearest its threshold: v_1 2.02e-05 of 5.00e-04
map ~/Library/Application Support/Siliconed/store/qwen-image-2.1-turbo-dit.v0.silicon · 14.2 GB · bfloat16 ×296, float32 ×1 (standard weights)
✓ qwen-image-2.1 trajectory-edit2 512 · 14 measures · nearest its threshold: v_1 7.68e-06 of 5.00e-04
✓ qwen-image-2.1 vae 512 · 2 measures · nearest its threshold: image RGBA 4.51e-07 of 1.18e-06
✓ qwen-image-2.1 vae-edit1 512 · 3 measures · nearest its threshold: image RGBA 1.00e-06 of 5.00e-04
✓ qwen-image-2.1 vae-edit2 512 · 2 measures · nearest its threshold: image RGBA 9.64e-07 of 5.00e-04
✓ qwen-image-2.1 vae-encode 512 · 7 measures · nearest its threshold: picture 1 latent 4.27e-06 of 7.16e-06
```

</details>

<details>
<summary>Krea 2</summary>

```
map ~/Library/Application Support/Siliconed/store/qwen3-vl-4b-encoder.v0.silicon · 7.8 GB · bfloat16 (standard weights)
map ~/Library/Application Support/Siliconed/store/krea2-turbo-dit.v0.silicon · 26.3 GB · bfloat16 ×256, float32 ×174 (standard weights)
✓ krea2 text · 7 measures · nearest its threshold: tap 35 2.54e-05 of 5.00e-04
map ~/Library/Application Support/Siliconed/store/krea2-turbo-dit.v0.silicon · 26.3 GB · bfloat16 ×256, float32 ×174 (standard weights)
✓ krea2 dit 512 · 11 measures · nearest its threshold: model_out per plane 1.04e-04 of 1.14e-04
map ~/Library/Application Support/Siliconed/store/krea2-turbo-dit.v0.silicon · 26.3 GB · bfloat16 ×256, float32 ×174 (standard weights)
✓ krea2 dit 1024 · 11 measures · nearest its threshold: block0_attn_out 3.57e-03 of 3.59e-03
map ~/Library/Application Support/Siliconed/store/krea2-turbo-dit.v0.silicon · 26.3 GB · bfloat16 ×256, float32 ×174 (standard weights)
✓ krea2 dit 768x512 · 11 measures · nearest its threshold: block0_out 9.44e-04 of 1.20e-03
map ~/Library/Application Support/Siliconed/store/krea2-turbo-dit.v0.silicon · 26.3 GB · bfloat16 ×256, float32 ×174 (standard weights)
✓ krea2 trajectory 512 · 9 measures · nearest its threshold: v_2 5.08e-04 of 5.14e-04
map ~/Library/Application Support/Siliconed/store/krea2-turbo-dit.v0.silicon · 26.3 GB · bfloat16 ×256, float32 ×174 (standard weights)
✓ krea2 trajectory 1024 · 9 measures · nearest its threshold: v_1 3.43e-03 of 4.80e-03
✓ krea2 vae 512 · 1 measure · nearest its threshold: image RGB 5.20e-07 of 1.48e-06
✓ krea2 vae 1024 · 1 measure · nearest its threshold: image RGB 2.89e-07 of 1.03e-06
✓ krea2 vae 768x512 · 2 measures · nearest its threshold: image RGB 3.23e-07 of 9.63e-07
```

</details>

Two Krea 2 measures pass with little to spare on our Mac: the velocity `v_2` of the 512² trajectory (5.08e-4 for an allowance of 5.14e-4) and `block0_attn_out` of the 1024² denoiser (3.57e-3 for 3.59e-3, 0.6 % to spare). The allowance is computed from your reference's own error, so it moves with your CPU's rounding, and these two may fail by a hair on another Mac. The rule is not loosened for them. If they fail on yours, an issue with the output of `siliconed check` helps us find which layer carries the gap.

<details>
<summary>MOSS-Image</summary>

```
✓ moss-image tokenizer · 11 measures
map ~/Library/Application Support/Siliconed/store/qwen3.5-9b-encoder.v0.silicon · 15.9 GB · bfloat16 ×249, float32 ×177 (standard weights)
✓ moss-image text-delta · 8 measures · nearest its threshold: layer 0 out 3.61e-06 of 5.00e-04
map ~/Library/Application Support/Siliconed/store/qwen3.5-9b-encoder.v0.silicon · 15.9 GB · bfloat16 ×249, float32 ×177 (standard weights)
✓ moss-image text-full · 15 measures · nearest its threshold: layer3 gated attention 2.39e-05 of 5.00e-04
map ~/Library/Application Support/Siliconed/store/qwen3.5-9b-encoder.v0.silicon · 15.9 GB · bfloat16 ×249, float32 ×177 (standard weights)
✓ moss-image text · 44 measures · nearest its threshold: layer3 gated attention 2.39e-05 of 5.00e-04
✓ moss-image schedule · 25 measures
map ~/Library/Application Support/Siliconed/store/moss-image-sft-dit.v0.silicon · 10.2 GB · bfloat16 (standard weights)
✓ moss-image dit 512 · 23 measures · nearest its threshold: model_out per plane 1.35e-06 of 1.75e-06
map ~/Library/Application Support/Siliconed/store/moss-image-sft-dit.v0.silicon · 10.2 GB · bfloat16 (standard weights)
✓ moss-image dit-lora 512 · 22 measures · nearest its threshold: model_out per plane 2.72e-06 of 4.17e-06
map ~/Library/Application Support/Siliconed/store/moss-image-sft-dit.v0.silicon · 10.2 GB · bfloat16 (standard weights)
✓ moss-image trajectory 512 · 11 measures · nearest its threshold: v_6 per plane 2.97e-05 of 5.00e-04
✓ moss-image vae 512 · 4 measures · nearest its threshold: image RGB (encoded test image) 2.34e-07 of 7.28e-07
```

</details>

## Are your goldens ours?

We do not publish the goldens (several gigabytes, and weights whose licenses forbid it). We publish their fingerprints: shape, type and statistics of every tensor.

```sh
oracle-env/bin/python tools/oracle/fingerprints.py compare
```

Integers (token identifiers, masks) must match exactly. Floats are compared within a tolerance (1e-3 on norms), not bit for bit: PyTorch on the CPU does not promise the same float32 bits from one machine to another, and two honest runs of the reference differ by rounding. That rounding is far below what the checks judge.

## The rule

A stage's figure is the **relative error per channel**: for each channel (the last axis of a tensor, or each plane of a latent), the L2 norm of the difference over the L2 norm of the reference. The worst channel is reported, never an average: a fault confined to one channel would vanish in a global figure. The same rule judges every boundary of every model:

    without a witness:   error(engine, reference)  ≤ 5e-4
    with a witness:      error(engine, witness)    ≤ max(floor, 1.25 · error(reference, witness))

The *witness* is the same computation in float64. Where it exists, it decides: float32 has its own rounding error, and on an ill-conditioned boundary the reference itself can be 4e-3 away from the exact value; the engine may then be no further from the exact value than the reference is, plus 25 %.

Each stage's line in `check list` gives its own rule. An identity (tokens, schedules) is judged bit for bit. Every model's stages hold the same floor, 5e-4. A rectangle is checked against its transpose, which must fail, so that a swapped height and width cannot pass.

Every decoder, Z-Image's included, is judged per plane of the image against a float64 witness of the reference's decoder (`vae-witness`).

Krea 2's denoiser has a few boundaries so ill-conditioned that a change of one unit in the last place of the input moves the output by more than the rule allows, the reference included. There, a measure that fails at the witness is judged by *clouds*: the reference and the engine are both run on inputs moved by one ulp, and the engine's median error must stay within 1.25 times the reference's worst. These are the stages whose rule says "clouds when the oracle has draws".

## Beyond `all`

Two checks need goldens that `all` does not write, each from one more script stage. `siliconed check <model>` leaves them out; they run stage by stage.

**Long prompts.** The app reads up to 1,024 tokens of prompt and refuses a longer one; the reference pipelines cut at 512 by default. `text --long` checks the stretch between the two: the script reads `tools/oracle/inputs/long-prompt.txt` (844 tokens for the Qwen tokenizers, 954 for Anima's T5) with the reference's sequence length raised so that none of it is cut, and the check reads the same prompt the way the app admits it.

```sh
oracle-env/bin/python tools/oracle/z-image.py text --long
siliconed check z-image text --long
```

**The Neural Engine.** The app can share the denoiser's matrix products with the Mac's Neural Engine, which computes in fp16 ([ENGINE](ENGINE.md#the-neural-engine)). `--neural-engine` checks that path: every widened product of the DiT with 512 rows or more gives half its rows to the ANE, the same split on every Mac and more than any map gives (a render leaves products under 1,024 rows to the GPU), and the stage is judged against the reference run in bf16 on the same inputs, the engine allowed bf16's own error plus 25 %. The verdict says how many products the ANE computed; on a Mac where it computes none, the stage is skipped. The `dit` stages of the five models take it.

```sh
oracle-env/bin/python tools/oracle/z-image.py bf16 --size 1024
siliconed check z-image dit --size 1024 --neural-engine
```

**Activations rounded to bf16.** Where this Mac's map turns it on, the DiT's products take their activation rounded to bf16 just before the product (the weights as published, the sums and the output in fp32). `--bf16-activations` checks that path on any Mac, for every product of 256 rows or more, and judges the stage like the Neural Engine's: against the reference run in bf16, the engine allowed bf16's own error plus 25 %. The verdict says how many products rounded their operand. The `dit` stages of the five models take it.

```sh
siliconed check z-image dit --size 512 --bf16-activations
```

Qwen-Image-2.1's editing, with one and two reference pictures, is part of its `all` and of `siliconed check qwen-image-2.1`; one stage alone is `siliconed check qwen-image-2.1 dit --edit 2`. Its Neural Engine bar is written by `qwen-image-2.1.py bf16 --edit N`.

**The Compact and the Light.** Their DiTs are third parties' files (8-bit GGUF Q8_0 and Comfy-Org's int8 convrot for the Compact, GGUF Q4_K_M for the Light; the app pins each by repository, revision and sha256). Two things are checked, both from outside.

*The dequantization*, value by value: `swift test` reads small pieces of each published file (`Tests/SiliconedTests/Fixtures/quant`, rows cut by HTTP `Range` at the pinned revision, bytes identical to the published ones) and compares the engine's widening with the publisher's own library at the bit: `gguf-py` (llama.cpp's reader, which diffusers' GGUF loader agrees with on every value), ComfyUI's `comfy_kitchen` for the convrot, torchao and SDNQ for the 8-bit float and int8 files. `tools/oracle/quant_fixtures.py` rewrites those fixtures from the published files (`python tools/oracle/quant_fixtures.py [slug ...]`; the manifests hold the sha256 of what it wrote, and a rerun must leave them unchanged).

*The denoiser*, whole: the reference DiT of diffusers, in float32 on the CPU, with the weights the third party published, dequantized by that library (`tools/oracle/quantized.py`), on the Standard's own inputs (same latent, same text states, same sigma). The oracle scripts take `--variant compact|light` for `dit` and `bf16`, and, for Z-Image and Krea 2, for `witness` (Krea 2 also for `witness-step --step 0` and `draw`); MOSS-Image has a Compact only, and its script takes `--variant compact` for `dit`, `witness` and `dit-bf16`; the variants of Anima and Qwen-Image-2.1 (generation only, no `--edit`) have no fp64 witness and are judged by the plain rule against the reference. The check takes `--variant` for `dit`, at 512, and reads the map of that version beside the Standard's (`--map` names another). A map of the other version is refused, said: a Light judged against a Compact's golden would be other weights.

```sh
oracle-env/bin/python tools/oracle/z-image.py dit --size 512 --variant compact     # downloads the 7 GB GGUF once
oracle-env/bin/python tools/oracle/z-image.py bf16 --size 512 --variant compact
oracle-env/bin/python tools/oracle/z-image.py witness --size 512 --variant compact
siliconed check z-image dit --size 512 --variant compact
siliconed check z-image dit --size 512 --variant compact --neural-engine
```

The variants' oracle also needs the publishers' libraries: `oracle-env/bin/pip install -r tools/oracle/requirements-quant.txt` first (it adds `gguf`, and `sdnq` and `torchao` for `quant_fixtures.py`, to the base requirements).

The Compact and the Light run on the Neural Engine by default (fp16 products), so `--neural-engine` is the bar that matters for them, judged like the Standard's: against the reference in bf16 on the same dequantized weights, the engine allowed bf16's own error plus 25 %. Those goldens need the Standard's `dit` golden at 512 first (they read its inputs); they are not part of `all` or of `siliconed check <model>`.

On our MacBook Pro M1 Pro, `dit --size 512` of each of the nine variants, in fp32 then with `--neural-engine`, passes: 18 runs out of 18. Below, the gap of `model_out` to the reference (for Qwen-Image-2.1, `model_out (full)`; for MOSS-Image, `model_out per plane`) in each run, and the bar the Neural Engine run is judged against.

| Variant | `model_out` in fp32, against the reference | Under the Neural Engine, against the reference | Its bar (bf16's own error + 25 %) |
|---|---|---|---|
| Z-Image, Compact | 7.833e-06 | 9.906e-03 | 5.89e-02 |
| Z-Image, Light | 7.096e-06 | 1.009e-02 | 8.75e-02 |
| Qwen-Image-2.1, Compact | 4.301e-06 | 4.562e-03 | 1.59e-02 |
| Qwen-Image-2.1, Light | 4.896e-06 | 4.103e-03 | 1.47e-02 |
| Anima, Compact | 6.233e-06 | 6.133e-03 | 8.57e-02 |
| Anima, Light | 7.918e-06 | 6.422e-03 | 4.81e-02 |
| Krea 2, Compact | 7.070e-05 | 4.861e-03 | 4.24e-02 |
| Krea 2, Light | 4.383e-05 | 4.741e-03 | 4.91e-02 |
| MOSS-Image, Compact | 1.296e-06 | 4.171e-03 | 2.89e-02 |

The fp32 column is the plain gap to the reference; Z-Image, Krea 2 and MOSS-Image are judged on it at the fp64 witness, as their Standards are. The complete lines:

<details>
<summary>Compact and Light, the 18 runs</summary>

```
$ siliconed check z-image dit --size 512 --variant compact
map ~/Library/Application Support/Siliconed/store/z-image-turbo-dit.compact.v0.silicon · 7.2 GB · bfloat16 ×274, int8 ×240, float32 ×7 (8-bit weights)
✓ z-image dit compact 512 · 11 measures · nearest its threshold: t_embedder_out 3.73e-04 of 5.00e-04
$ siliconed check z-image dit --size 512 --variant compact --neural-engine
map ~/Library/Application Support/Siliconed/store/z-image-turbo-dit.compact.v0.silicon · 7.2 GB · bfloat16 ×274, int8 ×240, float32 ×7 (8-bit weights)
✓ z-image dit compact 512 · 12 measures · nearest its threshold: x_embedder_out 1.03e-03 of 3.21e-03
$ siliconed check z-image dit --size 512 --variant light
map ~/Library/Application Support/Siliconed/store/z-image-turbo-dit.light.v0.silicon · 5.0 GB · bfloat16 ×274, q4_k ×112, q5_k ×92, q6_k ×36, float32 ×7 (packed weights)
✓ z-image dit light 512 · 11 measures · nearest its threshold: t_embedder_out 3.73e-04 of 5.00e-04
$ siliconed check z-image dit --size 512 --variant light --neural-engine
map ~/Library/Application Support/Siliconed/store/z-image-turbo-dit.light.v0.silicon · 5.0 GB · bfloat16 ×274, q4_k ×112, q5_k ×92, q6_k ×36, float32 ×7 (packed weights)
✓ z-image dit light 512 · 12 measures · nearest its threshold: x_embedder_out 1.03e-03 of 3.21e-03
$ siliconed check qwen-image-2.1 dit --variant compact
map ~/Library/Application Support/Siliconed/store/qwen-image-2.1-turbo-dit.compact.v0.silicon · 7.3 GB · int8 ×224, bfloat16 ×72, float32 ×1 (8-bit weights)
✓ qwen-image-2.1 dit compact 512 · 24 measures · nearest its threshold: temb 2.11e-04 of 5.00e-04
$ siliconed check qwen-image-2.1 dit --variant compact --neural-engine
map ~/Library/Application Support/Siliconed/store/qwen-image-2.1-turbo-dit.compact.v0.silicon · 7.3 GB · int8 ×224, bfloat16 ×72, float32 ×1 (8-bit weights)
✓ qwen-image-2.1 dit compact 512 · 25 measures · nearest its threshold: temb 2.11e-04 of 5.00e-04
$ siliconed check qwen-image-2.1 dit --variant light
map ~/Library/Application Support/Siliconed/store/qwen-image-2.1-turbo-dit.light.v0.silicon · 4.2 GB · q4_k ×195, bfloat16 ×68, q5_k ×28, int8 ×4, float32 ×1, q6_k ×1 (packed weights)
✓ qwen-image-2.1 dit light 512 · 24 measures · nearest its threshold: temb 2.14e-04 of 5.00e-04
$ siliconed check qwen-image-2.1 dit --variant light --neural-engine
map ~/Library/Application Support/Siliconed/store/qwen-image-2.1-turbo-dit.light.v0.silicon · 4.2 GB · q4_k ×195, bfloat16 ×68, q5_k ×28, int8 ×4, float32 ×1, q6_k ×1 (packed weights)
✓ qwen-image-2.1 dit light 512 · 25 measures · nearest its threshold: temb 2.14e-04 of 5.00e-04
$ siliconed check anima dit --size 512 --variant compact
map ~/Library/Application Support/Siliconed/store/anima-turbo-dit.compact.v0.silicon · 2.1 GB · int8 ×448, bfloat16 ×119 (8-bit weights)
✓ anima dit compact 512 · 11 measures · nearest its threshold: temb 4.14e-05 of 5.00e-04
$ siliconed check anima dit --size 512 --variant compact --neural-engine
map ~/Library/Application Support/Siliconed/store/anima-turbo-dit.compact.v0.silicon · 2.1 GB · int8 ×448, bfloat16 ×119 (8-bit weights)
✓ anima dit compact 512 · 12 measures · nearest its threshold: model_out 6.13e-03 of 8.57e-02
$ siliconed check anima dit --size 512 --variant light
map ~/Library/Application Support/Siliconed/store/anima-turbo-dit.light.v0.silicon · 1.3 GB · q4_k ×336, bfloat16 ×119, q6_k ×112 (packed weights)
✓ anima dit light 512 · 11 measures · nearest its threshold: temb 4.14e-05 of 5.00e-04
$ siliconed check anima dit --size 512 --variant light --neural-engine
map ~/Library/Application Support/Siliconed/store/anima-turbo-dit.light.v0.silicon · 1.3 GB · q4_k ×336, bfloat16 ×119, q6_k ×112 (packed weights)
✓ anima dit light 512 · 12 measures · nearest its threshold: model_out 6.42e-03 of 4.81e-02
$ siliconed check krea2 dit --size 512 --variant compact
map ~/Library/Application Support/Siliconed/store/krea2-turbo-dit.compact.v0.silicon · 14.1 GB · int8 ×224, float32 ×174, bfloat16 ×32 (8-bit weights)
model_out per plane         2.999e-05   1.94e-05       15   1.042e-04    1.19e-04  ✓  oracle at fp64 9.48e-05
✓ krea2 dit compact 512 · 11 measures · nearest its threshold: block27_out 5.36e-04 of 5.74e-04
$ siliconed check krea2 dit --size 512 --variant compact --neural-engine
map ~/Library/Application Support/Siliconed/store/krea2-turbo-dit.compact.v0.silicon · 14.1 GB · int8 ×224, float32 ×174, bfloat16 ×32 (8-bit weights)
model_out per plane         1.716e-03   1.51e-03        0   1.716e-03    1.67e-02  ✓  oracle at fp64 1.34e-02
✓ krea2 dit compact 512 · 12 measures · nearest its threshold: img_in_out 1.09e-03 of 3.19e-03
$ siliconed check krea2 dit --size 512 --variant light
map ~/Library/Application Support/Siliconed/store/krea2-turbo-dit.light.v0.silicon · 7.4 GB · q4_k ×201, float32 ×168, q5_k ×61 (packed weights)
model_out per plane         4.064e-05   1.32e-05        9   1.282e-04    1.36e-04  ✓  clouds: engine median of 5 draws, oracle worst of 4
✓ krea2 dit light 512 · 11 measures · nearest its threshold: block1_out 5.09e-04 of 5.25e-04
$ siliconed check krea2 dit --size 512 --variant light --neural-engine
map ~/Library/Application Support/Siliconed/store/krea2-turbo-dit.light.v0.silicon · 7.4 GB · q4_k ×201, float32 ×168, q5_k ×61 (packed weights)
model_out per plane         1.667e-03   1.56e-03        0   1.667e-03    2.03e-02  ✓  oracle at fp64 1.62e-02
✓ krea2 dit light 512 · 12 measures · nearest its threshold: img_in_out 1.09e-03 of 4.54e-03
$ siliconed check moss-image dit --variant compact
map ~/Library/Application Support/Siliconed/store/moss-image-sft-dit.compact.v0.silicon · 5.4 GB · int8 ×252, bfloat16 ×210 (8-bit weights)
model_out per plane         1.296e-06   1.02e-06       82   1.470e-06    1.91e-06  ✓  oracle at fp64 1.53e-06
✓ moss-image dit compact 512 · 22 measures · nearest its threshold: model_out per plane 1.47e-06 of 1.91e-06
$ siliconed check moss-image dit --variant compact --neural-engine
map ~/Library/Application Support/Siliconed/store/moss-image-sft-dit.compact.v0.silicon · 5.4 GB · int8 ×252, bfloat16 ×210 (8-bit weights)
model_out per plane         4.171e-03   3.51e-03      120   4.171e-03    2.89e-02  ✓  oracle at fp64 2.31e-02
✓ moss-image dit compact 512 · 23 measures · nearest its threshold: video_embed 1.43e-03 of 3.20e-03
```

</details>

Krea 2's variants are judged by clouds like its Standard, and **the Light's `model_out per plane` needs them.** Judged in a single draw, that line is 1.345e-4 for an allowance of 1.33e-4: it fails, by about 1 %, and it is the only measure of the nine variants that does. It passes once the clouds are applied to it. The rule already applied them to `model_out`; in 0.15 it was extended to `model_out per plane`, and with the reference's one-ulp draws the engine's median is 1.282e-4 for 1.36e-4 (the line above says: `clouds: engine median of 5 draws, oracle worst of 4`). So that one line is green because of the draws, not without them, and you can see it in the output of `krea2 dit --variant light`. The draws of a variant are `krea2.py draw --size 512 --step 0 --jitter 1 2 3 --variant compact|light` (step 0 only: the later latents belong to the Standard's trajectory).

LoRAs are not covered here: they are files you choose, not part of what Siliconed ships.

## Timings and peak memory

`siliconed check` judges precision and nothing else. Times and peak memory are measured on the app alone: `silicontrol measure` ([REMOTE-CONTROL](REMOTE-CONTROL.md#the-socket-and-the-protocol); `silicontrol help measure` lists the options) renders a series in the app, times each render, samples the app's memory and writes a TSV and its summary. The times and peaks in [ENGINE](ENGINE.md) are taken the same way (renders queued in the app with `silicontrol`), so you can redo them on your Mac with the app alone, without Python or an oracle.

The same seed gives the same pixels (the same md5) from one render to the next on one Mac, with one version of the app and one map of that machine. It does not give them from one Mac to another: the map picks other kernels on another chip, and other kernels sum in another order. What carries from Mac to Mac is the rule above, not the bits.

## Every stage

The output of `siliconed check list --markdown`. `{size}` is 512, 1024 or a rectangle such as 768x512; the stages that take no size are checked at the reference's own.

| Stage | Proves | Rule | Oracle | Weights |
|---|---|---|---|---|
| `kernels widen` | bf16 → fp32 widening and its inverse are exact, on the values a naive shift gets wrong | bit for bit | — | — |
| `kernels arena` | the arena hands out distinct page-aligned slices, refuses duplicates and growth | each property holds | — | — |
| `kernels elementwise` | the GPU's SwiGLU, gate and gated residual are as close to double precision as the CPU's; the residual rounds twice, as torch | GPU worst ulp ≤ max(CPU's, 1), mean ≤ CPU's + 5 %; residual bit for bit | — | — |
| `kernels gelu` | the GPU's GELU (erf) is as close to double precision as the CPU's, over one bit pattern in 64 of [−12, 12] | GPU worst deviation ≤ the CPU's, in ulp of the operand | — | — |
| `kernels sdpa` | the attention cut by heads (GPU + CPU), by query blocks, past maxMatrix and the matrix kernel return the whole graph's result | against the whole graph: query blocks ≤ 1e-6 per channel; head cut ≤ 1e-5 with the GPU heads bit for bit; matrix kernel ≤ 1e-5 | — | — |
| `kernels gemm-split` | the token-wise split of a GEMM between the GPU and the AMX stitches back cblas_sgemm's product | against cblas_sgemm, per row: the GPU's rows bit for bit, the AMX's ≤ 1e-5 (another summation order over k = 3 840) | — | — |
| `kernels rope` | every RoPE table of the engine is the reference's, value by value (13 tables, 5 bit for bit) | max \|Δ\| per table within its bound, in units of 2⁻²⁴: 0 where the reference is bit-reproducible, 1 to 64 where torch uses SLEEF's cos, sin, pow | `tools/oracle/kernels.py rope` | — |
| `kernels resample` | the spectral schedule's nearest-exact decimation and bilinear upsampling are torch's, on a square and a rectangle | decimation bit for bit (and the even decimation, its counter-test, must differ); bilinear within one ulp of the input's scale | `tools/oracle/kernels.py resample` | — |
| `z-image tokenizer` | the chat template and the tokenizer give transformers' 512 identifiers and mask, padding included, and the app's admission the same real tokens (with --long, past 512: 1024) | bit for bit | `tools/oracle/z-image.py text [--long]` | tokenizer |
| `z-image text` | the Qwen3-4B encoder gives the reference's hidden states, layer by layer, and its cap_feats (with --long, on a prompt past 512 tokens) | the rule, floor 5e-4 | `tools/oracle/z-image.py text [--long]` | text encoder |
| `z-image block` | block 0 matches the reference at its 19 internal boundaries: norms, projections, attention, SwiGLU | the rule, floor 5e-4 | `tools/oracle/z-image.py dit --size {size}` | DiT |
| `z-image dit` | one evaluation of the DiT, latent to velocity: embedders, refiners, 30 blocks, head, unpatchify | the rule, floor 5e-4; on a rectangle, the transposed golden must deviate; with --neural-engine or --bf16-activations, against bf16's own error + 25 % | `tools/oracle/z-image.py dit --size {size}; witness --size {size}; with --neural-engine or --bf16-activations, bf16 --size {size}; with --variant compact or light, the same three with --variant, at 512 (the Compact's or the Light's published weights, dequantized by their publisher's library)` | DiT |
| `z-image trajectory` | the 8-step sampler: each step's velocity and latent, from the reference's own latent at that step | the rule on the guided pass, floor 5e-4; the free pass is shown with its gain, not judged | `tools/oracle/z-image.py trajectory --size {size}` | DiT |
| `z-image vae` | Flux's decoder on a rectangular latent (the centered crop of the final 1024² latent), plane by plane: what a square cannot see | the rule per plane, floor 5e-4, at the fp64 witness (no floor there); on a rectangle, the transposed image must deviate | `tools/oracle/z-image.py vae --size {size}; vae-witness --size {size}` | VAE |
| `qwen-image-2.1 text` | Qwen3-VL-8B reads the prompt (and with --edit N, the N pictures; with --long, a prompt past 512 tokens): template, identifiers, positions, 36 layers, prompt_embeds | identifiers and masks bit for bit; the rule, floor 5e-4 (with images, the deep layers judged on their text rows) | `tools/oracle/qwen-image-2.1.py text [--edit N]; text --long` | text encoder |
| `qwen-image-2.1 vision` | the processor (Lanczos, patches, grids), the 3D positions of both edit templates, and the vision tower with its deepstack, image by image | pixels, grids, identifiers and positions bit for bit; the rule, floor 5e-4 | `tools/oracle/qwen-image-2.1.py vision` | text encoder |
| `qwen-image-2.1 references` | a reference with transparency, prepared as the pipeline does: the VAE's RGBA input and the vision copy over white (at 512, Pillow premultiplies around its Lanczos; at 443 it copies) | bit for bit; the vision copy on its bytes (the Pil and Fast processors differ by 6e-8 in float) | `tools/oracle/qwen-image-2.1.py references [--side 443]` | — |
| `qwen-image-2.1 dit` | one evaluation three ways — prefill of the conditions, the step from their K/V cache, the full pass under the block-causal mask — plus the layout and the modulation | layout bit for bit; the rule, floor 5e-4, at the fp64 witness where one exists; cache and full pass ≤ 1e-5 apart; a second step the same bits; with --neural-engine or --bf16-activations, against bf16's own error + 25 %, cache and full pass too | `tools/oracle/qwen-image-2.1.py dit [--edit N]; witness; with --neural-engine or --bf16-activations, bf16 [--edit N]; with --variant compact or light (no --edit), dit and bf16 with --variant (the Compact's or the Light's published weights, dequantized by their publisher's library)` | DiT |
| `qwen-image-2.1 trajectory` | the 8 steps: σ, timesteps and noise to the bit, then each step's velocity from the oracle's latent and encoder states; the product's free run from the real inputs is shown | schedule and noise bit for bit; the rule on each guided velocity, floor 5e-4; the free run's final latent and image are shown, not judged | `tools/oracle/qwen-image-2.1.py trajectory [--edit N]` | DiT, text encoder, VAE |
| `qwen-image-2.1 vae` | the decoder on the trajectory's final latent (with --edit N, the edit's): unpacking and de-normalization to the bit, then the RGBA image plane by plane | vae_in bit for bit; the rule per plane, floor 5e-4, at the fp64 witness where one exists; on a rectangle, the transposed image must deviate | `tools/oracle/qwen-image-2.1.py vae [--edit N]; vae-witness` | VAE |
| `qwen-image-2.1 vae-encode` | the encoder on both edit pictures (RGBA): the raw mean and the normalized latent, plane by plane; the product's module (RGB in, alpha added) gives the same bits | the rule per plane, floor 5e-4, at the fp64 witness where one exists; the module bit for bit | `tools/oracle/qwen-image-2.1.py vae-encode; vae-witness` | VAE |
| `anima tokenizer` | the T5 tokenizer against `tokenizers` on a corpus of edge cases: normalization, whitespace, special tokens, truncation | every case bit for bit | `tools/oracle/anima.py tokenizer` | tokenizer |
| `anima text` | the prompt through both tokenizers, Qwen3-0.6B (final norm included) and the conditioner, to the DiT's [512, 1024] context (with --long, a prompt past 512 tokens: more rows) | identifiers bit for bit; the rule, floor 5e-4 | `tools/oracle/anima.py text [--long]` | text encoder, DiT (the conditioner) |
| `anima dit` | one evaluation of the DiT at σ = 1, boundary by boundary, and its output per latent plane against the fp64 witness | the rule, floor 5e-4; model_out per latent plane at the fp64 witness; on a rectangle, the transposed golden must deviate; with --neural-engine or --bf16-activations, against bf16's own error + 25 % | `tools/oracle/anima.py dit --size {size}; witness --size {size} --step 0; with --neural-engine or --bf16-activations, bf16 --size {size}; with --variant compact or light, dit and bf16 with --variant, at 512 (the Compact's or the Light's published weights, dequantized by their publisher's library)` | DiT |
| `anima trajectory` | the schedule, then the 8 Euler steps: each step's velocity from the oracle's latent at that step | σ within 1e-6; the rule on each guided velocity, floor 5e-4; the free pass is shown with its gain, not judged | `tools/oracle/anima.py trajectory --size {size}` | DiT |
| `anima vae` | the Qwen-Image decoder on the final latent of the oracle's trajectory, plane by plane | the rule per plane, floor 5e-4, at the fp64 witness (no floor there); on a rectangle, the transposed image must deviate | `tools/oracle/anima.py vae --size {size}; vae-witness --size {size}` | VAE |
| `krea2 text` | the prompt's identifiers, the 12 Qwen3-VL taps (padding removed, judged against states computed with it), then the fusion and txt_in on the oracle's taps (with --long, a prompt past 512 tokens read whole, as the app admits it: no fusion golden for it) | identifiers bit for bit; the rule, floor 5e-4 | `tools/oracle/krea2.py text [--long]` | text encoder, DiT (the fusion) |
| `krea2 dit` | one evaluation of the DiT at σ = 1, boundary by boundary, judged at the fp64 witness where it exists | the rule, floor 5e-4, at the fp64 witness where it exists, clouds when the oracle has draws; model_out per latent plane at the witness; on a rectangle, the transposed golden must deviate; with --neural-engine or --bf16-activations, against bf16's own error + 25 % | `tools/oracle/krea2.py dit --size {size}; witness --size {size}; witness-step --size {size} --step 0; draw --size {size} --step 0 --jitter 1 2 3; with --neural-engine or --bf16-activations, bf16 --size {size}; with --variant compact or light, dit, bf16, witness, witness-step --step 0 and draw --step 0 --jitter 1 2 3 with --variant, at 512 (the Compact's or the Light's published weights, dequantized by their publisher's library)` | DiT |
| `krea2 trajectory` | the schedule, then the 8 Euler steps: each step's velocity from the oracle's latent at that step, at the step's fp64 witness where it exists | σ within 1e-6; the rule on each guided velocity, floor 5e-4, at the step's fp64 witness where it exists, clouds when the oracle has draws; the free pass is shown, not judged | `tools/oracle/krea2.py trajectory --size {size}; witness-step --size {size} --step 0 1 2 3 4 5 6 7` | DiT |
| `krea2 vae` | the Qwen-Image decoder with Krea's weights on the final latent of the oracle's trajectory, plane by plane | the rule per plane, floor 5e-4, at the fp64 witness (no floor there); on a rectangle, the transposed image must deviate | `tools/oracle/krea2.py vae --size {size}; vae-witness --size {size}` | VAE |
| `moss-image tokenizer` | the chat template (no system turn, thinking block open) and the Qwen2 tokenizer against `tokenizers` on six prompts: accents, a newline and tab, JSON, digits and punctuation, a long text; cut at 4096 | the template string and every identifier bit for bit | `tools/oracle/moss-image.py tokenizer` | tokenizer |
| `moss-image text-delta` | the embedding rows and layer 0, a Gated DeltaNet layer, part by part: input norm, fused projection, delta rule, gated norm, mixer, MLP, output | the rule, floor 5e-4 | `tools/oracle/moss-image.py text` | text encoder (layer 0) |
| `moss-image text-full` | layers 0 to 3: three Gated DeltaNet layers and the first full-attention layer (gated queries, partial RoPE, output gate), part by part | the rule, floor 5e-4 | `tools/oracle/moss-image.py text` | text encoder (layers 0–3) |
| `moss-image text` | the prompt through Qwen3.5-9B: all 32 layers' outputs and the final state, the DiT's `prompt_embeds` | the rule, floor 5e-4; the layers 0, 3, 15, 31 and the final state at the fp64 witness | `tools/oracle/moss-image.py text; text-witness` | text encoder |
| `moss-image schedule` | the sigmas of MOSSImageScheduler (shift 5 on the fp32 grid, snapped to the 1000-step training grid) and the timesteps the DiT receives, for 1 to 10, 20 and 50 steps | bit for bit | `tools/oracle/moss-image.py schedule` | — |
| `moss-image dit` | one evaluation of the DiT at σ = 1, boundary by boundary — time, modulations, RoPE, text path, block 0 inside out, blocks 1, 13, 27, 41, the head —, judged at the fp64 witness | the rule, floor 5e-4, at the fp64 witness where it exists; model_out per latent plane at the witness; with --neural-engine or --bf16-activations, against bf16's own error + 25 % | `tools/oracle/moss-image.py dit; witness; with --neural-engine or --bf16-activations, dit-bf16; with --variant compact, dit, witness and dit-bf16 with --variant (the Compact's or the Light's published weights, dequantized by their publisher's library)` | DiT |
| `moss-image dit-lora` | the same evaluation with the DMD 8-step LoRA at strength 1 (kohya `lora.safetensors`, rank 128, fp32), the golden having it merged in fp32 | the rule, floor 5e-4, at the fp64 witness where it exists; model_out per latent plane at the witness | `tools/oracle/moss-image.py dit --lora; witness --lora` | DiT, LoRA |
| `moss-image trajectory` | the 8 steps of the recipe: the seed's noise bit for bit, the σ, each step's velocity from the oracle's latent at that step (the DMD LoRA at strength 1), then the product's denoiser run free from the noise on the oracle's text — its latents shown against the oracle's, and against the free loop the stage runs itself | noise and σ bit for bit; each guided velocity per latent plane, floor 5e-4; the product's free run against the stage's own loop, 1e-5 (the same arithmetic); the free latents against the oracle's are shown, not judged (eight steps amplify the evaluation's rounding) | `tools/oracle/moss-image.py trajectory` | DiT, LoRA |
| `moss-image vae` | FLUX.2's decoder: the pipeline's de-normalization and 2×2 de-patchification, then the image, plane by plane — on an encoded test image's latent and on the final latent of the reference pipeline | the de-patchified latent per channel, floor 5e-4; the image per plane, floor 5e-4, at the fp64 witness (no floor there) | `tools/oracle/moss-image.py vae --size {size}; vae-witness --size {size}; vae-final --size {size}` | VAE |
