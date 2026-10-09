English · [Français](RELEASE-NOTES.fr.md)

# Siliconed — release notes

## 0.4

### Installation

- **Standard, Compact or Light.** Z-Image Turbo and Qwen-Image-2.1 install with the publisher's
  weights (Standard), with 8-bit weights published by third parties for the transformer and the text
  encoder (Compact: 11.7 GB instead of 20.4 for Z-Image, 19.4 GB instead of 33.2 for
  Qwen-Image-2.1), or with the transformer in 4 to 6 bits (GGUF Q4_K_M) and the Compact's text
  encoder (Light: 9.5 GB and 16.2 GB). The Light is preselected on a Mac with 8 GB of memory. The
  installed version keeps rendering until the new one is complete.
- **Downloads that resume.** Large files come in chunks, several at a time; a stalled chunk is asked
  again, and a cut download resumes where it stopped. Every file is still checked by sha256.
- **Signed and notarized.** The app and the disk image are signed with a Developer ID and notarized
  by Apple: they open with a double click.

### Editing

- **Edit from anywhere.** Right-click an image in the history and choose « Edit with
  Qwen-Image-2.1 », even when the rack is set to a model that does not edit: the rack switches to
  the editor and the image becomes image 1. A thumbnail dragged from the history onto the Edit box
  becomes a reference.
- **Original / edited curtain.** An edited image has a button that lays it over its original, with a
  line to drag across to see exactly what changed. Escape leaves it.

### Images

- **Detail at 1024².** « Detail: Normal · More · Most » now gives clean results at 1024² too, on
  Z-Image and Qwen-Image-2.1: its strength follows the size of the image, and it no longer leaves
  a grainy texture at high resolution.
- **Variations follow « Images ».** Subtle or strong variations of an image now make as many images
  as the rack's « Images » field (1 by default), instead of always four.

### Exploration

- **Pick the model in the panel.** The exploration panel has its own « Model » menu.
- **One status line.** The panel says where it is in a single line: « Sketch at n steps of N », or
  « Finished images, N steps » for Qwen-Image-2.1, which now explores with finished images (its
  early sketches were blurry).

### Imported models

- **8-bit stays 8-bit.** A fine-tune imported in 8 bits — fp8, int8, GGUF Q8_0 or int8 convrot — is
  kept in 8 bits on disk, about half the size of the 16-bit file (Z-Image: 6.2 GB instead of 12.3
  GB). The computation is still fp32, and every weight is checked bit for bit against the
  publisher's own conversion. The app imports `.gguf` files as well as `.safetensors`.
- **GGUF down to 4 bits.** A `.gguf` fine-tune in Q4_0 to Q6_K is kept as published, its blocks
  copied whole (Z-Image: about 5 GB in Q4_K_M instead of 12.3 GB). Files below 4 bits, and 4-bit
  files outside GGUF, are refused, with the reason.

### Speed and memory

- **Z-Image is faster at 1024²**: about 82 s instead of about 107 s, and 153 s instead of 168 s at
  1024×1536, still without any swap.
- **A memory check on the machine's real state**: before each render, the app reads the memory it can
  really have at that moment (not just the Mac's total) and refuses, with the figures, a render that
  would not fit — instead of letting it swap. If memory gets short during a render, the engine
  switches to a leaner plan: the same image, a little slower, rather than swapping.

### Languages

- **German, Spanish and Italian**, besides English and French: the app follows the system language.

Timings are measured on a MacBook Pro M1 Pro with 16 GB.
