# The oracle's input pictures

Both were made with Siliconed itself and Z-Image Turbo (Apache 2.0), 8 steps, Euler; each PNG carries its prompt and
seed in its metadata. No photograph, no real person. The stages that start from a picture or edit one read them
(`img2img`, `--edit N`, the writer's `describe`); the goldens written from them hold as long as these bytes do.

- `border-collie-beach.png`, 1024×1024, seed 2:
  « photo of a wet black and white border collie dog sitting on the sand of an Atlantic beach, looking at the camera, waves breaking behind, soft overcast light, realistic photography, 35mm film grain »
- `surfer-raincoat-cloudy-beach.png`, 1216×832, seed 1:
  « photo of two women standing on a beach in Brittany at low tide, the first woman on the left is Irish, with wavy auburn hair, fair freckled skin and green eyes, wearing a black shorty wetsuit and holding a white surfboard, a second woman on the right in a yellow raincoat, behind them a heavy grey cloudy overcast sky over the Atlantic, waves breaking, realistic photography, 35mm film grain »
- `rgba-512x384.png`: synthetic, drawn by a script (an alpha ramp, a fully transparent area): what a picture with
  transparency becomes on its way in.
- `long-prompt.txt`: a prompt past 512 tokens, written for the text stages.
