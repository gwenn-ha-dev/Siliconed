English · [Français](REMOTE-CONTROL.fr.md)

# silicontrol, the remote control of Siliconed: how it works

**The reference is in the app.** `silicontrol help` gives the overview, `silicontrol help <command>` or
`silicontrol <command> --help` one page, `silicontrol help all` the whole reference (contract, JSON
answers, error codes, recipes). To put an AI to work, tell it "look at `silicontrol help`". This file
only explains how the thing is built; it does not repeat the help, so the two cannot diverge.

## The idea

The open app holds the machine's only render queue. `silicontrol …` adds to it, waits, stops,
reads the history: the same functions as the buttons (`AppState.enqueue`, `cancel`, `remove`,
`stopAll`). What the command adds shows up in the window; what the mouse starts, the command sees.
There are never two engines inside the 16 GB.

**A license is accepted only in the window.** A job whose model's license the library does not
record is refused by the engine (`license_not_accepted`) and set aside, out of the queue, while the
app's sheet shows the license: `status` lists it under `"waiting_for_license"`, `wait` waits for it,
`cancel` removes it. Accepted, it goes back to the head of the queue; declined, it ends in
`"error"`. No command accepts on the user's behalf.

**A prompt's alternatives expand the same way on both sides.**
`"woman posing in a {library|greenhouse}"` makes two jobs, at the same seed, whether the prompt comes
from the app's field or from `silicontrol add`: both go through `PromptAlternatives` (the library,
`docs/API.md` §3.14), then through `AppState.enqueue`. The server redoes nothing: it refuses what the
parser refuses (error `prompt`), and each job carries its expanded prompt and its `choices`. The
syntax and the answer are in `silicontrol help add`.

**A grid is jobs, then an image.** `silicontrol grid --x … [--y …]` and the app's « Exploration »
window (Render > Exploration, ⌥⌘G) both check the grid with the library (`RenderGrid`, `docs/API.md`
§3.15) and queue it through `AppState.enqueueGrid`: one ordinary job per cell, in the sheet's reading
order, plus a `GridRun` that only remembers which job is which cell (`Sources/SiliconedApp/Grid.swift`).
The axes are a `{…}` group of the prompt, the seed, the steps, a LoRA's strength, LoRAs to compare
(`loras=none,flat,ink`: one added to the stack per cell, names resolved as `--lora` does), the model
(`model=z-image,qwen-image-2.1`: each cell at its model's own steps, refused with a `--lora`) and the
edit's images (`image`: each `--ref` alone, with the same instruction). `silicontrol grid --sketch`
queues the window's kind of exploration (sketches, in its window, no sheet); otherwise each
cell is a finished image that joins the history; when the last cell ends — done, failed, stopped or
removed — the app draws the sheet off the main thread and adds it to the history; the `Signal`
`.sheet` tells `wait g2`, `grid --wait` and `follow` (`"type": "sheet"`). A cell that did not render
is drawn empty, with why; no cell rendered, no sheet. A `{…}` group on no axis is refused (`grid`),
never multiplied. The syntax is in `silicontrol help grid`.

**The exploration sketches.** In the window (`Sources/SiliconedApp/Exploration.swift`), each cell
stops as a sketch (`Request.sketch`, `docs/API.md` §3.16) at 512 on the short side (512², 512×768,
768×512): about half the time of a finished image. Its sketches stay in the window, not in the
history, until the next exploration, which replaces it (its waiting cells leave the queue, the
running one stops). On a cell, « Finish This Image » renders it to its end — same seed, same size —
into the history, the image the sketch announced; « Use These Settings » gives the rack its model,
prompt, steps, LoRA strengths and seed, not its size. The sheet joins the history only on « Add Sheet
to History ».

**Variations are the menu's.** `silicontrol vary i7 subtle|strong [--batch N]` and an image's
« Variations » menu (the button under the image, its context menu) both call `AppState.vary`: as
many ordinary jobs as the rack's « Images » (1 by default; `--batch` overrides it), with exactly that image's settings and seed, each adding one variation seed drawn at random
(`Request.variations`, `docs/API.md` §3.18) at the family's strength (`Variation.Amount`), never a
number the command passes. A variation of a variation appends to the chain, so it varies *that*
image. Refused on a grid's sheet, a sketch, or a model that is not ready (`model`). Each image's
`variation` lists its variation seeds in order; `silicontrol help vary` has the answer.

## The command

`silicontrol` ships inside the app, at `Siliconed.app/Contents/Helpers/silicontrol` (target `Silicontrol`,
`Sources/Silicontrol/main.swift`). The app menu ("Install the silicontrol command…") links it into
`/usr/local/bin`. It is a pipe: Foundation only, it does not link the engine. If the app is not
running, any command but the help launches it and waits for its socket (30 s at most); `silicontrol open`
does only that. The help (`help`, `help <topic>`, `<command> --help`, no argument) is answered by the
command itself, without launching the app.

## The socket and the protocol

- **Unix socket** `<library>/silicontrol.sock` (by default
  `~/Library/Application Support/Siliconed/silicontrol.sock`, or under `SILICONED_ROOT`), mode `0600`:
  only the account that launched the app can open it. No network port. It disappears when the app
  quits; a dead socket is replaced at the next launch, and a second Siliconed on the same library
  does not listen.
- **One connection per command.** The client writes one JSON line,
  `{"argv": ["add", "--batch", "2", "…"], "cwd": "/current/folder"}`, and the app answers with JSON
  lines, then closes. Intermediate lines carry `"type"` (`added`, `image`, `sheet`, `start`, `stage`,
  `step`, `end`); the last one carries `"ok"` (`true` or `false`, with `"error"`, `"message"` and
  often `"hint"` on failure). Without the client:
  `printf '{"argv":["status"]}\n' | nc -U ~/Library/Application\ Support/Siliconed/silicontrol.sock`.
- **All parsing is in the app** (`Sources/SiliconedApp/RemoteControl.swift`): `argv` is the command
  line as typed and the client is only a pipe. Only `open` and the help are handled client-side,
  because they have to work when the app is closed. Relative paths resolve against `cwd`. The images of an edit
  (`--ref`, repeatable, in order: the first is image 1, the one edited) travel the same way, as
  paths or history names (`i7`), never as pixels: the app reads them itself.
- **Help has one source**, the module `SilicontrolHelp` (`Sources/SilicontrolHelp/RemoteControlHelp.swift`,
  English and French pages, Foundation only), linked by both sides: `silicontrol` answers from it
  without the app, and the app answers from it on its socket (`nc -U` gets the same pages). The
  language is the user's order between English and French. The answer is
  `{"ok": true, "help": "<text>"}` and the client prints the text raw; an unknown topic is a usage
  failure (`"error": "usage"`, the topics in `"hint"`, exit `2`). Commands, flags, JSON keys
  and error codes are English in both languages. When a command changes in the app, its help
  changes in the same commit.
- **Exit codes** of the client: `0` the last line says `"ok": true`; `1` it says `"ok": false`;
  `2` usage error; `3` the app cannot be reached (not running and not launchable, or not
  answering).
- **Signals.** The `AppState` emits `Signal` (`started`, `stage`, `step`, `image`, `finished` with
  its `Issue`, `sheet` when a grid's last cell ended) to whoever subscribes; `wait` and `follow` subscribe for the length of their
  connection and unsubscribe at the first write that fails (client gone).
- **SIGPIPE is ignored** (`signal(SIGPIPE, SIG_IGN)`, plus `SO_NOSIGPIPE` on each client socket),
  so that a client hanging up cannot kill the app: measured, a client that connected and closed
  without reading used to take the app down.
- **The plumbing stays off the main thread**: accepting and reading run on background queues
  (`Socket`, `Connection`), writing on each client's own queue, so a long `wait` blocks neither the
  app nor other clients. Only the commands run on the main thread, like a click.

## What was measured

- **Same bits**: the app's image equals a render of the library called directly with the same
  request (512², seed 42, identical decoded bytes). Likewise for a variation of a variation queued by
  `silicontrol vary`: redone from its PNG's recipe, it comes out to the bit.
- **App Nap**: in the background, macOS throttled the app — 64 s instead of 44 for a 512², one
  step at 85 s. The app now holds a "render" activity (`ProcessInfo.beginActivity`) from the first
  job started until the queue is empty: **45.9 s** in the background (44.3 s for the same render
  outside the app).

## Another process that renders

One render per machine: the engine takes a lock (`/private/tmp/siliconed-render.lock`) for each
render, so another program built on the library is refused with `render_already_running` while the
app renders, and the reverse (two models in 16 GB: the system would swap and both timings would mean
nothing). Such a program can queue its render in the app with `silicontrol add` instead.
