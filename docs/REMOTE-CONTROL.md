English · [Français](REMOTE-CONTROL.fr.md)

# silicontrol, the remote control of Siliconed: how it works

**The reference is in the app.** `silicontrol help` gives the overview, `silicontrol help <command>` or
`silicontrol <command> --help` one page, `silicontrol help all` the whole reference (contract, JSON
answers, error codes, recipes). To put an agent to work, tell it “look at `silicontrol help`”. This file
only explains how the thing is built; it does not repeat the help, so the two cannot diverge.

## The idea

The open app holds the machine's only render queue. `silicontrol …` adds to it, waits, stops,
reads the history: the same functions as the buttons (`AppState.enqueue`, `cancel`, `remove`,
`stopAll`). What the command adds shows up in the window; what the mouse starts, the command sees.
There are never two engines inside the 16 GB. When the window computes on another Mac (Render › Compute On),
the commands follow it: they still talk only to this Mac's socket, but act on the other Mac's queue and history,
as the buttons do; `install` (and `install writer`, `rewrite`, `describe`: the writer is this Mac's; `import`, `uninstall`, `library move|home`: the files and the click are that Mac's) refuses there, `diagnose` and `set faster-rendering|accelerators` act on the other Mac.

**A license is accepted only in the window.** A job whose model's license the library does not
record is refused by the engine (`license_not_accepted`) and set aside, out of the queue, while the
app's sheet shows the license: `status` lists it under `"waiting_for_license"`, `wait` waits for it,
`cancel` removes it. Accepted, it goes back to the head of the queue; declined, it ends in
`"error"`. No command accepts on the user's behalf.

**A prompt's alternatives expand the same way on both sides.**
`"woman posing in a {library|greenhouse}"` makes two jobs, at the same seed, whether the prompt comes
from the app's field or from `silicontrol add`: both go through `PromptAlternatives` (the library,
`Sources/Siliconed/PromptAlternatives.swift`), then through `AppState.enqueue`. The server redoes nothing: it refuses what the
parser refuses (error `prompt`), and each job carries its expanded prompt and its `choices`. The
syntax and the answer are in `silicontrol help add`.

**A grid is jobs, then an image.** `silicontrol grid --x … [--y …]` and the window's series (« Generate » with
something bracketed, `Sources/SiliconedApp/Series.swift`) both check the grid with the library (`RenderGrid`,
`Sources/Siliconed/RenderGrid.swift`) and queue it through `AppState.enqueueGrid`: one ordinary job per cell, in the sheet's reading
order, plus a `GridRun` that only remembers which job is which cell (`Sources/SiliconedApp/Grid.swift`).
The axes are a `{…}` group of the prompt, the seed, the steps, a LoRA's strength, LoRAs to compare
(`loras=none,flat,ink`: one added to the stack per cell, names resolved as `--lora` does), the model
(`model=z-image,qwen-image-2.1`: each cell at its model's own steps, refused with a `--lora`), the
edit's images (`image`: each `--ref` alone, with the same instruction), the solver, the starting image's strength
(`start=`) and Detail (`detail=`). Each cell is a finished image that joins the history; when the last cell ends — done,
failed, stopped or removed — the app draws the sheet off the main thread and adds it to the history; the `Signal`
`.sheet` tells `wait g2`, `grid --wait` and `follow` (`"type": "sheet"`). A cell that did not render
is drawn empty, with why; no cell rendered, no sheet. A `{…}` group on no axis is refused (`grid`),
never multiplied. The syntax is in `silicontrol help grid`.

**The window's series are grids without a sheet.** What varies — the prompt's groups, the Batch, the knobs bracketed
(`Bracket`, its pure rules in `Sources/Siliconed/SeriesLayout.swift`) — lays out on two axes at most, and « Generate » renders it
through `enqueueGrid` with `sheet: false`: the same cells, the same images, bit for bit, as `silicontrol grid --no-sheet`; the canvas
lays them out (`SeriesCanvas`), and the sheet joins the history only on « Add Sheet to History » (`sheet g2`). A Batch alone
stays one job, its prompt encoded once. `grid` gives the window its axes as a series where the window has them (`setSeries`);
`reuse` answers the rack's series (`"series"`, its axes as `--x` and `--y` read them).

**Variations are the menu's.** `silicontrol vary i7 subtle|strong [--batch N]` and an image's
“Variations” menu (the button under the image, its context menu) both call `AppState.vary`: as
many ordinary jobs as the rack's “Batch” (1 by default; `--batch` overrides it), with exactly that image's settings and seed, each adding one variation seed drawn at random
(`Request.variations`, `Sources/Siliconed/Variation.swift`) at the family's strength (`Variation.Amount`), never a
number the command passes. A variation of a variation appends to the chain, so it varies *that*
image. Refused on a grid's sheet, or a model that is not ready (`model`). Each image's
`variation` lists its variation seeds in order; `silicontrol help vary` has the answer.

**Enlarging is the menu's too.** `silicontrol enlarge i7 [--download]` and “Enlarge ×4” both call
`AppState.enlarged`: macOS's super-resolution (VideoToolbox, macOS 26), no prompt, outside the
queue — refused while a render runs —, answered with the new image once it is in the history.
Apple's model is downloaded by macOS only with `--download` (else `download_required`);
`silicontrol help enlarge` has the rest.

**A size is `WxH` or a proportion and a resolution.** `silicontrol add` and `grid` take `832x1216` as ever, or `3:4@2K`: the proportion (1:1, 4:5, 3:4, 2:3, 9:16, 5:4, 4:3, 3:2, 16:9), `@`, then `1K` or `2K`. Neither character means anything to a shell, so no quotes. It resolves to the pixels of the rack's Size in Two Scales (`SizeScales`, one function for both), whether or not that Beta option is on, and is refused (`format`) with its reason where the rack greys it: Anima's 2K, for instance, is the square alone (2048 px a side). The table is in [the app's page](APP.md#create).

**Solvers follow More Solvers.** `--solver` is accepted only with More Solvers on (Settings › Advanced › Beta) and on a model that offers more than Euler; otherwise the answer is a refusal (`solver`) that names the setting.

**Faster Rendering is the app's setting too, one per model, and a Beta option.** `silicontrol set faster-rendering on|off
[--model ID]` goes through `AppState.setFasterRendering(_:for:)`, keyed per installed version like the
Accelerators' choice, and the
window's switch follows, as a click; without `--model`, the model in the rack. `models` reports each
installed model's value, `status` the rack's (`"faster_rendering"`). `on` is refused (`faster_rendering`) while the Beta option (Settings › Advanced › Beta) is off, as `--solver` is without More Solvers; `off` always goes through. The value `models` and `status` report is what a render takes, so `false` while the option is off. Refused (`busy`) unless nothing
runs or waits: a job carries the value it was queued with (`RenderSettings.fasterRendering`), and one
planned under one value and finished under the other would mix two recipes. It persists like the
switch does; `silicontrol measure --faster-rendering on|off` sets it for the measured model, writes it
in the TSV and gives the value from before back at the end, even when the series fails. Off is full
size: the image checked against the reference.

**Nothing depends on a click, except pairing.** What the window does, a command does too, **by the same functions as the buttons** (`AppState`, the Prompt Lab's `LabModel`), never by a road of its own. What stays in the window on purpose: pairing a Mac, accepting a license, “Lend This Mac” and “Forget”. The Prompt Lab's bar is `lab`: what it holds, folded or not (`lab open|close`, ⌥⌘L), in a window of its own or over the image (`lab detach|attach`), an image in as a drop brings it, with the role a newcomer takes (`lab add`, `rack.addInput`), a role (`lab role N edited|reference|start|none`, `rack.setRole`, refused where the rack's model lacks it, as the bar's menu does not offer it), an image out (`lab remove`); its answer gives each image's role and whether the next render reads it (`"used"`, `"unused"`). Its two buttons are `lab rewrite` and `lab describe N` (`LabModel.rewrite`, `describe`): the bar unfolds, the button is pressed, and the text is written in the field token by token with its marks, as the click writes it, ⌘Z included — where `rewrite` and `describe` (below) answer on the socket and leave the field alone. They answer as soon as the writer starts (`lab` then says where it stands, `"writing"`), or with `--wait` once the text is written; the gesture is the window's, so a client that hangs up leaves it writing. Their refusals are `rewrite`'s, and `empty_text` for a text with nothing to rewrite. Its subjects are `subjects`. Siliconed › Settings… is `settings [general|macs|lab|advanced]` (`SettingsWindow.show`, the panes named in the library, `SettingsPane`), and `settings lab [rewrite|describe] [--for ID]` opens Prompt Lab on one instruction, as the Lab's “Your Instruction” does (`showInstruction`). A knob's yellow ± is `bracket seed|detail|solver|start|loras` or `bracket lora NAME` (`app.bracketing`, as the click sets it): its bubble opens beside it, refused where the window draws no ± that would open; `bracket close` closes it. The rack's scroll is `rack top|dit`: its head, or the DiT's settings with the chain pinned in small above them. The Prompt Lab's writer and Settings' switches are described next, and after them the gestures of the history, the series, the models, Compute On and the library.

**The Prompt Lab's writer is the window's.** `silicontrol rewrite "<text>" [--for ID]` and `silicontrol describe <image|i7> [--for ID]` call what the Lab's two buttons call: `EngineLabWriter` over `PromptWriter`, our instructions per model and gesture (`PromptWriterInstructions`), the image read as the Lab reads a dropped one (`PromptWriterImage`), and for a model that reads images the `@names` put aside and put back (`LabInputs.MentionGuard`, shared as `LabModel.rewriteInput` and `rewritten`). The model it writes for is the one the Lab would take (`WriterTarget.pick`: the rack's, else the first installed) unless `--for` names one. The refusals are the Lab's, in its order (`WriterPause.decide`, one rule for the sentence under its buttons and for the code): the window follows another Mac (`remote`), an image renders (`busy`), no writer, the head to download, a Mac too small (`writer_absent`, `writer_head`, `writer_memory`, `writer_preparing`); and a second gesture while one writes (`writer_busy`), from the terminal or from the window. The lines are `{"type": "started"}`, `{"type": "loading", "fraction"}` and `{"type": "text", "text"}`, which carries everything written so far (cumulative, as the Lab's stream is: the detokenizer may revise its last word); the verdict has the whole `text`, the `tokens`, and the `notice` and `lost_mentions` when the writer lost an `@name`. **A render that starts stops the gesture** (`stopped`), as it stops the Lab's. **A client that hangs up cancels it**: `Connection.isGone` sees the close without writing to the socket, and `LabModel.cancelRemote` ends the stream, whose termination stops the sampling (`EngineLabWriter.write`, `PromptWriter.cancel`) before the weights go back. Incognito: the text goes out on the socket and nowhere else. `silicontrol install writer` is “Install the Writer” (`LabModel.installHead`): the head of Qwen-Image-2.1's encoder, with no license of its own (it is the one the encoder's installation had accepted: `Installer.installWriterHead` checks it, and the command says so before it starts). All three act on this Mac: with the window on another one they are refused (`remote`). `silicontrol instructions` is Settings › Prompt Lab: what the writer is told for each model and gesture, listed, shown, replaced (`--set`, `--file`) or restored (`--restore`), through the same `LabModel.instructions` the pane's field writes (`PromptWriterInstructions.Edits`); the instructions are this Mac's too, and stay allowed with the window on another one.

**Accelerators and the Beta options are Settings' switches too.** `set accelerators on|off [--model ID]` (`neural-engine`, its former name, still read) goes through `AppState.setAccelerators(_:for:)`, as `set faster-rendering` does through its own, with the same wait (`busy`) and the same keys per installed version — the Neural Engine's before (`UserDefaults` key `neuralEngine`): a choice made then is kept; it is refused (`accelerators`) where this Mac's measure gives the accelerators nothing to do, which is when the switch is greyed (`DiTSwitches.acceleratorsUnavailable`). `models` and `status` report it (`"accelerators"`); `add --accelerators` / `--no-accelerators` (`--ane` / `--no-ane` before) set it for one job, as a click on the switch. `set beta [OPTION on|off]` writes the property each of Settings › Advanced › Beta's five switches is bound to (`developerMode` for More Solvers, `stepsBeta` for Choose Steps, `fasterRenderingBeta`, `sizeScalesBeta`, `negativeLoRAStrengths`), so their `didSet` do what a click does; `status` gives them as `"beta"`. They are this Mac's preferences, and the copy of the mirror the window shows is set with them. `set save-images FOLDER|off` is Settings › General's Save Every Image To…, off by default: each new image of the history (a render, a crop, an enlargement, a grid's sheet) is also written to that folder as a PNG with its metadata, under its `save` name; the folder must exist, it is never created; alone, `set save-images` reads it. The arguments of `rewrite`, `describe` and `set` are read in the library (`WriterOrder`, `SetOrder`, `BetaOption`), where the tests read them.

**The history, the series, the models and the library are the window's too.** Each of these calls the function the button or the menu item calls, with the guard the window puts on it: what is greyed out in the window is a refusal here, with the same sentence.
`remove i7 [i8 …]` is `AppState.delete` (⌫), `undo [--redo]` the window's own `UndoManager` (⌘Z, ⇧⌘Z: removals, crops, a prompt carried to the rack), `again <i7|file.png>` is `redo` (⌘R, Render Again) and, from a PNG that Siliconed wrote, `restoreSettings(fromPNG:)` as a drop does it, then Generate; `reuse` is Reuse These Settings, without the render. A gesture that queues jobs answers with them (on a followed Mac each is answered by that Mac, as `vary` does). `crop i7 --box X,Y,W,H | --ratio P:Q` asks the editor's own frame rules (`Crop.Frame`, `CropSpec` in the library) and the same `cropped` that Return calls; `save` writes PNG, JPEG and HEIC as the Save and Export panels do.
A grid's cell is named `g2c3` (the grid, then its rank in the sheet's reading order from 0, the same arithmetic as the job's `column` and `row`; its job `t5` names it too); `sheet g2` is `addSheet`, on the series the window shows and nothing else. `import FILE …` is `perform` (⌘O, a drop): a model whose family is not installed opens the Models sheet with the family's license, as the window does, and the command says so (`needs_family`) without ever accepting it. `uninstall ID|lora NAME|downloads` is the sheet's Uninstall, the trash of an imported model or a LoRA, and Discard; `install stop` is its Stop. `macs` lists the paired Macs and their state, `compute NAME|this` is the Compute On menu (`MainWindow.compute(on:)`), `macs connect|disconnect` are Settings › Other Macs' buttons. `library move DIR|home` is Move Library… and Move Back to This Mac, `open DIR` and `local` the two buttons beside them.
**What destroys or moves things needs `--yes`**: `uninstall` and `library move|home`. Without it the answer is a `dry_run` that says what would go, or what would be copied, and the room it frees or takes (`LibraryMove.plan`, the same function the move starts with, so the two cannot disagree). The refusals are the sheet's and Settings': a render, an installation or the measure running (`busy`), the library's disk absent (`library_absent`). **On a followed Mac** the sheet asks the other Mac and waits for a click on its screen; the commands do not wait for a click that is not theirs: `import`, `uninstall` and `library move|home` are refused there (`remote`). Never a pairing, a loan or a forgetting. What has no command is display, not a gesture: the comparison curtain, full screen, zoom, Statistics, Share and Copy, the rack's own fields (`add` is the form and Generate in one), and the window's sheets (they only show what a command answers).

## The command

`silicontrol` ships inside the app, at `Siliconed.app/Contents/Helpers/silicontrol` (target `Silicontrol`,
`Sources/Silicontrol/main.swift`). Settings › Advanced › Command Line (“Install…”) links it into
`/usr/local/bin`. It is a pipe: Foundation and the system's frameworks only, it does not link the engine. If the app is not
running, any command but the help launches it and waits for its socket (30 s at most); `silicontrol open`
does only that. The help (`help`, `help <topic>`, `<command> --help`, no argument) is answered by the
command itself, without launching the app.

## The socket and the protocol

- **Unix socket** `silicontrol.sock` in the app's space (by default
  `~/Library/Application Support/Siliconed/silicontrol.sock`, or under `SILICONED_ROOT`) — there even
  when the library was moved to another disk —, mode `0600`:
  only the account that launched the app can open it. No network port. It disappears when the app
  quits; a dead socket is replaced at the next launch, and a second Siliconed on the same library
  does not listen.
- **One connection per command.** The client writes one JSON line,
  `{"argv": ["add", "--batch", "2", "…"], "cwd": "/current/folder"}`, and the app answers with JSON
  lines, then closes. Intermediate lines carry `"type"` (`added`, `image`, `sheet`, `start`, `stage`,
  `step`, `end`; `started`, `loading`, `text` for `rewrite` and `describe`; `started`, `progress` for `library move`); the last one carries `"ok"` (`true` or `false`, with `"error"`, `"message"` and
  often `"hint"` on failure). Without the client:
  `printf '{"argv":["status"]}\n' | nc -U ~/Library/Application\ Support/Siliconed/silicontrol.sock`.
- **All parsing is in the app** (`Sources/SiliconedApp/RemoteControl.swift`): `argv` is the command
  line as typed and the client is only a pipe. Only `open`, the help and `measure` are
  handled client-side: the first two have to work when the app is closed; `measure` is a series of
  renders that the client drives itself, each an `add --wait` on the socket, timed, while it samples
  the app's memory (`Sources/Measurement`). Relative paths resolve against `cwd`. The images of an edit
  (`--ref`, repeatable, in order: the first is image 1, the one edited) travel the same way, as
  paths or history names (`i7`), never as pixels: the app reads them itself.
- **Help has one source**, the module `SilicontrolHelp` (`Sources/SilicontrolHelp/RemoteControlHelp.swift`,
  English and French pages, Foundation only), linked by both sides: `silicontrol` answers from it
  without the app, and the app answers from it on its socket (`nc -U` gets the same pages). The
  language follows the user's preferred order between English and French. The answer is
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

Taken on the reference M1 Pro, 16 GB, when the remote control was written, with an early version of the app and not retaken since:

- **Same bits**: the app's image equals a render of the library called directly with the same
  request (512², seed 42, identical decoded bytes). Likewise for a variation of a variation queued by
  `silicontrol vary`: redone from its PNG's recipe, it comes out to the bit.
- **App Nap**: in the background, macOS throttled the app — 64 s instead of 44 for a 512², one
  step at 85 s. The app now holds a "render" activity (`ProcessInfo.beginActivity`) from the first
  job started until the queue is empty: **45.9 s** in the background (44.3 s for the same render
  outside the app).

## Another process that renders

One render per machine: the engine takes a lock (`/private/tmp/siliconed-render.lock`) for each
render, so another process using the engine (a build from source, say) is refused with `render_already_running` while the
app renders, and the reverse (two models in 16 GB: the system would swap and both timings would mean
nothing). Such a program can queue its render in the app with `silicontrol add` instead.
