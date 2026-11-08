// **The remote control's help: the complete reference, in one place.**
//
// `silicontrol help`, `silicontrol help <topic>`, `silicontrol help all` and `silicontrol <command> --help`
// are answered with one line, `{"ok": true, "help": "<text>"}` (the client prints the field raw). An
// AI told "run `silicontrol help`" must be able to do everything without another document: what the
// help says, the server does.
//
// A module of its own, Foundation only, shared by the two sides: `silicontrol` answers the help
// itself — reading a manual must not launch the app — and the app answers it on its socket (`nc -U`
// gets the same pages). One text, one routing (`answer`), so the two cannot diverge.
//
// Two versions of the same pages — English, and a faithful French one, picked by the user's language
// order among the two (`preferred`). The protocol itself (commands, flags, JSON keys, error codes) is
// English in both: only the prose is translated.

import Foundation

// This module is no part of the engine: only the app and `silicontrol` link it.
package enum RemoteControlHelp {
    /// One language's pages: the overview, and a page per topic — and the refusal of an unknown topic.
    struct Pages: Sendable {
        let noHelp: @Sendable (_ topic: String) -> String
        let topics: @Sendable (_ list: String) -> String
        let overview: String
        let pages: [String: String]
    }

    /// The topics in reading order (`help all` follows it).
    static let order = ["contract", "open", "models", "install", "add", "grid", "edit", "wait", "status", "follow",
                        "cancel", "clear", "set", "history", "save", "vary", "enlarge", "crop", "again", "remove", "undo", "sheet", "rewrite", "describe",
                        "instructions", "settings", "start-image", "lab", "bracket", "rack", "subjects", "import", "uninstall", "macs", "library", "diagnose", "measure", "errors", "recipes"]
    static let aliases = ["examples": "recipes", "reuse": "again", "compute": "macs"]

    /// French when the user puts it before English. Not `Bundle.main.preferredLocalizations`: the app's
    /// bundle has French phrases, `silicontrol` (a bare tool in `Contents/Helpers`) has none, and both
    /// must answer in the same language. `forPreferences: nil` is the user's order: without the
    /// argument, a bare tool gets `["en"]` whatever the user's language.
    static var preferred: Pages {
        Bundle.preferredLocalizations(from: ["en", "fr"], forPreferences: nil).first == "fr" ? french : english
    }

    /// **The answer to a help request**, or `nil` if `argv` is not one: `help [topic]`, `-h`, `--help`,
    /// `<command> --help` (or `-h`), and nothing at all (the overview). A page is
    /// `{"ok": true, "help": …}`; an unknown topic, the usage failure (`"error": "usage"`, with the
    /// topics as `"hint"`).
    package static func answer(_ argv: [String]) -> [String: Any]? { answer(argv, preferred) }

    static func answer(_ argv: [String], _ pages: Pages) -> [String: Any]? {
        let subject: String?
        switch argv.first {
        case nil: subject = nil
        case "help", "-h", "--help": subject = argv.dropFirst().first
        case let command? where argv.dropFirst().contains("--help") || argv.dropFirst().contains("-h"): subject = command
        default: return nil
        }
        guard let text = text(for: subject, pages) else {
            return ["ok": false, "error": "usage", "message": pages.noHelp(subject ?? ""),
                    "hint": pages.topics(topicList)]
        }
        return ["ok": true, "help": text]
    }

    /// The text for `topic` (`nil`: the overview), or `nil` if there is no such topic.
    static func text(for topic: String?, _ pages: Pages = preferred) -> String? {
        guard let topic else { return pages.overview }
        let key = aliases[topic] ?? topic
        if key == "all" {
            return ([pages.overview] + order.compactMap { pages.pages[$0] })
                .joined(separator: "\n" + String(repeating: "─", count: 78) + "\n\n")
        }
        return pages.pages[key]
    }

    static var topicList: String { (order + ["all"]).joined(separator: ", ") }

    // MARK: - English

    static let english = Pages(noHelp: { "no help for \"\($0)\"" }, topics: { "topics: \($0)" }, overview: """
    silicontrol — drive Siliconed (the image generation app) from a terminal

    Siliconed is a macOS app that keeps ONE queue of renders and a history of images. These
    commands talk to the running app: what they add appears in its window, and what the mouse
    starts, they see. Only one app computes, so there are never two renders at once.
    The app is launched automatically if it is not running (any command but help does it).

    GETTING STARTED
      silicontrol open                         launches the app if needed and waits until it answers (≤ 30 s)
      silicontrol models                       what is installed: ids, formats, LoRAs, editing
      silicontrol add --wait --out out "a 30 year old woman posing in a library"
                                               one image, blocking; the line "type": "image" gives its "path"

    THE CONTRACT (details: help contract)
      stdout = JSON lines, one per line. The LAST one carries "ok": true|false.
      The lines before it (add --wait, grid --wait, wait, follow) carry "type".
      A failure: "error" (stable code, see help errors), "message", often "hint" (what to do).
      Exit: 0 ok · 1 ok:false · 2 usage · 3 the app cannot be reached.
      Exception: help answers {"ok": true, "help": "<text>"} and the client prints the text.
      t3 = a job (one request for one or more images) · i7 = an image of the history · g2 = a grid · g2c3 = its cell.

    THE COMMANDS (details: help <command>, or <command> --help)
      open                                     launches the app and waits until it answers
      models                                   installed models, formats, LoRAs, default steps
      install ID [--standard|--compact|--light] shows a model's license and size in the app, to accept there
      install writer                           the Prompt Lab's writer (1.25 GB), no license of its own
      diagnose [--issue|--stop]                measures this Mac (≈ 1 min, no model) → the report's JSON; --stop: the sheet's Stop
      measure [options] [-- add options]       times and peak memory of a series of renders → a TSV and its summary
      add [options] "<prompt>" [format] [seed] puts a request in the queue → its job (t3)
      grid --x AXIS [--y AXIS] [options] "<prompt>" …
                                               an XY grid: one job per cell, then a sheet → its grid (g2)
      wait [t3 …] [--out D]                    blocks until done; writes the PNGs into D
      wait g2 [--out D]                        blocks until a grid's sheet
      status                                   what runs, what waits, the time left
      follow                                   live events until the queue is empty
      cancel [t3 …]                            stops the running render, or removes jobs
      clear                                    empties the queue and stops the running render
      set faster-rendering on|off [--model ID] a model's Faster Rendering, as a click (queue empty)
      set accelerators on|off [--model ID]     a model's accelerators, as a click (queue empty)
      set beta [OPTION on|off]                 the five Beta options of Settings, as a click; alone, reads them
      set save-images [FOLDER|off]             Save Every Image To… of Settings: each new image as a PNG there; alone, reads it
      history [N]                              the session's images (most recent first)
      save i7 [i8 …] DEST [--format F]         writes images (.png/.jpg/.heic file, or folder in png|jpeg|heic)
      vary i7 subtle|strong [--batch N]        variations of an image (same settings, same seed)
      enlarge i7 [--download]                  the image ×4, by macOS's super-resolution → its image (i8)
      rewrite "<text>" [--for ID]              the Prompt Lab's Rewrite: a prompt rewritten for a model → its text
      describe <image|i7> [--for ID]           the Prompt Lab's Describe: an image told as a prompt → its text
      instructions [rewrite|describe] [--for ID] [--set "<text>" | --file PATH | --restore]
                                               what the writer is told, per model and gesture (Settings › Prompt Lab)
      start-image [<image|i7>] [--strength S] · start-image --remove
                                               the starting image (img2img): an image of the Prompt Lab, and its strength
      lab [open|close] · lab add|role|remove … the Prompt Lab: its images, a role each; folded or not
      lab rewrite · lab describe N [--wait]    its Rewrite and Describe, written in its text as its buttons write
      settings [PANE] · settings lab [rewrite|describe] [--for ID]
                                               opens Settings on a pane, or Prompt Lab on one instruction
      bracket KNOB · bracket close             a knob's ± clicked: its bubble open (seed, detail, solver, start, loras, lora NAME)
      rack top|dit                             the rack scrolled to its head, or to the DiT's settings
      remove i7 [i8 …]                         Remove from History · undo [--redo]: ⌘Z, ⇧⌘Z
      again <i7|file.png> · reuse <i7|file.png> Render Again (Same Seed) · Reuse These Settings (an image, or a PNG of another day)
      crop i7 --box X,Y,W,H | --ratio P:Q      Crop… → a new image (i8)
      sheet g2                                 Add Sheet to History (the series shown, its images side by side)
      import FILE.safetensors|FILE.gguf …      Import a Model or LoRA… (⌘O); a family not installed shows its license in the app
      install stop                             Stop the installation or import in progress
      uninstall ID|lora NAME|downloads [--yes] Uninstall / Remove / Discard: without --yes, says what would go and the room freed
      macs [connect|disconnect NAME] · compute NAME|this
                                               the paired Macs and their state · Render › Compute On
      library [move DIR|home [--yes]|open DIR|local]
                                               where the models live: Move Library…, Move Back, Open, Use This Mac's
      help [topic]                             this reference

    NOTHING DEPENDS ON A CLICK, but pairing a Mac, accepting a license, « Lend This Mac » and « Forget »: those stay in the window.
    A command that removes or moves models acts only with --yes.

    EDITING: silicontrol add --model qwen-image-2.1 --ref photo.png --ref dog.png "put the dog of image 2 next to her"
      up to 3 --ref, in order; the first is image 1, the one edited (help edit)

    STARTING FROM AN IMAGE (img2img): silicontrol add --image photo.png --strength 0.65 "a lighthouse at dusk"
      Z-Image, Anima and Krea 2; 0 keeps the image, 1 ignores it (help start-image)

    OTHER PAGES: help edit · help errors · help recipes · help all (the whole reference at once)
    """, pages: [
        "contract": """
        THE CONTRACT — what holds for every command

        1. stdout only contains JSON lines, one per line (ASCII keys, sorted: jq '.job' works).
           Nothing else is written there. The one exception is the client printing the "help"
           text of a help answer raw.
        2. The LAST line is the verdict: {"ok": true, …} or {"ok": false, …}.
        3. The lines before it exist only for add --wait, grid --wait, wait and follow; they carry
           "type": added, image, sheet, start, stage, step, end. rewrite, describe and lab rewrite|describe --wait
           have their own: started, loading, text (help rewrite); library move, started and progress (help library).
        4. A failure: {"ok": false, "error": "<code>", "message": "<sentence>", "hint": "<what to do>"}.
           The code is stable (help errors); "hint" is not always there. "message" and "hint" are
           in the app's language; only "error" is meant for programs.
        5. Exit code: 0 verdict ok · 1 verdict ok:false · 2 usage error · 3 the app cannot be
           reached (not running and not launchable, or not answering).
        6. Short names, numbered within the app's session and lost when it quits:
           t3 = a job (one request: 1 to 8 images) · i7 = an image of the history ·
           g2 = a grid (one job per cell, then its sheet, an image of the history) ·
           g2c3 = a cell of a grid, by its rank in the sheet's reading order, from 0.
           A bare number is also accepted for a job (3 = t3).
        7. Nothing is written to disk without --out or save: the history lives in the app.
        8. Relative paths resolve against the terminal's current directory.
        9. Same arguments, same seed, same model: the same image, bit for bit.
        10. Closing the app's window empties its queue and stops the render; the app stays open.
        """,

        "open": """
        silicontrol open

        Launches Siliconed.app in the background (without bringing it to the front) if it is not
        answering, then waits until it answers (30 s at most). No effect if it is already open.
        Every other command does this too when the app is not running: `open` is the way to do
        only that.

        Answer: {"ok": true, "message": "Siliconed open" | "Siliconed was already open", "socket": "<socket>"}
        Failures: app_not_found (the app cannot be found) · app_not_running (it does not
                  answer after 30 s). Exit 3.

        This command is handled by the client, not by the app.
        """,

        "models": """
        silicontrol models

        The models the app shows, installed or not: Z-Image Turbo, Qwen-Image-2.1 Turbo, Anima Turbo
        and Krea 2 Turbo, plus the models you imported (« add » refuses a model the app does not show). Read it
        before « add »: the ids, the LoRAs and the formats come from here.

        Answer:
          {"ok": true, "models": [
            {"id": "z-image", "name": "Z-Image Turbo", "ready": true, "default": true, "default_steps": 8,
             "formats": ["1024x1024", "832x1216", …], "commercial_license": true,
             "license": "Apache 2.0", "license_accepted": true,
             "license_urls": ["https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/blob/<sha>/README.md"],
             "edit": false, "max_references": 0, "variant": "compact",
             "variants": [{"variant": "standard", "disk_bytes": 20357263360},
                          {"variant": "compact", "disk_bytes": 11749122176},
                          {"variant": "light", "disk_bytes": 9536600064}],
             "loras": [{"lora": "zimage-flat-color-v2-1", "name": "zimage flat color v2.1"}, …]},
            {"id": "qwen-image-2.1", "ready": false, "default": false, "edit": null, "max_references": null,
             "variant": null, "missing": "<the absent file>", "install": "silicontrol install qwen-image-2.1", …}, …]}

        id          → --model of « add »; "default": true = the one taken without --model
        variants    → the versions a publisher's model installs in, with their space on disk:
                      "standard" (the publisher's weights), "compact" (the default; 8-bit weights
                      published by third parties: smaller on disk, an image slightly different) and
                      "light" (the image model in 4 to 8 bits, Siliconed's own 4-bit format (SQ): the
                      smallest; another set of weights, an image that may compose differently).
                      Z-Image's, Qwen-Image-2.1's and Krea 2's Compact and Light read an 8-bit text
                      encoder; Anima's three versions read the publisher's (bf16), only the image
                      model changes. Absent for an imported model
        variant     → the version on disk; null while the model is not installed (and for an
                      imported model, which has its own weights)
        loras.lora  → --lora of « add » (only the LoRAs of THIS model)
        edit        → the model accepts --ref (editing an image with a prompt); null while it is
                      not installed
        max_references → how many --ref it reads at most (Qwen-Image-2.1 3, the others 0);
                      null while it is not installed
        start_image → "start_image": true when the model starts from an image (img2img: add --image,
                      start-image) — Z-Image, Anima, Krea 2; absent while it is not installed
        formats     → the suggested formats; any valid WxH is accepted (help add)
        accelerators → the DiT box's « Accelerators » for the installed version: true = part of the
                      DiT on the Neural Engine and the GPU's tensor units, where this Mac's map
                      chooses them; false = fp32 throughout (--accelerators, --no-accelerators of
                      « add »; help set). Only on a Mac whose measure offers one, and once the model
                      is ready
        faster_rendering → Faster Rendering for the installed version (help set): what a render takes,
                      so false while its Beta option is off; off by default; once the model is ready
        license_urls → the license file(s) on Hugging Face, at the revision the installation takes:
                      the model's, then each component under a license of its own (a Compact: the
                      publisher's, then each third party's model card). Z-Image has none: its model card, which
                      declares Apache 2.0.
        commercial_license → false: the model is for non-commercial use (Qwen-Image-2.1: Qwen
                      Research; Anima: CircleStone Labs Non-Commercial — its images may still be
                      used commercially, the license says so). Krea 2 Community is true with
                      conditions: under $1M of annual revenue, a content filter wherever the model
                      is deployed
        license_accepted → false: the first render of this model waits until the user accepts its
                      license in the app (a sheet shows it); declined, the job ends in "error"
                      with "code": "license_not_accepted"
        A model with "ready": false cannot render; its "install" field is the command that offers
        it in the app (help install) — the user reads the license and the size, and accepts there.
        """,

        "install": """
        silicontrol install ID [--standard|--compact|--light]
        silicontrol install writer
        silicontrol install stop

        Opens the app's Models sheet on the model ID (an "id" of « models »), with its license and
        the space it will take on disk. NOTHING is downloaded by the command: the user reads, then
        clicks « Accept and Install » in the app (tens of GB, several minutes). The answer comes at
        once:
          {"ok": true, "model": "qwen-image-2.1", "installed": false, "variant": "standard", "message": "…"}
          {"ok": true, "model": "z-image", "installed": true, "variant": "standard", "message": "already installed"}
        Then « silicontrol models » says "ready": true once the installation is done.

        Without a flag, the version installed is kept (none installed: Compact, or Light on a Mac
        of 8 GB — preselected, the user sees all three): a command that names no version never
        switches it.
        --standard  preselects the Standard version (the publisher's weights);
        --compact   the Compact version, --light the Light version (every model of the app —
                    the "variants" of « models »).
                    One version of a model is installed at a time: asking for another one of an
                    installed model offers to replace it, and the user still accepts in the app. The
                    installed one keeps rendering until the new one is complete, then goes — unless
                    the disk cannot hold both, which the app says before the user accepts.
        An imported model's ID installs what it lacks (encoder, VAE) in its family's version
        installed: a flag that contradicts that version is refused.

        install writer is the Prompt Lab's « Install the Writer » button, by the same function: the one thing the Lab
        needs that a model's installation does not bring, the language model's head (1.25 GB, two byte ranges of
        Qwen-Image-2.1's text encoder, each checked by sha256). It has no license of its own and no sheet: it is the
        license of Qwen-Image-2.1, which the encoder's installation had accepted (else "license_not_accepted", and
        install qwen-image-2.1 shows it). The download starts at once and the answer comes at once:
          {"ok": true, "installed": false, "installing": true, "download_bytes": 1244668…, "message": "…"}
          {"ok": true, "installed": true, "message": "already installed"}
        "status" says the progress ("installing") and the end ("writer": {"state": "ready"}); a failure is in
        "writer"."failure". Stop it with « install stop ». Like the install of a model, it acts on this Mac: it is refused
        ("remote") while the window computes on another one.
        install stop is « Stop » in the Models sheet and the sidebar: the installation, or the import, in progress stops; the files
        already downloaded are kept, the one in progress starts over. It works on a Mac followed too (the lender's forge, or this Mac's
        upload). Answer: {"ok": true, "stopped": true | false}.
        FAILURES (install writer): writer_absent (Qwen-Image-2.1 is not installed: no encoder to give a head to),
        writer_memory (this Mac is too small for the writer), writer_preparing, license_not_accepted, busy (a render,
        an installation or the diagnostic is running), remote.

        FAILURES: usage (also: --compact or --light for a model that has neither, two version flags
        together, a flag that contradicts the version an imported model's family has), model
        (unknown), busy (a render, an installation or the
        diagnostic is running: wait, then retry).
        """,

        "diagnose": """
        silicontrol diagnose [--issue]
        silicontrol diagnose --stop

        « Measure This Mac » (the app's Help menu), from the terminal. The sheet opens in
        the app and the diagnostic runs — about a minute, five at most, no model needed: what
        the silicon announces (the GPU's cores and clock, the CPU's clusters, the Neural Engine),
        then each
        engine alone and in sets on matrix products and on attention, the forms of the weights and
        their preparation, the heat and the memory; each output checked against the exact result,
        and a route that computed wrong is left out. After each step it checks that the Mac was
        free (the threads still had their cores, the GPU answered promptly, identical calls took
        the same time) and measures the step again, while the ceiling leaves seconds; what stayed
        busy is kept, its chains read on their quick side (a disturbance only adds time), and said
        in "findings" ("disturbed") with how far identical calls differed ("regime"."spreads") and
        the cores the others used ("regime"."foreign"). It runs in a
        process of its own: a route whose load kills that process (a program the Neural Engine
        refuses) is left out, the measure starts again without it, and "judge" says
        {"excluded": true, "reason": "died"}. The queue waits for it, and it for the
        queue. The report is the model of this Mac: from the next render, the engine works out
        from its "model" how many rows each engine takes at every product and every attention;
        before it, the defaults. A finished measure is always applied: its "findings" say, each with
        its effect on the plan ("fact", "left out", "bounded", "fast side"), what was missing, what
        was aberrant and what was uncertain — never a reason to refuse it.

          {"type": "started", "stages": ["weight preparation", …]}
          {"ok": true, "report": {"schema": 17, "machine": {…, "inventory": {…}},
            "model": {"lines": {"gpu/fp32": {"t0": 0.0003, "rate": 3.5e12}, …},
                      "sharing": {"gpu+ane": {"keep": {"gpu": 0.88, "ane": 0.74}, …}, …}, "crest": {…}, "heat": {…}, …},
            "measurements": {…}, "judge": {"ane/fp16": {"error": …, "threshold": …, "excluded": false}, …},
            "memory": {…}, "stages": {…}, "findings": [], …}}

        --issue   then opens GitHub's issue form in the browser, prefilled with the report (the
                  sheet's « Open the Issue »); the user submits it. "issue_url_length" says how
                  long the link was; beyond ~8 000 characters the JSON goes to the clipboard
                  instead and "message" says so.

        diagnose --stop is « Stop » in the sheet, from another terminal: the measure running stops (the diagnose that waits answers
        "stopped"). Answer: {"ok": true, "stopped": true | false}.

        FAILURES: usage, busy (a render or an installation is running), render_already_running
        (another process renders), stopped (« Stop » in the sheet).
        """,

        "add": """
        silicontrol add [options] "<prompt>" [512|1024|WxH|P:Q@1K|2K] [seed]

        Puts a request in the app's queue and returns at once (unless --wait). This page is
        complete. Options and arguments go in any order; the first argument that is not an option
        is the prompt.

        ARGUMENTS
          "<prompt>"          ONE argument: always in quotes. An extra word is refused.
                              {a|b|c} in it: one job per alternative (ALTERNATIVES below).
          512 | 1024 | WxH    format. 512 = 512x512, 1024 = 1024x1024; WxH = width x height
                              (832x1216 = portrait, the "formats" of « models »). Each side
                              ≥ 512 and a multiple of 16 (32 for qwen-image-2.1), ≤ 8192
                              (≤ 2048 for anima); no area limit, the memory decides
                              ("insufficient_memory"). Default: 1024x1024.
          P:Q@1K | P:Q@2K     the same, as a proportion and a resolution: the app's Size in Two Scales,
                              accepted whether or not that Beta option is on (no quotes needed). 1K is
                              about 1 megapixel, 2K about 4. Proportions: 1:1, 4:5, 3:4, 2:3, 9:16,
                              5:4, 4:3, 3:2, 16:9. The pixels are the app's menu's:
                                1K   1:1 1024x1024  4:5 896x1120  3:4 864x1152  2:3 832x1248
                                     9:16 768x1344 (really 4:7, 1.6 % off)
                                2K   1:1 2048x2048  4:5 1792x2240  3:4 1728x2304  2:3 1664x2496
                                     9:16 1440x2560 (exact)
                              The landscapes are the portraits turned (3:2 = 1248x832 at 1K). The model's
                              limits apply as to WxH: on anima only 1:1 exists at 2K (2048 px a side).
          seed                integer. Absent: drawn at random, returned in "seeds".

        OPTIONS
          --model ID          an "id" of « models ». Default: z-image.
          --lora NAME[:STRENGTH]
                              repeatable, 3 at most. NAME = a "loras.lora" of « models » (or its
                              displayed name, or a path). STRENGTH: 1 by default, 0.8 attenuates,
                              > 1 emphasizes; below 0 (a « slider » LoRA), with Experimental →
                              Negative LoRA Strengths on in the app's Settings.
          --steps N           number of steps, 1 to 50. Default: the model's "default_steps". A
                              model may have a schedule for some counts only (Qwen-Image-2.1: 8
                              only).
          --detail more|most  the rack's Detail: finer texture and small details, same time;
                              most can over-sharpen. Default: normal (the image as the model
                              makes it). Not on Anima nor Krea 2. Jobs and images say
                              "detail" when it is not normal.
          --solver NAME       the rack's Solver, with More Solvers on only: euler (default, the one
                              the model was distilled for), euler-ancestral, dpmpp-2m or unipc.
                              Same steps, same time, another image. Jobs and images say "solver"
                              when it is not euler.
          --accelerators, --no-accelerators
                              the DiT box's « Accelerators », as a click: part of the DiT on the
                              Neural Engine (fp16) and on the GPU's tensor units (bf16, fp16),
                              where this Mac's map chooses them — less time or energy, an image a
                              little different from fp32's —, or fp32 throughout. Kept for the
                              model's installed version, like the switch; the job keeps what it
                              was given. Default: the model's choice ("accelerators" of
                              « models »). --accelerators only on a Mac whose measure offers one.
                              --ane, --no-ane: their former names. Jobs say "accelerators"
                              there; images say what could run: "neural_engine": true,
                              "tensor_units": "bf16".
          --batch N           N images (1 to 8) with seeds s, s+1, …; the prompt is encoded once
                              for the whole batch (faster than N requests).
          --ref IMAGE|i7      editing by instruction: an image file or an image of the history.
                              Repeatable, in order, up to the model's "max_references" (1 to 3
                              for Qwen-Image-2.1). Only for a model with "edit": true.
                              THE FIRST --ref IS IMAGE 1, THE ONE EDITED; the others are what the
                              prompt borrows from. The prompt says what changes, and may name the
                              images by number (« image 1 », « image 2 »): Qwen-Image-2.1 reads
                              them as <image1>, <image2>. Leave the format out: the output then
                              follows image 1 (help edit).
          --image IMAGE|i7    img2img: the render starts from this image (a file, or an image of the
                              history) instead of noise, and keeps its composition. Only for a
                              model with "start_image": true (Z-Image, Anima, Krea 2); not with
                              --ref. Leave the format out: the output takes the model's format
                              nearest the image's proportions; any format is accepted, the image
                              then filled and its centre kept. The rack takes the image, as
                              « start-image » does (help start-image).
          --strength S        with --image: how far the render moves away from it — diffusers'
                              strength, 0.65 or 65%, above 0 and at most 1 (1 ignores the image).
                              Default: the model's. Jobs and images say "strength" and "start_image".
          --preview, --no-preview
                              a preview at every step in the app (writes nothing). Default: the
                              app's « Preview at every step » switch.
          --wait              blocks until done, then behaves like « wait t<n> ».
          --out D             with --wait: writes the PNGs into D (created if needed).

        Everything is checked BEFORE entering the queue (model ready, LoRA of the right model,
        format, reference, alternatives): an accepted request can only fail during the computation.

        ALTERNATIVES — "woman posing in a {library|greenhouse}" is two jobs, as in the app
          {a|b|c}             one job per alternative, all at the SAME seed(s) (the series
                              compares prompts); --batch N gives each job its N images.
          several groups      every combination, in reading order, the first group varying the
                              slowest: "{a|b} x {c|d}" → "a x c", "a x d", "b x c", "b x d".
          {|red }dress        an empty alternative is allowed: "dress", "red dress". Nothing is
                              trimmed: the spaces belong to the alternative that carries them.
          \\{  \\}  \\|          the characters themselves. Outside a group | is a plain character:
                              a prompt without braces is rendered as typed.
          Refused (error "prompt"): a { never closed, a } that closes nothing, a group inside a
          group; more than 64 combinations. Each image carries its expanded "prompt", and so do
          the PNG's metadata. For one image laying them side by side: « silicontrol grid ».

        ANSWER
          {"ok": true, "job": "t3", "position": 1, "seeds": [42], "model": "z-image",
           "prompt": "…", "width": 832, "height": 1216, "steps": 8, "edit": false, "references": 0,
           "loras": [{"lora": "zimage-flat-color-v2-1", "strength": 0.8}], "estimate_s": 45.2}
          position   : 0 = already running, 1 = the next one, …
          estimate_s : null until the app has rendered this model at this format, with as many
                       references, in the session (each reference lengthens the render).
          With --wait: this line arrives with "type": "added" instead of "ok", then the lines of
          « wait » (help wait).
          With alternatives (2 jobs or more):
          {"ok": true, "jobs": [{"job": "t3", "prompt": "woman posing in a library",
            "choices": ["library"], "position": 1, "estimate_s": 45.2, …}, {"job": "t4", …}],
           "estimate_s": 90.4}
          choices    : the alternative taken in each group, in order (what a grid labels).
          estimate_s : the sum, null if one job has no estimate yet.
          With --wait: one "type": "added" line per job, then the lines of « wait » for all.

        FAILURES: usage, prompt, model, solver, detail, accelerators, lora, format, steps, edit, ref, img2img
        (--image on a model that does not start from an image, or a strength that leaves no step), subject (a named subject the prompt cites, help
        subjects), queue_full (64 waiting, or not enough room for every alternative), busy (an
        installation is running: no render until it is done), write.

        Indicative times (MacBook Pro M1 Pro 16 GB, Z-Image, 8 steps): ~34 s at 512², 113–119 s at 1024².
        """,

        "grid": """
        silicontrol grid --x AXIS [--y AXIS] [--no-sheet] [options] "<prompt>" [512|1024|WxH|P:Q@1K|2K] [seed]

        An XY grid: one or two settings swept, ONE JOB PER CELL in the queue — each an ordinary
        job, its image in the history like any other —, then, when the last cell ends, a SHEET:
        one image with every cell side by side, the axes' values as labels and the prompt as its
        title, added to the history (i…). Nothing is written to disk without --out or save.
        Options and arguments are those of « add » (help add), except --batch: a cell is one image.
        The window's series (« Bracket… » on a knob, a {…} group in the prompt, the lot) are grids too, drawn on its canvas: the
        same cells, the same images, the sheet only on « Add Sheet to History » (« sheet g2 »): --no-sheet renders a grid so,
        its verdict with "sheet": null. The window takes the grid's axes as its series; a models, steps, listed seeds or images
        axis stays silicontrol's. « reuse » answers the rack's series ("series": its x and y as --x and --y read them, "excess"
        beyond two, "renders" the jobs « Generate » would queue).

        AXES — kind[:which][=values]; values separated by commas, decimals with a point
          prompt              the prompt's {…} group      prompt:2      its second group
          seed=42,7,1000      these seeds                 seeds=4       4 seeds from the seed: s, s+1, …
          steps=4,8,12        these step counts (each one the model has a schedule for)
          lora=0.4,0.7,1      strengths of the first --lora
          lora:NAME=0.4,1     strengths of the --lora NAME (or lora:2=… for the second)
          model=z-image,qwen-image-2.1   these models, each at its own steps (no --lora: a LoRA is
                              made for one model)
          image               each --ref image alone, with the same instruction (an edit)
          loras=none,flat,ink one LoRA added to the stack per cell (none: the cell without), at
          loras:0.6=…         strength 1 — or the one given; names as --lora takes them
          solver=euler,dpmpp-2m   these solvers (euler, euler-ancestral, dpmpp-2m, unipc), same seed
                              and steps; More Solvers, like --solver
          start=0.45,0.6,0.75 strengths of the starting image (--image), from 0 (kept) to 1 (ignored)
          detail=normal,more,most   these levels of Detail, where the model offers it
          --x gives the columns, --y the rows (without --y: one row). X and Y vary two different
          settings. EVERY {…} GROUP OF THE PROMPT MUST BE AN AXIS: a group left out would stack
          several images in a cell (error "grid"); write \\{ \\} for braces that are not
          alternatives. The cells render row by row, X varying the fastest. 64 cells at most, and
          the queue must have room for all of them.

        THE SHEET
          Each cell is reduced to 512 px on its long side at most — smaller beyond 16 cells: all
          the cells together stay within 12 megapixels (432 px for 8×8 at 512²). A cell that did not
          render (error, stopped, removed) is drawn empty, with why; no cell rendered, no sheet.
          The images of the cells stay full size in the history.
          File name: grid-<model>-<columns>x<rows>-<W>x<H>[-<seed>].png — the base seed, left
          out when the seeds are listed (seed=…).

        ANSWER
          {"ok": true, "grid": "g2", "columns": 2, "rows": 2, "estimate_s": 128.4,
           "jobs": [{"job": "t5", "grid": "g2", "cell": "g2c0", "column": 0, "row": 0, "prompt": "…",
                     "seeds": [42], "position": 1, "estimate_s": 32.1, …}, …]}
          estimate_s: the cells' sum, null until the model has rendered at this format.
          With --wait: one "type": "added" line per cell, then the lines of « wait g2 »:
            {"type": "image", "image": "i7", "job": "t5", …, "path": …}   each cell, as it comes
            {"type": "sheet", "image": "i9", "grid": "g2", "sheet": true, "columns": 2, "rows": 2,
             "empty_cells": 0, "seconds": 130.2, …, "path": …}            the sheet (seconds: the cells' sum)
            {"ok": true, "grid": "g2", "sheet": "i9", "jobs": [{"job": "t5", "outcome": "done", …}, …]}
          A cell not "done": "ok": false, "error": "render" — the sheet is made all the same, that
          cell empty ("sheet": null if no cell rendered).
          « wait g2 [--out D] » follows a grid later: at once if it has ended.
          A cell has a name of its own, g2c3 (its rank in the sheet's reading order, from 0 — row × columns + column), and its
          job's, t5. « sheet g2 » is « Add Sheet to History » on the series shown (help sheet).

        FAILURES: usage (no --x, --batch, --x given to « add »), grid (an axis that does not read or
        does not make a grid, more than 64 cells), prompt, model, solver, detail, accelerators, lora,
        format, steps, edit, ref, subject (a grid does not cite named subjects), queue_full, busy (an installation
        is running), write.
        """,

        "wait": """
        silicontrol wait [t3 t4 …] [--out D]
        silicontrol wait g2 [--out D]

        Blocks until the jobs are done — or, for a grid (g2, « grid »), until its sheet: its cells'
        lines, then a "type": "sheet" line, then the verdict with "grid" and "sheet" (help grid). With no job named: all those running or waiting at the
        time of the call — including those started by others (the mouse); naming your jobs
        (t3 t4) waits only for yours. A job that is already done answers at once, even long
        after. Ctrl-C only stops the wait: the renders go on in the app.

        LINES, as they come — one per image:
          {"type": "image", "image": "i7", "job": "t3", "model": "z-image", "prompt": "…",
           "width": 832, "height": 1216, "seed": 42, "steps": 8, "evaluations": 8,
           "seconds": 45.86, "loras": [], "edit": false, "references": 0, "path": "/abs/D/….png"}
          "path" only exists with --out; "write_error" if it could not be written.
          Without --out, the image is ONLY in the app's memory (history, lost when it quits):
          « save i7 DEST » writes it later.

        FILE NAMES: <model>[-lora-<name>[@<strength>]…][-p<steps>][-edit]-<W>x<H>-<seed>[-v<seed>x<strength>…].png
          (the strength if it is not 1, the steps if they are not the model's; a variation, each
          variation seed and its strength in hundredths). An existing file is never overwritten: -2, -3… are appended.

        VERDICT:
          {"ok": true, "jobs": [{"job": "t3", "outcome": "done", "images": ["i7"]}]}
          outcome: done · stopped (stopped mid-computation; the images already made stay)
                 · removed (removed from the queue before running) · error (with "message",
                 the engine's sentence in the app's language, and "code", its stable key:
                 "license_not_accepted", "insufficient_memory", "disk_full"…). An image whose
                 pixels came out unusable (black or blank) is never "done": "error", code
                 "non_finite_image".
          "warnings": the engine's own lines during the job's render (English), when it wrote any.
          If a job is not "done": "ok": false, "error": "render", exit 1.

        FAILURES: usage, unknown_job, write.
        """,

        "status": """
        silicontrol status

        What the app is doing now. Does not block.

        Answer:
          {"ok": true,
           "running": {"job": "t3", "stage": "denoising", "batch_image": 1, "step": 4,
                       "steps_total": 8, "fraction": 0.52, "remaining_s": 21.3, "model": "z-image",
                       "prompt": "…", "seeds": [42], …} | null,
           "queue": [{"job": "t4", "estimate_s": 45.2, …}, …],
           "waiting_for_license": [{"job": "t5", "state": "waiting_for_license",
                                    "license": "Qwen Research (non-commercial)", "license_urls": […], "model": …}, …],
           "installing": {"title": "Installing Z-Image Turbo", "last": "forging the DiT → …"} | null,
           "queue_remaining_s": 66.5, "remaining_complete": true,
           "history_images": 12, "library": "/Users/…/Siliconed", "faster_rendering": true,
           "accelerators": true, "beta": {"more_solvers": false, "faster_rendering": true, …},
           "writer": {"state": "ready", "installing": false, "failure": null, "bytes": 17…},
           "settings": {"read": true, "map": "2026-10-26T14:43:50Z", "warnings": []}}

        faster_rendering   : Faster Rendering of the model in the rack (help set).
        accelerators       : the « Accelerators » choice of the model in the rack (help set); only on a Mac whose measure offers one.
        beta               : the five Beta options of Settings (help set): more_solvers, steps, faster_rendering, size_scales, negative_lora.
        writer             : the Prompt Lab's writer (help rewrite): "state" is "ready", or the code rewrite would refuse with
                             (writer_absent, writer_head with "download_bytes", writer_memory with "gigabytes", writer_preparing);
                             "installing" while the head comes (help install); "failure" the sentence of an install that failed.
        settings           : the engine settings renders run on — read: false until this Mac's measure is; map: the
                             date of that measure (null: the defaults); warnings: why a measure was not
                             taken (one of another Mac or build). A measure is applied as it stands, whatever it found (see help diagnose). Two
                             series compared across launches compare their measures too.
        stage              : text, image (encoding a reference), denoising, decoding.
        steps_total        : the model's evaluations, not the requested steps.
        remaining_s        : null until the app has a measurement for this render.
        remaining_complete : false if a waiting job has no estimate (queue_remaining_s is a minimum).
        installing         : an installation or an import in progress, and its last journal line;
                             renders wait until it ends (add, grid and vary answer "busy").
        waiting_for_license: jobs the engine refused because their model's license is not accepted
                             (help models, "license_accepted"). They wait out of the queue for the
                             user's answer in the app's sheet: accepted, they go back to the head of
                             the queue; declined, they end in "error", "code": "license_not_accepted".
                             « wait » waits for them, « cancel t5 » removes them. Nothing can accept
                             from the terminal: the user reads the license in the app.
        """,

        "follow": """
        silicontrol follow

        The app's events, live, until the queue is empty (at once if nothing runs:
        {"ok": true, "message": "nothing is running or waiting"}). To watch; to collect images,
        « wait ».

        LINES:
          {"type": "start", "job": "t3", …settings…}
          {"type": "stage", "job": "t3", "stage": "denoising", "batch_image": 1}
          {"type": "step", "job": "t3", "batch_image": 1, "step": 4, "steps_total": 8, "seconds": 5.8}
          {"type": "image", "image": "i7", …}           (as in « wait », without "path")
          {"type": "end", "job": "t3", "outcome": "done" | "stopped" | "removed" | "error"}   (+ "warnings", as in « wait »)
          {"type": "sheet", "image": "i9", "grid": "g2", …}   (a grid's last cell ended; "image": null without a sheet)
        VERDICT: {"ok": true, "message": "queue empty"}
        """,

        "cancel": """
        silicontrol cancel [t3 t4 …]

        With no argument: stops the running render. With jobs: stops the one that runs, removes
        from the queue those that wait. The stop takes effect in under half a second; the images
        of a batch already made stay in the history.

        Answer: {"ok": true, "stopped": "t3" | null, "removed": ["t4"], "already_finished": []}
        FAILURES: usage, unknown_job.
        """,

        "clear": """
        silicontrol clear

        Empties the queue AND stops the running render — like « Stop everything » in the app.

        Answer: {"ok": true, "stopped": "t3" | null, "removed": ["t4", "t5"]}
        """,

        "set": """
        silicontrol set faster-rendering on|off [--model ID]
        silicontrol set accelerators on|off [--model ID]
        silicontrol set beta [OPTION on|off]
        silicontrol set save-images [FOLDER|off]

        Settings' switches, from the terminal. Each goes through the function its switch calls, so the choice is
        written and the window follows, exactly as a click.

        FASTER-RENDERING flips « Faster Rendering » for one model. It is chosen per installed version, like the
        accelerators, and off by default. It is a Beta option (Settings ▸ Advanced ▸ Beta): "on" is
        refused ("faster_rendering") while the option is off, and no render is a faster one then. On, from
        1024 px on each side the first evaluations run on a grid of half the size: faster, another image than at
        full size. Off: full size, the image checked against the reference.

        ACCELERATORS flips « Accelerators » for one model (the DiT box's switch): part of the DiT on the Neural
        Engine (fp16) and on the GPU's tensor units (bf16, fp16), where this Mac's map chooses them — less time or
        energy, an image a little different, checked against the reference in bf16; off, the DiT in fp32
        throughout, checked against the reference in fp32. Also per installed version; unless chosen, on, on every
        version. "neural-engine", its former name, still works. Refused ("accelerators"), on or off, on a Mac whose
        measure gives the accelerators nothing to do: the window greys the switch there with that reason, and there
        is nothing to choose. For one render only: add --accelerators / --no-accelerators.

        Both: without --model, the model in the rack. Only with nothing running or waiting (a job planned under
        one value and finished under the other would mix two recipes): else "busy". « models » tells each model's
        values, « status » the rack's.

        BETA is the five switches of Settings ▸ Advanced ▸ Beta, off by default:
          more-solvers      the solvers Euler a, DPM++ 2M and UniPC (--solver of « add »)
          steps             Choose Steps: the step count in the DiT box, for step-distilled LoRAs (--steps of « add »
                            takes any count the model has a schedule for, on or off)
          faster-rendering  the Faster Rendering switch in the DiT box — the master of the per-model choices above
          size-scales       Size in Two Scales: the Size menu as a proportion, then 1K or 2K (3:4@2K, help add)
          negative-lora     LoRA strengths from −1.5 to 1.5, for "slider" LoRAs (--lora NAME:-0.5)
        "set beta" alone reads them; "set beta OPTION on|off" sets one (the setting's own name works too:
        size-in-two-scales, negative-lora-strengths, choose-steps). The effects are the switch's: turning More Solvers
        off puts a solver back to Euler; turning Choose Steps off puts the rack's steps back to the model's recipe;
        turning Negative LoRA Strengths off brings the strengths already set back to 0.
        Like the switches, they wait for nothing and apply to what is queued next. They are this Mac's preferences; when
        the window computes on another Mac, its copy is set too. « status » gives them as "beta".

        SAVE-IMAGES is Settings ▸ General's « Save Every Image To… », off by default: given a folder (relative to the current
        directory; one named "off" is ./off), each new image of the history — a render, a crop, an enlargement, a grid's
        sheet — is also written there as a PNG with its prompt, seed and model, under its
        « save » name, never over another file. "off" turns it off; alone, it is read. The folder must exist ("write"
        otherwise): it is never created. This Mac's preference, whichever Mac computes.

        Answer: {"ok": true, "model": "krea2", "faster_rendering": false, "previous": true}
                {"ok": true, "model": "z-image", "accelerators": true, "previous": false}
                {"ok": true, "option": "size-scales", "enabled": true, "previous": false,
                 "beta": {"more_solvers": false, "faster_rendering": false, "size_scales": true, "negative_lora": false}}
                {"ok": true, "save_images": "/Users/me/Pictures/Siliconed", "previous": null}
        FAILURES: usage (also: an unknown Beta option, "hint" lists them), model, busy, faster_rendering ("on" without
        its Beta option), accelerators (no accelerator to choose on this Mac), write (save-images: not a folder).
        """,

        "history": """
        silicontrol history [N]

        The session's images, most recent first (the first N if N is given). The same object as
        the "image" lines of « wait », without "path": the history is in memory.

        Answer: {"ok": true, "images": [{"image": "i7", "job": "t3", "seed": 42, …}, …]}
        FAILURES: usage (N not an integer ≥ 1).
        """,

        "vary": """
        silicontrol vary i7 subtle|strong [--batch N]

        « Variations » on a history image, as its menu in the app: N jobs (1 to 8; default: the
        rack's « Images », 1 unless changed) with exactly its settings and its seed, each with a
        variation seed drawn at random. The starting noise is
        turned a little towards the variation seed's: subtle keeps the scene and the pose, strong
        keeps the idea and moves the composition. A variation of a variation varies THAT image.
        Each image's "variation" lists its variation seeds in order; its PNG redoes it.

        Answer: {"ok": true, "jobs": [{"job": "t5", "seeds": [42], "variation": [{"seed": 81…, "strength": 0.1}], …}, …]}
        Then: silicontrol wait t5
        FAILURES: usage, unknown_image, model, queue_full, busy (an installation is running).
        """,

        "enlarge": """
        silicontrol enlarge i7 [--download]

        « Enlarge ×4 » on a history image, as in the app: macOS's super-resolution (VideoToolbox,
        macOS 26), which sees the image and nothing else — no prompt, no denoising. 1024×1024 becomes
        4096×4096 in a second or two; the image joins the history beside its original, which stays.
        Its PNG keeps the original's metadata, plus "enlarged". It does not go through the queue: it
        is refused while a render runs, and answers once done (no wait).
        The model is Apple's, kept by macOS: missing, the command refuses (download_required) unless
        --download lets macOS fetch it first. An image already enlarged, or wider than 1920, is refused.

        Answer: {"ok": true, "image": {"image": "i8", "width": 4096, "height": 4096, "enlarged": 4, …}}
        Then: silicontrol save i8 big.png
        FAILURES: usage, unknown_image, refused (macOS before 26, a grid, already enlarged, a render
          running), download_required, download_failed, internal.
        """,

        "rewrite": """
        silicontrol rewrite "<text>" [--for ID]

        The Prompt Lab's « Rewrite » (View ▸ Show Prompt Lab), from the terminal: a language model installed with the app
        (Qwen3-VL-8B, the text encoder of Qwen-Image-2.1 made to write) rewrites a prompt the way the model it is aimed
        at reads best. The writer, its instructions for each model, the protection of the @names and the refusals are
        the Lab's own: a command and a click give the same text. Nothing is written to disk — the text comes back on
        the socket only (incognito). The text is one argument, in quotes, like « add »'s prompt.

          silicontrol rewrite "woman posing in a library"
          silicontrol rewrite --for anima "1girl, library, solo"
          silicontrol rewrite --for qwen-image-2.1 "@alice reading in a library"

        --for ID   the model to write for: an "id" of « models » (or its family) among the models installed. Without it,
                   what the Lab takes: the model chosen in its window, else the one in the rack, else the first installed.
        @names     where the model aimed at reads images (Qwen-Image-2.1), each @name is put aside and put back after the
                   writer: one it lost or deformed is told ("lost_mentions", "notice"), never dropped in silence. Elsewhere
                   @ is the model's own (Anima's artists) and the text goes as it is.

        LINES — the first text takes a few seconds, the weights being read; they go back when it ends:
          {"type": "started", "gesture": "rewrite", "for": "qwen-image-2.1"}
          {"type": "loading", "fraction": 0.4}        the weights being read (null: not known); once
          {"type": "text", "text": "A woman…"}        EVERYTHING written so far — cumulative, not a delta: the writer may revise its last word
        VERDICT:
          {"ok": true, "gesture": "rewrite", "for": "qwen-image-2.1", "text": "…", "tokens": 112, "lost_mentions": []}
          with "notice" (a sentence) when a mention was lost or deformed — the text is given anyway, as the Lab gives it.

        One gesture at a time: the writer is the Lab's too. An image that renders has priority, as in the Lab: a render
        that starts ends the gesture ("stopped"). A client that hangs up cancels it: the sampling stops, the weights go back.
        FAILURES: usage, model (--for names none of the models installed), busy (an image renders or waits), remote (the
        window computes on another Mac: the writer is this Mac's), writer_absent (no installed model carries a writer),
        writer_head (its head is to download: install writer), writer_memory (this Mac is too small), writer_preparing,
        writer_busy (another Rewrite or Describe is under way, from the terminal or the Lab's window), stopped,
        prompt_too_long, writer_failed (it failed or wrote nothing; "message" says why), and the engine's codes
        (render_already_running: another process renders).
        """,

        "describe": """
        silicontrol describe <image|i7> [--for ID]

        The Prompt Lab's « Describe », from the terminal: the writer looks at an image and writes the prompt that would
        make it, in the manner of the model it is aimed at. IMAGE is an image of the history (i7, as « history » names
        them) or a file — whatever macOS reads, its orientation applied, its transparency laid over white, as the Lab
        reads a dropped image. Same writer, same instructions, same refusals and same lines as « rewrite » (help rewrite);
        the verdict has "image" (the history name, or the path) where « rewrite »'s has "lost_mentions".

          silicontrol describe i7
          silicontrol describe photo.jpg --for anima

          {"ok": true, "gesture": "describe", "for": "z-image", "image": "i7", "text": "A woman…", "tokens": 138}

        --for ID   as in « rewrite ».
        FAILURES: those of « rewrite »; ref (the file is unreadable or missing, or not an image); unknown_image (an i… that is
        not in the history).
        """,

        "instructions": """
        silicontrol instructions [--for ID]
        silicontrol instructions rewrite|describe [--for ID] [--set "<text>" | --file PATH | --restore]

        Settings › Prompt Lab, from the terminal: the instruction the writer is sent before each Rewrite and each Describe,
        one per model and gesture — Siliconed's, or yours in its place. The whole text, in English, as it is sent. What you
        leave untouched follows Siliconed's from version to version; a text equal to Siliconed's is not kept as yours.

          silicontrol instructions                                  the ten: whose each one is
          silicontrol instructions rewrite --for krea2              one: its text
          silicontrol instructions rewrite --for krea2 --file mine.txt
          silicontrol instructions describe --for anima --set "Write danbooru tags only."
          silicontrol instructions rewrite --for krea2 --restore    Siliconed's back

        --for ID    a model's id or a family, installed or not. Without it, the model the Lab writes for.
        --set TEXT  yours, one argument in quotes; --file PATH reads it from a UTF-8 file. A blank one is kept as typed,
                    and Siliconed's is sent.
        --restore   Siliconed's, as « Restore » in Settings.

        Answer, one instruction:
          {"ok": true, "for": "krea2", "gesture": "rewrite", "edited": true, "sent": "yours", "text": "…", "instruction": "…"}
          "text" is what Settings shows, "instruction" what the writer is sent. The list: {"ok": true, "instructions":
          [{"for": "z-image", "gesture": "rewrite", "edited": false, "sent": "ours"}, …]}.
        Allowed while the window computes on another Mac: the instructions are this Mac's, as its writer is.
        To see one in Settings › Prompt Lab: settings lab rewrite --for krea2 (help settings).
        FAILURES: usage (a change names its gesture), model (--for names no model), unreadable (--file).
        """,

        "settings": """
        silicontrol settings [general|macs|lab|advanced]
        silicontrol settings lab [rewrite|describe] [--for ID]

        Siliconed › Settings… (⌘,), opened on one of its four panes: general (where the library is, Save Every Image To…),
        macs (this Mac, its measure, the other Macs), lab (Prompt Lab: the writer's instructions), advanced (Command Line,
        Beta). Without a pane, the one last shown. The window comes forward as the menu brings it; nothing in it changes.

          silicontrol settings macs
          silicontrol settings lab rewrite --for krea2      Prompt Lab on Krea 2's Rewrite instruction
          silicontrol settings lab describe                 the same model, its Describe

        lab with a gesture or --for: Prompt Lab on one instruction, its two menus set as a click on the Prompt Lab's
        « Your Instruction » sets them. --for ID is a model's id or a family, installed or not; without it, the model the
        Lab writes for. A menu not named stays as it is.

        Answer: {"ok": true, "settings": {"pane": "lab", "for": "krea2", "gesture": "rewrite"}}   ("for", "gesture": lab only)
        What the panes set, commands set: set (beta, save-images), library, macs, compute, instructions.
        Allowed while the window computes on another Mac: Settings are this Mac's.
        FAILURES: usage (a pane that does not read; a gesture or --for on a pane other than lab), model (--for names no model).
        """,

        "bracket": """
        silicontrol bracket seed|detail|solver|start|loras
        silicontrol bracket lora <name>
        silicontrol bracket close

        A click on a knob's yellow ± (or right-click › « Bracket… »): its bubble opens beside it, with the values the knob would
        take (three around the current one to begin with) — the window's own bubble, to be read or captured; nothing varies
        until it is committed there. seed: the Seed line's ± (the lot); start: the starting image's strength; loras: the LoRA
        stack's « Compare LoRAs… »; lora <name>: one LoRA's strength, by its name in the stack as "loras" prints it (lab, status).
        close: the bubble closed, the values as they were. To make a series from the terminal: grid (its axes become the
        window's series, help grid).

          silicontrol bracket detail
          silicontrol bracket lora incase
          silicontrol bracket close

        Answer: {"ok": true, "bracketing": "detail"}   ("bracketing": null after close)
        FAILURES: usage, lora (no LoRA of that name in the stack: the names it holds are listed), bracket (no ± that would open
        under the current model and settings: Detail on a model without it, no starting image, nothing to compare, a LoRA off).
        """,

        "rack": """
        silicontrol rack top|dit

        The rack's column scrolled as the hand scrolls it: top, to its head (the ports and the encoders); dit, to the DiT's box
        and its settings — the encoders then out of sight, the chain pinned in small at the top of the column, its stage at work
        lit, a click on it scrolling back. Nothing in the rack changes.

        Answer: {"ok": true, "rack": "dit"}
        FAILURES: usage (a place that does not read).
        """,

        "lab": """
        silicontrol lab [open|close|detach|attach]
        silicontrol lab add <image|i7> …
        silicontrol lab role <n> edited|reference|start|none
        silicontrol lab remove <n>
        silicontrol lab rewrite [--wait]
        silicontrol lab describe <n> [--wait]

        THE PROMPT LAB — the bar under the image, where the next render's text and images are prepared:
        what a drop on the canvas, ⌘V, « Send to the Prompt Lab » and the role menus of its images do. Its
        text is the rack's prompt (add and reuse write it); its images are numbered as the bar shows
        them, from 1.

          lab                          what it holds: folded or not, the text, the images and their roles
          lab open · lab close         unfolds or folds the bar (⌥⌘L); folding stops a Rewrite or a Describe
                                       under way and gives the writer's memory back
          lab detach · lab attach      unfolded, the bar in a window of its own, resizable — or back over the image;
                                       folded, it is the line under the image either way ("detached")
          lab add IMAGE|i7 …           files, or images of the history: each takes the role a newcomer
                                       takes — the next place under Qwen-Image-2.1 (edited, then references,
                                       three at most with the subjects the text cites), none elsewhere
          lab role N ROLE              edited · reference (Qwen-Image-2.1) · start (Z-Image, Anima, Krea 2,
                                       MOSS-Image: the render starts from it, the size takes the model's
                                       format nearest its proportions) · none. One image to edit and one
                                       start: choosing another sends the previous image to edit to the
                                       references, the previous start to none.
          lab remove N                 takes image N out
          lab rewrite                  the bar's « Rewrite »: its text rewritten for the model it writes for (the rack's),
                                       IN THE TEXT, token by token with its marks (additions, removals struck) — ⌘Z
                                       brings the text back. « rewrite » answers with a text and leaves the bar alone.
          lab describe N               image N's « Describe »: what it shows, written in the text where the caret is (at
                                       its end when the field has no caret), or as the whole text when it is empty
          --wait                       the answer once the text is written, after the lines of « rewrite » (help rewrite):
                                       started, loading, text — "text" being the field's whole text as it will read

        Both unfold the bar first (Rewrite and Describe are the unfolded plate's), then press the button: the writer, its
        instructions and the refusals are those of « rewrite ». Without --wait they answer as soon as the writer starts,
          {"ok": true, "gesture": "describe", "for": "z-image", "image": 2, "writing": true, "lab": {…}}
        and « lab » says where it stands ("writing": {"gesture": "describe", "phase": "loading"|"writing", "image": 2}, null
        once written). With --wait: {"ok": true, "gesture": "rewrite", "for": "z-image", "text": "<the field's text>",
        "lab": {…}}, and "notice" when a mention was lost. The gesture is the window's: a client that hangs up leaves it
        writing; the bar's Stop or « lab close » stops it, a render that starts too ("stopped").

        A CHANGE OF MODEL never gives a role, nor turns one into another: a reference keeps its role under
        Z-Image, without effect ("used": false), and works again back on Qwen-Image-2.1. "unused" counts
        what the next render will not read, as the line beside Generate does.

        ANSWER
          {"ok": true, "lab": {"unfolded": false, "detached": false, "model": "qwen-image-2.1", "text": "@alice on the quay",
            "images": [{"n": 1, "name": "quay.jpg", "width": 1536, "height": 1024, "role": "edited",
                        "used": true}], "unused": 0, "writing": null}}
        FAILURES: usage, image (no image N), role (a role the rack's model lacks), ref (image N cannot be read);
        lab rewrite and lab describe: those of « rewrite » (remote, busy, writer_absent, writer_head, writer_memory,
        writer_preparing, writer_busy), empty_text (lab rewrite on an empty text); with --wait, stopped, writer_failed,
        prompt_too_long.
        """,

        "start-image": """
        silicontrol start-image [<image|i7>] [--strength S]
        silicontrol start-image --remove

        THE STARTING IMAGE (img2img) — an image of the Prompt Lab given the start's role (« Start From This
        Image », or « lab role N start »), and its strength, a setting of the rack. Only for a model that
        starts from an image ("start_image": true in « models »: Z-Image, Anima, Krea 2, MOSS-Image).
        Qwen-Image-2.1 edits by reference instead (--ref, help edit).

          start-image                  what the rack holds
          start-image IMAGE|i7         brings an image into the lab as the start (a file, or an image of the
                                       history): the next renders start from it. The size is set to the model's format
                                       nearest its proportions, at the rack's scale — and stays free.
          … --strength S               with it, its strength; alone, retouches the one carried
          start-image --remove         takes it out: the next renders start from noise

        THE STRENGTH is diffusers' strength: how far the render moves away from the image. 0.65 or 65%,
        above 0 and at most 1: low keeps its composition and colours, 1 ignores it. It counts in steps:
        at 8 steps, 0.65 redraws 6 of them (0.63 and 0.74 too). Default: Krea 2 0.65, Z-Image and
        Anima 0.75, MOSS-Image 0.80. A strength that redraws every step (at 8 steps, above 0.875)
        starts from noise alone: the image is ignored ("image_ignored": true), the render goes on.
        Kept in hundredths: the PNG writes it ("Denoising strength"), « reuse » brings it back — the
        image itself is not in the file.

        ANSWER
          {"ok": true, "rack": {"model": "z-image", "width": 1024, "height": 1024, …,
            "start_image": {"name": "photo.png", "width": 3024, "height": 4032, "strength": 0.65,
                            "used": true, "redrawn_steps": 6, "of_steps": 8, "image_ignored": false}}}
          "start_image": null when the rack holds none; "used": false when the rack's model does not
          start from an image (it is then dropped at the next change of model). « status » says the same.

        FAILURES: img2img (the rack's model does not start from an image), no_start_image (--strength
        alone, with no image carried), ref (unreadable or missing image), unknown_image, usage, remote
        (the window computes on another Mac: not offered there yet).
        """,

        "subjects": """
        NAMED SUBJECTS — silicontrol subjects [add … | remove NAME]

        The same library as the Prompt Lab's (Window → Prompt Lab). A subject is a picture (or two of the same subject) with a name, a kind and a frame; the
        prompt cites it as @name, and its pictures go to the model as references, after the --ref
        images. A model that reads images only (Qwen-Image-2.1); 3 images at most per render.
        In the app's memory, like the history: the subjects go when the app quits (incognito).

          silicontrol subjects                                     the library
          silicontrol subjects add IMAGE --name NAME --kind KIND [--frame x,y,w,h] [--second IMAGE2]
          silicontrol subjects remove NAME                         forgets it
          silicontrol add --model qwen-image-2.1 "@alice in @red-coat, posing in a library"

        KIND: person, garment, object, place or style — what the model takes from the picture.
        NAME: letters, digits, - and _ (« red-coat »); @Red-Coat cites it too (case is ignored).
        --frame: the part of the image to look at, in fractions (0,0,1,1 = the whole image).
        The model's prompt (each subject tied to its image) is the job's "model_prompt".

        Answer: {"ok": true, "subjects": [{"name": "alice", "kind": "person", "pictures": [{"frame": [0, 0, 1, 1]}]}]}
        FAILURES: usage, ref (an image unreadable or missing), subject (unknown name, a name taken, too many images, a model that reads none).
        """,

        "remove": """
        silicontrol remove i7 [i8 …]

        « Remove from History » (⌫), as the menu does it: the images leave the history, in memory — nothing on disk is touched.
        The window undoes it with ⌘Z, and so does « silicontrol undo ». An image that was never saved is gone for good once
        the app quits (« save » first).

        Answer: {"ok": true, "removed": ["i7", "i8"], "history_images": 5,
                 "undo": {"can_undo": true, "action": "Remove 2 Images", "can_redo": false, "redo_action": ""}}
        On a Mac followed (Compute On), the other Mac's images are removed, and undone, the same way.
        FAILURES: usage, unknown_image.
        """,

        "undo": """
        silicontrol undo [--redo]

        Edit ▸ Undo (⌘Z), or with --redo Edit ▸ Redo (⇧⌘Z): whatever the window's undo manager would undo now, nothing more. What
        it knows: removing images from the history, a crop, a prompt carried to the rack from the Prompt Lab. A removal of images
        on a Mac followed is undone by asking the images back from that Mac while the session lasts.

        Answer: {"ok": true, "done": "undo", "action": "Remove Image", "history_images": 6,
                 "undo": {"can_undo": false, "action": "", "can_redo": true, "redo_action": "Remove Image"}}
        FAILURES: usage, nothing_to_undo (nothing to undo, or to redo).
        """,

        "again": """
        silicontrol again <i7|file.png>
        silicontrol reuse <i7|file.png>

        again is « Render Again (Same Seed) » (⌘R): the rack takes the image's settings (model, prompt, size, steps, LoRAs, edit
        images, detail, solver, and its seed), then « Generate » renders it, at that seed. On a sheet (a grid's image) the rack takes
        its series and renders it again at the sheet's seed. Same settings, same seed, same model: the same image, bit for bit.
        With a PNG that Siliconed wrote — an image of another day, a file from out/ — it is a PNG dropped on the canvas, then
        « Generate »: the rack reads the file's recipe (model, prompt, size, steps, LoRAs found in this library, seed, variations),
        and what could not come back is in "notes" (a model not installed, a LoRA missing, a solver that needs More Solvers…).
        The rack keeps what it took, as after a click: « silicontrol add » is unaffected, it never reads the rack.
        reuse is « Reuse These Settings »: the same filling of the rack, with no render.

        Answer (again): {"ok": true, "jobs": [{"job": "t6", "model": "z-image", "seeds": [42], …}]}
        Answer (reuse): {"ok": true, "rack": {"model": "z-image", "prompt": "…", "steps": 8, "width": 1024, "height": 1024,
                                              "seed": 42, "loras": [], "series": {"x": null, "y": null, "excess": [], "renders": 1}},
                         "notes": "…" (only when something did not come back). A sheet's series comes back with it (help grid).
        FAILURES: usage, unknown_image, ref (a file that is unreadable, or not a PNG Siliconed wrote), and for again the reasons « Generate » is
          greyed out: model (not installed), library_absent, busy (an installation is running), queue_full, link (the session with the
          other Mac is down), refused (anything else it says: "message" is the window's sentence — a memory refusal, a step count the
          model has no schedule for…). On a Mac followed, each job is answered by that Mac.
        """,

        "crop": """
        silicontrol crop i7 --box X,Y,W,H
        silicontrol crop i7 --ratio P:Q

        « Crop… » on a history image, with the frame given instead of dragged. --box is the frame in pixels of the image, the origin at
        the top left (x, y, width, height); --ratio is the largest frame of that proportion, centered (4:5, 16:9, 1:1…), which is where the
        editor's proportion menu starts. The frame's pixels become a NEW image at the head of the history; the original stays.
        « silicontrol undo » removes the crop, as ⌘Z does. The rules are the editor's: a side is never under 512 px, the whole image
        is no crop, a grid's sheet is not cropped. The new image keeps the original's metadata plus "cropped".

        Answer: {"ok": true, "image": {"image": "i8", "width": 1024, "height": 1024, …, "crop": {"x": 96, "y": 0, "width": 1024,
                 "height": 1024}}, "undo": {…}}
        FAILURES: usage, unknown_image, crop (the frame leaves the image, a side under 512 px, the whole image, a proportion the image
          cannot hold, a sheet), busy (the crop editor is open in the window).
        """,

        "sheet": """
        silicontrol sheet <g2>

        « Add Sheet to History » on the series shown on the canvas — a « Generate » with something bracketed, or a « grid » —:
        its images side by side, labelled, as one image of the history, in memory. Each call adds one. A « grid » draws its
        own when its last cell ends; a series of the window only when asked. Answered once the sheet is drawn.
          {"ok": true, "image": {"image": "i12", "grid": "g2", "sheet": true, "columns": 2, "rows": 2, "empty_cells": 0, …}}
        The series' images are in the history already, one by one, as any render's: the sheet only lays them out. A cell is
        named gNcM (help grid), and its job tN.
        Only the series shown — the newest: a new one takes the canvas.
        FAILURES: usage, unknown_job (no series is shown, or this is not the one shown), cell (no cell has a picture yet: wait
          for it), busy (cells are still on their way: « wait g2 »).
        """,

        "import": """
        silicontrol import FILE.safetensors|FILE.gguf …

        « Import a Model or LoRA… » (⌘O), with the files given instead of chosen — what dropping them on the window does. A model (DiT, as
        published on Civitai: bf16, fp16, fp32, 8 bits, or GGUF of 4 bits or more) or a LoRA. The same engine, the same rules, the same
        result in the library: « models » lists the model, with its family's version, and the LoRA under the models it fits. The file
        you give may be deleted afterwards. The answer comes when the import is done:
          {"ok": true, "files": ["flat.safetensors"], "models": [], "loras": ["flat"]}
        "models" and "loras" are what is new in the library. A VAE or an encoder file only completes the family: both lists are empty.
        **A model whose family is not installed**: the app shows the Models sheet on that family, with its license and the space it takes,
        and the import follows once the user has accepted there — the command never accepts a license, and answers at once:
          {"ok": true, "imported": [], "needs_family": "z-image", "install": "silicontrol install z-image", "message": "…"}
        Stop it with « install stop ». On a Mac followed (Compute On) the files would go up to the other Mac, whose owner confirms there:
        refused (remote).
        FAILURES: usage, ref (file unreadable or missing), import (not a .safetensors or a .gguf, or the engine refused the file: "message"
          says why, in the window's words), busy (a render, an installation or the diagnostic is running), library_absent, remote.
        """,

        "uninstall": """
        silicontrol uninstall ID [--yes]
        silicontrol uninstall lora NAME [--yes]
        silicontrol uninstall downloads [--yes]

        The Models sheet's removals. ID is a model of « models »: a publisher's model (z-image…) is « Uninstall… » — its DiT, encoder
        and VAE go, its imported models and LoRAs stay, as the sheet says —; an imported model is its trash; lora NAME is a LoRA's
        ("lora" in « models »); downloads is « Discard » on the interrupted downloads. Reinstalling downloads again; an imported model
        or a LoRA comes back by importing its file again.
        **Without --yes nothing is removed**: the answer says what would go and the room it frees.
          {"ok": true, "dry_run": true, "target": "z-image", "kind": "family", "frees_bytes": 11749122176, "stays": ["my-finetune"],
           "message": "nothing was removed: add --yes to remove it"}
        With --yes: the same answer with "removed": true instead of "dry_run".
        Refused as the sheet greys it out: a render running or waiting, an installation or the measure under way (busy); the library's
        disk absent (library_absent). On a Mac followed, the sheet asks the other Mac and waits for a click on its screen: refused here
        (remote) — the removal belongs to whoever sits at the Mac whose models they are.
        FAILURES: usage, model (unknown, or nothing of it installed, or no interrupted download), lora, busy, library_absent, remote.
        """,

        "macs": """
        silicontrol macs
        silicontrol macs connect NAME | disconnect NAME
        silicontrol compute NAME|this

        Render ▸ Compute On, from the terminal. macs lists the Macs paired to compute on, as the menu does, and their state; NAME is a Mac's
        name (case and accents aside) or its "id". compute NAME makes the window compute on that Mac — its queue and its history are shown,
        and every command follows (help contract) — and compute this brings it back to this Mac. A queue running here goes on. The switch
        is made as the menu makes it; the command answers once the session is up, or lost, or after 15 s ("link" says where it stands).
        macs connect and disconnect are Settings ▸ Other Macs' buttons: the session kept, or dropped.
        Never a pairing, a loan or a forgetting: those are accepted by the person in front of the screens, in the window.

        Answer (macs): {"ok": true, "computing_on": null | "Mac mini", "pairings_read": true,
                        "macs": [{"name": "Mac mini", "id": "…", "state": "connected", "computing": false, "machine": "M4, 16 GB…"}],
                        "lending": false, "lent_to": [{"name": "MacBook", "connected": false}]}
          state: connected, connecting, lost (trying again), on_the_network, not_on_the_network, refused_forgotten (that Mac forgot this
          one: pair again, in the window), refused_version (the two must run the same version of Siliconed).
          "pairings_read" false: the app has not read the Keychain yet (it does at launch if a Mac was paired, never without the user): the
          list is empty until Settings ▸ Other Macs has been opened.
        Answer (compute): {"ok": true, "computing_on": "Mac mini", "changed": true, "link": "connected", "message": "…" (if not yet)}
        FAILURES: usage, unknown_mac (no such paired Mac, or two of that name: use the "id"), link (that Mac refused the session).
        """,

        "library": """
        silicontrol library
        silicontrol library move DIR [--yes]
        silicontrol library home [--yes]
        silicontrol library open DIR
        silicontrol library local

        Settings ▸ General's library buttons. Alone: where the library is, its size and the room left.
        move is « Move Library… »: DIR is a folder on another disk (a « Siliconed » folder is made in it, unless DIR is one or already holds
        a library you are moving). The files are copied and read back one by one, the old place is emptied last; the licenses you accepted
        and this Mac's settings stay on this Mac. home is « Move Back to This Mac ».
        **Without --yes nothing is copied**: the answer says what would be and the room there is.
          {"ok": true, "dry_run": true, "destination": "/Volumes/X/Siliconed", "files": 14, "bytes": 11749122176, "free_bytes": 400000000000,
           "message": "nothing was moved: add --yes to move it"}
        With --yes: {"type": "started", …}, then {"type": "progress", "percent": 37} lines, then the verdict with "library".
        A move stopped in the app leaves the library where it was ("stopped").
        open is « Open Library… » (a folder that already holds a library: nothing is moved), local « Use This Mac’s Library » (nothing is moved).
        Refused as Settings greys them out: a render, an installation or the measure under way (busy), the disk absent (library_absent), and for
        move and home a Mac followed (remote: the library shown is the other Mac's).
        Answer: {"ok": true, "library": {"path": "…", "elsewhere": true, "reachable": true, "volume": "X", "free_bytes": …, "used_bytes": …,
                 "this_mac": "…"}}
        FAILURES: usage, busy, library_absent, remote, library (already on this Mac; or the engine refused the destination — the library
          itself, a disk that is full, one that cannot be written to, another library's models: "message" says which), stopped.
        """,

        "save": """
        silicontrol save i7 [i8 …] DEST [--format png|jpeg|heic]

        Writes images of the history to disk, in the three formats of the window's Save and Export panels.
          DEST = a .png, .jpg (.jpeg) or .heic file (a single image): written, or REPLACED if it exists; the extension decides.
          DEST = a folder (created if needed): naming and no-overwrite as in « wait ». --format says the format of its files (png by
                 default; the panel's pop-up remembers the last one used, the command does not).
        The PNG carries its metadata (prompt, seed, model, LoRAs); JPEG (0.9) and HEIC (0.85) are plain pictures, without metadata.

        Answer: {"ok": true, "files": [{"image": "i7", "path": "/abs/…"}]}
        FAILURES: usage, unknown_image, write.
        """,

        "edit": """
        EDITING BY INSTRUCTION — silicontrol add --ref … "<what changes>"

        A model with "edit": true in « models » edits images: the prompt is an instruction
        (« replace the cloudy sky with a blue sky »), not a description. The images are not
        re-noised: the model reads them, and the new image starts from noise.

        THE IMAGES, IN ORDER — one --ref each, 1 to "max_references" (Qwen-Image-2.1: 3):
          --ref A --ref B --ref C   →   image 1 = A, image 2 = B, image 3 = C
          image 1 is THE ONE EDITED; the others are what the prompt borrows from.
          A --ref is a file (PNG, JPEG, HEIC…; relative to the current directory) or an image
          of the history (i7: « edit the last image » is --ref i7).

        THE PROMPT — says what changes, and names the images by number:
          "change her jacket to red"
          "remove the second person from image 1 and put the dog of image 2 in her place"
          Qwen-Image-2.1's encoder sees the images: it reads them as <image1>, <image2>… in front
          of the prompt.

        THE OUTPUT'S SIZE — leave the format out, the output follows image 1:
          Qwen-Image-2.1: about 1024² (1 megapixel) at image 1's proportions, each side a
          multiple of 32 and at least 512.
          The app edits up to 1024² of AREA: a very elongated image 1 has its long side
          shortened, and a typed format beyond 1024² is refused ("format").
          The answer's "width"/"height" say it. A typed format overrides it.

        COST — each reference lengthens the render (the sequence grows by its tokens). The
        app's estimate ("estimate_s") is learned per model, format and number of references:
        null until that combination has rendered once in the session.

        No mask, no inpainting: the prompt alone says what changes.
        """,

        "measure": """
        silicontrol measure [--model ID] [--size 512|1024|WxH] [--renders N] [--seed S] [--prompt TEXT]…
                            [--tsv FILE] [--out DIR] [--settle SECONDS] [--tag TEXT] [--faster-rendering on|off] [-- add options]

        Repeats our timings and peaks on your Mac, with this app alone: a series of renders, each timed
        from the request to the PNG, the app's peak memory sampled while it renders, the pixels hashed.
        Defaults: z-image, 1024x1024, 3 renders, seed 42, "a 30 year old woman posing in a library",
        15 s of rest first. The measured model's Faster Rendering is written in the TSV for each
        render; --faster-rendering on|off sets it for the series (« set faster-rendering --model », as
        a click) and gives the value from before back at the end, even when the series fails. Without
        it the model's value is kept: a number published "with it off" is --faster-rendering off.

        The series: the queue must be idle (else "busy"); a load render (the weights come in; its time
        is reported apart); a witness; the N renders; a witness again. The witness is the default prompt
        at seed 42; the machine drifts by a few percent over an hour, and the two witnesses bound that
        drift: a series whose witness moved by more than 13 % is not comparable to another. The Mac
        is kept awake for the series. Each render:
          wall_s       from the request to the PNG, on this command's clock (what we publish)
          engine_s     the app's own "seconds" for the image
          peak_GiB     the app's phys_footprint, read every 100 ms, the maximum (what we publish)
          start_GiB    the footprint just before the request: what the app held already
          swapouts     pages swapped out during the render (vm_stat's Swapouts); disk_read_GB the app's reads
          md5          of the decoded 8-bit RGB pixels: same model, prompt, seed, size and settings must give
                       the same md5 on every run, every day (a mismatch is a defect: report it)
        beside the model, variant, steps, seed, accelerators (column neural_engine), app version, chip, memory, macOS, power
        source and thermal state. A render's prompt is encoded once and cached: with several renders on
        one prompt the encoder runs only the first time; give several --prompt to include it each time.

        Output: stdout, JSON lines — one {"type": "render", …} per render, then the verdict
        {"ok": true, "summary": {"median_seconds", "witness_head_seconds", "witness_tail_seconds",
        "witness_drift", "peak_gib", "swapouts", "bits_stable", "notes", …}, "tsv": "<path>"}.
        stderr: the same, for a person. The TSV (default siliconed-measure-<date>-<model>-<size>.tsv in
        the current folder) has one line per render, the witnesses and the load included.
        --out keeps the PNGs there; otherwise they are deleted after they are hashed.
        --tag labels the series in the TSV. After --, options go to "add" as they are (--steps, --lora,
        --accelerators, --no-accelerators); not --wait, --out, --batch, --ref, --model, --preview.

        For a comparison to mean something: the adapter plugged in, no other job on the Mac, the app
        freshly opened for a memory figure (a peak read after other models ran includes what they left).

        FAILURES: usage, busy (a render or an installation is running), unsupported (--faster-rendering on an app
        that does not report it), model (unknown or not installed),
        render (the render failed: its "error" and message), write. The rows measured before a failure stay in the TSV.
        """,

        "errors": """
        THE ERROR CODES ("error" in a verdict with "ok": false)

          app_not_running      the app cannot be reached (exit 3)           → silicontrol open
          app_not_found        « open » did not find the app (exit 3)       → build or install Siliconed.app
          usage                malformed command, option or argument (exit 2)
                                                                           → the prompt in quotes; help <command>
          prompt               {a|b} alternatives: a { never closed, a } alone, nested groups,
                               more than 64 combinations                   → help add (ALTERNATIVES)
          model                unknown or not installed model
                                                                           → "hint" lists the ready ones; models
          busy                 (install, diagnose, set, import, uninstall, library) a render, an installation or the diagnostic is
                               running; (crop) the crop editor is open; (sheet) cells are still on their way; (rewrite, describe, lab rewrite|describe) an image renders or waits; (add, grid, vary) an installation is running;
                               (any) more than 8 connections still sending their command
                                                                           → wait, then retry
          library_absent       (add, grid, vary) the library is on a disk that is not connected
                                                                           → connect it, then retry
          lora                 LoRA not found for this model, or > 3        → "hint" lists the compatible ones
          format               unreadable or out-of-bounds format; an edit beyond 1024² of area
                                                                           → WxH or P:Q@1K|2K, sides ≥ 512, multiples of 16 (32: qwen-image-2.1)
          steps                a step count the model has no schedule for   → its "default_steps"
          edit                 --ref on a model without editing, or more --ref than it reads
                                                                           → a model with "edit": true; its "max_references"
          ref                  unreadable reference image, or image to describe → check the path
          img2img              (add, start-image) --image on a model that does not start from an image,
                               or a strength that leaves no step           → a model "start_image": true; a higher --strength
          no_start_image       (start-image --strength) the rack holds no starting image
                                                                           → start-image <image|i7>
          solver               --solver without More Solvers, or on a model that walks Euler only
                                                                           → leave it out
          detail               --detail more|most on a model without Detail (Anima, Krea 2)
                                                                           → leave it out
          accelerators         --accelerators, or set accelerators, on a Mac whose measure offers no
                               accelerator                                 → leave it out: nothing to choose there
                               set accelerators on a Mac not measured yet  → render once (it measures), or diagnose
          subject              a named subject: unknown, a name taken, too many images, a model
                               that reads none, a grid citing one          → help subjects
          grid                 an axis that does not read, two axes on one setting, a {…} group
                               on no axis, more than 64 cells              → help grid
          queue_full           64 jobs waiting, or no room for every alternative or cell
                                                                           → wait, then retry
          remote               (install, import, rewrite, describe, lab rewrite|describe) the window computes on another Mac; these act on
                               this one; (uninstall, library move|home) a removal or a move there waits for a click on
                               the other Mac                               → Render › Compute On › This Mac
          link                 the session with the other Mac is down (add, grid, vary, again); (compute, macs)
                               that Mac refused the session                → macs
          refused              (again, vary, enlarge) the window would not do it, and says why in "message"
          nothing_to_undo      (undo) nothing to undo, or to redo           → the window's own undo list
          crop                 (crop) a frame the image cannot give: it leaves the image, a side under 512 px, the whole
                               image, a proportion it cannot hold, a sheet → help crop
          cell                 (sheet) no cell of the series has a picture yet → wait for it
          import               (import) not a .safetensors or a .gguf, or the engine refused the file → "message"
          unknown_mac          (macs, compute) no paired Mac of that name, or two → macs; the "id"
          library              (library) the library is already where you ask, or the engine refused the destination
                                                                           → "message"
          writer_absent        (rewrite, describe, lab rewrite|describe, install writer) no installed model carries a writer
                                                                           → install qwen-image-2.1
          writer_head          (rewrite, describe, lab rewrite|describe) the writer's head is to download (1.25 GB)
                                                                           → install writer
          writer_memory        (rewrite, describe, lab rewrite|describe, install writer) this Mac is too small for the writer
          writer_preparing     (rewrite, describe, lab rewrite|describe, install writer) the writer is being prepared → status
          writer_busy          (rewrite, describe, lab rewrite|describe) another Rewrite or Describe is under way, from the terminal or
                               the Lab's window                            → wait, then retry
          writer_failed        (rewrite, describe, lab rewrite|describe) the writer failed or wrote nothing; the engine's own code
                               (render_already_running…) when it has one   → "message"
          empty_text           (lab rewrite) the Prompt Lab's text is empty  → reuse <i7>, or lab describe <n>
          prompt_too_long      (rewrite, lab rewrite) the text is longer than the writer reads → shorten it
          license_not_accepted (install writer) Qwen-Image-2.1's license is not accepted yet
                                                                           → install qwen-image-2.1
          unsupported          (measure) --faster-rendering on an app older than silicontrol → same build
          unknown_job          this t… (or g…) does not exist in this session → status
          unknown_image        this i… is not in the history                → history
          write                folder or file cannot be written             → check the path
          render               (wait) a job did not finish                  → "outcome", "message" and "code" of each job
          stopped              (diagnose) the diagnostic was stopped (« Stop » in the sheet); (rewrite,
                               describe) a render started, which has priority; (lab rewrite|describe --wait) the same,
                               or Stop or lab close in the bar             → diagnose again, or wait and retry
          protocol, internal   should not happen                            → report it
        """,

        "recipes": """
        RECIPES

        One image, blocking, and its path:
          silicontrol add --wait --out out "a 30 year old woman posing in a library" 1024 7 \\
            | tail -1 | jq .ok

        A pile of requests, then their files (only yours, even if others are waiting):
          P="a 30 year old woman posing in a library"
          a=$(silicontrol add "$P" 832x1216 42 | jq -r .job)
          b=$(silicontrol add --model qwen-image-2.1 "$P" 1024 42 | jq -r .job)
          c=$(silicontrol add --lora zimage-flat-color-v2-1:0.8 --steps 10 "$P" 1024 42 | jq -r .job)
          silicontrol wait $a $b $c --out out/pile | jq -r 'select(.type=="image") | .path'

        Sweep a LoRA strength (same seed: only the strength changes):
          for f in 0.4 0.7 1.0; do silicontrol add --lora zimage-flat-color-v2-1:$f "$P" 1024 42; done
          silicontrol wait --out out/sweep

        Several seeds of the same prompt: --batch (faster than separate requests):
          silicontrol add --batch 4 --wait --out out/seeds "$P" 1024 100

        Compare places at the same seed (one job per alternative, help add):
          silicontrol add --wait --out out/places "a 30 year old woman posing in a {library|greenhouse}" 512 42 \\
            | jq -r 'select(.type=="image") | .prompt + " → " + .path'

        The same, as a sheet: places in columns, two seeds in rows (help grid):
          silicontrol grid --x prompt --y seeds=2 --wait --out out/grid \\
            "a 30 year old woman posing in a {library|greenhouse}" 512 42 | jq -r 'select(.type=="sheet") | .path'

        A LoRA's strength against seeds, one sheet, without waiting:
          g=$(silicontrol grid --lora zimage-flat-color-v2-1 --x lora=0.4,0.7,1 --y seed=42,7 "$P" 1024 | jq -r .grid)
          silicontrol wait $g --out out/lora

        Edit the last image (model with "edit": true) — it becomes image 1:
          i=$(silicontrol history 1 | jq -r '.images[0].image')
          silicontrol add --wait --out out --model qwen-image-2.1 --ref "$i" "change her jacket to red"

        Edit by instruction with two images (help edit):
          silicontrol add --wait --out out --model qwen-image-2.1 --ref photo.png --ref dog.png \\
            "remove the second person from image 1 and put the dog of image 2 in her place" 42

        Stop everything: silicontrol clear
        """,
    ])

    // MARK: - Français

    static let french = Pages(noHelp: { "pas d’aide pour « \($0) »" }, topics: { "sujets : \($0)" }, overview: """
    silicontrol — piloter Siliconed (l'app de génération d'images) depuis un terminal

    Siliconed est une app macOS qui tient UNE file de rendus et un historique d'images. Ces
    commandes parlent à l'app ouverte : ce qu'elles ajoutent apparaît dans sa fenêtre, ce que la
    souris lance, elles le voient. Une seule app calcule, il n'y a jamais deux rendus en même temps.
    L'app se lance toute seule si elle n'est pas ouverte (n'importe quelle commande sauf l'aide le fait).

    POUR COMMENCER
      silicontrol open                         lance l'app si besoin et attend qu'elle réponde (≤ 30 s)
      silicontrol models                       ce qui est installé : ids, formats, LoRA, édition
      silicontrol add --wait --out out "a 30 year old woman posing in a library"
                                               une image, bloquant ; la ligne "type": "image" donne son "path"

    LE CONTRAT (détails : help contract)
      stdout = des lignes JSON, une par ligne. La DERNIÈRE porte "ok": true|false.
      Les lignes d'avant (add --wait, grid --wait, wait, follow) portent "type".
      Échec : "error" (code stable, voir help errors), "message", souvent "hint" (quoi faire).
      Sortie : 0 ok · 1 ok:false · 2 usage · 3 l'app est injoignable.
      Exception : help répond {"ok": true, "help": "<texte>"} et le client imprime le texte.
      t3 = un travail (une demande d'une ou plusieurs images) · i7 = une image de l'historique · g2 = une grille · g2c3 = sa case.

    LES COMMANDES (détails : help <commande>, ou <commande> --help)
      open                                     lance l'app et attend qu'elle réponde
      models                                   modèles installés, formats, LoRA, pas par défaut
      install ID [--standard|--compact|--light] montre dans l'app la licence et la place d'un modèle, à accepter là
      install writer                           le rédacteur de l'Atelier de prompts (1,25 Go), sans licence propre
      diagnose [--issue|--stop]                mesure ce Mac (≈ 1 min, sans modèle) → le JSON du rapport ; --stop : l'Arrêter de la feuille
      measure [options] [-- options de add]    temps et pic mémoire d'une série de rendus → un TSV et son résumé
      add [options] "<prompt>" [format] [seed] met une demande en file → son travail (t3)
      grid --x AXE [--y AXE] [options] "<prompt>" …
                                               une grille XY : un travail par case, puis une planche → sa grille (g2)
      wait [t3 …] [--out D]                    bloque jusqu'à la fin ; écrit les PNG dans D
      wait g2 [--out D]                        bloque jusqu'à la planche d'une grille
      status                                   ce qui tourne, ce qui attend, le temps restant
      follow                                   les évènements en direct jusqu'à file vide
      cancel [t3 …]                            arrête le rendu en cours, ou retire des travaux
      clear                                    vide la file et arrête le rendu en cours
      set faster-rendering on|off [--model ID] le rendu plus rapide d'un modèle, comme un clic (file vide)
      set accelerators on|off [--model ID]     les accélérateurs d'un modèle, comme un clic (file vide)
      set beta [OPTION on|off]                 les cinq options Bêta des Réglages, comme un clic ; seul, les lit
      set save-images [FOLDER|off]             Enregistrer chaque image dans… des Réglages : chaque image nouvelle en PNG ; seul, le lit
      history [N]                              les images de la session (la plus récente d'abord)
      save i7 [i8 …] DEST [--format F]         écrit des images (fichier .png/.jpg/.heic, ou dossier en png|jpeg|heic)
      vary i7 subtle|strong [--batch N]        variations d'une image (mêmes réglages, même graine)
      enlarge i7 [--download]                  l'image ×4, par la super-résolution de macOS → son image (i8)
      rewrite "<texte>" [--for ID]             « Réécrire » de l'Atelier : un prompt réécrit pour un modèle → son texte
      describe <image|i7> [--for ID]           « Décrire » de l'Atelier : une image racontée en prompt → son texte
      instructions [rewrite|describe] [--for ID] [--set "<texte>" | --file CHEMIN | --restore]
                                               ce qu'on dit au rédacteur, par modèle et par geste (Réglages › Atelier de prompts)
      start-image [<image|i7>] [--strength S] · start-image --remove
                                               l'image de départ (img2img) : une image de l'Atelier, et sa force
      lab [open|close] · lab add|role|remove … l'Atelier : ses images, un rôle chacune ; replié ou non
      lab rewrite · lab describe N [--wait]    son Réécrire et son Décrire, écrits dans son texte comme ses boutons l'écrivent
      settings [VOLET] · settings lab [rewrite|describe] [--for ID]
                                               ouvre les Réglages sur un volet, ou l'Atelier de prompts sur une consigne
      bracket RÉGLAGE · bracket close          un clic sur le ± d'un réglage : sa bulle ouverte (seed, detail, solver, start, loras, lora NOM)
      rack top|dit                             le rack défilé jusqu'à sa tête, ou jusqu'aux réglages du DiT
      remove i7 [i8 …]                         Retirer de l'historique · undo [--redo] : ⌘Z, ⇧⌘Z
      again <i7|fichier.png> · reuse <i7|fichier.png> Rendre à nouveau (même graine) · Reprendre ces réglages (une image, ou un PNG d'un autre jour)
      crop i7 --box X,Y,W,H | --ratio P:Q      Recadrer… → une nouvelle image (i8)
      sheet g2                                 Ajouter la planche à l'historique (la série montrée, ses images côte à côte)
      import FICHIER.safetensors|FICHIER.gguf … Importer un modèle ou une LoRA… (⌘O) ; une famille non installée montre sa licence dans l'app
      install stop                             arrête l'installation ou l'import en cours
      uninstall ID|lora NOM|downloads [--yes]  Désinstaller / Retirer / Jeter : sans --yes, dit ce qui partirait et la place libérée
      macs [connect|disconnect NOM] · compute NOM|this
                                               les Mac appairés et leur état · Rendu › Calculer sur
      library [move DOSSIER|home [--yes]|open DOSSIER|local]
                                               où vivent les modèles : Déplacer la bibliothèque…, Ramener, Ouvrir, Utiliser celle de ce Mac
      help [sujet]                             cette référence

    RIEN NE DÉPEND D'UN CLIC, sauf appairer un Mac, accepter une licence, « Prêter ce Mac » et « Oublier » : ceux-là restent dans la fenêtre.
    Une commande qui retire ou déplace des modèles n'agit qu'avec --yes.

    ÉDITER : silicontrol add --model qwen-image-2.1 --ref photo.png --ref chien.png "put the dog of image 2 next to her"
      jusqu'à 3 --ref, dans l'ordre ; le premier est l'image 1, celle qu'on édite (help edit)

    PARTIR D'UNE IMAGE (img2img) : silicontrol add --image photo.png --strength 0.65 "a lighthouse at dusk"
      Z-Image, Anima et Krea 2 ; 0 garde l'image, 1 l'ignore (help start-image)

    AUTRES PAGES : help edit · help errors · help recipes · help all (toute la référence d'un coup)
    Les commandes, options et clés JSON sont en anglais ; seuls les textes sont traduits.
    """, pages: [
        "contract": """
        LE CONTRAT — ce qui vaut pour toutes les commandes

        1. stdout ne contient que des lignes JSON, une par ligne (clés ASCII, triées :
           jq '.job' marche). Rien d'autre n'y est écrit, sauf le client qui imprime tel quel
           le texte "help" d'une réponse d'aide.
        2. La DERNIÈRE ligne est le verdict : {"ok": true, …} ou {"ok": false, …}.
        3. Les lignes d'avant n'existent que pour add --wait, grid --wait, wait et follow ; elles
           portent "type" : added, image, sheet, start, stage, step, end. rewrite, describe et lab rewrite|describe
           --wait ont les leurs : started, loading, text (help rewrite) ; library move, started et progress (help library).
        4. Un échec : {"ok": false, "error": "<code>", "message": "<phrase>", "hint": "<quoi faire>"}.
           Le code est stable (help errors) ; "hint" n'est pas toujours là. "message" et "hint"
           sont dans la langue de l'app ; seul "error" est fait pour les programmes.
        5. Code de sortie : 0 verdict ok · 1 verdict ok:false · 2 erreur d'usage · 3 l'app est
           injoignable (pas ouvert et impossible à lancer, ou ne répond pas).
        6. Noms courts, numérotés dans la session de l'app et perdus quand elle quitte :
           t3 = un travail (une demande : 1 à 8 images) · i7 = une image de l'historique ·
           g2 = une grille (un travail par case, puis sa planche, une image de l'historique) ·
           g2c3 = une case d'une grille, par son rang dans l'ordre de lecture de la planche, depuis 0.
           Un nombre nu est aussi accepté pour un travail (3 = t3).
        7. Rien n'est écrit sur disque sans --out ou save : l'historique vit dans l'app.
        8. Les chemins relatifs se résolvent contre le répertoire courant du terminal.
        9. Mêmes arguments, même graine, même modèle : la même image, au bit près.
        10. Fermer la fenêtre de l'app vide sa file et arrête le rendu ; l'app reste ouverte.
        """,

        "open": """
        silicontrol open

        Lance Siliconed.app en arrière-plan (sans lui donner le premier plan) si elle ne répond pas,
        puis attend qu'elle réponde (30 s au plus). Sans effet si elle est déjà ouverte.
        Toute autre commande en fait autant quand l'app n'est pas ouverte : `open` sert à ne
        faire que cela.

        Réponse : {"ok": true, "message": "Siliconed open" | "Siliconed was already open", "socket": "<socket>"}
        Échecs  : app_not_found (l'app est introuvable) · app_not_running (elle ne répond
                  pas après 30 s). Sortie 3.

        Cette commande est traitée par le client, pas par l'app.
        """,

        "models": """
        silicontrol models

        Les modèles que l'app montre, installés ou non : Z-Image Turbo, Qwen-Image-2.1 Turbo, Anima
        Turbo et Krea 2 Turbo, plus les modèles que vous avez importés (« add » refuse un modèle que l'app ne montre
        pas). À lire avant « add » : les ids, les LoRA et les formats viennent d'ici.

        Réponse :
          {"ok": true, "models": [
            {"id": "z-image", "name": "Z-Image Turbo", "ready": true, "default": true, "default_steps": 8,
             "formats": ["1024x1024", "832x1216", …], "commercial_license": true,
             "license": "Apache 2.0", "license_accepted": true,
             "license_urls": ["https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/blob/<sha>/README.md"],
             "edit": false, "max_references": 0, "variant": "compact",
             "variants": [{"variant": "standard", "disk_bytes": 20357263360},
                          {"variant": "compact", "disk_bytes": 11749122176},
                          {"variant": "light", "disk_bytes": 9536600064}],
             "loras": [{"lora": "zimage-flat-color-v2-1", "name": "zimage flat color v2.1"}, …]},
            {"id": "qwen-image-2.1", "ready": false, "default": false, "edit": null, "max_references": null,
             "variant": null, "missing": "<le fichier absent>", "install": "silicontrol install qwen-image-2.1", …}, …]}

        id          → --model de « add » ; "default": true = celui pris sans --model
        variants    → les versions dans lesquelles s'installe un modèle d'éditeur, avec leur place sur
                      le disque : "standard" (les poids de l'éditeur), "compact" (le défaut ; poids
                      8 bits publiés par des tiers : plus léger sur le disque, une image légèrement
                      différente) et "light" (le modèle d'image en 4 à 6 bits tel qu'un tiers le
                      publie : la plus petite ; un autre jeu de poids, une image qui peut se composer
                      autrement). Les Compact et Légère de Z-Image, de Qwen-Image-2.1 et de Krea 2
                      lisent un encodeur de texte 8 bits ; les trois versions d'Anima lisent celui de
                      l'éditeur (bf16), seul le modèle d'image change. Absent pour un modèle importé
        variant     → la version sur le disque ; null tant que le modèle n'est pas installé (et pour
                      un modèle importé, qui a ses propres poids)
        loras.lora  → --lora de « add » (seulement les LoRA de CE modèle)
        edit        → le modèle accepte --ref (édition d'une image par un prompt) ; null tant
                      qu'il n'est pas installé
        max_references → combien de --ref il lit au plus (Qwen-Image-2.1 3, les autres 0) ;
                      null tant qu'il n'est pas installé
        start_image → "start_image": true quand le modèle part d'une image (img2img : add --image,
                      start-image) — Z-Image, Anima, Krea 2 ; absent tant qu'il n'est pas installé
        formats     → les formats conseillés ; tout WxH valide est accepté (help add)
        accelerators → les « Accélérateurs » de la boîte DiT pour la version installée : true = une
                      part du DiT sur le Neural Engine et les unités tensorielles du GPU, là où la
                      carte de ce Mac les choisit ; false = fp32 de bout en bout (--accelerators,
                      --no-accelerators de « add » ; help set). Seulement sur un Mac dont la mesure
                      en offre un, et le modèle prêt
        faster_rendering → le rendu plus rapide de la version installée (help set) : ce qu'un rendu
                      prend, donc faux tant que son option Bêta est éteinte ; éteint par défaut ; le modèle prêt
        license_urls → le(s) fichier(s) de licence sur Hugging Face, à la révision que l'installation
                      prend : celui du modèle, puis chaque composant sous une licence propre
                      (une Compact : celui de l'éditeur, puis la fiche de chaque tiers). Z-Image n'en a pas : sa
                      fiche, qui déclare Apache 2.0.
        commercial_license → false : le modèle est d'usage non commercial (Qwen-Image-2.1 : Qwen
                      Research ; Anima : CircleStone Labs Non-Commercial — ses images restent
                      utilisables commercialement, la licence le dit). Krea 2 Community est true
                      sous conditions : moins d'un million de dollars de chiffre d'affaires annuel,
                      un filtre de contenu partout où le modèle est déployé
        license_accepted → false : le premier rendu de ce modèle attend que l'utilisateur accepte
                      sa licence dans l'app (une feuille la montre) ; refusée, le travail finit en
                      "error" avec "code": "license_not_accepted"
        Un modèle "ready": false ne peut pas rendre ; son champ "install" est la commande qui le
        propose dans l'app (help install) — l'utilisateur y lit la licence et la place, et accepte.
        """,

        "install": """
        silicontrol install ID [--standard|--compact|--light]
        silicontrol install writer
        silicontrol install stop

        Ouvre la feuille Modèles de l'app sur le modèle ID (un "id" de « models »), avec sa licence
        et la place qu'il prendra sur le disque. La commande ne télécharge RIEN : l'utilisateur lit,
        puis clique « Accepter et installer » dans l'app (des dizaines de Go, plusieurs minutes). La
        réponse vient tout de suite :
          {"ok": true, "model": "qwen-image-2.1", "installed": false, "variant": "standard", "message": "…"}
          {"ok": true, "model": "z-image", "installed": true, "variant": "standard", "message": "already installed"}
        Ensuite « silicontrol models » dit "ready": true quand l'installation est finie.

        Sans drapeau, la version installée est gardée (aucune installée : Compact, ou Légère sur
        un Mac de 8 Go — présélectionnée, l'utilisateur voit les trois) : une commande qui ne nomme
        pas de version n'en change jamais.
        --standard  présélectionne la version Standard (les poids de l'éditeur) ;
        --compact   la version Compact, --light la version Légère (tous les modèles de l'app —
                    les "variants" de « models »).
                    Une seule version d'un modèle est installée à la fois : en demander une autre d'un
                    modèle installé propose de la remplacer, et l'utilisateur accepte toujours dans
                    l'app. L'installée continue de rendre jusqu'à ce que la nouvelle soit complète,
                    puis part — sauf si le disque ne peut pas tenir les deux, ce que l'app dit avant
                    l'acceptation.
        L'ID d'un modèle importé installe ce qui lui manque (encodeur, VAE) dans la version
        installée de sa famille : un drapeau qui la contredit est refusé.

        install writer est le bouton « Installer le rédacteur » de l'Atelier de prompts, par la même fonction : ce que l'Atelier
        demande et que l'installation d'un modèle n'apporte pas, la tête du modèle de langage (1,25 Go, deux plages d'octets
        de l'encodeur de Qwen-Image-2.1, chacune vérifiée par sha256). Elle n'a pas de licence propre ni de feuille : c'est
        la licence de Qwen-Image-2.1, que l'installation de l'encodeur a fait accepter (sinon « license_not_accepted », et
        install qwen-image-2.1 la montre). Le téléchargement commence aussitôt et la réponse vient aussitôt :
          {"ok": true, "installed": false, "installing": true, "download_bytes": 1244668…, "message": "…"}
          {"ok": true, "installed": true, "message": "already installed"}
        « status » dit l'avancement ("installing") et la fin ("writer" : {"state": "ready"}) ; un échec est dans
        "writer"."failure". L'arrêter se fait par « install stop ». Comme l'installation d'un modèle, elle agit sur ce Mac : refusée
        ("remote") tant que la fenêtre calcule sur un autre.
        install stop est « Arrêter » dans la feuille Modèles et la barre latérale : l'installation, ou l'import, en cours s'arrête ; les
        fichiers déjà téléchargés sont gardés, celui en cours repartira de zéro. Il marche aussi sur un Mac suivi (la forge du prêteur, ou
        le téléversement de ce Mac). Réponse : {"ok": true, "stopped": true | false}.
        ÉCHECS (install writer) : writer_absent (Qwen-Image-2.1 n'est pas installé : pas d'encodeur à qui donner une tête),
        writer_memory (ce Mac est trop petit pour le rédacteur), writer_preparing, license_not_accepted, busy (un rendu,
        une installation ou le diagnostic tourne), remote.

        ÉCHECS : usage (aussi : --compact ou --light pour un modèle qui n'en a pas, deux drapeaux de
        version ensemble, un drapeau qui contredit la version de la famille d'un modèle importé), model
        (inconnu), busy (un rendu, une installation ou le
        diagnostic tourne : attendre, puis réessayer).
        """,

        "diagnose": """
        silicontrol diagnose [--issue]
        silicontrol diagnose --stop

        « Mesurer ce Mac » (menu Aide de l'app), depuis le terminal. La feuille s'ouvre
        dans l'app et le diagnostic tourne — une minute environ, cinq au plus, sans modèle : ce
        que le silicium annonce (les cœurs et la fréquence du GPU, les grappes du CPU, le Neural
        Engine), puis
        chaque moteur seul et en ensembles sur des produits matriciels et sur l'attention, les formes
        des poids et leur préparation, la chaleur et la mémoire ; chaque sortie vérifiée contre le
        résultat exact, et une route qui a mal calculé est écartée. Après chaque étape elle vérifie
        que le Mac était libre (les fils avaient encore leurs cœurs, le GPU répondait sans attendre,
        des appels identiques prenaient le même temps) et refait l'étape tant que le plafond laisse
        des secondes ; ce qui est resté occupé est gardé, ses chaînes lues du côté rapide (une
        perturbation n'ajoute que du temps), et dit dans "findings" ("disturbed") avec l'écart des
        appels identiques ("regime"."spreads") et les cœurs qu'ont pris les autres ("regime"."foreign").
        Elle tourne dans un processus à part : une route dont le
        chargement tue ce processus (un programme que le Neural Engine refuse) est écartée, la mesure
        repart sans elle, et "judge" dit {"excluded": true, "reason": "died"}. La file l'attend, et
        lui attend la file. Le rapport est le modèle de ce Mac : dès le lancement suivant, le moteur
        tire de son "model" combien de lignes prend chaque moteur à chaque produit et à chaque
        attention ; avant, les défauts. Une mesure finie est toujours appliquée : ses "findings" disent,
        chacun avec son effet sur le plan ("fact", "left out", "bounded", "fast side"), ce qui
        manquait, ce qui détonnait et ce qui restait incertain — jamais une raison de la refuser.

          {"type": "started", "stages": ["weight preparation", …]}
          {"ok": true, "report": {"schema": 17, "machine": {…, "inventory": {…}},
            "model": {"lines": {"gpu/fp32": {"t0": 0.0003, "rate": 3.5e12}, …},
                      "sharing": {"gpu+ane": {"keep": {"gpu": 0.88, "ane": 0.74}, …}, …}, "crest": {…}, "heat": {…}, …},
            "measurements": {…}, "judge": {"ane/fp16": {"error": …, "threshold": …, "excluded": false}, …},
            "memory": {…}, "stages": {…}, "findings": [], …}}

        --issue   ouvre ensuite le formulaire d'issue GitHub dans le navigateur, prérempli avec le
                  rapport (« Ouvrir le signalement » de la feuille) ; c'est l'utilisateur qui l'envoie.
                  "issue_url_length" dit la longueur du lien ; au-delà de ~8 000 caractères le
                  JSON va dans le presse-papiers et "message" le dit.

        diagnose --stop est « Arrêter » dans la feuille, depuis un autre terminal : la mesure en cours s'arrête (le diagnose qui attend répond
        "stopped"). Réponse : {"ok": true, "stopped": true | false}.

        ÉCHECS : usage, busy (un rendu ou une installation tourne), render_already_running (un
        autre processus fait un rendu), stopped (« Arrêter » dans la feuille).
        """,

        "add": """
        silicontrol add [options] "<prompt>" [512|1024|WxH|P:Q@1K|2K] [seed]

        Met une demande dans la file de l'app et rend la main tout de suite (sauf --wait).
        Cette page est complète. Options et arguments vont dans n'importe quel ordre ; le premier
        argument qui n'est pas une option est le prompt.

        ARGUMENTS
          "<prompt>"          UN argument : toujours entre guillemets. Un mot en trop est refusé.
                              {a|b|c} dedans : un travail par alternative (ALTERNATIVES plus bas).
          512 | 1024 | WxH    format. 512 = 512x512, 1024 = 1024x1024 ; WxH = largeur x hauteur
                              (832x1216 = portrait, les "formats" de « models »). Chaque côté
                              ≥ 512 et multiple de 16 (32 pour qwen-image-2.1), ≤ 8192
                              (≤ 2048 pour anima) ; pas de limite de surface, la mémoire décide
                              ("insufficient_memory"). Défaut : 1024x1024.
          P:Q@1K | P:Q@2K     la même chose, en proportion et résolution : la Taille en deux échelles de
                              l'app, acceptée que cette option Bêta soit allumée ou non (sans guillemets).
                              1K vaut environ 1 mégapixel, 2K environ 4. Proportions : 1:1, 4:5, 3:4,
                              2:3, 9:16, 5:4, 4:3, 3:2, 16:9. Les pixels sont ceux du menu de l'app :
                                1K   1:1 1024x1024  4:5 896x1120  3:4 864x1152  2:3 832x1248
                                     9:16 768x1344 (en fait 4:7, 1,6 % d'écart)
                                2K   1:1 2048x2048  4:5 1792x2240  3:4 1728x2304  2:3 1664x2496
                                     9:16 1440x2560 (exact)
                              Les paysages sont les portraits couchés (3:2 = 1248x832 en 1K). Les limites
                              du modèle valent comme pour WxH : sur anima, seul 1:1 existe en 2K
                              (2048 px par côté).
          seed                entier. Absente : tirée au hasard, rendue dans "seeds".

        OPTIONS
          --model ID          un "id" de « models ». Défaut : z-image.
          --lora NOM[:FORCE]  répétable, 3 au plus. NOM = un "loras.lora" de « models » (ou son
                              nom affiché, ou un chemin). FORCE : 1 par défaut, 0.8 atténue,
                              > 1 appuie.
          --steps N           nombre de pas, de 1 à 50. Défaut : "default_steps" du modèle. Un
                              modèle peut n'avoir de calendrier que pour certains (Qwen-Image-2.1 :
                              8 seulement).
          --detail more|most  le Détail du rack : texture fine et petits détails, même durée ;
                              most peut trop accentuer. Défaut : normal (l'image telle que le
                              modèle la fait). Ni sur Anima ni sur Krea 2. Travaux et images
                              disent "detail" s'il n'est pas normal.
          --solver NOM        le Solveur du rack, en mode développeur seulement : euler (défaut,
                              celui pour lequel le modèle a été distillé), euler-ancestral,
                              dpmpp-2m ou unipc. Mêmes pas, même durée, une autre image. Travaux
                              et images disent "solver" s'il n'est pas euler.
          --accelerators, --no-accelerators
                              les « Accélérateurs » de la boîte DiT, comme un clic : une part du
                              DiT sur le Neural Engine (fp16) et sur les unités tensorielles du GPU
                              (bf16, fp16), là où la carte de ce Mac les choisit — moins de temps
                              ou d'énergie, une image un peu différente de celle du fp32 —, ou le
                              fp32 de bout en bout. Gardé pour la version installée du modèle,
                              comme l'interrupteur ; le travail garde ce qu'on lui a donné.
                              Défaut : le choix du modèle ("accelerators" de « models »).
                              --accelerators seulement sur un Mac dont la mesure en offre un.
                              --ane, --no-ane : leurs anciens noms. Les travaux y disent
                              "accelerators" ; les images, ce qui pouvait tourner :
                              "neural_engine": true, "tensor_units": "bf16".
          --batch N           N images (1 à 8) aux graines s, s+1, … ; le prompt est encodé une
                              fois pour tout le lot (plus rapide que N demandes).
          --ref IMAGE|i7      édition par instruction : un fichier image ou une image de l'historique.
                              Répétable, dans l'ordre, jusqu'au "max_references" du modèle (1 à 3
                              pour Qwen-Image-2.1). Seulement pour un modèle "edit": true.
                              LE PREMIER --ref EST L'IMAGE 1, CELLE QU'ON ÉDITE ; les autres sont ce
                              où le prompt puise. Le prompt dit ce qui change, et peut nommer les
                              images par leur numéro (« image 1 », « image 2 ») : Qwen-Image-2.1
                              les lit comme <image1>, <image2>. Ne pas donner de format : la sortie
                              suit alors l'image 1 (help edit).
          --image IMAGE|i7    img2img : le rendu part de cette image (un fichier, ou une image de
                              l'historique) au lieu du bruit, et garde sa composition. Seulement
                              pour un modèle "start_image": true (Z-Image, Anima, Krea 2) ; pas
                              avec --ref. Ne pas donner de format : la sortie prend le format du
                              modèle le plus proche des proportions de l'image ; tout format est
                              accepté, l'image le remplit alors et son centre est gardé. Le rack
                              prend l'image, comme « start-image » (help start-image).
          --strength S        avec --image : jusqu'où le rendu s'en éloigne — la strength de
                              diffusers, 0.65 ou 65%, au-dessus de 0 et au plus 1 (1 ignore
                              l'image). Défaut : celle du modèle. Travaux et images disent "strength" et
                              "start_image".
          --preview, --no-preview
                              un aperçu à chaque pas dans l'app (n'écrit rien). Défaut : l'interrupteur
                              « Aperçu à chaque pas » de l'app.
          --wait              bloque jusqu'à la fin, puis se comporte comme « wait t<n> ».
          --out D             avec --wait : écrit les PNG dans D (créé au besoin).

        Tout est vérifié AVANT d'entrer en file (modèle prêt, LoRA du bon modèle, format, référence,
        alternatives) : une demande acceptée ne peut plus échouer que pendant le calcul.

        ALTERNATIVES — "woman posing in a {library|greenhouse}" fait deux travaux, comme dans l'app
          {a|b|c}             un travail par alternative, tous à la MÊME graine (la série compare
                              des prompts) ; --batch N donne à chaque travail ses N images.
          plusieurs groupes   toutes les combinaisons, dans l'ordre de lecture, le premier groupe
                              variant le plus lentement : "{a|b} x {c|d}" → "a x c", "a x d",
                              "b x c", "b x d".
          {|red }dress        une alternative vide est permise : "dress", "red dress". Rien n'est
                              rogné : les espaces appartiennent à l'alternative qui les porte.
          \\{  \\}  \\|          les caractères eux-mêmes. Hors d'un groupe, | est un caractère
                              ordinaire : un prompt sans accolades est rendu tel quel.
          Refusé (erreur "prompt") : une { jamais fermée, une } qui ne ferme rien, un groupe dans
          un groupe ; plus de 64 combinaisons. Chaque image porte son "prompt" développé, et les
          métadonnées du PNG aussi. Pour une image qui les pose côte à côte : « silicontrol grid ».

        RÉPONSE
          {"ok": true, "job": "t3", "position": 1, "seeds": [42], "model": "z-image",
           "prompt": "…", "width": 832, "height": 1216, "steps": 8, "edit": false, "references": 0,
           "loras": [{"lora": "zimage-flat-color-v2-1", "strength": 0.8}], "estimate_s": 45.2}
          position   : 0 = il tourne déjà, 1 = le prochain, …
          estimate_s : null tant que l'app n'a pas rendu ce modèle à ce format, avec autant de
                       références, dans la session (chaque référence allonge le rendu).
          Avec --wait : cette ligne arrive avec "type": "added" au lieu de "ok", puis les
          lignes de « wait » (help wait).
          Avec des alternatives (2 travaux ou plus) :
          {"ok": true, "jobs": [{"job": "t3", "prompt": "woman posing in a library",
            "choices": ["library"], "position": 1, "estimate_s": 45.2, …}, {"job": "t4", …}],
           "estimate_s": 90.4}
          choices    : l'alternative prise dans chaque groupe, dans l'ordre (ce qu'une grille étiquette).
          estimate_s : la somme, null si un travail n'a pas encore d'estimation.
          Avec --wait : une ligne "type": "added" par travail, puis les lignes de « wait » pour tous.

        ÉCHECS : usage, prompt, model, solver, detail, accelerators, lora, format, steps, edit, ref, img2img
        (--image sur un modèle qui ne part pas d'une image, ou une force qui ne laisse aucun pas), subject (un sujet nommé que le prompt cite,
        help subjects), queue_full (64 en attente, ou pas assez de place pour toutes les
        alternatives), busy (une installation tourne : pas de rendu avant qu'elle soit finie), write.

        Temps indicatifs (MacBook Pro M1 Pro 16 Go, Z-Image, 8 pas) : ~34 s à 512², 113–119 s à 1024².
        """,

        "grid": """
        silicontrol grid --x AXE [--y AXE] [--no-sheet] [options] "<prompt>" [512|1024|WxH|P:Q@1K|2K] [seed]

        Une grille XY : un ou deux réglages balayés, UN TRAVAIL PAR CASE dans la file — chacun un
        travail ordinaire, son image dans l'historique comme les autres —, puis, quand la dernière
        case finit, une PLANCHE : une image avec toutes les cases côte à côte, les valeurs des axes
        en étiquettes et le prompt en titre, ajoutée à l'historique (i…). Rien n'est écrit sur
        disque sans --out ou save.
        Options et arguments sont ceux de « add » (help add), sauf --batch : une case est une image.
        Les séries de la fenêtre (« Bracketer… » sur une molette, un groupe {…} dans le prompt, le lot) sont aussi des grilles,
        dessinées sur son canevas : les mêmes cases, les mêmes images, la planche seulement sur « Ajouter la planche à
        l'historique » (« sheet g2 ») : --no-sheet rend ainsi une grille, son verdict avec "sheet": null. La fenêtre prend les
        axes de la grille pour sa série ; un axe de modèles, de pas, de graines listées ou d'images reste à silicontrol.
        « reuse » répond la série du rack ("series" : son x et son y comme --x et --y les lisent, "excess" au-delà de deux,
        "renders" les travaux que « Générer » mettrait en file).

        AXES — sorte[:lequel][=valeurs] ; valeurs séparées par des virgules, décimales avec un point
          prompt              le groupe {…} du prompt     prompt:2      son deuxième groupe
          seed=42,7,1000      ces graines                 seeds=4       4 graines depuis la graine : s, s+1, …
          steps=4,8,12        ces nombres de pas (chacun doit avoir un calendrier pour le modèle)
          lora=0.4,0.7,1      forces de la première --lora
          lora:NOM=0.4,1      forces de la --lora NOM (ou lora:2=… pour la deuxième)
          model=z-image,qwen-image-2.1   ces modèles, chacun à ses propres pas (pas de --lora : une
                              LoRA est faite pour un seul modèle)
          image               chaque image --ref seule, avec la même instruction (une édition)
          loras=none,flat,ink une LoRA ajoutée à la pile par case (none : la case sans), à la
          loras:0.6=…         force 1 — ou celle donnée ; noms comme --lora les prend
          solver=euler,dpmpp-2m   ces solveurs (euler, euler-ancestral, dpmpp-2m, unipc), même graine
                              et mêmes pas ; Plus de solveurs, comme --solver
          start=0.45,0.6,0.75 forces de l'image de départ (--image), de 0 (gardée) à 1 (ignorée)
          detail=normal,more,most   ces niveaux de Détail, là où le modèle l'offre
          --x donne les colonnes, --y les rangées (sans --y : une rangée). X et Y font varier deux
          réglages différents. TOUT GROUPE {…} DU PROMPT DOIT ÊTRE UN AXE : un groupe oublié
          empilerait plusieurs images dans une case (erreur "grid") ; écrire \\{ \\} pour des
          accolades qui ne sont pas des alternatives. Les cases se rendent rangée par rangée, X
          variant le plus vite. 64 cases au plus, et la file doit avoir la place pour toutes.

        LA PLANCHE
          Chaque case est réduite à 512 px sur son grand côté au plus — moins au-delà de 16 cases :
          toutes les cases ensemble tiennent dans 12 mégapixels (432 px pour 8×8 en 512²). Une case
          qui n'a pas été rendue (erreur, arrêt, retrait) est dessinée vide, avec la raison ; aucune
          case rendue, pas de planche. Les images des cases restent en taille réelle dans l'historique.
          Nom de fichier : grid-<modèle>-<colonnes>x<rangées>-<W>x<H>[-<graine>].png — la graine
          de base, omise quand les graines sont listées (seed=…).

        RÉPONSE
          {"ok": true, "grid": "g2", "columns": 2, "rows": 2, "estimate_s": 128.4,
           "jobs": [{"job": "t5", "grid": "g2", "cell": "g2c0", "column": 0, "row": 0, "prompt": "…",
                     "seeds": [42], "position": 1, "estimate_s": 32.1, …}, …]}
          estimate_s : la somme des cases, null tant que le modèle n'a pas rendu à ce format.
          Avec --wait : une ligne "type": "added" par case, puis les lignes de « wait g2 » :
            {"type": "image", "image": "i7", "job": "t5", …, "path": …}   chaque case, au fil de l'eau
            {"type": "sheet", "image": "i9", "grid": "g2", "sheet": true, "columns": 2, "rows": 2,
             "empty_cells": 0, "seconds": 130.2, …, "path": …}            la planche (seconds : la somme des cases)
            {"ok": true, "grid": "g2", "sheet": "i9", "jobs": [{"job": "t5", "outcome": "done", …}, …]}
          Une case pas "done" : "ok": false, "error": "render" — la planche est faite quand même,
          cette case vide ("sheet": null si aucune case n'a été rendue).
          « wait g2 [--out D] » suit une grille plus tard : tout de suite si elle est finie.
          Une case a son propre nom, g2c3 (son rang dans l'ordre de lecture de la planche, depuis 0 — ligne × colonnes + colonne), et
          celui de son job, t5. « sheet g2 » est « Ajouter la planche à l'historique » sur la série montrée (help sheet).

        ÉCHECS : usage (pas de --x, --batch, --x donné à « add »), grid (un axe illisible ou qui ne
        fait pas une grille, plus de 64 cases), prompt, model, solver, detail, accelerators, lora,
        format, steps, edit, ref, subject (une grille ne cite pas de sujets nommés), queue_full, busy (une
        installation tourne), write.
        """,

        "wait": """
        silicontrol wait [t3 t4 …] [--out D]
        silicontrol wait g2 [--out D]

        Bloque jusqu'à ce que les travaux aient fini — ou, pour une grille (g2, « grid »), jusqu'à sa
        planche : les lignes de ses cases, puis une ligne "type": "sheet", puis le verdict avec "grid"
        et "sheet" (help grid). Sans travail nommé : tous ceux qui tournent
        ou attendent au moment de l'appel — y compris ceux lancés par d'autres (la souris) ; nommer
        ses travaux (t3 t4) n'attend que les siens. Un travail déjà fini répond tout de suite, même
        longtemps après. Ctrl-C n'arrête que l'attente : les rendus continuent dans l'app.

        LIGNES, au fil de l'eau — une par image :
          {"type": "image", "image": "i7", "job": "t3", "model": "z-image", "prompt": "…",
           "width": 832, "height": 1216, "seed": 42, "steps": 8, "evaluations": 8,
           "seconds": 45.86, "loras": [], "edit": false, "references": 0, "path": "/abs/D/….png"}
          "path" n'existe qu'avec --out ; "write_error" s'il n'a pas pu être écrit.
          Sans --out, l'image n'est QUE dans la mémoire de l'app (historique, perdu quand elle
          quitte) : « save i7 DEST » l'écrit plus tard.

        NOMS DES FICHIERS : <model>[-lora-<nom>[@<force>]…][-p<pas>][-edit]-<W>x<H>-<seed>[-v<graine>x<force>…].png
          (la force si elle n'est pas 1, les pas s'ils ne sont pas ceux du modèle ; une variation,
          chaque graine de variation et sa force en centièmes). Un fichier présent n'est jamais écrasé : -2, -3…
          sont ajoutés.

        VERDICT :
          {"ok": true, "jobs": [{"job": "t3", "outcome": "done", "images": ["i7"]}]}
          outcome : done · stopped (arrêté en plein calcul ; les images déjà faites restent)
                  · removed (retiré de la file avant de tourner) · error (avec "message", la
                  phrase du moteur dans la langue de l'app, et "code", sa clé stable :
                  "license_not_accepted", "insufficient_memory", "disk_full"…). Une image dont
                  les pixels sont sortis inutilisables (noire ou vide) n'est jamais "done" :
                  "error", code "non_finite_image".
          "warnings" : les lignes du moteur pendant le rendu du travail (en anglais), s'il en a écrit.
          Si un travail n'est pas "done" : "ok": false, "error": "render", sortie 1.

        ÉCHECS : usage, unknown_job, write.
        """,

        "status": """
        silicontrol status

        Ce que fait l'app maintenant. Ne bloque pas.

        Réponse :
          {"ok": true,
           "running": {"job": "t3", "stage": "denoising", "batch_image": 1, "step": 4,
                       "steps_total": 8, "fraction": 0.52, "remaining_s": 21.3, "model": "z-image",
                       "prompt": "…", "seeds": [42], …} | null,
           "queue": [{"job": "t4", "estimate_s": 45.2, …}, …],
           "waiting_for_license": [{"job": "t5", "state": "waiting_for_license",
                                    "license": "Qwen Research (non-commercial)", "license_urls": […], "model": …}, …],
           "installing": {"title": "Installing Z-Image Turbo", "last": "forging the DiT → …"} | null,
           "queue_remaining_s": 66.5, "remaining_complete": true,
           "history_images": 12, "library": "/Users/…/Siliconed", "faster_rendering": true,
           "accelerators": true, "beta": {"more_solvers": false, "faster_rendering": true, …},
           "writer": {"state": "ready", "installing": false, "failure": null, "bytes": 17…},
           "settings": {"read": true, "map": "2026-10-26T14:43:50Z", "warnings": []}}

        faster_rendering   : le rendu plus rapide du modèle du rack (help set).
        accelerators       : le choix des « Accélérateurs » du modèle du rack (help set) ; seulement sur un Mac dont la mesure en offre un.
        beta               : les cinq options Bêta des Réglages (help set) : more_solvers, steps, faster_rendering, size_scales, negative_lora.
        writer             : le rédacteur de l'Atelier (help rewrite) : "state" vaut "ready", ou le code dont rewrite refuserait
                             (writer_absent, writer_head avec "download_bytes", writer_memory avec "gigabytes", writer_preparing) ;
                             "installing" tant que la tête arrive (help install) ; "failure" la phrase d'une installation qui a échoué.
        settings           : les réglages du moteur des rendus — read : false tant que la mesure de ce Mac ne l'est
                             pas ; map : la date de cette mesure (null : les défauts) ; warnings : pourquoi une mesure n'a
                             pas été prise (celle d'un autre Mac ou d'un autre build). Une mesure s'applique telle quelle, quoi qu'elle ait trouvé (voir
                             help diagnose). Deux séries de deux lancements comparent aussi leurs mesures.
        stage              : text, image (encodage d'une référence), denoising, decoding.
        steps_total        : les évaluations du modèle, pas les pas demandés.
        remaining_s        : null tant que l'app n'a pas de mesure pour ce rendu.
        remaining_complete : false si un travail en attente n'a pas d'estimation (queue_remaining_s est un minimum).
        installing         : une installation ou un import en cours, et la dernière ligne de son
                             journal ; les rendus attendent sa fin (add, grid et vary répondent "busy").
        waiting_for_license: les travaux que le moteur a refusés parce que la licence de leur modèle
                             n'est pas acceptée (help models, "license_accepted"). Ils attendent hors
                             de la file la réponse de l'utilisateur dans la feuille de l'app : acceptée,
                             ils reviennent en tête de file ; refusée, ils finissent en "error",
                             "code": "license_not_accepted". « wait » les attend, « cancel t5 » les
                             retire. Rien ne peut accepter depuis le terminal : l'utilisateur lit la
                             licence dans l'app.
        """,

        "follow": """
        silicontrol follow

        Les évènements de l'app en direct, jusqu'à ce que la file soit vide (tout de suite si rien
        ne tourne : {"ok": true, "message": "nothing is running or waiting"}). Pour regarder ; pour
        récupérer des images, « wait ».

        LIGNES :
          {"type": "start", "job": "t3", …réglages…}
          {"type": "stage", "job": "t3", "stage": "denoising", "batch_image": 1}
          {"type": "step", "job": "t3", "batch_image": 1, "step": 4, "steps_total": 8, "seconds": 5.8}
          {"type": "image", "image": "i7", …}           (comme dans « wait », sans "path")
          {"type": "end", "job": "t3", "outcome": "done" | "stopped" | "removed" | "error"}   (+ "warnings", comme dans « wait »)
          {"type": "sheet", "image": "i9", "grid": "g2", …}   (la dernière case d'une grille a fini ; "image": null sans planche)
        VERDICT : {"ok": true, "message": "queue empty"}
        Les textes "message" et "hint" suivent la langue de l'app ; ceux montrés ici sont les anglais.
        """,

        "cancel": """
        silicontrol cancel [t3 t4 …]

        Sans argument : arrête le rendu en cours. Avec des travaux : arrête celui qui tourne,
        retire de la file ceux qui attendent. L'arrêt prend effet en moins d'une demi-seconde ;
        les images déjà faites d'un lot restent dans l'historique.

        Réponse : {"ok": true, "stopped": "t3" | null, "removed": ["t4"], "already_finished": []}
        ÉCHECS  : usage, unknown_job.
        """,

        "clear": """
        silicontrol clear

        Vide la file ET arrête le rendu en cours — comme « Tout arrêter » dans l'app.

        Réponse : {"ok": true, "stopped": "t3" | null, "removed": ["t4", "t5"]}
        """,

        "set": """
        silicontrol set faster-rendering on|off [--model ID]
        silicontrol set accelerators on|off [--model ID]
        silicontrol set beta [OPTION on|off]
        silicontrol set save-images [FOLDER|off]

        Les interrupteurs des Réglages, depuis le terminal. Chacun passe par la fonction que son interrupteur appelle : le
        choix est écrit et la fenêtre suit, exactement comme un clic.

        FASTER-RENDERING bascule le « rendu plus rapide » d'un modèle. Il se choisit par version installée, comme les
        accélérateurs, et éteint par défaut. C'est une option Bêta (Réglages ▸ Avancé ▸ Bêta) :
        « on » est refusé (« faster_rendering ») tant qu'elle est éteinte, et aucun rendu n'est alors
        plus rapide. Allumé, dès 1024 px de chaque côté les premières évaluations se font sur une grille de
        demi-taille : plus vite, une autre image qu'en pleine taille. Éteint : pleine taille, l'image
        contrôlée contre la référence.

        ACCELERATORS bascule les « Accélérateurs » d'un modèle (l'interrupteur de la boîte DiT) : une part du DiT sur le
        Neural Engine (fp16) et sur les unités tensorielles du GPU (bf16, fp16), là où la carte de ce Mac les choisit —
        moins de temps ou d'énergie, une image un peu différente, contrôlée contre la référence en bf16 ; éteint, le DiT en
        fp32 de bout en bout, contrôlé contre la référence en fp32. Aussi par version installée ; faute de choix, allumé,
        sur toutes les versions. « neural-engine », son ancien nom, marche encore. Refusé (« accelerators »), allumé ou
        éteint, sur un Mac dont la mesure ne donne rien à faire aux accélérateurs : la fenêtre y grise l'interrupteur avec
        cette raison, et il n'y a rien à choisir. Pour un seul rendu : add --accelerators / --no-accelerators.

        Les deux : sans --model, le modèle du rack. Seulement quand rien ne tourne ni n'attend (un travail planifié sous
        une valeur et fini sous l'autre mélangerait deux recettes) : sinon « busy ». « models » donne les valeurs de chaque
        modèle, « status » celles du rack.

        BETA, ce sont les cinq interrupteurs de Réglages ▸ Avancé ▸ Bêta, éteints par défaut :
          more-solvers      les solveurs Euler a, DPM++ 2M et UniPC (--solver de « add »)
          steps             Choisir les pas : le nombre de pas dans la boîte DiT, pour les LoRA distillées en moins de
                            pas (--steps de « add » prend tout nombre que le modèle sait faire, allumé ou non)
          faster-rendering  l'interrupteur du rendu plus rapide dans la boîte DiT — le maître des choix par modèle ci-dessus
          size-scales       Taille en deux échelles : le menu Taille en proportion, puis 1K ou 2K (3:4@2K, help add)
          negative-lora     forces de LoRA de −1,5 à 1,5, pour les LoRA « curseur » (--lora NOM:-0.5)
        « set beta » seul les lit ; « set beta OPTION on|off » en règle une (le nom du réglage marche aussi :
        size-in-two-scales, negative-lora-strengths, choose-steps). Les effets sont ceux de l'interrupteur : éteindre More
        Solvers remet un solveur sur Euler ; éteindre Choose Steps remet les pas du rack à la recette du modèle ; éteindre
        Negative LoRA Strengths ramène à 0 les forces déjà réglées. Comme les interrupteurs, elles
        n'attendent rien et valent pour ce qui se met en file ensuite. Ce sont les préférences de ce Mac ; quand la fenêtre
        calcule sur un autre, sa copie est réglée aussi. « status » les donne sous "beta".

        SAVE-IMAGES, c'est « Enregistrer chaque image dans… » de Réglages ▸ Général, éteint par défaut : avec un dossier
        (relatif au répertoire courant ; un dossier nommé « off » s'écrit ./off), chaque image nouvelle de l'historique — un
        rendu, un recadrage, un agrandissement, la planche d'une grille — s'y écrit
        aussi en PNG avec son prompt, sa graine et son modèle, sous son nom de « save », jamais sur un autre fichier. « off »
        l'éteint ; seul, il est lu. Le dossier doit exister (sinon « write ») : il n'est jamais créé. Préférence de ce Mac,
        quel que soit le Mac qui calcule.

        Réponse : {"ok": true, "model": "krea2", "faster_rendering": false, "previous": true}
                  {"ok": true, "model": "z-image", "accelerators": true, "previous": false}
                  {"ok": true, "option": "size-scales", "enabled": true, "previous": false,
                   "beta": {"more_solvers": false, "faster_rendering": false, "size_scales": true, "negative_lora": false}}
                  {"ok": true, "save_images": "~/Pictures/Siliconed", "previous": null}
        ÉCHECS : usage (aussi : une option Bêta inconnue, "hint" les liste), model, busy, faster_rendering (« on » sans son
        option Bêta), accelerators (pas d'accélérateur à choisir sur ce Mac), write (save-images : pas un dossier).
        """,

        "history": """
        silicontrol history [N]

        Les images de la session, la plus récente d'abord (les N premières si N est donné). Même
        objet que les lignes "image" de « wait », sans "path" : l'historique est en mémoire.

        Réponse : {"ok": true, "images": [{"image": "i7", "job": "t3", "seed": 42, …}, …]}
        ÉCHECS  : usage (N pas un entier ≥ 1).
        """,

        "vary": """
        silicontrol vary i7 subtle|strong [--batch N]

        « Variations » sur une image de l'historique, comme son menu dans l'app : N tâches (1 à 8 ;
        par défaut le champ « Images » du rack, 1 sauf changement) avec exactement ses réglages et sa
        graine, chacune avec une graine de variation tirée au hasard.
        Le bruit de départ est tourné un peu vers celui de la graine de variation : subtle garde la
        scène et la pose, strong garde l'idée et bouge la composition. Une variation d'une variation
        varie CETTE image. Le "variation" de chaque image donne ses graines de variation dans l'ordre ;
        son PNG la refait.

        Réponse : {"ok": true, "jobs": [{"job": "t5", "seeds": [42], "variation": [{"seed": 81…, "strength": 0.1}], …}, …]}
        Puis : silicontrol wait t5
        ÉCHECS : usage, unknown_image, model, queue_full, busy (une installation tourne).
        """,

        "enlarge": """
        silicontrol enlarge i7 [--download]

        « Agrandir ×4 » sur une image de l'historique, comme dans l'app : la super-résolution de macOS
        (VideoToolbox, macOS 26), qui ne voit que l'image — ni prompt, ni débruitage. 1024×1024 devient
        4096×4096 en une ou deux secondes ; l'image rejoint l'historique à côté de son original, qui
        reste. Son PNG garde les métadonnées de l'original, plus "enlarged". Elle ne passe pas par la
        file : refusée pendant un rendu, elle répond une fois faite (pas de wait).
        Le modèle est celui d'Apple, gardé par macOS : absent, la commande refuse (download_required)
        sauf si --download laisse macOS le chercher d'abord. Une image déjà agrandie, ou plus large que
        1920, est refusée.

        Réponse : {"ok": true, "image": {"image": "i8", "width": 4096, "height": 4096, "enlarged": 4, …}}
        Puis : silicontrol save i8 grande.png
        ÉCHECS : usage, unknown_image, refused (macOS avant 26, une grille, déjà agrandie, un rendu en
          cours), download_required, download_failed, internal.
        """,

        "rewrite": """
        silicontrol rewrite "<texte>" [--for ID]

        « Réécrire » de l'Atelier de prompts (Présentation ▸ Afficher l'Atelier de prompts), depuis le terminal : un modèle de langage installé
        avec l'app (Qwen3-VL-8B, l'encodeur de texte de Qwen-Image-2.1 mis à écrire) réécrit un prompt comme le modèle visé le lit
        le mieux. Le rédacteur, ses consignes pour chaque modèle, la protection des @noms et les refus sont ceux de l'Atelier :
        une commande et un clic donnent le même texte. Rien n'est écrit sur disque — le texte ne revient que par la socket
        (incognito). Le texte est un seul argument, entre guillemets, comme le prompt de « add ».

          silicontrol rewrite "woman posing in a library"
          silicontrol rewrite --for anima "1girl, library, solo"
          silicontrol rewrite --for qwen-image-2.1 "@alice reading in a library"

        --for ID   le modèle pour lequel écrire : un "id" de « models » (ou sa famille) parmi les modèles installés. Sans lui,
                   ce que prend l'Atelier : le modèle choisi dans sa fenêtre, sinon celui du rack, sinon le premier installé.
        @noms      là où le modèle visé lit des images (Qwen-Image-2.1), chaque @nom est mis de côté puis remis après le
                   rédacteur : un nom qu'il a perdu ou déformé est dit ("lost_mentions", "notice"), jamais supprimé en silence.
                   Ailleurs, @ est celui du modèle (les artistes d'Anima) et le texte passe tel quel.

        LIGNES — le premier texte prend quelques secondes, les poids se lisent ; ils repartent à la fin :
          {"type": "started", "gesture": "rewrite", "for": "qwen-image-2.1"}
          {"type": "loading", "fraction": 0.4}        les poids en lecture (null : inconnu) ; une fois
          {"type": "text", "text": "A woman…"}        TOUT ce qui est écrit jusque-là — cumulatif, pas un delta : le rédacteur peut réviser son dernier mot
        VERDICT :
          {"ok": true, "gesture": "rewrite", "for": "qwen-image-2.1", "text": "…", "tokens": 112, "lost_mentions": []}
          avec "notice" (une phrase) quand une mention a été perdue ou déformée — le texte est donné quand même, comme le fait l'Atelier.

        Un geste à la fois : le rédacteur est aussi celui de l'Atelier. Une image qui se rend est prioritaire, comme dans
        l'Atelier : un rendu qui démarre met fin au geste ("stopped"). Un client qui raccroche l'annule : l'échantillonnage
        s'arrête, les poids repartent.
        ÉCHECS : usage, model (--for nomme aucun des modèles installés), busy (une image se rend ou attend), remote (la fenêtre
        calcule sur un autre Mac : le rédacteur est celui de ce Mac), writer_absent (aucun modèle installé n'a de rédacteur),
        writer_head (sa tête est à télécharger : install writer), writer_memory (ce Mac est trop petit), writer_preparing,
        writer_busy (un autre Réécrire ou Décrire est en cours, du terminal ou de la fenêtre de l'Atelier), stopped,
        prompt_too_long, writer_failed (il a échoué ou n'a rien écrit ; "message" dit pourquoi), et les codes du moteur
        (render_already_running : un autre processus rend).
        """,

        "describe": """
        silicontrol describe <image|i7> [--for ID]

        « Décrire » de l'Atelier de prompts, depuis le terminal : le rédacteur regarde une image et écrit le prompt qui la
        ferait, à la manière du modèle visé. IMAGE est une image de l'historique (i7, comme « history » les nomme) ou un
        fichier — tout ce que macOS lit, son orientation appliquée, sa transparence posée sur du blanc, comme l'Atelier lit
        une image déposée. Même rédacteur, mêmes consignes, mêmes refus et mêmes lignes que « rewrite » (help rewrite) ;
        le verdict a "image" (le nom de l'historique, ou le chemin) là où celui de « rewrite » a "lost_mentions".

          silicontrol describe i7
          silicontrol describe photo.jpg --for anima

          {"ok": true, "gesture": "describe", "for": "z-image", "image": "i7", "text": "A woman…", "tokens": 138}

        --for ID   comme dans « rewrite ».
        ÉCHECS : ceux de « rewrite » ; ref (le fichier est illisible ou absent, ou n'est pas une image) ; unknown_image (un i… qui
        n'est pas dans l'historique).
        """,

        "instructions": """
        silicontrol instructions [--for ID]
        silicontrol instructions rewrite|describe [--for ID] [--set "<texte>" | --file CHEMIN | --restore]

        Réglages › Atelier de prompts, depuis le terminal : la consigne envoyée au rédacteur avant chaque Réécrire et chaque
        Décrire, une par modèle et par geste — celle de Siliconed, ou la vôtre à sa place. Le texte entier, en anglais, tel qu'il
        part. Ce que vous ne touchez pas suit celle de Siliconed d'une version à l'autre ; un texte égal à celui de Siliconed
        n'est pas gardé comme le vôtre.

          silicontrol instructions                                  les dix : à qui chacune
          silicontrol instructions rewrite --for krea2              une : son texte
          silicontrol instructions rewrite --for krea2 --file la-mienne.txt
          silicontrol instructions describe --for anima --set "Write danbooru tags only."
          silicontrol instructions rewrite --for krea2 --restore    celle de Siliconed revient

        --for ID      l'id d'un modèle ou une famille, installé ou non. Sans lui, le modèle pour lequel écrit l'Atelier.
        --set TEXTE   la vôtre, un seul argument entre guillemets ; --file CHEMIN la lit dans un fichier UTF-8. Vide, elle
                      est gardée telle quelle, et c'est celle de Siliconed qui part.
        --restore     celle de Siliconed, comme « Rétablir » dans les Réglages.

        Réponse, une consigne :
          {"ok": true, "for": "krea2", "gesture": "rewrite", "edited": true, "sent": "yours", "text": "…", "instruction": "…"}
          "text" est ce que montrent les Réglages, "instruction" ce que reçoit le rédacteur. La liste : {"ok": true,
          "instructions": [{"for": "z-image", "gesture": "rewrite", "edited": false, "sent": "ours"}, …]}.
        Permis quand la fenêtre calcule sur un autre Mac : les consignes sont celles de ce Mac, comme son rédacteur.
        Pour en voir une dans Réglages › Atelier de prompts : settings lab rewrite --for krea2 (help settings).
        ÉCHECS : usage (un changement nomme son geste), model (--for ne nomme aucun modèle), unreadable (--file).
        """,

        "settings": """
        silicontrol settings [general|macs|lab|advanced]
        silicontrol settings lab [rewrite|describe] [--for ID]

        Siliconed › Réglages… (⌘,), ouvert sur l'un de ses quatre volets : general (Général : où est la bibliothèque,
        Enregistrer chaque image dans…), macs (Mac : ce Mac, sa mesure, les autres Mac), lab (Atelier de prompts : les
        consignes du rédacteur), advanced (Avancé : Ligne de commande, Bêta). Sans volet, le dernier montré. La fenêtre
        passe devant comme le menu l'amène ; rien n'y change.

          silicontrol settings macs
          silicontrol settings lab rewrite --for krea2      l'Atelier de prompts sur la consigne Réécrire de Krea 2
          silicontrol settings lab describe                 le même modèle, son Décrire

        lab avec un geste ou --for : l'Atelier de prompts sur une consigne, ses deux menus posés comme les pose un clic sur
        « Votre consigne » dans l'Atelier. --for ID est l'id d'un modèle ou une famille, installé ou non ; sans lui, le modèle
        pour lequel écrit l'Atelier. Un menu qu'on ne nomme pas reste comme il est.

        Réponse : {"ok": true, "settings": {"pane": "lab", "for": "krea2", "gesture": "rewrite"}}   ("for", "gesture" : lab seulement)
        Ce que règlent les volets, des commandes le règlent : set (beta, save-images), library, macs, compute, instructions.
        Permis quand la fenêtre calcule sur un autre Mac : les Réglages sont ceux de ce Mac.
        ÉCHECS : usage (un volet qui ne se lit pas ; un geste ou --for sur un autre volet que lab), model (--for ne nomme
        aucun modèle).
        """,

        "bracket": """
        silicontrol bracket seed|detail|solver|start|loras
        silicontrol bracket lora <nom>
        silicontrol bracket close

        Un clic sur le ± jaune d'un réglage (ou clic droit › « Bracketer… ») : sa bulle s'ouvre à côté, avec les valeurs que
        prendrait le réglage (trois autour de l'actuelle pour commencer) — la bulle de la fenêtre elle-même, à lire ou à
        capturer ; rien ne varie avant qu'on la valide. seed : le ± de la ligne Graine (le lot) ; start : la force de l'image de
        départ ; loras : « Comparer des LoRA… » de la pile ; lora <nom> : la force d'une LoRA, par son nom dans la pile tel que
        l'écrit "loras" (lab, status). close : la bulle fermée, les valeurs comme avant. Pour faire une série depuis le terminal :
        grid (ses axes deviennent la série de la fenêtre, help grid).

          silicontrol bracket detail
          silicontrol bracket lora incase
          silicontrol bracket close

        Réponse : {"ok": true, "bracketing": "detail"}   ("bracketing" : null après close)
        ÉCHECS : usage, lora (aucune LoRA de ce nom dans la pile : les noms qu'elle tient sont listés), bracket (aucun ± qui
        s'ouvrirait sous le modèle et les réglages du moment : Détail sur un modèle qui n'en a pas, pas d'image de départ, rien
        à comparer, une LoRA éteinte).
        """,

        "rack": """
        silicontrol rack top|dit

        La colonne du rack défilée comme la main la fait défiler : top, jusqu'à sa tête (les ports et les encodeurs) ; dit,
        jusqu'à la boîte du DiT et ses réglages — les encodeurs alors hors de vue, la chaîne épinglée en petit en haut de la
        colonne, l'étage qui travaille allumé, un clic dessus ramenant en haut. Rien ne change dans le rack.

        Réponse : {"ok": true, "rack": "dit"}
        ÉCHECS : usage (une place qui ne se lit pas).
        """,

        "lab": """
        silicontrol lab [open|close|detach|attach]
        silicontrol lab add <image|i7> …
        silicontrol lab role <n> edited|reference|start|none
        silicontrol lab remove <n>
        silicontrol lab rewrite [--wait]
        silicontrol lab describe <n> [--wait]

        L'ATELIER — la barre sous l'image, où se préparent le texte et les images du prochain rendu : ce que
        font un dépôt sur le canevas, ⌘V, « Envoyer à l'atelier » et les menus de rôle de ses images. Son
        texte est le prompt du rack (add et reuse l'écrivent) ; ses images sont numérotées comme la
        barre les montre, à partir de 1.

          lab                          ce qu'il tient : replié ou non, le texte, les images et leurs rôles
          lab open · lab close         déplie ou replie la barre (⌥⌘L) ; replier arrête un Réécrire ou un
                                       Décrire en cours et rend la mémoire du rédacteur
          lab detach · lab attach      dépliée, la barre dans une fenêtre à elle, redimensionnable — ou de nouveau sur
                                       l'image ; repliée, c'est la ligne sous l'image dans les deux cas ("detached")
          lab add IMAGE|i7 …           des fichiers, ou des images de l'historique : chacune prend le rôle
                                       d'une arrivante — la place suivante sous Qwen-Image-2.1 (éditée, puis
                                       références, trois au plus avec les sujets que cite le texte), aucun
                                       ailleurs
          lab role N RÔLE              edited (à éditer) · reference (Qwen-Image-2.1) · start (Z-Image,
                                       Anima, Krea 2, MOSS-Image : le rendu part d'elle, la taille prend le
                                       format du modèle le plus proche de ses proportions) · none (aucun).
                                       Une image à éditer et un départ : en choisir une autre envoie la
                                       précédente image à éditer aux références, le départ précédent à aucun.
          lab remove N                 retire l'image N
          lab rewrite                  « Réécrire » de la barre : son texte réécrit pour le modèle pour lequel elle écrit
                                       (celui du rack), DANS LE TEXTE, jeton par jeton avec ses marques (ajouts, retraits
                                       barrés) — ⌘Z ramène le texte. « rewrite » répond par un texte et laisse la barre.
          lab describe N               « Décrire » de l'image N : ce qu'elle montre, écrit dans le texte au curseur (à la
                                       fin quand le champ n'a pas le curseur), ou comme tout le texte quand il est vide
          --wait                       la réponse une fois le texte écrit, après les lignes de « rewrite » (help rewrite) :
                                       started, loading, text — "text" étant tout le texte du champ tel qu'il se lira

        Les deux déplient d'abord la barre (Réécrire et Décrire sont du plateau déplié), puis pressent le bouton : le
        rédacteur, ses consignes et les refus sont ceux de « rewrite ». Sans --wait, ils répondent dès que le rédacteur part,
          {"ok": true, "gesture": "describe", "for": "z-image", "image": 2, "writing": true, "lab": {…}}
        et « lab » dit où il en est ("writing" : {"gesture": "describe", "phase": "loading"|"writing", "image": 2}, null une
        fois écrit). Avec --wait : {"ok": true, "gesture": "rewrite", "for": "z-image", "text": "<le texte du champ>",
        "lab": {…}}, et "notice" quand une mention s'est perdue. Le geste est celui de la fenêtre : un client qui raccroche
        le laisse écrire ; Arrêter dans la barre ou « lab close » l'arrête, un rendu qui démarre aussi ("stopped").

        UN CHANGEMENT DE MODÈLE ne donne jamais de rôle et n'en change aucun en un autre : une référence
        garde son rôle sous Z-Image, sans effet ("used": false), et resert de retour sous Qwen-Image-2.1.
        "unused" compte ce que le prochain rendu ne lira pas, comme la ligne à côté de Générer.

        RÉPONSE
          {"ok": true, "lab": {"unfolded": false, "detached": false, "model": "qwen-image-2.1", "text": "@alice on the quay",
            "images": [{"n": 1, "name": "quai.jpg", "width": 1536, "height": 1024, "role": "edited",
                        "used": true}], "unused": 0, "writing": null}}
        ÉCHECS : usage, image (pas d'image N), role (un rôle que le modèle du rack n'a pas), ref (l'image N ne se lit
        pas) ; lab rewrite et lab describe : ceux de « rewrite » (remote, busy, writer_absent, writer_head, writer_memory,
        writer_preparing, writer_busy), empty_text (lab rewrite sur un texte vide) ; avec --wait, stopped, writer_failed,
        prompt_too_long.
        """,

        "start-image": """
        silicontrol start-image [<image|i7>] [--strength S]
        silicontrol start-image --remove

        L'IMAGE DE DÉPART (img2img) — une image de l'Atelier à qui on donne le rôle de départ (« Partir de
        cette image », ou « lab role N start »), et sa force, un réglage du rack. Seulement pour un modèle
        qui part d'une image ("start_image": true dans « models » : Z-Image, Anima, Krea 2, MOSS-Image).
        Qwen-Image-2.1 édite par référence à la place (--ref, help edit).

          start-image                  ce que le rack tient
          start-image IMAGE|i7         met une image dans l'Atelier comme départ (un fichier, ou une image de
                                       l'historique) : les rendus suivants partent d'elle. La taille prend le format du
                                       modèle le plus proche de ses proportions, à l'échelle du rack —
                                       et reste libre.
          … --strength S               avec elle, sa force ; seul, retouche celle de l'image portée
          start-image --remove         la retire : les rendus suivants partent du bruit

        LA FORCE est la strength de diffusers : jusqu'où le rendu s'éloigne de l'image. 0.65 ou 65%,
        au-dessus de 0 et au plus 1 : basse, elle garde sa composition et ses couleurs ; 1 l'ignore. Elle
        compte en pas : à 8 pas, 0.65 en redessine 6 (0.63 et 0.74 aussi). Défaut : Krea 2 0.65, Z-Image et
        Anima 0.75, MOSS-Image 0.80. Une force qui redessine tous les pas (à 8 pas, au-delà de 0.875)
        part du bruit seul : l'image est ignorée ("image_ignored": true), le rendu se fait quand même.
        Gardée au centième : le PNG l'écrit ("Denoising strength"), « reuse » la rend — l'image
        elle-même n'est pas dans le fichier.

        RÉPONSE
          {"ok": true, "rack": {"model": "z-image", "width": 1024, "height": 1024, …,
            "start_image": {"name": "photo.png", "width": 3024, "height": 4032, "strength": 0.65,
                            "used": true, "redrawn_steps": 6, "of_steps": 8, "image_ignored": false}}}
          "start_image" : null quand le rack n'en tient pas ; "used": false quand le modèle du rack ne
          part pas d'une image (elle part alors au prochain changement de modèle). « status » dit pareil.

        ÉCHECS : img2img (le modèle du rack ne part pas d'une image), no_start_image (--strength seul,
        sans image portée), ref (image illisible ou absente), unknown_image, usage, remote (la fenêtre
        calcule sur un autre Mac : pas encore offert là).
        """,

        "subjects": """
        SUJETS NOMMÉS — silicontrol subjects [add … | remove NOM]

        La même bibliothèque que celle de l'atelier (Fenêtre → Atelier de prompts). Un sujet est une image (ou deux du même sujet) avec un nom, un genre et
        un cadre ; le prompt le cite par @nom, et ses images vont au modèle comme références, après
        celles des --ref. Un modèle qui lit les images seulement (Qwen-Image-2.1) ; 3 images au plus
        par rendu. En mémoire dans l'app, comme l'historique : les sujets partent quand l'app quitte
        (incognito).

          silicontrol subjects                                     la bibliothèque
          silicontrol subjects add IMAGE --name NOM --kind GENRE [--frame x,y,l,h] [--second IMAGE2]
          silicontrol subjects remove NOM                          l'oublie
          silicontrol add --model qwen-image-2.1 "@alice in @red-coat, posing in a library"

        GENRE : person, garment, object, place ou style — ce que le modèle prend de l'image.
        NOM : lettres, chiffres, - et _ (« red-coat ») ; @Red-Coat le cite aussi (casse ignorée).
        --frame : la partie de l'image à regarder, en fractions (0,0,1,1 = l'image entière).
        Le prompt du modèle (chaque sujet relié à son image) est le "model_prompt" de la tâche.

        Réponse : {"ok": true, "subjects": [{"name": "alice", "kind": "person", "pictures": [{"frame": [0, 0, 1, 1]}]}]}
        ÉCHECS : usage, ref (image illisible ou absente), subject (nom inconnu ou déjà pris, trop d'images, modèle qui n'en lit pas).
        """,

        "remove": """
        silicontrol remove i7 [i8 …]

        « Retirer de l'historique » (⌫), comme le fait le menu : les images quittent l'historique, en mémoire — rien n'est touché sur le disque.
        La fenêtre l'annule par ⌘Z, et « silicontrol undo » aussi. Une image jamais enregistrée est perdue pour de bon quand l'app
        quitte (« save » d'abord).

        Réponse : {"ok": true, "removed": ["i7", "i8"], "history_images": 5,
                   "undo": {"can_undo": true, "action": "Retirer 2 images", "can_redo": false, "redo_action": ""}}
        Sur un Mac suivi (Calculer sur), les images de l'autre Mac sont retirées, et rendues, de la même façon.
        ÉCHECS : usage, unknown_image.
        """,

        "undo": """
        silicontrol undo [--redo]

        Édition ▸ Annuler (⌘Z), ou avec --redo Édition ▸ Rétablir (⇧⌘Z) : ce que le gestionnaire d'annulation de la fenêtre annulerait
        maintenant, rien de plus. Ce qu'il connaît : le retrait d'images de l'historique, un recadrage, un prompt porté au rack depuis
        l'Atelier de prompts. Un retrait d'images sur un Mac suivi s'annule en redemandant les images à ce Mac, tant que la session dure.

        Réponse : {"ok": true, "done": "undo", "action": "Retirer l'image", "history_images": 6,
                   "undo": {"can_undo": false, "action": "", "can_redo": true, "redo_action": "Retirer l'image"}}
        ÉCHECS : usage, nothing_to_undo (rien à annuler, ni à rétablir).
        """,

        "again": """
        silicontrol again <i7|fichier.png>
        silicontrol reuse <i7|fichier.png>

        again est « Rendre à nouveau (même graine) » (⌘R) : le rack prend les réglages de l'image (modèle, prompt, taille, pas, LoRA, images à
        éditer, détail, solveur, et sa graine), puis « Générer » la rend, à cette graine. Sur une planche (l'image d'une grille), le rack
        prend sa série et la rend à nouveau à la graine de la planche. Mêmes réglages, même graine, même modèle : la même image, bit pour bit.
        Avec un PNG que Siliconed a écrit — une image d'un autre jour, un fichier de out/ — c'est un PNG déposé sur le canevas, puis
        « Générer » : le rack lit la recette du fichier (modèle, prompt, taille, pas, LoRA retrouvées dans cette bibliothèque, graine,
        variations), et ce qui n'a pas pu revenir est dans "notes" (un modèle non installé, une LoRA absente, un solveur qui demande
        Plus de solveurs…). Le rack garde ce qu'il a pris, comme après un clic : « silicontrol add » n'en est pas touché, il ne lit
        jamais le rack.
        reuse est « Reprendre ces réglages » : le même remplissage du rack, sans rendu.

        Réponse (again) : {"ok": true, "jobs": [{"job": "t6", "model": "z-image", "seeds": [42], …}]}
        Réponse (reuse) : {"ok": true, "rack": {"model": "z-image", "prompt": "…", "steps": 8, "width": 1024, "height": 1024,
                                                "seed": 42, "loras": [], "series": {"x": null, "y": null, "excess": [], "renders": 1}},
                          "notes": "…" (seulement si quelque chose n'est pas revenu). La série d'une planche revient avec elle (help grid).
        ÉCHECS : usage, unknown_image, ref (un fichier illisible, ou qui n'est pas un PNG écrit par Siliconed), et pour again les raisons pour
          lesquelles « Générer » est grisé : model (non installé), library_absent, busy (une installation tourne), queue_full, link (la
          session avec l'autre Mac est coupée), refused (tout le reste : "message" est la phrase de la fenêtre — un refus mémoire, un
          nombre de pas sans calendrier pour le modèle…). Sur un Mac suivi, chaque job reçoit la réponse de ce Mac.
        """,

        "crop": """
        silicontrol crop i7 --box X,Y,W,H
        silicontrol crop i7 --ratio P:Q

        « Recadrer… » sur une image de l'historique, le cadre donné au lieu d'être tiré. --box est le cadre en pixels de l'image, l'origine en
        haut à gauche (x, y, largeur, hauteur) ; --ratio est le plus grand cadre de cette proportion, centré (4:5, 16:9, 1:1…), d'où part le
        menu des proportions de l'éditeur. Les pixels du cadre deviennent une NOUVELLE image en tête de l'historique ; l'originale reste.
        « silicontrol undo » retire le recadrage, comme ⌘Z. Les règles sont celles de l'éditeur : un côté n'est jamais sous 512 px, l'image
        entière n'est pas un recadrage, la planche d'une grille ne se recadre pas. La nouvelle image garde les
        métadonnées de l'originale, plus "cropped".

        Réponse : {"ok": true, "image": {"image": "i8", "width": 1024, "height": 1024, …, "crop": {"x": 96, "y": 0, "width": 1024,
                   "height": 1024}}, "undo": {…}}
        ÉCHECS : usage, unknown_image, crop (le cadre sort de l'image, un côté sous 512 px, l'image entière, une proportion que l'image ne
          peut pas tenir, une planche), busy (l'éditeur de recadrage est ouvert dans la fenêtre).
        """,

        "sheet": """
        silicontrol sheet <g2>

        « Ajouter la planche à l'historique » sur la série montrée sur le canevas — un « Générer » avec quelque chose de bracketé,
        ou une « grid » — : ses images côte à côte, étiquetées, en une image de l'historique, en mémoire. Chaque appel en ajoute
        une. Une « grid » dessine la sienne quand sa dernière case finit ; une série de la fenêtre seulement sur demande. Réponse
        quand la planche est dessinée.
          {"ok": true, "image": {"image": "i12", "grid": "g2", "sheet": true, "columns": 2, "rows": 2, "empty_cells": 0, …}}
        Les images de la série sont déjà dans l'historique, une à une, comme celles de tout rendu : la planche ne fait que les
        disposer. Une case se nomme gNcM (help grid), et son job tN.
        Seulement la série montrée — la plus récente : une nouvelle prend le canevas.
        ÉCHECS : usage, unknown_job (aucune série montrée, ou ce n'est pas celle-là), cell (aucune case n'a encore d'image :
          l'attendre), busy (des cases sont encore en route : « wait g2 »).
        """,

        "import": """
        silicontrol import FICHIER.safetensors|FICHIER.gguf …

        « Importer un modèle ou une LoRA… » (⌘O), avec les fichiers donnés au lieu d'être choisis — ce que fait leur dépôt sur la fenêtre.
        Un modèle (DiT, tel que publié sur Civitai : bf16, fp16, fp32, 8 bits, ou GGUF de 4 bits au moins) ou une LoRA. Le même moteur, les
        mêmes règles, le même résultat dans la bibliothèque : « models » liste le modèle, avec la version de sa famille, et la LoRA sous les
        modèles qu'elle sert. Le fichier donné peut être supprimé ensuite. La réponse vient quand l'import est fini :
          {"ok": true, "files": ["flat.safetensors"], "models": [], "loras": ["flat"]}
        "models" et "loras" sont ce qui est nouveau dans la bibliothèque. Un fichier de VAE ou d'encodeur ne fait que compléter la famille :
        les deux listes sont vides.
        **Un modèle dont la famille n'est pas installée** : l'app montre la feuille Modèles sur cette famille, avec sa licence et la place
        qu'elle prend, et l'import suit quand l'utilisateur a accepté là — la commande n'accepte jamais une licence, et répond aussitôt :
          {"ok": true, "imported": [], "needs_family": "z-image", "install": "silicontrol install z-image", "message": "…"}
        L'arrêter se fait par « install stop ». Sur un Mac suivi (Calculer sur), les fichiers monteraient sur l'autre Mac, dont le
        propriétaire confirme là : refusé (remote).
        ÉCHECS : usage, ref (fichier illisible ou absent), import (ni .safetensors ni .gguf, ou le moteur a refusé le fichier : "message"
          dit pourquoi, dans les mots de la fenêtre), busy (un rendu, une installation ou le diagnostic tourne), library_absent, remote.
        """,

        "uninstall": """
        silicontrol uninstall ID [--yes]
        silicontrol uninstall lora NOM [--yes]
        silicontrol uninstall downloads [--yes]

        Les retraits de la feuille Modèles. ID est un modèle de « models » : un modèle d'éditeur (z-image…) est « Désinstaller… » — son DiT,
        son encodeur et son VAE partent, ses modèles importés et ses LoRA restent, comme la feuille le dit — ; un modèle importé est sa
        corbeille ; lora NOM est une LoRA ("lora" dans « models ») ; downloads est « Jeter » sur les téléchargements interrompus. Réinstaller
        télécharge de nouveau ; un modèle importé ou une LoRA revient en important son fichier de nouveau.
        **Sans --yes rien n'est retiré** : la réponse dit ce qui partirait et la place libérée.
          {"ok": true, "dry_run": true, "target": "z-image", "kind": "family", "frees_bytes": 11749122176, "stays": ["my-finetune"],
           "message": "nothing was removed: add --yes to remove it"}
        Avec --yes : la même réponse avec "removed": true au lieu de "dry_run".
        Refusé comme la feuille le grise : un rendu en cours ou en attente, une installation ou la mesure en route (busy) ; le disque de la
        bibliothèque absent (library_absent). Sur un Mac suivi, la feuille interroge l'autre Mac et attend un clic sur son écran : refusé ici
        (remote) — le retrait revient à qui est assis devant le Mac dont ce sont les modèles.
        ÉCHECS : usage, model (inconnu, ou rien d'installé, ou aucun téléchargement interrompu), lora, busy, library_absent, remote.
        """,

        "macs": """
        silicontrol macs
        silicontrol macs connect NOM | disconnect NOM
        silicontrol compute NOM|this

        Rendu ▸ Calculer sur, depuis le terminal. macs liste les Mac appairés pour calculer, comme le menu, et leur état ; NOM est le nom d'un
        Mac (casse et accents mis à part) ou son "id". compute NOM fait calculer la fenêtre sur ce Mac — sa file et son historique sont
        montrés, et toutes les commandes suivent (help contract) — et compute this la ramène sur ce Mac. Une file qui tourne ici continue.
        Le basculement est fait comme le menu le fait ; la commande répond quand la session est établie, ou perdue, ou après 15 s ("link" dit
        où elle en est). macs connect et disconnect sont les boutons de Réglages ▸ Autres Mac : la session gardée, ou lâchée.
        Jamais un appairage, un prêt ni un oubli : ceux-là sont acceptés par la personne devant les écrans, dans la fenêtre.

        Réponse (macs) : {"ok": true, "computing_on": null | "Mac mini", "pairings_read": true,
                          "macs": [{"name": "Mac mini", "id": "…", "state": "connected", "computing": false, "machine": "M4, 16 GB…"}],
                          "lending": false, "lent_to": [{"name": "MacBook", "connected": false}]}
          state : connected, connecting, lost (nouvel essai), on_the_network, not_on_the_network, refused_forgotten (ce Mac a oublié celui-ci :
          appairer de nouveau, dans la fenêtre), refused_version (les deux doivent faire tourner la même version de Siliconed).
          "pairings_read" false : l'app n'a pas encore lu le Trousseau (elle le fait au lancement si un Mac a été appairé, jamais sans
          l'utilisateur) : la liste est vide tant que Réglages ▸ Autres Mac n'a pas été ouvert.
        Réponse (compute) : {"ok": true, "computing_on": "Mac mini", "changed": true, "link": "connected", "message": "…" (si pas encore)}
        ÉCHECS : usage, unknown_mac (aucun Mac appairé de ce nom, ou deux : prendre l'"id"), link (ce Mac a refusé la session).
        """,

        "library": """
        silicontrol library
        silicontrol library move DOSSIER [--yes]
        silicontrol library home [--yes]
        silicontrol library open DOSSIER
        silicontrol library local

        Les boutons de bibliothèque de Réglages ▸ Général. Seul : où est la bibliothèque, sa taille et la place qui reste.
        move est « Déplacer la bibliothèque… » : DOSSIER est un dossier d'un autre disque (un dossier « Siliconed » y est créé, sauf si DOSSIER
        l'est déjà ou contient une bibliothèque qu'on déplace). Les fichiers sont copiés et relus un à un, l'ancien lieu est vidé en dernier ;
        les licences acceptées et les réglages de ce Mac restent sur ce Mac. home est « Ramener sur ce Mac ».
        **Sans --yes rien n'est copié** : la réponse dit ce qui le serait et la place qu'il y a.
          {"ok": true, "dry_run": true, "destination": "/Volumes/X/Siliconed", "files": 14, "bytes": 11749122176, "free_bytes": 400000000000,
           "message": "nothing was moved: add --yes to move it"}
        Avec --yes : {"type": "started", …}, puis des lignes {"type": "progress", "percent": 37}, puis le verdict avec "library".
        Un déplacement arrêté dans l'app laisse la bibliothèque où elle était ("stopped").
        open est « Ouvrir une bibliothèque… » (un dossier qui contient déjà une bibliothèque : rien n'est déplacé), local « Utiliser la
        bibliothèque de ce Mac » (rien n'est déplacé).
        Refusés comme Réglages les grise : un rendu, une installation ou la mesure en route (busy), le disque absent (library_absent), et pour
        move et home un Mac suivi (remote : la bibliothèque montrée est celle de l'autre Mac).
        Réponse : {"ok": true, "library": {"path": "…", "elsewhere": true, "reachable": true, "volume": "X", "free_bytes": …, "used_bytes": …,
                   "this_mac": "…"}}
        ÉCHECS : usage, busy, library_absent, remote, library (déjà sur ce Mac ; ou le moteur a refusé la destination — la bibliothèque
          elle-même, un disque plein, un dossier où l'on ne peut pas écrire, les modèles d'une autre bibliothèque : "message" dit laquelle), stopped.
        """,

        "save": """
        silicontrol save i7 [i8 …] DEST [--format png|jpeg|heic]

        Écrit des images de l'historique sur le disque, dans les trois formats des panneaux Enregistrer et Exporter de la fenêtre.
          DEST = un fichier .png, .jpg (.jpeg) ou .heic (une seule image) : écrit, ou REMPLACÉ s'il existe ; l'extension décide.
          DEST = un dossier (créé au besoin) : nommage et non-écrasement de « wait ». --format dit le format de ses fichiers (png par
                 défaut ; le menu du panneau retient le dernier utilisé, la commande non).
        Le PNG porte ses métadonnées (prompt, graine, modèle, LoRA) ; le JPEG (0,9) et le HEIC (0,85) sont des images simples, sans métadonnées.

        Réponse : {"ok": true, "files": [{"image": "i7", "path": "/abs/…"}]}
        ÉCHECS  : usage, unknown_image, write.
        """,

        "edit": """
        L'ÉDITION PAR INSTRUCTION — silicontrol add --ref … "<ce qui change>"

        Un modèle "edit": true dans « models » édite des images : le prompt est une instruction
        (« replace the cloudy sky with a blue sky »), pas une description. Les images ne sont pas
        rebruitées : le modèle les lit, et la nouvelle image part du bruit.

        LES IMAGES, DANS L'ORDRE — un --ref chacune, de 1 à "max_references" (Qwen-Image-2.1 : 3) :
          --ref A --ref B --ref C   →   image 1 = A, image 2 = B, image 3 = C
          l'image 1 est CELLE QU'ON ÉDITE ; les autres sont ce où le prompt puise.
          Un --ref est un fichier (PNG, JPEG, HEIC… ; relatif au répertoire courant) ou une image
          de l'historique (i7 : « éditer la dernière image », c'est --ref i7).

        LE PROMPT — dit ce qui change, et nomme les images par leur numéro :
          "change her jacket to red"
          "remove the second person from image 1 and put the dog of image 2 in her place"
          L'encodeur de Qwen-Image-2.1 voit les images : il les lit comme <image1>, <image2>… devant
          le prompt.

        LA TAILLE DE SORTIE — sans format, la sortie suit l'image 1 :
          Qwen-Image-2.1 : environ 1024² (1 mégapixel) aux proportions de l'image 1, chaque côté
          multiple de 32 et d'au moins 512.
          L'app édite jusqu'à 1024² de SURFACE : une image 1 très allongée voit son grand côté
          raccourci, et un format tapé au-delà de 1024² est refusé ("format").
          "width"/"height" de la réponse la disent. Un format tapé l'emporte.

        LE COÛT — chaque référence allonge le rendu (la séquence grandit de ses jetons).
        L'estimation de l'app ("estimate_s") s'apprend par modèle, format et nombre de références :
        null tant que cette combinaison n'a pas rendu une fois dans la session.

        Ni masque, ni inpainting : le prompt seul dit ce qui change.
        """,

        "measure": """
        silicontrol measure [--model ID] [--size 512|1024|WxH] [--renders N] [--seed S] [--prompt TEXTE]…
                            [--tsv FICHIER] [--out DOSSIER] [--settle SECONDES] [--tag TEXTE] [--faster-rendering on|off] [-- options de add]

        Refait nos temps et nos pics chez vous, avec cette seule app : une série de rendus, chacun chronométré
        de la demande au PNG, le pic mémoire de l'app échantillonné pendant qu'elle rend, les pixels hachés.
        Défauts : z-image, 1024x1024, 3 rendus, graine 42, « a 30 year old woman posing in a library »,
        15 s de repos d'abord. Le rendu plus rapide du modèle mesuré est écrit dans le TSV à chaque
        rendu ; --faster-rendering on|off le pose pour la série (« set faster-rendering --model »,
        comme un clic) et rend la valeur d'avant à la fin, même si la série échoue. Sans lui, la valeur
        du modèle reste : un chiffre publié « éteint » est --faster-rendering off.

        La série : la file doit être libre (sinon « busy ») ; un rendu de chargement (les poids entrent ; son temps
        est donné à part) ; un témoin ; les N rendus ; un témoin encore. Le témoin est le prompt par défaut à la
        graine 42 ; la machine dérive de quelques pour cent en une heure, et les deux témoins bornent cette
        dérive : une série dont le témoin a bougé de plus de 13 % ne se compare pas à une autre. Le Mac reste
        éveillé pendant la série. Pour chaque rendu :
          wall_s       de la demande au PNG, sur l'horloge de cette commande (ce que nous publions)
          engine_s     les « seconds » de l'app pour l'image
          peak_GiB     le phys_footprint de l'app, lu toutes les 100 ms, le maximum (ce que nous publions)
          start_GiB    l'empreinte juste avant la demande : ce que l'app tenait déjà
          swapouts     pages écrites en swap pendant le rendu (Swapouts de vm_stat) ; disk_read_GB les lectures de l'app
          md5          des pixels décodés en RGB 8 bits : même modèle, prompt, graine, format et réglages doivent
                       donner le même md5 à chaque essai, chaque jour (un écart est un défaut : le signaler)
        avec le modèle, la version, les pas, la graine, les accélérateurs (colonne neural_engine), la version de l'app, la puce, la mémoire,
        macOS, la source d'alimentation et l'état thermique. Le prompt d'un rendu est encodé une fois puis gardé :
        avec plusieurs rendus du même prompt, l'encodeur ne tourne que la première fois ; plusieurs --prompt
        l'incluent à chaque fois.

        Sortie : stdout, des lignes JSON — une {"type": "render", …} par rendu, puis le verdict
        {"ok": true, "summary": {"median_seconds", "witness_head_seconds", "witness_tail_seconds",
        "witness_drift", "peak_gib", "swapouts", "bits_stable", "notes", …}, "tsv": "<chemin>"}.
        stderr : la même chose, pour une personne. Le TSV (par défaut siliconed-measure-<date>-<modèle>-<format>.tsv
        dans le dossier courant) a une ligne par rendu, témoins et chargement compris.
        --out garde les PNG là ; sinon ils sont effacés une fois hachés.
        --tag nomme la série dans le TSV. Après --, les options vont à « add » telles quelles (--steps, --lora,
        --accelerators, --no-accelerators) ; pas --wait, --out, --batch, --ref, --model, --preview.

        Pour qu'une comparaison veuille dire quelque chose : le secteur branché, aucun autre travail sur le Mac,
        l'app fraîchement ouverte pour un chiffre de mémoire (un pic lu après d'autres modèles inclut ce qu'ils ont laissé).

        ÉCHECS : usage, busy (un rendu ou une installation tourne), unsupported (--faster-rendering avec une
        app qui ne le déclare pas), model (inconnu ou pas installé),
        render (le rendu a échoué : son « error » et son message), write. Les lignes déjà mesurées restent dans le TSV.
        """,

        "errors": """
        LES CODES D'ERREUR ("error" dans un verdict "ok": false)

          app_not_running      l'app est injoignable (sortie 3)             → silicontrol open
          app_not_found        « open » n'a pas trouvé l'app (sortie 3)     → construire ou installer Siliconed.app
          usage                commande, option ou argument mal formé (sortie 2)
                                                                           → le prompt entre guillemets ; help <commande>
          prompt               alternatives {a|b} : une { jamais fermée, une } seule, des groupes
                               imbriqués, plus de 64 combinaisons          → help add (ALTERNATIVES)
          model                modèle inconnu ou pas installé
                                                                           → "hint" liste les prêts ; models
          busy                 (install, diagnose, set, import, uninstall, library) un rendu, une installation ou le diagnostic
                               tourne ; (crop) l'éditeur de recadrage est ouvert ; (sheet) des cases sont encore en route ; (rewrite, describe, lab rewrite|describe) une image se rend ou attend ; (add, grid, vary) une installation tourne ;
                               (toutes) plus de 8 connexions qui envoient encore leur commande
                                                                           → attendre, puis réessayer
          library_absent       (add, grid, vary) la bibliothèque est sur un disque qui n'est pas branché
                                                                           → le brancher, puis réessayer
          lora                 LoRA introuvable pour ce modèle, ou > 3      → "hint" liste les compatibles
          format               format illisible ou hors bornes ; une édition au-delà de 1024² de surface
                                                                           → WxH ou P:Q@1K|2K, côtés ≥ 512, multiples de 16
          steps                un nombre de pas sans calendrier pour ce modèle → son "default_steps"
          edit                 --ref sur un modèle sans édition, ou plus de --ref qu'il n'en lit
                                                                           → un modèle "edit": true ; son "max_references"
          ref                  image de référence, ou à décrire, illisible  → vérifier le chemin
          img2img              (add, start-image) --image sur un modèle qui ne part pas d'une image,
                               ou une force qui ne laisse aucun pas        → un modèle "start_image": true ; une --strength plus haute
          no_start_image       (start-image --strength) le rack ne tient aucune image de départ
                                                                           → start-image <image|i7>
          solver               --solver hors du mode développeur, ou sur un modèle qui ne fait
                               qu'Euler                                    → le retirer
          detail               --detail more|most sur un modèle sans Détail (Anima, Krea 2)
                                                                           → le retirer
          accelerators         --accelerators, ou set accelerators, sur un Mac dont la mesure n'offre
                               aucun accélérateur                          → le retirer : rien à choisir là
                               set accelerators sur un Mac pas encore mesuré → un rendu (il mesure), ou diagnose
          subject              un sujet nommé : inconnu, nom déjà pris, trop d'images, modèle qui
                               n'en lit pas, une grille qui en cite un     → help subjects
          grid                 un axe illisible, deux axes sur un même réglage, un groupe {…} sur
                               aucun axe, plus de 64 cases                 → help grid
          queue_full           64 travaux en attente, ou pas de place pour toutes les alternatives ou cases
                                                                           → wait, puis réessayer
          remote               (install, import, rewrite, describe, lab rewrite|describe) la fenêtre calcule sur un autre Mac ; ces commandes
                               agissent sur celui-ci ; (uninstall, library move|home) un retrait ou un déplacement
                               attend là un clic sur l'autre Mac           → Rendu › Calculer sur › Ce Mac
          link                 la session avec l'autre Mac est coupée (add, grid, vary, again) ; (compute, macs)
                               ce Mac a refusé la session                  → macs
          refused              (again, vary, enlarge) la fenêtre ne le ferait pas, et dit pourquoi dans "message"
          nothing_to_undo      (undo) rien à annuler, ni à rétablir         → la liste d'annulation de la fenêtre
          crop                 (crop) un cadre que l'image ne peut pas donner : il sort de l'image, un côté sous 512 px, l'image
                               entière, une proportion qu'elle ne tient pas, une planche → help crop
          cell                 (sheet) aucune case de la série n'a encore d'image → l'attendre
          import               (import) ni .safetensors ni .gguf, ou le moteur a refusé le fichier → "message"
          unknown_mac          (macs, compute) aucun Mac appairé de ce nom, ou deux → macs ; l'"id"
          library              (library) la bibliothèque est déjà où on la demande, ou le moteur a refusé la destination
                                                                           → "message"
          writer_absent        (rewrite, describe, lab rewrite|describe, install writer) aucun modèle installé n'a de rédacteur
                                                                           → install qwen-image-2.1
          writer_head          (rewrite, describe, lab rewrite|describe) la tête du rédacteur est à télécharger (1,25 Go)
                                                                           → install writer
          writer_memory        (rewrite, describe, lab rewrite|describe, install writer) ce Mac est trop petit pour le rédacteur
          writer_preparing     (rewrite, describe, lab rewrite|describe, install writer) le rédacteur se prépare → status
          writer_busy          (rewrite, describe, lab rewrite|describe) un autre Réécrire ou Décrire est en cours, du terminal ou de
                               la fenêtre de l'Atelier                     → attendre, puis réessayer
          writer_failed        (rewrite, describe, lab rewrite|describe) le rédacteur a échoué ou n'a rien écrit ; le code du moteur
                               (render_already_running…) quand il en a un  → "message"
          empty_text           (lab rewrite) le texte de l'Atelier est vide → reuse <i7>, ou lab describe <n>
          prompt_too_long      (rewrite, lab rewrite) le texte est plus long que ce que le rédacteur lit → le raccourcir
          license_not_accepted (install writer) la licence de Qwen-Image-2.1 n'est pas encore acceptée
                                                                           → install qwen-image-2.1
          unsupported          (measure) --faster-rendering avec une app plus ancienne que silicontrol → même build
          unknown_job          ce t… (ou g…) n'existe pas dans cette session → status
          unknown_image        ce i… n'est pas dans l'historique            → history
          write                dossier ou fichier impossible à écrire       → vérifier le chemin
          render               (wait) un travail n'a pas fini               → "outcome", "message" et "code" de chaque travail
          stopped              (diagnose) le diagnostic a été arrêté (« Arrêter » dans la feuille) ; (rewrite,
                               describe) un rendu a démarré, il est prioritaire ; (lab rewrite|describe --wait) de même,
                               ou Arrêter ou lab close dans la barre        → relancer diagnose, ou attendre et réessayer
          protocol, internal   ne devrait pas arriver                       → le signaler
        """,

        "recipes": """
        RECETTES

        Une image, bloquante, et son chemin :
          silicontrol add --wait --out out "a 30 year old woman posing in a library" 1024 7 \\
            | tail -1 | jq .ok

        Une pile de demandes, puis leurs fichiers (seulement les siens, même si d'autres attendent) :
          P="a 30 year old woman posing in a library"
          a=$(silicontrol add "$P" 832x1216 42 | jq -r .job)
          b=$(silicontrol add --model qwen-image-2.1 "$P" 1024 42 | jq -r .job)
          c=$(silicontrol add --lora zimage-flat-color-v2-1:0.8 --steps 10 "$P" 1024 42 | jq -r .job)
          silicontrol wait $a $b $c --out out/pile | jq -r 'select(.type=="image") | .path'

        Balayer une force de LoRA (même graine : seule la force change) :
          for f in 0.4 0.7 1.0; do silicontrol add --lora zimage-flat-color-v2-1:$f "$P" 1024 42; done
          silicontrol wait --out out/sweep

        Plusieurs graines d'un même prompt : --batch (plus rapide que des demandes séparées) :
          silicontrol add --batch 4 --wait --out out/seeds "$P" 1024 100

        Comparer des lieux à la même graine (un travail par alternative, help add) :
          silicontrol add --wait --out out/places "a 30 year old woman posing in a {library|greenhouse}" 512 42 \\
            | jq -r 'select(.type=="image") | .prompt + " → " + .path'

        La même chose en planche : les lieux en colonnes, deux graines en rangées (help grid) :
          silicontrol grid --x prompt --y seeds=2 --wait --out out/grid \\
            "a 30 year old woman posing in a {library|greenhouse}" 512 42 | jq -r 'select(.type=="sheet") | .path'

        La force d'une LoRA contre des graines, une planche, sans attendre :
          g=$(silicontrol grid --lora zimage-flat-color-v2-1 --x lora=0.4,0.7,1 --y seed=42,7 "$P" 1024 | jq -r .grid)
          silicontrol wait $g --out out/lora

        Éditer la dernière image (modèle "edit": true) — elle devient l'image 1 :
          i=$(silicontrol history 1 | jq -r '.images[0].image')
          silicontrol add --wait --out out --model qwen-image-2.1 --ref "$i" "change her jacket to red"

        Éditer par instruction avec deux images (help edit) :
          silicontrol add --wait --out out --model qwen-image-2.1 --ref photo.png --ref chien.png \\
            "remove the second person from image 1 and put the dog of image 2 in her place" 42

        Tout arrêter : silicontrol clear
        """,
    ])
}
