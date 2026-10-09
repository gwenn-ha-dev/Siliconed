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

// `public`, not `package`: `tools/translations.sh` compiles the app's files alone, without the
// package's name, and a `package` symbol would not be seen there. This module is no part of the
// engine's API (`docs/API.md`): only the app and `silicontrol` link it.
public enum RemoteControlHelp {
    /// One language's pages: the overview, and a page per topic — and the refusal of an unknown topic.
    struct Pages: Sendable {
        let noHelp: @Sendable (_ topic: String) -> String
        let topics: @Sendable (_ list: String) -> String
        let overview: String
        let pages: [String: String]
    }

    /// The topics in reading order (`help all` follows it).
    static let order = ["contract", "open", "models", "install", "add", "grid", "edit", "wait", "status", "follow",
                        "cancel", "clear", "history", "save", "vary", "diagnose", "errors", "recipes"]
    static let aliases = ["examples": "recipes"]

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
    public static func answer(_ argv: [String]) -> [String: Any]? { answer(argv, preferred) }

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
      t3 = a job (one request for one or more images) · i7 = an image of the history.

    THE COMMANDS (details: help <command>, or <command> --help)
      open                                     launches the app and waits until it answers
      models                                   installed models, formats, LoRAs, default steps
      install ID [--standard|--compact|--light] shows a model's license and size in the app, to accept there
      diagnose [--issue]                       measures this Mac (≈ 15 s per model) → the report's JSON
      add [options] "<prompt>" [format] [seed] puts a request in the queue → its job (t3)
      grid --x AXIS [--y AXIS] [options] "<prompt>" …
                                               an XY grid: one job per cell, then a sheet → its grid (g2)
      wait [t3 …] [--out D]                    blocks until done; writes the PNGs into D
      wait g2 [--out D]                        blocks until a grid's sheet
      status                                   what runs, what waits, the time left
      follow                                   live events until the queue is empty
      cancel [t3 …]                            stops the running render, or removes jobs
      clear                                    empties the queue and stops the running render
      history [N]                              the session's images (most recent first)
      save i7 [i8 …] DEST                      writes images (.png/.jpg file, or folder)
      vary i7 subtle|strong [--batch N]        variations of an image (same settings, same seed)
      help [topic]                             this reference

    EDITING: silicontrol add --model qwen-image-2.1 --ref photo.png --ref dog.png "put the dog of image 2 next to her"
      up to 3 --ref, in order; the first is image 1, the one edited (help edit)

    OTHER PAGES: help edit · help errors · help recipes · help all (the whole reference at once)
    """, pages: [
        "contract": """
        THE CONTRACT — what holds for every command

        1. stdout only contains JSON lines, one per line (ASCII keys, sorted: jq '.job' works).
           Nothing else is written there. The one exception is the client printing the "help"
           text of a help answer raw.
        2. The LAST line is the verdict: {"ok": true, …} or {"ok": false, …}.
        3. The lines before it exist only for add --wait, grid --wait, wait and follow; they carry
           "type": added, image, sheet, start, stage, step, end.
        4. A failure: {"ok": false, "error": "<code>", "message": "<sentence>", "hint": "<what to do>"}.
           The code is stable (help errors); "hint" is not always there. "message" and "hint" are
           in the app's language; only "error" is meant for programs.
        5. Exit code: 0 verdict ok · 1 verdict ok:false · 2 usage error · 3 the app cannot be
           reached (not running and not launchable, or not answering).
        6. Short names, numbered within the app's session and lost when it quits:
           t3 = a job (one request: 1 to 8 images) · i7 = an image of the history ·
           g2 = a grid (one job per cell, then its sheet, an image of the history).
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

        The models the app shows, installed or not: Z-Image Turbo and Qwen-Image-2.1 Turbo, plus
        the models you imported (« add » refuses a model the app does not show). Read it before
        « add »: the ids, the LoRAs and the formats come from here.

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
                      "standard" (the publisher's weights, the default) and, for Z-Image and
                      Qwen-Image-2.1, "compact" (8-bit weights published by third parties: smaller
                      on disk, an image slightly different, about as fast) and "light" (the
                      image model in 4 to 6 bits as a third party publishes it, the Compact's text
                      encoder: the smallest; another set of weights, an image that may compose
                      differently). Absent for an imported model
        variant     → the version on disk; null while the model is not installed (and for an
                      imported model, which has its own weights)
        loras.lora  → --lora of « add » (only the LoRAs of THIS model)
        edit        → the model accepts --ref (editing an image with a prompt); null while it is
                      not installed
        max_references → how many --ref it reads at most (Qwen-Image-2.1 3, Z-Image 0);
                      null while it is not installed
        formats     → the suggested formats; any valid WxH is accepted (help add)
        license_urls → the license file(s) on Hugging Face, at the revision the installation takes:
                      the model's, then each component under a license of its own (Qwen-Image-2.1:
                      the model, then the turbo LoRA). Z-Image has none: its model card, which
                      declares Apache 2.0.
        license_accepted → false: the first render of this model waits until the user accepts its
                      license in the app (a sheet shows it); declined, the job ends in "error"
                      with "code": "license_not_accepted"
        A model with "ready": false cannot render; its "install" field is the command that offers
        it in the app (help install) — the user reads the license and the size, and accepts there.
        """,

        "install": """
        silicontrol install ID [--standard|--compact|--light]

        Opens the app's Models sheet on the model ID (an "id" of « models »), with its license and
        the space it will take on disk. NOTHING is downloaded by the command: the user reads, then
        clicks « Accept and Install » in the app (tens of GB, several minutes). The answer comes at
        once:
          {"ok": true, "model": "qwen-image-2.1", "installed": false, "variant": "standard", "message": "…"}
          {"ok": true, "model": "z-image", "installed": true, "variant": "standard", "message": "already installed"}
        Then « silicontrol models » says "ready": true once the installation is done.

        Without a flag, the version installed is kept (none installed: Standard, or Light on a Mac
        of 8 GB — preselected, the user sees all three): a command that names no version never
        switches it.
        --standard  preselects the Standard version (the publisher's weights);
        --compact   the Compact version, --light the Light version (Z-Image, Qwen-Image-2.1 — the
                    "variants" of « models »).
                    One version of a model is installed at a time: asking for another one of an
                    installed model offers to replace it, and the user still accepts in the app. The
                    installed one keeps rendering until the new one is complete, then goes — unless
                    the disk cannot hold both, which the app says before the user accepts.
        An imported model's ID installs what it lacks (encoder, VAE) in its family's version
        installed: a flag that contradicts that version is refused.

        FAILURES: usage (also: --compact or --light for a model that has neither, two version flags
        together, a flag that contradicts the version an imported model's family has), model
        (unknown), busy (a render, an installation or the
        diagnostic is running: wait, then retry).
        """,

        "diagnose": """
        silicontrol diagnose [--issue]

        « Report My Configuration » (the app's Help menu), from the terminal. The sheet opens in
        the app and the diagnostic runs on the visible installed models, at 512²: the text
        encoder, two DiT steps (the second judged against the fp32 reference when the model has
        one), the decoder — about 15 s per model. The queue waits for it, and it for the queue.

          {"type": "started", "models": ["z-image", "qwen-image-2.1"]}
          {"ok": true, "report": {"machine": {…}, "models": [{"model": "z-image",
            "seconds": {…}, "estimatedRenderSeconds": 31.6, "peakFootprintBytes": …,
            "swapouts": 0, "deviation": null | {"worst": …, "threshold": …, "pass": true}, …}], …}}

        --issue   then opens GitHub's issue form in the browser, prefilled with the report (the
                  sheet's « Open the Issue »); the user submits it. "issue_url_length" says how
                  long the link was; beyond ~8 000 characters the JSON goes to the clipboard
                  instead and "message" says so.

        FAILURES: usage, busy (a render or an installation is running, or nothing is installed),
        stopped (« Stop » in the sheet).
        """,

        "add": """
        silicontrol add [options] "<prompt>" [512|1024|WxH] [seed]

        Puts a request in the app's queue and returns at once (unless --wait). This page is
        complete. Options and arguments go in any order; the first argument that is not an option
        is the prompt.

        ARGUMENTS
          "<prompt>"          ONE argument: always in quotes. An extra word is refused.
                              {a|b|c} in it: one job per alternative (ALTERNATIVES below).
          512 | 1024 | WxH    format. 512 = 512x512, 1024 = 1024x1024; WxH = width x height
                              (832x1216 = portrait, the "formats" of « models »). Each side
                              ≥ 512 and a multiple of 16, area ≤ 1024×1536. Default: 1024x1024.
          seed                integer. Absent: drawn at random, returned in "seeds".

        OPTIONS
          --model ID          an "id" of « models ». Default: z-image.
          --lora NAME[:STRENGTH]
                              repeatable, 3 at most. NAME = a "loras.lora" of « models » (or its
                              displayed name, or a path). STRENGTH: 1 by default, 0.8 attenuates,
                              > 1 emphasizes.
          --steps N           number of steps, 1 to 50. Default: the model's "default_steps". A
                              model may have a schedule for some counts only (Qwen-Image-2.1: 5,
                              6, 7, 9).
          --detail more|most  the rack's Detail: finer texture and small details, same time;
                              most can over-sharpen. Default: normal (the image as the model
                              makes it). Jobs and images say "detail" when it is not normal.
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

        FAILURES: usage, prompt, model, lora, format, steps, edit, ref, img2img (--image/--strength: the
        app does none), queue_full (64 waiting, or not
        enough room for every alternative), busy (an installation is running: no render until it
        is done), write.

        Indicative times (MacBook Pro M1 Pro 16 GB, Z-Image): ~45 s at 512², ~115 s at 1024².
        """,

        "grid": """
        silicontrol grid --x AXIS [--y AXIS] [options] "<prompt>" [512|1024|WxH] [seed]

        An XY grid: one or two settings swept, ONE JOB PER CELL in the queue — each an ordinary
        job, its image in the history like any other —, then, when the last cell ends, a SHEET:
        one image with every cell side by side, the axes' values as labels and the prompt as its
        title, added to the history (i…). Nothing is written to disk without --out or save.
        Options and arguments are those of « add » (help add), except --batch: a cell is one image.
        --sketch: an EXPLORATION — each cell stops as soon as its image is readable (σ ≤ 0.8: 3
          evaluations of 7 for z-image, about half the time) and shows the model's estimate of the
          final image (qwen-image-2.1 does not sketch: its cells are finished images); the cells go to the app's Exploration window and
          their "type": "image" lines (with "sketch": n), not to the history, and no sheet is drawn.
          A new exploration replaces the last one's cells still waiting. Keep 512 on the short side.

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
           "jobs": [{"job": "t5", "grid": "g2", "column": 0, "row": 0, "prompt": "…",
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

        FAILURES: usage (no --x, --batch, --x given to « add »), grid (an axis that does not read or
        does not make a grid, more than 64 cells), prompt, model, lora, format, steps, edit, ref,
        queue_full, busy (an installation is running), write.
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
           "width": 832, "height": 1216, "seed": 42, "steps": 8, "evaluations": 7,
           "seconds": 45.86, "loras": [], "edit": false, "references": 0, "path": "/abs/D/….png"}
          "path" only exists with --out; "write_error" if it could not be written.
          Without --out, the image is ONLY in the app's memory (history, lost when it quits):
          « save i7 DEST » writes it later.

        FILE NAMES: <model>[-lora-<name>[@<strength>]…][-p<steps>][-edit]-<W>x<H>-<seed>.png
          (the strength if it is not 1, the steps if they are not the model's). An existing file
          is never overwritten: -2, -3… are appended.

        VERDICT:
          {"ok": true, "jobs": [{"job": "t3", "outcome": "done", "images": ["i7"]}]}
          outcome: done · stopped (stopped mid-computation; the images already made stay)
                 · removed (removed from the queue before running) · error (with "message",
                 the engine's sentence in the app's language, and "code", its stable key:
                 "license_not_accepted", "insufficient_memory", "disk_full"…).
          If a job is not "done": "ok": false, "error": "render", exit 1.

        FAILURES: usage, unknown_job, write.
        """,

        "status": """
        silicontrol status

        What the app is doing now. Does not block.

        Answer:
          {"ok": true,
           "running": {"job": "t3", "stage": "denoising", "batch_image": 1, "step": 4,
                       "steps_total": 7, "fraction": 0.52, "remaining_s": 21.3, "model": "z-image",
                       "prompt": "…", "seeds": [42], …} | null,
           "queue": [{"job": "t4", "estimate_s": 45.2, …}, …],
           "waiting_for_license": [{"job": "t5", "state": "waiting_for_license",
                                    "license": "Qwen Research (non-commercial)", "license_urls": […], "model": …}, …],
           "installing": {"title": "Installing Z-Image Turbo", "last": "forging the DiT → …"} | null,
           "queue_remaining_s": 66.5, "remaining_complete": true,
           "history_images": 12, "library": "/Users/…/Siliconed"}

        stage              : text, image (encoding a reference), denoising, decoding.
        steps_total        : the model's evaluations (often steps − 1), not the requested steps.
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
          {"type": "step", "job": "t3", "batch_image": 1, "step": 4, "steps_total": 7, "seconds": 5.8}
          {"type": "image", "image": "i7", …}           (as in « wait », without "path")
          {"type": "end", "job": "t3", "outcome": "done" | "stopped" | "removed" | "error"}
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

        "history": """
        silicontrol history [N]

        The session's images, most recent first (the first N if N is given). The same object as
        the "image" lines of « wait », without "path": the history is in memory.

        Answer: {"ok": true, "images": [{"image": "i7", "job": "t3", "seed": 42, …}, …]}
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

        "save": """
        silicontrol save i7 [i8 …] DEST

        Writes images of the history to disk.
          DEST = a .png or .jpg file (a single image): written, or REPLACED if it exists.
          DEST = a folder (created if needed): naming and no-overwrite as in « wait ».
        The PNG carries its metadata (prompt, seed, model, LoRAs).

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
          busy                 (install, diagnose) a render, an installation or the diagnostic is
                               running; (add, grid, vary) an installation is running;
                               (any) more than 8 connections still sending their command
                                                                           → wait, then retry
          lora                 LoRA not found for this model, or > 3        → "hint" lists the compatible ones
          format               unreadable or out-of-bounds format; an edit beyond 1024² of area
                                                                           → WxH, sides ≥ 512, multiples of 16
          steps                a step count the model has no schedule for   → its "default_steps"
          edit                 --ref on a model without editing, or more --ref than it reads
                                                                           → a model with "edit": true; its "max_references"
          ref                  unreadable reference image                   → check the path
          img2img              --image / --strength: not in the app         → leave them out
          grid                 an axis that does not read, two axes on one setting, a {…} group
                               on no axis, more than 64 cells              → help grid
          queue_full           64 jobs waiting, or no room for every alternative or cell
                                                                           → wait, then retry
          unknown_job          this t… (or g…) does not exist in this session → status
          unknown_image        this i… is not in the history                → history
          write                folder or file cannot be written             → check the path
          render               (wait) a job did not finish                  → "outcome", "message" and "code" of each job
          stopped              (diagnose) the diagnostic was stopped (« Stop » in the sheet)
                                                                           → diagnose again
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
      t3 = un travail (une demande d'une ou plusieurs images) · i7 = une image de l'historique.

    LES COMMANDES (détails : help <commande>, ou <commande> --help)
      open                                     lance l'app et attend qu'elle réponde
      models                                   modèles installés, formats, LoRA, pas par défaut
      install ID [--standard|--compact|--light] montre dans l'app la licence et la place d'un modèle, à accepter là
      diagnose [--issue]                       mesure ce Mac (≈ 15 s par modèle) → le JSON du rapport
      add [options] "<prompt>" [format] [seed] met une demande en file → son travail (t3)
      grid --x AXE [--y AXE] [options] "<prompt>" …
                                               une grille XY : un travail par case, puis une planche → sa grille (g2)
      wait [t3 …] [--out D]                    bloque jusqu'à la fin ; écrit les PNG dans D
      wait g2 [--out D]                        bloque jusqu'à la planche d'une grille
      status                                   ce qui tourne, ce qui attend, le temps restant
      follow                                   les évènements en direct jusqu'à file vide
      cancel [t3 …]                            arrête le rendu en cours, ou retire des travaux
      clear                                    vide la file et arrête le rendu en cours
      history [N]                              les images de la session (la plus récente d'abord)
      save i7 [i8 …] DEST                      écrit des images (fichier .png/.jpg, ou dossier)
      vary i7 subtle|strong [--batch N]        variations d'une image (mêmes réglages, même graine)
      help [sujet]                             cette référence

    ÉDITER : silicontrol add --model qwen-image-2.1 --ref photo.png --ref chien.png "put the dog of image 2 next to her"
      jusqu'à 3 --ref, dans l'ordre ; le premier est l'image 1, celle qu'on édite (help edit)

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
           portent "type" : added, image, sheet, start, stage, step, end.
        4. Un échec : {"ok": false, "error": "<code>", "message": "<phrase>", "hint": "<quoi faire>"}.
           Le code est stable (help errors) ; "hint" n'est pas toujours là. "message" et "hint"
           sont dans la langue de l'app ; seul "error" est fait pour les programmes.
        5. Code de sortie : 0 verdict ok · 1 verdict ok:false · 2 erreur d'usage · 3 l'app est
           injoignable (pas ouvert et impossible à lancer, ou ne répond pas).
        6. Noms courts, numérotés dans la session de l'app et perdus quand elle quitte :
           t3 = un travail (une demande : 1 à 8 images) · i7 = une image de l'historique ·
           g2 = une grille (un travail par case, puis sa planche, une image de l'historique).
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

        Les modèles que l'app montre, installés ou non : Z-Image Turbo et Qwen-Image-2.1 Turbo, plus
        les modèles que vous avez importés (« add » refuse un modèle que l'app ne montre pas). À lire
        avant « add » : les ids, les LoRA et les formats viennent d'ici.

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
                      le disque : "standard" (les poids de l'éditeur, le défaut) et, pour Z-Image et
                      Qwen-Image-2.1, "compact" (poids 8 bits publiés par des tiers : plus léger sur
                      le disque, une image légèrement différente, à peu près aussi rapide) et "light"
                      (le modèle d'image en 4 à 6 bits tel qu'un tiers le publie, l'encodeur de la
                      Compact : la plus petite ; un autre jeu de poids, une image qui peut se composer
                      autrement). Absent pour un modèle importé
        variant     → la version sur le disque ; null tant que le modèle n'est pas installé (et pour
                      un modèle importé, qui a ses propres poids)
        loras.lora  → --lora de « add » (seulement les LoRA de CE modèle)
        edit        → le modèle accepte --ref (édition d'une image par un prompt) ; null tant
                      qu'il n'est pas installé
        max_references → combien de --ref il lit au plus (Qwen-Image-2.1 3, Z-Image 0) ;
                      null tant qu'il n'est pas installé
        formats     → les formats conseillés ; tout WxH valide est accepté (help add)
        license_urls → le(s) fichier(s) de licence sur Hugging Face, à la révision que l'installation
                      prend : celui du modèle, puis chaque composant sous une licence propre
                      (Qwen-Image-2.1 : le modèle, puis la LoRA turbo). Z-Image n'en a pas : sa
                      fiche, qui déclare Apache 2.0.
        license_accepted → false : le premier rendu de ce modèle attend que l'utilisateur accepte
                      sa licence dans l'app (une feuille la montre) ; refusée, le travail finit en
                      "error" avec "code": "license_not_accepted"
        Un modèle "ready": false ne peut pas rendre ; son champ "install" est la commande qui le
        propose dans l'app (help install) — l'utilisateur y lit la licence et la place, et accepte.
        """,

        "install": """
        silicontrol install ID [--standard|--compact|--light]

        Ouvre la feuille Modèles de l'app sur le modèle ID (un "id" de « models »), avec sa licence
        et la place qu'il prendra sur le disque. La commande ne télécharge RIEN : l'utilisateur lit,
        puis clique « Accepter et installer » dans l'app (des dizaines de Go, plusieurs minutes). La
        réponse vient tout de suite :
          {"ok": true, "model": "qwen-image-2.1", "installed": false, "variant": "standard", "message": "…"}
          {"ok": true, "model": "z-image", "installed": true, "variant": "standard", "message": "already installed"}
        Ensuite « silicontrol models » dit "ready": true quand l'installation est finie.

        Sans drapeau, la version installée est gardée (aucune installée : Standard, ou Légère sur
        un Mac de 8 Go — présélectionnée, l'utilisateur voit les trois) : une commande qui ne nomme
        pas de version n'en change jamais.
        --standard  présélectionne la version Standard (les poids de l'éditeur) ;
        --compact   la version Compact, --light la version Légère (Z-Image, Qwen-Image-2.1 — les
                    "variants" de « models »).
                    Une seule version d'un modèle est installée à la fois : en demander une autre d'un
                    modèle installé propose de la remplacer, et l'utilisateur accepte toujours dans
                    l'app. L'installée continue de rendre jusqu'à ce que la nouvelle soit complète,
                    puis part — sauf si le disque ne peut pas tenir les deux, ce que l'app dit avant
                    l'acceptation.
        L'ID d'un modèle importé installe ce qui lui manque (encodeur, VAE) dans la version
        installée de sa famille : un drapeau qui la contredit est refusé.

        ÉCHECS : usage (aussi : --compact ou --light pour un modèle qui n'en a pas, deux drapeaux de
        version ensemble, un drapeau qui contredit la version de la famille d'un modèle importé), model
        (inconnu), busy (un rendu, une installation ou le
        diagnostic tourne : attendre, puis réessayer).
        """,

        "diagnose": """
        silicontrol diagnose [--issue]

        « Signaler ma configuration » (menu Aide de l'app), depuis le terminal. La feuille s'ouvre
        dans l'app et le diagnostic tourne sur les modèles visibles installés, en 512² : l'encodeur
        de texte, deux pas du DiT (le second jugé contre la référence fp32 quand le modèle en a
        une), le décodeur — environ 15 s par modèle. La file l'attend, et lui attend la file.

          {"type": "started", "models": ["z-image", "qwen-image-2.1"]}
          {"ok": true, "report": {"machine": {…}, "models": [{"model": "z-image",
            "seconds": {…}, "estimatedRenderSeconds": 31.6, "peakFootprintBytes": …,
            "swapouts": 0, "deviation": null | {"worst": …, "threshold": …, "pass": true}, …}], …}}

        --issue   ouvre ensuite le formulaire d'issue GitHub dans le navigateur, prérempli avec le
                  rapport (« Ouvrir le signalement » de la feuille) ; c'est l'utilisateur qui l'envoie.
                  "issue_url_length" dit la longueur du lien ; au-delà de ~8 000 caractères le
                  JSON va dans le presse-papiers et "message" le dit.

        ÉCHECS : usage, busy (un rendu ou une installation tourne, ou rien n'est installé),
        stopped (« Arrêter » dans la feuille).
        """,

        "add": """
        silicontrol add [options] "<prompt>" [512|1024|WxH] [seed]

        Met une demande dans la file de l'app et rend la main tout de suite (sauf --wait).
        Cette page est complète. Options et arguments vont dans n'importe quel ordre ; le premier
        argument qui n'est pas une option est le prompt.

        ARGUMENTS
          "<prompt>"          UN argument : toujours entre guillemets. Un mot en trop est refusé.
                              {a|b|c} dedans : un travail par alternative (ALTERNATIVES plus bas).
          512 | 1024 | WxH    format. 512 = 512x512, 1024 = 1024x1024 ; WxH = largeur x hauteur
                              (832x1216 = portrait, les "formats" de « models »). Chaque côté
                              ≥ 512 et multiple de 16, surface ≤ 1024×1536. Défaut : 1024x1024.
          seed                entier. Absente : tirée au hasard, rendue dans "seeds".

        OPTIONS
          --model ID          un "id" de « models ». Défaut : z-image.
          --lora NOM[:FORCE]  répétable, 3 au plus. NOM = un "loras.lora" de « models » (ou son
                              nom affiché, ou un chemin). FORCE : 1 par défaut, 0.8 atténue,
                              > 1 appuie.
          --steps N           nombre de pas, de 1 à 50. Défaut : "default_steps" du modèle. Un
                              modèle peut n'avoir de calendrier que pour certains (Qwen-Image-2.1 :
                              5, 6, 7, 9).
          --detail more|most  le Détail du rack : texture fine et petits détails, même durée ;
                              most peut trop accentuer. Défaut : normal (l'image telle que le
                              modèle la fait). Travaux et images disent "detail" s'il n'est pas normal.
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

        ÉCHECS : usage, prompt, model, lora, format, steps, edit, ref, img2img (--image/--strength : l'app
        n'en fait pas), queue_full (64 en attente, ou pas assez
        de place pour toutes les alternatives), busy (une installation tourne : pas de rendu avant
        qu'elle soit finie), write.

        Temps indicatifs (MacBook Pro M1 Pro 16 Go, Z-Image) : ~45 s à 512², ~115 s à 1024².
        """,

        "grid": """
        silicontrol grid --x AXE [--y AXE] [options] "<prompt>" [512|1024|WxH] [seed]

        Une grille XY : un ou deux réglages balayés, UN TRAVAIL PAR CASE dans la file — chacun un
        travail ordinaire, son image dans l'historique comme les autres —, puis, quand la dernière
        case finit, une PLANCHE : une image avec toutes les cases côte à côte, les valeurs des axes
        en étiquettes et le prompt en titre, ajoutée à l'historique (i…). Rien n'est écrit sur
        disque sans --out ou save.
        Options et arguments sont ceux de « add » (help add), sauf --batch : une case est une image.
        --sketch : une EXPLORATION — chaque case s'arrête dès que son image se lit (σ ≤ 0,8 : 3
          évaluations sur 7 pour z-image, environ la moitié du temps) et montre l'estimation par le
          modèle de l'image finale (qwen-image-2.1 n'esquisse pas : ses cases sont des images finies) ; les cases vont dans la fenêtre
          Exploration de l'app et dans leurs lignes "type": "image" (avec "sketch": n), pas dans
          l'historique, et aucune planche n'est dessinée. Une nouvelle exploration remplace les cases
          encore en attente de la précédente. Garder 512 sur le petit côté.

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
           "jobs": [{"job": "t5", "grid": "g2", "column": 0, "row": 0, "prompt": "…",
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

        ÉCHECS : usage (pas de --x, --batch, --x donné à « add »), grid (un axe illisible ou qui ne
        fait pas une grille, plus de 64 cases), prompt, model, lora, format, steps, edit, ref,
        queue_full, busy (une installation tourne), write.
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
           "width": 832, "height": 1216, "seed": 42, "steps": 8, "evaluations": 7,
           "seconds": 45.86, "loras": [], "edit": false, "references": 0, "path": "/abs/D/….png"}
          "path" n'existe qu'avec --out ; "write_error" s'il n'a pas pu être écrit.
          Sans --out, l'image n'est QUE dans la mémoire de l'app (historique, perdu quand elle
          quitte) : « save i7 DEST » l'écrit plus tard.

        NOMS DES FICHIERS : <model>[-lora-<nom>[@<force>]…][-p<pas>][-edit]-<W>x<H>-<seed>.png
          (la force si elle n'est pas 1, les pas s'ils ne sont pas ceux du modèle). Un fichier
          présent n'est jamais écrasé : -2, -3… sont ajoutés.

        VERDICT :
          {"ok": true, "jobs": [{"job": "t3", "outcome": "done", "images": ["i7"]}]}
          outcome : done · stopped (arrêté en plein calcul ; les images déjà faites restent)
                  · removed (retiré de la file avant de tourner) · error (avec "message", la
                  phrase du moteur dans la langue de l'app, et "code", sa clé stable :
                  "license_not_accepted", "insufficient_memory", "disk_full"…).
          Si un travail n'est pas "done" : "ok": false, "error": "render", sortie 1.

        ÉCHECS : usage, unknown_job, write.
        """,

        "status": """
        silicontrol status

        Ce que fait l'app maintenant. Ne bloque pas.

        Réponse :
          {"ok": true,
           "running": {"job": "t3", "stage": "denoising", "batch_image": 1, "step": 4,
                       "steps_total": 7, "fraction": 0.52, "remaining_s": 21.3, "model": "z-image",
                       "prompt": "…", "seeds": [42], …} | null,
           "queue": [{"job": "t4", "estimate_s": 45.2, …}, …],
           "waiting_for_license": [{"job": "t5", "state": "waiting_for_license",
                                    "license": "Qwen Research (non-commercial)", "license_urls": […], "model": …}, …],
           "installing": {"title": "Installing Z-Image Turbo", "last": "forging the DiT → …"} | null,
           "queue_remaining_s": 66.5, "remaining_complete": true,
           "history_images": 12, "library": "/Users/…/Siliconed"}

        stage              : text, image (encodage d'une référence), denoising, decoding.
        steps_total        : les évaluations du modèle (souvent pas − 1), pas les pas demandés.
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
          {"type": "step", "job": "t3", "batch_image": 1, "step": 4, "steps_total": 7, "seconds": 5.8}
          {"type": "image", "image": "i7", …}           (comme dans « wait », sans "path")
          {"type": "end", "job": "t3", "outcome": "done" | "stopped" | "removed" | "error"}
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

        "history": """
        silicontrol history [N]

        Les images de la session, la plus récente d'abord (les N premières si N est donné). Même
        objet que les lignes "image" de « wait », sans "path" : l'historique est en mémoire.

        Réponse : {"ok": true, "images": [{"image": "i7", "job": "t3", "seed": 42, …}, …]}
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

        "save": """
        silicontrol save i7 [i8 …] DEST

        Écrit des images de l'historique sur le disque.
          DEST = un fichier .png ou .jpg (une seule image) : écrit, ou REMPLACÉ s'il existe.
          DEST = un dossier (créé au besoin) : nommage et non-écrasement de « wait ».
        Le PNG porte ses métadonnées (prompt, graine, modèle, LoRA).

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
          busy                 (install, diagnose) un rendu, une installation ou le diagnostic
                               tourne ; (add, grid, vary) une installation tourne ;
                               (toutes) plus de 8 connexions qui envoient encore leur commande
                                                                           → attendre, puis réessayer
          lora                 LoRA introuvable pour ce modèle, ou > 3      → "hint" liste les compatibles
          format               format illisible ou hors bornes ; une édition au-delà de 1024² de surface
                                                                           → WxH, côtés ≥ 512, multiples de 16
          steps                un nombre de pas sans calendrier pour ce modèle → son "default_steps"
          edit                 --ref sur un modèle sans édition, ou plus de --ref qu'il n'en lit
                                                                           → un modèle "edit": true ; son "max_references"
          ref                  image de référence illisible                 → vérifier le chemin
          img2img              --image / --strength : pas dans l'app        → les retirer
          grid                 un axe illisible, deux axes sur un même réglage, un groupe {…} sur
                               aucun axe, plus de 64 cases                 → help grid
          queue_full           64 travaux en attente, ou pas de place pour toutes les alternatives ou cases
                                                                           → wait, puis réessayer
          unknown_job          ce t… (ou g…) n'existe pas dans cette session → status
          unknown_image        ce i… n'est pas dans l'historique            → history
          write                dossier ou fichier impossible à écrire       → vérifier le chemin
          render               (wait) un travail n'a pas fini               → "outcome", "message" et "code" de chaque travail
          stopped              (diagnose) le diagnostic a été arrêté (« Arrêter » dans la feuille)
                                                                           → relancer diagnose
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
