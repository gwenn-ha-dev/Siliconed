// The help's pages of the contract: the overview, what holds for every command, following a render, the errors, the recipes.

extension RemoteControlHelp {
    /// What `silicontrol help` alone prints: every command in one screen.
    static let overview = Page(en: """
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
      install ID [--standard|--compact]        shows a model's license and size in the app, to accept there
      import FILE.safetensors…                 brings a model or a LoRA, as published, into the library
      measure [options] [-- add options]       times and peak memory of a series of renders → a TSV and its summary
      add [options] "<prompt>" [format] [seed] puts a request in the queue → its job (t3)
      grid --x AXIS [--y AXIS] [options] "<prompt>" …
                                               an XY grid: one job per cell, a series of the history → its grid (g2)
      wait [t3 …] [--out D]                    blocks until done; writes the PNGs into D
      wait g2 [--out D]                        blocks until a grid's last cell; with --out, its sheet written there too
      status                                   what runs, what waits, the time left
      follow                                   live events until the queue is empty
      cancel [t3 …]                            stops the running render, or removes jobs
      clear                                    empties the queue and stops the running render
      set beta [OPTION on|off]                 the four Beta options of Settings, as a click; alone, reads them
      set neural-engine [on|off]               Neural Engine of Settings: the engine gives it the work it does faster; alone, reads it
      set save-images [FOLDER|off]             Save Every Image To… of Settings: each new image as a PNG there; alone, reads it
      set glass [on|off]                       Glass of Settings: the columns let a zoomed image show through; alone, reads it
      set updates [on|off]                     Check at Launch of Settings: each launch asks GitHub for the latest version; alone, reads it
      set memory [GB|default]                  Memory Limit of Settings › Macs: the most a render holds (B); alone, reads it
      set preview [on|off]                     Preview every step, in the DiT box: the image taking shape after each step; alone, reads it
      compiled [erase]                         what Core AI compiled for this Mac weighs; erase: erases it, as Settings › Macs does
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
      lab · lab add|role|remove …              the Prompt Lab (the Text and Images boxes at the head of the chain): its images, a role each
      lab text "<text>" · lab generate         its text as typed, and a click on its « Generate »
      lab rewrite · lab describe N [--wait]    its Rewrite and Describe, written in its text as its buttons write
      subjects · subjects add IMAGE … · subjects remove NAME
                                               the named subjects a prompt cites as @name (help subjects)
      settings [PANE] · settings lab [rewrite|describe] [--for ID]
                                               opens Settings on a pane, or Prompt Lab on one instruction
      bracket KNOB · bracket close             a knob's ± clicked: its bubble open (seed, detail, solver, start, loras, lora NAME)
      rack top|dit|fold|unfold|detach|dock     the chain's column scrolled to its head or to the DiT's settings, hidden (⌥⌘R), shown,
                                               in a window of its own (Detach the Chain), back in the main one (Dock the Chain)
      zoom in|out|fit                          View › Zoom In, Zoom Out, Zoom to Fit (⌘+ ⌘− ⌘0): the image, or a series' cells
      snapshot FILE.png [--window W] [--appearance light|dark]
                                               a window drawn by the app into a PNG (no screen capture, works over ssh)
      remove i7 [i8 …] · remove g2             Remove from History, an image or a series · undo [--redo]: ⌘Z, ⇧⌘Z
      again <i7|file.png> · reuse <i7|g2|file.png>
                                               Render Again (Same Seed) · Reuse These Settings (an image, a PNG of another day), Reuse This Grid
      crop i7 --box X,Y,W,H | --ratio P:Q      Crop… → a new image (i8)
      sheet g2 FILE.png|DIR                    Export Sheet… (a series' images side by side, labelled, written there)
      install stop                             Stop the installation in progress
      uninstall ID [--variant V]|downloads|legacy [--yes]
                                               Uninstall / Remove a version / Discard / Free… (version 0.21's files): without --yes, says what
                                               would go and the room freed
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
      Z-Image, Anima, Krea 2 and MOSS-Image; 0 keeps the image, 1 ignores it (help start-image)

    OTHER PAGES: help edit · help errors · help recipes · help all (the whole reference at once)
    """, fr: """
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
      install ID [--standard|--compact]        montre dans l'app la licence et la place d'un modèle, à accepter là
      import FICHIER.safetensors…              fait entrer dans la bibliothèque un modèle ou un LoRA, tel que publié
      measure [options] [-- options de add]    temps et pic mémoire d'une série de rendus → un TSV et son résumé
      add [options] "<prompt>" [format] [seed] met une demande en file → son travail (t3)
      grid --x AXE [--y AXE] [options] "<prompt>" …
                                               une grille XY : un travail par case, une série de l'historique → sa grille (g2)
      wait [t3 …] [--out D]                    bloque jusqu'à la fin ; écrit les PNG dans D
      wait g2 [--out D]                        bloque jusqu'à la dernière case d'une grille ; avec --out, y écrit aussi sa planche
      status                                   ce qui tourne, ce qui attend, le temps restant
      follow                                   les évènements en direct jusqu'à file vide
      cancel [t3 …]                            arrête le rendu en cours, ou retire des travaux
      clear                                    vide la file et arrête le rendu en cours
      set beta [OPTION on|off]                 les quatre options Bêta des Réglages, comme un clic ; seul, les lit
      set neural-engine [on|off]               Neural Engine des Réglages : le moteur lui confie le travail qu'il fait plus vite ; seul, le lit
      set save-images [FOLDER|off]             Enregistrer chaque image dans… des Réglages : chaque image nouvelle en PNG ; seul, le lit
      set glass [on|off]                       Verre des Réglages : les colonnes laissent voir une image zoomée ; seul, le lit
      set updates [on|off]                     Vérifier au lancement, des Réglages : chaque lancement demande à GitHub la dernière version ; seul, le lit
      set memory [Go|default]                  Limite de mémoire, de Réglages › Mac : le plus qu'un rendu tient (B) ; seul, la lit
      set preview [on|off]                     Aperçu à chaque pas, dans la boîte du DiT : l'image qui se forme après chaque pas ; seul, le lit
      compiled [erase]                         ce que Core AI a compilé pour ce Mac pèse ; erase : l'efface, comme Réglages › Mac
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
      lab · lab add|role|remove …              l'Atelier (les boîtes Texte et Images en tête de la chaîne) : ses images, un rôle chacune
      lab text "<texte>" · lab generate        son texte comme tapé, et un clic sur son « Générer »
      lab rewrite · lab describe N [--wait]    son Réécrire et son Décrire, écrits dans son texte comme ses boutons l'écrivent
      subjects · subjects add IMAGE … · subjects remove NOM
                                               les sujets nommés qu'un prompt cite par @nom (help subjects)
      settings [VOLET] · settings lab [rewrite|describe] [--for ID]
                                               ouvre les Réglages sur un volet, ou l'Atelier de prompts sur une consigne
      bracket RÉGLAGE · bracket close          un clic sur le ± d'un réglage : sa bulle ouverte (seed, detail, solver, start, loras, lora NOM)
      rack top|dit|fold|unfold|detach|dock     la colonne de la chaîne défilée jusqu'à sa tête ou aux réglages du DiT, masquée (⌥⌘R), affichée,
                                               dans une fenêtre à elle (Détacher la chaîne), rendue à la fenêtre (Ancrer la chaîne)
      zoom in|out|fit                          Présentation › Zoom avant, Zoom arrière, Taille ajustée (⌘+ ⌘− ⌘0) : l'image, ou les cases d'une série
      snapshot FICHIER.png [--window F] [--appearance light|dark]
                                               une fenêtre dessinée par l'app dans un PNG (sans capture d'écran, marche en ssh)
      remove i7 [i8 …] · remove g2             Retirer de l'historique, une image ou une série · undo [--redo] : ⌘Z, ⇧⌘Z
      again <i7|fichier.png> · reuse <i7|g2|fichier.png>
                                               Rendre à nouveau (même graine) · Reprendre ces réglages (une image, un PNG d'un autre jour),
                                               Reprendre cette grille
      crop i7 --box X,Y,W,H | --ratio P:Q      Recadrer… → une nouvelle image (i8)
      sheet g2 FICHIER.png|DOSSIER             Exporter la planche… (les images d'une série côte à côte, légendées, écrites là)
      install stop                             arrête l'installation en cours
      uninstall ID [--variant V]|downloads|legacy [--yes]
                                               Désinstaller / Retirer une version / Jeter / Libérer… (les fichiers de la 0.21) : sans --yes, dit
                                               ce qui partirait et la place libérée
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
      Z-Image, Anima, Krea 2 et MOSS-Image ; 0 garde l'image, 1 l'ignore (help start-image)

    AUTRES PAGES : help edit · help errors · help recipes · help all (toute la référence d'un coup)
    Les commandes, options et clés JSON sont en anglais ; seuls les textes sont traduits.
    """)

    static let contractPages: [String: Page] = [
        "contract": Page(en: """
        THE CONTRACT — what holds for every command

        1. stdout only contains JSON lines, one per line (ASCII keys, sorted: jq '.job' works).
           Nothing else is written there. The one exception is the client printing the "help"
           text of a help answer raw.
        2. The LAST line is the verdict: {"ok": true, …} or {"ok": false, …}.
        3. The lines before it exist only for add --wait, grid --wait, wait and follow; they carry
           "type": added, image, sheet (a grid's, written with --out), start, stage, step, end. rewrite, describe and lab rewrite|describe --wait
           have their own: started, loading, text (help rewrite); library move|home --yes, started and progress (help library);
           measure, render (help measure).
        4. A failure: {"ok": false, "error": "<code>", "message": "<sentence>", "hint": "<what to do>"}.
           The code is stable (help errors); "hint" is not always there. "message" and "hint" are
           in the app's language; only "error" is meant for programs.
        5. Exit code: 0 verdict ok · 1 verdict ok:false · 2 usage error · 3 the app cannot be
           reached (not running and not launchable, or not answering).
        6. Short names, numbered within the app's session and lost when it quits:
           t3 = a job (one request: 1 to 8 images) · i7 = an image of the history ·
           g2 = a grid, or a series of the window (one job per cell, each cell an image of the history) ·
           g2c3 = a cell of a grid, by its rank in reading order (row × columns + column), from 0.
           A bare number is also accepted for a job (3 = t3).
        7. Nothing is written to disk without --out or save: the history lives in the app.
        8. Relative paths resolve against the terminal's current directory.
        9. Same arguments, same seed, same model, with the Neural Engine off (help set) and the same macOS: the same image,
           bit for bit. With it on the engine may move work to the Neural Engine from one render to the next, and the
           pixels may differ slightly.
        10. Closing the app's window empties its queue and stops the render; the app stays open.
        11. A word of two dashes is a flag, wherever it is: one the command does not take is refused, by name ("usage").
            A lone -- ends the flags: a text, a file or a folder whose name begins with -- is written after it:
            silicontrol add -- "--ar 16:9 a lighthouse at dusk". --help or -h before it asks for the command's page;
            after it, they are words.
        """, fr: """
        LE CONTRAT — ce qui vaut pour toutes les commandes

        1. stdout ne contient que des lignes JSON, une par ligne (clés ASCII, triées :
           jq '.job' marche). Rien d'autre n'y est écrit, sauf le client qui imprime tel quel
           le texte "help" d'une réponse d'aide.
        2. La DERNIÈRE ligne est le verdict : {"ok": true, …} ou {"ok": false, …}.
        3. Les lignes d'avant n'existent que pour add --wait, grid --wait, wait et follow ; elles
           portent "type" : added, image, sheet (celle d'une grille, écrite avec --out), start, stage, step, end. rewrite, describe et lab rewrite|describe
           --wait ont les leurs : started, loading, text (help rewrite) ; library move|home --yes, started et progress (help library) ;
           measure, render (help measure).
        4. Un échec : {"ok": false, "error": "<code>", "message": "<phrase>", "hint": "<quoi faire>"}.
           Le code est stable (help errors) ; "hint" n'est pas toujours là. "message" et "hint"
           sont dans la langue de l'app ; seul "error" est fait pour les programmes.
        5. Code de sortie : 0 verdict ok · 1 verdict ok:false · 2 erreur d'usage · 3 l'app est
           injoignable (pas ouvert et impossible à lancer, ou ne répond pas).
        6. Noms courts, numérotés dans la session de l'app et perdus quand elle quitte :
           t3 = un travail (une demande : 1 à 8 images) · i7 = une image de l'historique ·
           g2 = une grille, ou une série de la fenêtre (un travail par case, chaque case une image de l'historique) ·
           g2c3 = une case d'une grille, par son rang dans l'ordre de lecture (ligne × colonnes + colonne), depuis 0.
           Un nombre nu est aussi accepté pour un travail (3 = t3).
        7. Rien n'est écrit sur disque sans --out ou save : l'historique vit dans l'app.
        8. Les chemins relatifs se résolvent contre le répertoire courant du terminal.
        9. Mêmes arguments, même graine, même modèle, le Neural Engine éteint (help set) et sous le même macOS : la même
           image, au bit près. Allumé, le moteur peut confier du travail au Neural Engine d'un rendu à l'autre, et les
           pixels peuvent différer légèrement.
        10. Fermer la fenêtre de l'app vide sa file et arrête le rendu ; l'app reste ouverte.
        11. Un mot de deux tirets est une option, où qu'il soit : une option que la commande ne prend pas est refusée,
            nommée (« usage »). Un -- seul clôt les options : un texte, un fichier ou un dossier dont le nom commence par --
            s'écrit après lui : silicontrol add -- "--ar 16:9 a lighthouse at dusk". --help ou -h avant lui demande la page
            de la commande ; après lui, ce sont des mots.
        """),

        "wait": Page(en: """
        silicontrol wait [t3 t4 …] [--out D]
        silicontrol wait g2 [--out D]

        Blocks until the jobs are done — or, for a grid (g2: a « grid » or a series of the window), until its last cell: its
        cells' lines, then, with --out, its sheet written there too and a "type": "sheet" line, then the verdict with "grid"
        (help grid). With no job named: all those running or waiting at the
        time of the call — including those started by others (the mouse); naming your jobs
        (t3 t4) waits only for yours. A job that is already done answers at once, even long
        after. Ctrl-C only stops the wait: the renders go on in the app.

        LINES, as they come — one per image:
          {"type": "image", "image": "i7", "job": "t3", "model": "z-image", "prompt": "…",
           "width": 832, "height": 1216, "seed": 42, "steps": 8, "evaluations": 8,
           "seconds": 45.86, "loras": [], "edit": false, "references": 0, "path": "/abs/D/….png"}
          "path" only exists with --out; "write_error" if it could not be written. A cell of a grid or of a
          series also has "grid" (g2) and "cell" (g2c3).
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
                 pixels came out unusable (black or blank) is never kept: a job that loses them all
                 is "error", code "non_finite_image", its message naming where (the DiT or the VAE);
                 a job of several seeds that loses some is "done" with the others, and each one lost
                 is a line of its "warnings": "non_finite_image: seed 43: <where>". A render whose
                 engine stopped (past this Mac's budget B, or a crash) is "error", code
                 "engine_stopped"; one silent for ten minutes, "engine_silent"; one of another build
                 than the app, "engine_mismatch". The jobs waiting behind it go on, in a new engine.
          "warnings": the engine's own lines during the job's render (English), when it wrote any.
          If a job is not "done": "ok": false, "error": "render", exit 1.

        FAILURES: usage, unknown_job, write.
        """, fr: """
        silicontrol wait [t3 t4 …] [--out D]
        silicontrol wait g2 [--out D]

        Bloque jusqu'à ce que les travaux aient fini — ou, pour une grille (g2 : une « grid » ou une série
        de la fenêtre), jusqu'à sa dernière case : les lignes de ses cases, puis, avec --out, sa planche
        écrite là aussi et une ligne "type": "sheet", puis le verdict avec "grid" (help grid). Sans travail nommé : tous ceux qui tournent
        ou attendent au moment de l'appel — y compris ceux lancés par d'autres (la souris) ; nommer
        ses travaux (t3 t4) n'attend que les siens. Un travail déjà fini répond tout de suite, même
        longtemps après. Ctrl-C n'arrête que l'attente : les rendus continuent dans l'app.

        LIGNES, au fil de l'eau — une par image :
          {"type": "image", "image": "i7", "job": "t3", "model": "z-image", "prompt": "…",
           "width": 832, "height": 1216, "seed": 42, "steps": 8, "evaluations": 8,
           "seconds": 45.86, "loras": [], "edit": false, "references": 0, "path": "/abs/D/….png"}
          "path" n'existe qu'avec --out ; "write_error" s'il n'a pas pu être écrit. Une case d'une grille ou
          d'une série a aussi "grid" (g2) et "cell" (g2c3).
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
                  les pixels sont sortis inutilisables (noire ou vide) n'est jamais gardée : un
                  travail qui les perd toutes est "error", code "non_finite_image", son message
                  disant où (le DiT ou le VAE) ; un travail de plusieurs graines qui en perd
                  quelques-unes est "done" avec les autres, et chaque image perdue est une ligne de
                  ses "warnings" : "non_finite_image: seed 43: <où>". Un
                  rendu dont le moteur s'est arrêté (au-delà du budget B de ce Mac, ou un plantage)
                  est "error", code "engine_stopped" ; muet dix minutes, "engine_silent" ; d'une
                  autre version que l'app, "engine_mismatch". Les travaux qui attendaient derrière
                  lui continuent, dans un moteur neuf.
          "warnings" : les lignes du moteur pendant le rendu du travail (en anglais), s'il en a écrit.
          Si un travail n'est pas "done" : "ok": false, "error": "render", sortie 1.

        ÉCHECS : usage, unknown_job, write.
        """),

        "status": Page(en: """
        silicontrol status

        What the app is doing now. Does not block.

        Answer:
          {"ok": true,
           "running": {"job": "t3", "stage": "denoising", "compiling": false, "calibrating": false, "batch_image": 1, "step": 4,
                       "steps_total": 8, "fraction": 0.52, "remaining_s": 21.3, "model": "z-image",
                       "prompt": "…", "seeds": [42], …} | null,
           "queue": [{"job": "t4", "estimate_s": 45.2, …}, …],
           "waiting_for_license": [{"job": "t5", "state": "waiting_for_license",
                                    "license": "Qwen Research (non-commercial)", "license_urls": […], "model": …}, …],
           "installing": {"title": "Installing Z-Image Turbo", "last": "Downloading z_image_turbo_bf16.safetensors: 42 %",
                          "received_bytes": 5170000000, "total_bytes": 12310000000} | null,
           "queue_remaining_s": 66.5, "remaining_complete": true,
           "history_images": 12, "library": "/Users/…/Siliconed",
           "beta": {"more_solvers": false, "steps": false, …}, "neural_engine": true, "compiling": false,
           "writer": {"state": "ready"}, "start_image": {…} | null, "computing_on": "Mac mini" | null,
           "memory": {"budget_bytes": 10737418240, "chosen": false, "default_bytes": 10737418240,
                      "range_bytes": [4294967296, 12884901888]}}

        beta               : the four Beta options of Settings (help set): more_solvers, steps, size_scales, negative_lora.
        neural_engine      : Settings' Neural Engine switch (help set): true or false.
        writer             : the Prompt Lab's writer (help rewrite): "state" is "ready", or "writer_absent", the code rewrite
                             would refuse with (no installed model carries a writer: install qwen-image-2.1).
        start_image        : the rack's starting image (help start-image): {"name", "width", "height", "strength", "used", …}, or null.
        computing_on       : the other Mac the window computes on (Render › Compute On), or null for this Mac.
        memory             : B, the most a render holds on this Mac (help set): the user's, or the default (¾ of the memory
                             less 2 GB), and the range it may be set in.
        stage              : text, image (encoding a reference), denoising, decoding.
        compiling          : true while Core AI compiles a function its cache does not hold (a model's first render, a new
                             size): what the render waits for then is paid once, and remaining_s does not count it. At the
                             top, also while no render runs: the app compiles ahead what the form's next render needs (its
                             model, size, LoRA, Neural Engine), and every installed model after an update.
        calibrating        : true while the engine looks for this Mac's Neural Engine share (a model's first render here);
                             the Chain inspector shows each step's share and seconds.
        steps_total        : the model's evaluations, not the requested steps.
        remaining_s        : null until the app has a measurement for this render.
        remaining_complete : false if a waiting job has no estimate (queue_remaining_s is a minimum).
        installing         : an installation in progress: {"title", "last" (the line the window shows),
                             "received_bytes", "total_bytes" (of this installation's files; absent before
                             the first one starts)}; renders wait until it ends (add, grid and vary answer "busy").
        waiting_for_license: jobs the engine refused because their model's license is not accepted
                             (help models, "license_accepted"). They wait out of the queue for the
                             user's answer in the app's sheet: accepted, they go back to the head of
                             the queue; declined, they end in "error", "code": "license_not_accepted".
                             « wait » waits for them, « cancel t5 » removes them. Nothing can accept
                             from the terminal: the user reads the license in the app.
        """, fr: """
        silicontrol status

        Ce que fait l'app maintenant. Ne bloque pas.

        Réponse :
          {"ok": true,
           "running": {"job": "t3", "stage": "denoising", "compiling": false, "calibrating": false, "batch_image": 1, "step": 4,
                       "steps_total": 8, "fraction": 0.52, "remaining_s": 21.3, "model": "z-image",
                       "prompt": "…", "seeds": [42], …} | null,
           "queue": [{"job": "t4", "estimate_s": 45.2, …}, …],
           "waiting_for_license": [{"job": "t5", "state": "waiting_for_license",
                                    "license": "Qwen Research (non-commercial)", "license_urls": […], "model": …}, …],
           "installing": {"title": "Installing Z-Image Turbo", "last": "Downloading z_image_turbo_bf16.safetensors: 42 %",
                          "received_bytes": 5170000000, "total_bytes": 12310000000} | null,
           "queue_remaining_s": 66.5, "remaining_complete": true,
           "history_images": 12, "library": "/Users/…/Siliconed",
           "beta": {"more_solvers": false, "steps": false, …}, "neural_engine": true, "compiling": false,
           "writer": {"state": "ready"}, "start_image": {…} | null, "computing_on": "Mac mini" | null,
           "memory": {"budget_bytes": 10737418240, "chosen": false, "default_bytes": 10737418240,
                      "range_bytes": [4294967296, 12884901888]}}

        beta               : les quatre options Bêta des Réglages (help set) : more_solvers, steps, size_scales, negative_lora.
        neural_engine      : l'interrupteur Neural Engine des Réglages (help set) : true ou false.
        writer             : le rédacteur de l'Atelier (help rewrite) : "state" vaut "ready", ou "writer_absent", le code dont
                             rewrite refuserait (aucun modèle installé n'a de rédacteur : install qwen-image-2.1).
        start_image        : l'image de départ du rack (help start-image) : {"name", "width", "height", "strength", "used", …}, ou null.
        computing_on       : l'autre Mac sur lequel la fenêtre calcule (Rendu › Calculer sur), ou null pour ce Mac.
        memory             : B, le plus qu'un rendu tient sur ce Mac (help set) : celui de l'utilisateur, ou le défaut (¾ de
                             la mémoire moins 2 Go), et la plage où il se règle.
        stage              : text, image (encodage d'une référence), denoising, decoding.
        compiling          : true tant que Core AI compile une fonction que son cache n'a pas (le premier rendu d'un
                             modèle, une taille neuve) : ce que le rendu attend alors se paie une fois, remaining_s ne le compte pas.
                             En tête, aussi quand aucun rendu ne tourne : l'app compile d'avance ce dont le prochain rendu du
                             formulaire a besoin (son modèle, sa taille, son LoRA, le Neural Engine), et chaque modèle installé
                             après une mise à jour.
        calibrating        : true tant que le moteur cherche la part du Neural Engine de ce Mac (le premier rendu d'un
                             modèle ici) ; l'inspecteur Chaîne montre la part et les secondes de chaque pas.
        steps_total        : les évaluations du modèle, pas les pas demandés.
        remaining_s        : null tant que l'app n'a pas de mesure pour ce rendu.
        remaining_complete : false si un travail en attente n'a pas d'estimation (queue_remaining_s est un minimum).
        installing         : une installation en cours : {"title", "last" (la ligne que la fenêtre montre),
                             "received_bytes", "total_bytes" (des fichiers de cette installation ; absents avant
                             le premier)} ; les rendus attendent sa fin (add, grid et vary répondent "busy").
        waiting_for_license: les travaux que le moteur a refusés parce que la licence de leur modèle
                             n'est pas acceptée (help models, "license_accepted"). Ils attendent hors
                             de la file la réponse de l'utilisateur dans la feuille de l'app : acceptée,
                             ils reviennent en tête de file ; refusée, ils finissent en "error",
                             "code": "license_not_accepted". « wait » les attend, « cancel t5 » les
                             retire. Rien ne peut accepter depuis le terminal : l'utilisateur lit la
                             licence dans l'app.
        """),

        "follow": Page(en: """
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
          A grid's cells come as any job's ("grid" and "cell" on their lines); no sheet is made (help sheet).
        VERDICT: {"ok": true, "message": "queue empty"}
        """, fr: """
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
          Les cases d'une grille viennent comme tout travail ("grid" et "cell" sur leurs lignes) ; aucune planche n'est faite (help sheet).
        VERDICT : {"ok": true, "message": "queue empty"}
        Les textes "message" et "hint" suivent la langue de l'app ; ceux montrés ici sont les anglais.
        """),

        "errors": Page(en: """
        THE ERROR CODES ("error" in a verdict with "ok": false)

          app_not_running      the app cannot be reached (exit 3)           → silicontrol open
          app_not_found        « open » did not find the app (exit 3)       → build or install Siliconed.app
          usage                malformed command, option or argument (exit 2)
                                                                           → the prompt in quotes; help <command>
          prompt               {a|b} alternatives: a { never closed, a } alone, nested groups,
                               more than 64 combinations                   → help add (ALTERNATIVES)
          model                unknown or not installed model
                                                                           → "hint" lists the ready ones; models
          busy                 (install, set, uninstall, library) a render or an installation is
                               running; (crop) the crop editor is open; (rewrite, describe, lab rewrite|describe) an image renders or waits; (add, grid, vary) an installation is running;
                               (any) more than 8 connections still sending their command
                                                                           → wait, then retry
          library_absent       (add, grid, vary, again, uninstall, library move|home) the library is on a disk that is not connected
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
          detail               --detail more|most on a model without Detail
                                                                           → leave it out
          subject              a named subject: unknown, a name taken, too many images, a model
                               that reads none, a grid citing one          → help subjects
          grid                 an axis that does not read, two axes on one setting, a {…} group
                               on no axis, more than 64 cells              → help grid
          queue_full           64 jobs waiting, or no room for every alternative or cell
                                                                           → wait, then retry
          remote               (install, import, rewrite, describe, lab rewrite|describe) the window computes on another Mac;
                               these act on this one; (uninstall, library move|home) a removal or a move there waits for a click on
                               the other Mac                               → Render › Compute On › This Mac
          link                 the session with the other Mac is down (add, grid, vary, again); (compute, macs)
                               that Mac refused the session                → macs
          refused              (again, vary, enlarge, lab generate; add, grid on a Mac followed) the window, or that Mac, would
                               not do it, and says why in "message"
          nothing_to_undo      (undo) nothing to undo, or to redo           → the window's own undo list
          crop                 (crop) a frame the image cannot give: it leaves the image, a side under 512 px, the whole
                               image, a proportion it cannot hold         → help crop
          cell                 (sheet) no cell of the series has a picture yet → wait for it
          unknown_mac          (macs, compute) no paired Mac of that name, or two → macs; the "id"
          bracket              (bracket) that knob has no ± under the current model and settings
                                                                           → help bracket
          image, role          (lab) no Prompt Lab image of that number, or no such role for it
                                                                           → lab
          window, snapshot     (snapshot) no window of that title or number is shown, or it could not be drawn
                                                                           → "hint" lists the windows
          download_required    (enlarge) macOS does not have Apple's enlargement model yet
                                                                           → enlarge <i7> --download
          download_failed      (enlarge --download) the enlargement model could not be downloaded
                                                                           → retry
          unreadable           (instructions --file) the file is not UTF-8 text
                                                                           → check the file
          import_refused       (import) neither a LoRA nor a model of a supported family; the engine's own code
                               (file_missing…) when it has one             → help import
          library              (library) the library is already where you ask, or the engine refused the destination
                                                                           → "message"
          writer_absent        (rewrite, describe, lab rewrite|describe) no installed model carries a writer
                                                                           → install qwen-image-2.1
          writer_busy          (rewrite, describe, lab rewrite|describe) another Rewrite or Describe is under way, from the terminal or
                               the Text box                                → wait, then retry
          writer_failed        (rewrite, describe, lab rewrite|describe) the writer failed or wrote nothing; the engine's own code
                               (insufficient_memory, render_already_running…) when it has one → "message"
          empty_text           (lab rewrite) the Prompt Lab's text is empty  → reuse <i7>, or lab describe <n>
          prompt_too_long      (rewrite, lab rewrite) the text is longer than the writer reads → shorten it
          stopped              (rewrite, describe) a render started, which has priority; (lab rewrite|describe --wait) the same,
                               or Stop in the Text box, or rack fold       → wait, then retry
          unknown_job          this t… (or g…) does not exist in this session → status
          unknown_image        this i… is not in the history                → history
          write                folder or file cannot be written             → check the path
          render               (wait) a job did not finish                  → "outcome", "message" and "code" of each job
          protocol, internal   should not happen                            → report it
        """, fr: """
        LES CODES D'ERREUR ("error" dans un verdict "ok": false)

          app_not_running      l'app est injoignable (sortie 3)             → silicontrol open
          app_not_found        « open » n'a pas trouvé l'app (sortie 3)     → construire ou installer Siliconed.app
          usage                commande, option ou argument mal formé (sortie 2)
                                                                           → le prompt entre guillemets ; help <commande>
          prompt               alternatives {a|b} : une { jamais fermée, une } seule, des groupes
                               imbriqués, plus de 64 combinaisons          → help add (ALTERNATIVES)
          model                modèle inconnu ou pas installé
                                                                           → "hint" liste les prêts ; models
          busy                 (install, set, uninstall, library) un rendu ou une installation
                               tourne ; (crop) l'éditeur de recadrage est ouvert ; (rewrite, describe, lab rewrite|describe) une image se rend ou attend ; (add, grid, vary) une installation tourne ;
                               (toutes) plus de 8 connexions qui envoient encore leur commande
                                                                           → attendre, puis réessayer
          library_absent       (add, grid, vary, again, uninstall, library move|home) la bibliothèque est sur un disque qui n'est pas branché
                                                                           → le brancher, puis réessayer
          lora                 LoRA introuvable pour ce modèle, ou > 3      → "hint" liste les compatibles
          format               format illisible ou hors bornes ; une édition au-delà de 1024² de surface
                                                                           → WxH ou P:Q@1K|2K, côtés ≥ 512, multiples de 16 (32 : qwen-image-2.1)
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
          detail               --detail more|most sur un modèle sans Détail
                                                                           → le retirer
          subject              un sujet nommé : inconnu, nom déjà pris, trop d'images, modèle qui
                               n'en lit pas, une grille qui en cite un     → help subjects
          grid                 un axe illisible, deux axes sur un même réglage, un groupe {…} sur
                               aucun axe, plus de 64 cases                 → help grid
          queue_full           64 travaux en attente, ou pas de place pour toutes les alternatives ou cases
                                                                           → wait, puis réessayer
          remote               (install, import, rewrite, describe, lab rewrite|describe) la fenêtre calcule sur un autre Mac ;
                               ces commandes agissent sur celui-ci ; (uninstall, library move|home) un retrait ou un déplacement
                               attend là un clic sur l'autre Mac           → Rendu › Calculer sur › Ce Mac
          link                 la session avec l'autre Mac est coupée (add, grid, vary, again) ; (compute, macs)
                               ce Mac a refusé la session                  → macs
          refused              (again, vary, enlarge, lab generate ; add, grid sur un Mac suivi) la fenêtre, ou ce Mac, ne le
                               ferait pas, et dit pourquoi dans "message"
          nothing_to_undo      (undo) rien à annuler, ni à rétablir         → la liste d'annulation de la fenêtre
          crop                 (crop) un cadre que l'image ne peut pas donner : il sort de l'image, un côté sous 512 px, l'image
                               entière, une proportion qu'elle ne tient pas → help crop
          cell                 (sheet) aucune case de la série n'a encore d'image → l'attendre
          unknown_mac          (macs, compute) aucun Mac appairé de ce nom, ou deux → macs ; l'"id"
          bracket              (bracket) ce bouton n'a pas de ± sous le modèle et les réglages actuels
                                                                           → help bracket
          image, role          (lab) aucune image de l'Atelier de ce numéro, ou pas ce rôle pour elle
                                                                           → lab
          window, snapshot     (snapshot) aucune fenêtre de ce titre ou de ce numéro n'est affichée, ou elle n'a pas pu être dessinée
                                                                           → "hint" liste les fenêtres
          download_required    (enlarge) macOS n'a pas encore le modèle d'agrandissement d'Apple
                                                                           → enlarge <i7> --download
          download_failed      (enlarge --download) le modèle d'agrandissement n'a pas pu être téléchargé
                                                                           → réessayer
          unreadable           (instructions --file) le fichier n'est pas du texte UTF-8
                                                                           → vérifier le fichier
          import_refused       (import) ni un LoRA ni un modèle d'une famille connue ; le code du moteur
                               (file_missing…) quand il en a un            → help import
          library              (library) la bibliothèque est déjà où on la demande, ou le moteur a refusé la destination
                                                                           → "message"
          writer_absent        (rewrite, describe, lab rewrite|describe) aucun modèle installé n'a de rédacteur
                                                                           → install qwen-image-2.1
          writer_busy          (rewrite, describe, lab rewrite|describe) un autre Réécrire ou Décrire est en cours, du terminal ou de
                               la boîte Texte                              → attendre, puis réessayer
          writer_failed        (rewrite, describe, lab rewrite|describe) le rédacteur a échoué ou n'a rien écrit ; le code du moteur
                               (insufficient_memory, render_already_running…) quand il en a un → "message"
          empty_text           (lab rewrite) le texte de l'Atelier est vide → reuse <i7>, ou lab describe <n>
          prompt_too_long      (rewrite, lab rewrite) le texte est plus long que ce que le rédacteur lit → le raccourcir
          stopped              (rewrite, describe) un rendu a démarré, il est prioritaire ; (lab rewrite|describe --wait) de même,
                               ou Arrêter dans la boîte Texte, ou rack fold → attendre et réessayer
          unknown_job          ce t… (ou g…) n'existe pas dans cette session → status
          unknown_image        ce i… n'est pas dans l'historique            → history
          write                dossier ou fichier impossible à écrire       → vérifier le chemin
          render               (wait) un travail n'a pas fini               → "outcome", "message" et "code" de chaque travail
          protocol, internal   ne devrait pas arriver                       → le signaler
        """),

        "recipes": Page(en: """
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
        """, fr: """
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
        """),
    ]
}
