// The help's pages of the renders and the history: add, grid, edit, the history's gestures, the queue.

extension RemoteControlHelp {
    static let renderPages: [String: Page] = [
        "add": Page(en: """
        silicontrol add [options] "<prompt>" [512|1024|WxH|P:Q@1K|2K] [seed]

        Puts a request in the app's queue and returns at once (unless --wait). This page is
        complete. Options and arguments go in any order; the first argument that is not an option
        is the prompt. A prompt that begins with -- goes after a lone -- (add -- "--ar 16:9 a cat"):
        before it, a word of two dashes is an option, and one « add » does not take is refused.

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
          --model ID          an "id" of « models ». Default: the one it marks "default".
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
                              makes it). Every model offers it, each at its own setting. Jobs and
                              images say "detail" when it is not normal.
          --solver NAME       the rack's Solver, with More Solvers on only: euler (default, the one
                              the model was distilled for), euler-ancestral, dpmpp-2m or unipc.
                              Same steps, same time, another image. Jobs and images say "solver"
                              when it is not euler.
          --batch N           N images (1 to 8) with seeds s, s+1, …; the prompt is encoded once
                              for the whole batch (faster than N requests). One job, not a series:
                              the window's Seeds is one (help grid).
          --ref IMAGE|i7     editing by instruction: an image file or an image of the history.
                              Repeatable, in order, up to the model's "max_references" (1 to 3
                              for Qwen-Image-2.1). Only for a model with "edit": true.
                              THE FIRST --ref IS IMAGE 1, THE ONE EDITED; the others are what the
                              prompt borrows from. The prompt says what changes, and may name the
                              images by number (« image 1 », « image 2 »): Qwen-Image-2.1 reads
                              them as <image1>, <image2>. Leave the format out: the output then
                              follows image 1 (help edit).
          --image IMAGE|i7    img2img: the render starts from this image (a file, or an image of the
                              history) instead of noise, and keeps its composition. Only for a
                              model with "start_image": true (Z-Image, Anima, Krea 2, MOSS-Image); not with
                              --ref. Leave the format out: the output takes the model's format
                              nearest the image's proportions; any format is accepted, the image
                              then filled and its centre kept. The rack takes the image, as
                              « start-image » does (help start-image).
          --strength S        with --image: how far the render moves away from it — diffusers'
                              strength, 0.65 or 65%, above 0 and at most 1 (1 ignores the image).
                              Default: the model's. Jobs and images say "strength" and "start_image".
          --preview, --no-preview
                              a preview at every step in the app (writes nothing). Default: the
                              DiT box's « Preview every step » switch.
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

        FAILURES: usage, prompt, model, solver, detail, lora, format, steps, edit, ref, img2img
        (--image on a model that does not start from an image, or a strength that leaves no step), subject (a named subject the prompt cites, help
        subjects), queue_full (64 waiting, or not enough room for every alternative), busy (an
        installation is running: no render until it is done), library_absent (the library's disk is not connected), write;
        on a Mac followed also link, remote (--image) and refused (help errors).
        """, fr: """
        silicontrol add [options] "<prompt>" [512|1024|WxH|P:Q@1K|2K] [seed]

        Met une demande dans la file de l'app et rend la main tout de suite (sauf --wait).
        Cette page est complète. Options et arguments vont dans n'importe quel ordre ; le premier
        argument qui n'est pas une option est le prompt. Un prompt qui commence par -- s'écrit après un
        -- seul (add -- "--ar 16:9 a cat") : avant lui, un mot de deux tirets est une option, et une
        option que « add » ne prend pas est refusée.

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
          --model ID          un "id" de « models ». Défaut : celui qu'il marque "default".
          --lora NOM[:FORCE]  répétable, 3 au plus. NOM = un "loras.lora" de « models » (ou son
                              nom affiché, ou un chemin). FORCE : 1 par défaut, 0.8 atténue,
                              > 1 appuie.
          --steps N           nombre de pas, de 1 à 50. Défaut : "default_steps" du modèle. Un
                              modèle peut n'avoir de calendrier que pour certains (Qwen-Image-2.1 :
                              8 seulement).
          --detail more|most  le Détail du rack : texture fine et petits détails, même durée ;
                              most peut trop accentuer. Défaut : normal (l'image telle que le
                              modèle la fait). Tous les modèles l'offrent, chacun à son réglage.
                              Travaux et images disent "detail" s'il n'est pas normal.
          --solver NOM        le Solveur du rack, en mode développeur seulement : euler (défaut,
                              celui pour lequel le modèle a été distillé), euler-ancestral,
                              dpmpp-2m ou unipc. Mêmes pas, même durée, une autre image. Travaux
                              et images disent "solver" s'il n'est pas euler.
          --batch N           N images (1 à 8) aux graines s, s+1, … ; le prompt est encodé une
                              fois pour tout le lot (plus rapide que N demandes). Un seul travail, pas
                              une série : les Graines de la fenêtre en sont une (help grid).
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
                              pour un modèle "start_image": true (Z-Image, Anima, Krea 2, MOSS-Image) ; pas
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
                              « Aperçu à chaque pas » de la boîte DiT.
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

        ÉCHECS : usage, prompt, model, solver, detail, lora, format, steps, edit, ref, img2img
        (--image sur un modèle qui ne part pas d'une image, ou une force qui ne laisse aucun pas), subject (un sujet nommé que le prompt cite,
        help subjects), queue_full (64 en attente, ou pas assez de place pour toutes les
        alternatives), busy (une installation tourne : pas de rendu avant qu'elle soit finie), library_absent (le disque de la
        bibliothèque n'est pas branché), write ; sur un Mac suivi aussi link, remote (--image) et refused (help errors).
        """),

        "grid": Page(en: """
        silicontrol grid --x AXIS [--y AXIS] [--no-sheet] [options] "<prompt>" [512|1024|WxH|P:Q@1K|2K] [seed]

        An XY grid: one or two settings swept, ONE JOB PER CELL in the queue — each an ordinary
        job, its image in the history like any other, carrying "grid" and "cell". Every series of the
        session is kept: the history's column shows its tile (its grid in small, n/N), its cells under
        it, foldable; a click opens its grid on the canvas. The SHEET — every cell side by side, the
        axes' values as labels, the prompt as its title — is an export, never an image of the history:
        « grid --wait --out D » writes it into D after the cells (--no-sheet goes without), « sheet g2 »
        writes it anywhere, any time (help sheet), the tile's menu has Export Sheet… and Copy Sheet.
        Nothing is written to disk without --out, save or sheet.
        Options and arguments are those of « add » (help add), except --batch: a cell is one image.
        Neighbouring cells that differ only by their seed (up to 8) render in one pass: the prompt encoded once,
        every denoising, then every decoding — their images come together, at the end of the pass.
        The window's series (« Bracket… » on a knob, a {…} group in the prompt, the Seeds) are grids too, with the same cells and
        the same tile: « Generate » with N seeds queues a grid whose cells differ only by the seed, rendered in one pass.
        The window takes the grid's axes as its series; a models, steps, listed seeds or images
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
          render (error, stopped, removed), or not yet, is drawn empty, with why; no cell rendered, no sheet.
          The images of the cells stay full size in the history.
          File name: grid-<model>-<columns>x<rows>-<W>x<H>[-<seed>].png — the base seed, left
          out when the seeds are listed (seed=…).

        ANSWER
          {"ok": true, "grid": "g2", "columns": 2, "rows": 2, "estimate_s": 128.4,
           "jobs": [{"job": "t5", "grid": "g2", "cell": "g2c0", "column": 0, "row": 0, "prompt": "…",
                     "seeds": [42], "position": 1, "estimate_s": 32.1, …}, …]}
          estimate_s: the cells' sum, null until the model has rendered at this format.
          With --wait: one "type": "added" line per cell, then the lines of « wait g2 »:
            {"type": "image", "image": "i7", "job": "t5", "grid": "g2", "cell": "g2c0", …, "path": …}   each cell, as it comes
            {"type": "sheet", "grid": "g2", "path": "/abs/D/grid-z-image-2x2-512x512-42.png"}   with --out, unless --no-sheet
            {"ok": true, "grid": "g2", "sheet": "/abs/D/grid-….png", "jobs": [{"job": "t5", "outcome": "done", …}, …]}
          "sheet" is in the verdict only when one was to be written: null if no cell rendered, with "write_error"
          if it could not be written. A cell not "done": "ok": false, "error": "render" — the sheet is
          written all the same, that cell empty.
          « wait g2 [--out D] » follows a grid later: at once if it has ended.
          A cell has a name of its own, g2c3 (its rank in reading order, from 0 — row × columns + column), and its
          job's, t5.

        FAILURES: usage (no --x, --batch, --x given to « add »), grid (an axis that does not read or
        does not make a grid, more than 64 cells), prompt, model, solver, detail, lora,
        format, steps, edit, ref, subject (a grid does not cite named subjects), queue_full, busy (an installation
        is running), write.
        """, fr: """
        silicontrol grid --x AXE [--y AXE] [--no-sheet] [options] "<prompt>" [512|1024|WxH|P:Q@1K|2K] [seed]

        Une grille XY : un ou deux réglages balayés, UN TRAVAIL PAR CASE dans la file — chacun un
        travail ordinaire, son image dans l'historique comme les autres, avec "grid" et "cell". Toutes
        les séries de la session se gardent : la colonne de l'historique montre leur tuile (la grille en
        petit, n/N), leurs cases dessous, repliables ; un clic ouvre sa grille sur le canevas. La
        PLANCHE — toutes les cases côte à côte, les valeurs des axes en étiquettes, le prompt en titre —
        est un export, jamais une image de l'historique : « grid --wait --out D » l'écrit dans D après
        les cases (--no-sheet s'en passe), « sheet g2 » l'écrit n'importe où, n'importe quand (help
        sheet), le menu de la tuile a Exporter la planche… et Copier la planche.
        Rien n'est écrit sur disque sans --out, save ou sheet.
        Options et arguments sont ceux de « add » (help add), sauf --batch : une case est une image.
        Des cases voisines qui ne diffèrent que par la graine (8 au plus) se rendent en une passe : le prompt encodé
        une fois, tous les débruitages, puis tous les décodages — leurs images arrivent ensemble, à la fin de la passe.
        Les séries de la fenêtre (« Bracketer… » sur une molette, un groupe {…} dans le prompt, les Graines) sont aussi des grilles,
        avec les mêmes cases et la même tuile : « Générer » avec N graines met en file une grille dont les cases ne diffèrent
        que par la graine, rendue en une passe. La fenêtre prend les
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
          qui n'a pas été rendue (erreur, arrêt, retrait), ou pas encore, est dessinée vide, avec la raison ; aucune
          case rendue, pas de planche. Les images des cases restent en taille réelle dans l'historique.
          Nom de fichier : grid-<modèle>-<colonnes>x<rangées>-<W>x<H>[-<graine>].png — la graine
          de base, omise quand les graines sont listées (seed=…).

        RÉPONSE
          {"ok": true, "grid": "g2", "columns": 2, "rows": 2, "estimate_s": 128.4,
           "jobs": [{"job": "t5", "grid": "g2", "cell": "g2c0", "column": 0, "row": 0, "prompt": "…",
                     "seeds": [42], "position": 1, "estimate_s": 32.1, …}, …]}
          estimate_s : la somme des cases, null tant que le modèle n'a pas rendu à ce format.
          Avec --wait : une ligne "type": "added" par case, puis les lignes de « wait g2 » :
            {"type": "image", "image": "i7", "job": "t5", "grid": "g2", "cell": "g2c0", …, "path": …}   chaque case, au fil de l'eau
            {"type": "sheet", "grid": "g2", "path": "/abs/D/grid-z-image-2x2-512x512-42.png"}   avec --out, sauf --no-sheet
            {"ok": true, "grid": "g2", "sheet": "/abs/D/grid-….png", "jobs": [{"job": "t5", "outcome": "done", …}, …]}
          "sheet" n'est dans le verdict que quand une planche était à écrire : null si aucune case n'a été
          rendue, avec "write_error" si elle n'a pas pu être écrite. Une case pas "done" : "ok": false,
          "error": "render" — la planche est écrite quand même, cette case vide.
          « wait g2 [--out D] » suit une grille plus tard : tout de suite si elle est finie.
          Une case a son propre nom, g2c3 (son rang dans l'ordre de lecture, depuis 0 — ligne × colonnes + colonne), et
          celui de son job, t5.

        ÉCHECS : usage (pas de --x, --batch, --x donné à « add »), grid (un axe illisible ou qui ne
        fait pas une grille, plus de 64 cases), prompt, model, solver, detail, lora,
        format, steps, edit, ref, subject (une grille ne cite pas de sujets nommés), queue_full, busy (une
        installation tourne), write.
        """),

        "edit": Page(en: """
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
        """, fr: """
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
        """),

        "cancel": Page(en: """
        silicontrol cancel [t3 t4 …]

        With no argument: stops the running render. With jobs: stops the one that runs, removes
        from the queue those that wait. The stop takes effect in under half a second; the images
        of a batch already made stay in the history.

        Answer: {"ok": true, "stopped": "t3" | null, "removed": ["t4"], "already_finished": []}
        FAILURES: usage, unknown_job.
        """, fr: """
        silicontrol cancel [t3 t4 …]

        Sans argument : arrête le rendu en cours. Avec des travaux : arrête celui qui tourne,
        retire de la file ceux qui attendent. L'arrêt prend effet en moins d'une demi-seconde ;
        les images déjà faites d'un lot restent dans l'historique.

        Réponse : {"ok": true, "stopped": "t3" | null, "removed": ["t4"], "already_finished": []}
        ÉCHECS  : usage, unknown_job.
        """),

        "clear": Page(en: """
        silicontrol clear

        Empties the queue AND stops the running render — like Render › « Clear Queue and Stop » (⌥⌘.) or the queue's « Stop All » in the app.

        Answer: {"ok": true, "stopped": "t3" | null, "removed": ["t4", "t5"]}
        """, fr: """
        silicontrol clear

        Vide la file ET arrête le rendu en cours — comme Rendu › « Vider la file et arrêter » (⌥⌘.) ou « Tout arrêter » de la file dans l'app.

        Réponse : {"ok": true, "stopped": "t3" | null, "removed": ["t4", "t5"]}
        """),

        "history": Page(en: """
        silicontrol history [N]

        The session's images, most recent first (the first N if N is given). The same object as
        the "image" lines of « wait », without "path": the history is in memory.

        Answer: {"ok": true, "images": [{"image": "i7", "job": "t3", "seed": 42, …}, …]}
        FAILURES: usage (N not an integer ≥ 1).
        """, fr: """
        silicontrol history [N]

        Les images de la session, la plus récente d'abord (les N premières si N est donné). Même
        objet que les lignes "image" de « wait », sans "path" : l'historique est en mémoire.

        Réponse : {"ok": true, "images": [{"image": "i7", "job": "t3", "seed": 42, …}, …]}
        ÉCHECS  : usage (N pas un entier ≥ 1).
        """),

        "save": Page(en: """
        silicontrol save i7 [i8 …] DEST [--format png|jpeg|heic]

        Writes images of the history to disk, in the three formats of the window's Save and Export panels.
          DEST = a .png, .jpg (.jpeg) or .heic file (a single image): written, or REPLACED if it exists; the extension decides.
          DEST = a folder (created if needed): naming and no-overwrite as in « wait ». --format says the format of its files (png by
                 default; the panel's pop-up remembers the last one used, the command does not).
        The PNG carries its metadata (prompt, seed, model, LoRAs); JPEG (0.9) and HEIC (0.85) are plain pictures, without metadata.

        Answer: {"ok": true, "files": [{"image": "i7", "path": "/abs/…"}]}
        FAILURES: usage, unknown_image, write.
        """, fr: """
        silicontrol save i7 [i8 …] DEST [--format png|jpeg|heic]

        Écrit des images de l'historique sur le disque, dans les trois formats des panneaux Enregistrer et Exporter de la fenêtre.
          DEST = un fichier .png, .jpg (.jpeg) ou .heic (une seule image) : écrit, ou REMPLACÉ s'il existe ; l'extension décide.
          DEST = un dossier (créé au besoin) : nommage et non-écrasement de « wait ». --format dit le format de ses fichiers (png par
                 défaut ; le menu du panneau retient le dernier utilisé, la commande non).
        Le PNG porte ses métadonnées (prompt, graine, modèle, LoRA) ; le JPEG (0,9) et le HEIC (0,85) sont des images simples, sans métadonnées.

        Réponse : {"ok": true, "files": [{"image": "i7", "path": "/abs/…"}]}
        ÉCHECS  : usage, unknown_image, write.
        """),

        "vary": Page(en: """
        silicontrol vary i7 subtle|strong [--batch N]

        « Variations » on a history image, as its menu in the app: N jobs (1 to 8; default: the
        rack's « Seeds », 1 unless changed) with exactly its settings and its seed, each with a
        variation seed drawn at random. The starting noise is
        turned a little towards the variation seed's: subtle keeps the scene and the pose, strong
        keeps the idea and moves the composition. A variation of a variation varies THAT image.
        Each image's "variation" lists its variation seeds in order; its PNG redoes it.

        Answer: {"ok": true, "jobs": [{"job": "t5", "seeds": [42], "variation": [{"seed": 81…, "strength": 0.1}], …}, …]}
        Then: silicontrol wait t5
        FAILURES: usage, unknown_image, model, queue_full, busy (an installation is running), library_absent.
        """, fr: """
        silicontrol vary i7 subtle|strong [--batch N]

        « Variations » sur une image de l'historique, comme son menu dans l'app : N tâches (1 à 8 ;
        par défaut le champ « Graines » du rack, 1 sauf changement) avec exactement ses réglages et sa
        graine, chacune avec une graine de variation tirée au hasard.
        Le bruit de départ est tourné un peu vers celui de la graine de variation : subtle garde la
        scène et la pose, strong garde l'idée et bouge la composition. Une variation d'une variation
        varie CETTE image. Le "variation" de chaque image donne ses graines de variation dans l'ordre ;
        son PNG la refait.

        Réponse : {"ok": true, "jobs": [{"job": "t5", "seeds": [42], "variation": [{"seed": 81…, "strength": 0.1}], …}, …]}
        Puis : silicontrol wait t5
        ÉCHECS : usage, unknown_image, model, queue_full, busy (une installation tourne), library_absent.
        """),

        "enlarge": Page(en: """
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
        FAILURES: usage, unknown_image, refused (no super-resolution on this Mac, already enlarged, wider than
          1920, another enlargement under way, a render running), download_required, download_failed, internal.
        """, fr: """
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
        ÉCHECS : usage, unknown_image, refused (pas de super-résolution sur ce Mac, déjà agrandie, plus large que
          1920, un autre agrandissement en cours, un rendu en cours), download_required, download_failed, internal.
        """),

        "crop": Page(en: """
        silicontrol crop i7 --box X,Y,W,H
        silicontrol crop i7 --ratio P:Q

        « Crop… » on a history image, with the frame given instead of dragged. --box is the frame in pixels of the image, the origin at
        the top left (x, y, width, height); --ratio is the largest frame of that proportion, centered (4:5, 16:9, 1:1…), which is where the
        editor's proportion menu starts. The frame's pixels become a NEW image at the head of the history; the original stays.
        « silicontrol undo » removes the crop, as ⌘Z does. The rules are the editor's: a side is never under 512 px, the whole image
        is no crop. The new image keeps the original's metadata plus "cropped".

        Answer: {"ok": true, "image": {"image": "i8", "width": 1024, "height": 1024, …, "crop": {"x": 96, "y": 0, "width": 1024,
                 "height": 1024}}, "undo": {…}}
        FAILURES: usage, unknown_image, crop (the frame leaves the image, a side under 512 px, the whole image, a proportion the image
          cannot hold), busy (the crop editor is open in the window).
        """, fr: """
        silicontrol crop i7 --box X,Y,W,H
        silicontrol crop i7 --ratio P:Q

        « Recadrer… » sur une image de l'historique, le cadre donné au lieu d'être tiré. --box est le cadre en pixels de l'image, l'origine en
        haut à gauche (x, y, largeur, hauteur) ; --ratio est le plus grand cadre de cette proportion, centré (4:5, 16:9, 1:1…), d'où part le
        menu des proportions de l'éditeur. Les pixels du cadre deviennent une NOUVELLE image en tête de l'historique ; l'originale reste.
        « silicontrol undo » retire le recadrage, comme ⌘Z. Les règles sont celles de l'éditeur : un côté n'est jamais sous 512 px, l'image
        entière n'est pas un recadrage. La nouvelle image garde les
        métadonnées de l'originale, plus "cropped".

        Réponse : {"ok": true, "image": {"image": "i8", "width": 1024, "height": 1024, …, "crop": {"x": 96, "y": 0, "width": 1024,
                   "height": 1024}}, "undo": {…}}
        ÉCHECS : usage, unknown_image, crop (le cadre sort de l'image, un côté sous 512 px, l'image entière, une proportion que l'image ne
          peut pas tenir), busy (l'éditeur de recadrage est ouvert dans la fenêtre).
        """),

        "again": Page(en: """
        silicontrol again <i7|file.png>
        silicontrol reuse <i7|g2|file.png>

        again is « Render Again (Same Seed) » (⌘R): the rack takes the image's settings (model, prompt, size, steps, LoRAs, edit
        images, detail, solver, and its seed), then « Generate » renders it, at that seed — a grid's cell as one image, not its
        series (« Reuse This Grid » is the series tile's). Same settings, same seed, same model, with the Neural Engine off and the same macOS: the same image, bit for bit.
        With a PNG that Siliconed wrote — an image of another day, a file from out/ — it is that PNG dropped in the Images box,
        « Reuse These Settings » on its tile, then « Generate »: the rack reads the file's recipe (model, prompt, size, steps, LoRAs found in this library, seed, variations),
        and what could not come back is in "notes" (a model not installed, a LoRA missing, a solver that needs More Solvers…).
        The rack keeps what it took, as after a click: « silicontrol add » is unaffected, it never reads the rack.
        reuse is « Reuse These Settings »: the same filling of the rack, with no render. reuse g2 is a series tile's « Reuse This
        Grid »: the rack takes what the cells share and the grid's axes, as the window brackets them, the prompt as typed; what
        the window cannot bracket (the grid's models, steps or edit images) is left out, and said in "notes".

        Answer (again): {"ok": true, "jobs": [{"job": "t6", "model": "z-image", "seeds": [42], …}]}
        Answer (reuse): {"ok": true, "rack": {"model": "z-image", "prompt": "…", "steps": 8, "width": 1024, "height": 1024,
                                              "seed": 42, "loras": [], "series": {"x": null, "y": null, "excess": [], "renders": 1}},
                         "notes": "…" (only when something did not come back).
        FAILURES: usage, unknown_image, unknown_job (reuse g2: no such grid in this session), ref (a file that is unreadable, or
          not a PNG Siliconed wrote), and for again the reasons « Generate » is
          greyed out: model (not installed), library_absent, busy (an installation is running), queue_full, link (the session with the
          other Mac is down), refused (anything else it says: "message" is the window's sentence — a memory refusal, a step count the
          model has no schedule for…). On a Mac followed, each job is answered by that Mac.
        """, fr: """
        silicontrol again <i7|fichier.png>
        silicontrol reuse <i7|g2|fichier.png>

        again est « Rendre à nouveau (même graine) » (⌘R) : le rack prend les réglages de l'image (modèle, prompt, taille, pas, LoRA, images à
        éditer, détail, solveur, et sa graine), puis « Générer » la rend, à cette graine — la case d'une grille comme une image, pas
        sa série (« Reprendre cette grille » est à la tuile de la série). Mêmes réglages, même graine, même modèle, le Neural Engine éteint et sous le même macOS : la même image, bit pour bit.
        Avec un PNG que Siliconed a écrit — une image d'un autre jour, un fichier de out/ — c'est ce PNG déposé dans la boîte Images,
        « Reprendre ces réglages » sur sa tuile, puis « Générer » : le rack lit la recette du fichier (modèle, prompt, taille, pas, LoRA retrouvées dans cette bibliothèque, graine,
        variations), et ce qui n'a pas pu revenir est dans "notes" (un modèle non installé, une LoRA absente, un solveur qui demande
        Plus de solveurs…). Le rack garde ce qu'il a pris, comme après un clic : « silicontrol add » n'en est pas touché, il ne lit
        jamais le rack.
        reuse est « Reprendre ces réglages » : le même remplissage du rack, sans rendu. reuse g2 est le « Reprendre cette
        grille » de la tuile d'une série : le rack prend ce que les cases partagent et les axes de la grille, comme la fenêtre les
        décline, le prompt tel que tapé ; ce que la fenêtre ne sait pas décliner (les modèles, les pas ou les images à éditer de la
        grille) est laissé de côté, et dit dans "notes".

        Réponse (again) : {"ok": true, "jobs": [{"job": "t6", "model": "z-image", "seeds": [42], …}]}
        Réponse (reuse) : {"ok": true, "rack": {"model": "z-image", "prompt": "…", "steps": 8, "width": 1024, "height": 1024,
                                                "seed": 42, "loras": [], "series": {"x": null, "y": null, "excess": [], "renders": 1}},
                          "notes": "…" (seulement si quelque chose n'est pas revenu).
        ÉCHECS : usage, unknown_image, unknown_job (reuse g2 : pas de telle grille dans cette session), ref (un fichier illisible,
          ou qui n'est pas un PNG écrit par Siliconed), et pour again les raisons pour
          lesquelles « Générer » est grisé : model (non installé), library_absent, busy (une installation tourne), queue_full, link (la
          session avec l'autre Mac est coupée), refused (tout le reste : "message" est la phrase de la fenêtre — un refus mémoire, un
          nombre de pas sans calendrier pour le modèle…). Sur un Mac suivi, chaque job reçoit la réponse de ce Mac.
        """),

        "remove": Page(en: """
        silicontrol remove i7 [i8 …]
        silicontrol remove g2

        « Remove from History » (⌫), as the menu does it: the images leave the history, in memory — nothing on disk is touched.
        The window undoes it with ⌘Z, and so does « silicontrol undo ». An image that was never saved is gone for good once
        the app quits (« save » first).
        A series (g2, alone) is its tile's « Remove Series from History » (its ✕): its cells leave the history, undone the same
        way; its cells still waiting leave the queue, and one running stops.

        Answer: {"ok": true, "removed": ["i7", "i8"], "history_images": 5,
                 "undo": {"can_undo": true, "action": "Remove 2 Images", "can_redo": false, "redo_action": ""}}
                {"ok": true, "grid": "g2", "removed": ["i3", "i4", "i5", "i6"], "history_images": 2, "undo": {…}}
        On a Mac followed (Compute On), the other Mac's images are removed, and undone, the same way.
        FAILURES: usage (also: a series beside other images), unknown_image, unknown_job (no such grid in this session).
        """, fr: """
        silicontrol remove i7 [i8 …]
        silicontrol remove g2

        « Retirer de l'historique » (⌫), comme le fait le menu : les images quittent l'historique, en mémoire — rien n'est touché sur le disque.
        La fenêtre l'annule par ⌘Z, et « silicontrol undo » aussi. Une image jamais enregistrée est perdue pour de bon quand l'app
        quitte (« save » d'abord).
        Une série (g2, seule) est le « Retirer la série de l'historique » de sa tuile (son ✕) : ses cases quittent l'historique,
        et s'annulent de même ; ses cases encore en attente quittent la file, celle qui tourne s'arrête.

        Réponse : {"ok": true, "removed": ["i7", "i8"], "history_images": 5,
                   "undo": {"can_undo": true, "action": "Retirer 2 images", "can_redo": false, "redo_action": ""}}
                  {"ok": true, "grid": "g2", "removed": ["i3", "i4", "i5", "i6"], "history_images": 2, "undo": {…}}
        Sur un Mac suivi (Calculer sur), les images de l'autre Mac sont retirées, et rendues, de la même façon.
        ÉCHECS : usage (aussi : une série à côté d'autres images), unknown_image, unknown_job (pas de telle grille dans cette session).
        """),

        "undo": Page(en: """
        silicontrol undo [--redo]

        Edit › Undo (⌘Z), or with --redo Edit › Redo (⇧⌘Z): whatever the window's undo manager would undo now, nothing more. What
        it knows: removing images from the history (a series removed by its tile's ✕ included), a crop, and the edits of the
        Text box's field, which shares it (typing, a Rewrite or a Describe written there). A removal of images
        on a Mac followed is undone by asking the images back from that Mac while the session lasts.

        Answer: {"ok": true, "done": "undo", "action": "Remove Image", "history_images": 6,
                 "undo": {"can_undo": false, "action": "", "can_redo": true, "redo_action": "Remove Image"}}
        FAILURES: usage, nothing_to_undo (nothing to undo, or to redo).
        """, fr: """
        silicontrol undo [--redo]

        Édition › Annuler (⌘Z), ou avec --redo Édition › Rétablir (⇧⌘Z) : ce que le gestionnaire d'annulation de la fenêtre annulerait
        maintenant, rien de plus. Ce qu'il connaît : le retrait d'images de l'historique (une série retirée par le ✕ de sa tuile
        comprise), un recadrage, et les modifications du champ de la boîte Texte, qui le partage (la frappe, un Réécrire ou un
        Décrire écrit là). Un retrait d'images sur un Mac suivi s'annule en redemandant les images à ce Mac, tant que la session dure.

        Réponse : {"ok": true, "done": "undo", "action": "Retirer l'image", "history_images": 6,
                   "undo": {"can_undo": false, "action": "", "can_redo": true, "redo_action": "Retirer l'image"}}
        ÉCHECS : usage, nothing_to_undo (rien à annuler, ni à rétablir).
        """),

        "sheet": Page(en: """
        silicontrol sheet <g2> <file.png|folder>

        « Export Sheet… » on a series' tile (the history's column), without the panel: the series' images side by side,
        labelled — the axes' values, the prompt as title —, written as a PNG. A .png path is written, or replaced; a
        folder (it must exist) takes it under its own name, grid-<model>-<columns>x<rows>-<W>x<H>[-<seed>].png, never over
        a file (-2, -3…). Any series of the session, a « grid » or one of the window, ended or not: a cell not rendered yet,
        or removed from the history, is drawn empty, with why. Answered once the file is written:
          {"ok": true, "grid": "g2", "path": "/abs/out/grid-z-image-2x2-512x512-42.png"}
        The sheet is never an image of the history: the series' images are there already, one by one, as any render's
        (each with "grid" and "cell"); the sheet only lays them out. A cell is named gNcM (help grid), and its job tN.
        « grid --wait --out D » and « wait g2 --out D » write it too, after the last cell.
        FAILURES: usage, unknown_job (no such grid in this session), cell (no cell has a picture yet: wait for it), write.
        """, fr: """
        silicontrol sheet <g2> <fichier.png|dossier>

        « Exporter la planche… » sur la tuile d'une série (la colonne de l'historique), sans le panneau : les images de la
        série côte à côte, étiquetées — les valeurs des axes, le prompt en titre —, écrites en PNG. Un chemin en .png est
        écrit, ou remplacé ; un dossier (il doit exister) la prend sous son propre nom,
        grid-<modèle>-<colonnes>x<rangées>-<W>x<H>[-<graine>].png, jamais sur un fichier (-2, -3…). N'importe quelle série
        de la session, une « grid » ou une de la fenêtre, finie ou non : une case pas encore rendue, ou retirée de
        l'historique, est dessinée vide, avec la raison. Réponse une fois le fichier écrit :
          {"ok": true, "grid": "g2", "path": "/abs/out/grid-z-image-2x2-512x512-42.png"}
        La planche n'est jamais une image de l'historique : les images de la série y sont déjà, une à une, comme celles de
        tout rendu (chacune avec "grid" et "cell") ; la planche ne fait que les disposer. Une case se nomme gNcM (help grid),
        et son job tN. « grid --wait --out D » et « wait g2 --out D » l'écrivent aussi, après la dernière case.
        ÉCHECS : usage, unknown_job (pas de telle grille dans la session), cell (aucune case n'a encore d'image : l'attendre),
          write.
        """),
    ]
}
