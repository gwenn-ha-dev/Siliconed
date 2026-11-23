// The help's pages of the settings and the window: set, Settings, the starting image, a bracket, the rack, a snapshot, measure, open.

extension RemoteControlHelp {
    static let settingsPages: [String: Page] = [
        "open": Page(en: """
        silicontrol open

        Launches Siliconed.app in the background (without bringing it to the front) if it is not
        answering, then waits until it answers (30 s at most). No effect if it is already open.
        Every other command does this too when the app is not running: `open` is the way to do
        only that.

        Answer: {"ok": true, "message": "Siliconed open" | "Siliconed was already open", "socket": "<socket>"}
        Failures: app_not_found (the app cannot be found) · app_not_running (it does not
                  answer after 30 s). Exit 3.

        This command is handled by the client, not by the app.
        """, fr: """
        silicontrol open

        Lance Siliconed.app en arrière-plan (sans lui donner le premier plan) si elle ne répond pas,
        puis attend qu'elle réponde (30 s au plus). Sans effet si elle est déjà ouverte.
        Toute autre commande en fait autant quand l'app n'est pas ouverte : `open` sert à ne
        faire que cela.

        Réponse : {"ok": true, "message": "Siliconed open" | "Siliconed was already open", "socket": "<socket>"}
        Échecs  : app_not_found (l'app est introuvable) · app_not_running (elle ne répond
                  pas après 30 s). Sortie 3.

        Cette commande est traitée par le client, pas par l'app.
        """),

        "set": Page(en: """
        silicontrol set beta [OPTION on|off]
        silicontrol set neural-engine [on|off]
        silicontrol set save-images [FOLDER|off]
        silicontrol set glass [on|off]
        silicontrol set updates [on|off]
        silicontrol set memory [GB|default]
        silicontrol set preview [on|off]

        Settings' switches, and the DiT box's « Preview every step », from the terminal. Each goes through the function its switch calls, so the choice is
        written and the window follows, exactly as a click.

        BETA is the four switches of Settings › Advanced › Beta, off by default:
          more-solvers      the solvers Euler a, DPM++ 2M and UniPC (--solver of « add »)
          steps             Choose Steps: the step count in the DiT box, for step-distilled LoRAs (--steps of « add »
                            takes any count the model has a schedule for, on or off)
          size-scales       Size in Two Scales: the Size menu as a proportion, then 1K or 2K (3:4@2K, help add)
          negative-lora     LoRA strengths from −1.5 to 1.5, for "slider" LoRAs (--lora NAME:-0.5)
        "set beta" alone reads them; "set beta OPTION on|off" sets one (the setting's own name works too:
        size-in-two-scales, negative-lora-strengths, choose-steps). The effects are the switch's: turning More Solvers
        off puts a solver back to Euler; turning Choose Steps off puts the rack's steps back to the model's recipe;
        turning Negative LoRA Strengths off brings the strengths already set back to 0.
        Like the switches, they wait for nothing and apply to what is queued next. They are this Mac's preferences; when
        the window computes on another Mac, its copy is set too. « status » gives them as "beta".

        NEURAL-ENGINE is Settings › General's « Neural Engine » switch, on by default: the engine gives the Neural Engine
        the work it does faster on this Mac. It decides alone how much it gives, from the stage's measured time on this
        Mac, and gives none where it does not help; off, the GPU alone computes. Either way the values lie as close to
        their references. "on" or "off"; alone, it is read.
        This Mac's preference, for what this Mac computes next: a render it lends to another Mac follows its switch, a
        render it borrows from another Mac follows that Mac's. « status » gives it as "neural_engine"; each image's line
        carries the Neural Engine's share of the render as "neural_engine" (0 to 1), absent when it was not used.

        SAVE-IMAGES is Settings › General's « Save Every Image To… », off by default: given a folder (relative to the current
        directory; one named "off" is ./off), each new image of the history — a render, a grid's cell, a crop, an
        enlargement — is also written there as a PNG with its prompt, seed and model, under its
        « save » name, never over another file. "off" turns it off; alone, it is read. The folder must exist ("write"
        otherwise): it is never created. This Mac's preference, whichever Mac computes.

        GLASS is Settings › General › Window's « Glass », on by default: the chain's column and the history's lie over the
        image as glass, and a zoomed image shows through them; off, they are opaque, and nothing is composed again under
        them as the image moves (the glass costs a little time then, more on a small Mac). "on" or "off"; alone, it is read.
        This Mac's preference.

        UPDATES is Settings › General › Updates' « Check at Launch », on by default: each launch asks GitHub for the number of
        the latest version — one request, nothing of the user's in it — and a newer one is said in a small panel, with its
        disk image and the site. Siliconed › Check for Updates… asks whatever it says. "on" or "off"; alone, it is read.

        MEMORY is Settings › Macs › This Mac's « Memory Limit »: B, the most the engine holds while it renders, in
        gigabytes (of 2³⁰ bytes). By default ¾ of the memory less 2 GB; it may be set from 4 GB to the memory less 4 — a
        Mac of 8 GB has no choice, one of 16 goes from 4 to 12. "default" gives the default back; alone, it is read. A
        render never goes past it: more lets larger images and models run whole, less leaves more to the other apps. The
        engine's next process takes it; a render under way keeps the B it began under. « status » gives it as "memory".

        PREVIEW is the DiT box's « Preview every step », on by default: the canvas shows the image taking shape after each step;
        it does not change the image. "on" or "off"; alone, it is read. It is the window's, for the session (this Mac's, or the
        Mac it computes on), and applies to the renders queued next; « add --preview » and « --no-preview » still decide for one job.

        Answer: {"ok": true, "option": "size-scales", "enabled": true, "previous": false,
                 "beta": {"more_solvers": false, "steps": false, "size_scales": true, "negative_lora": false}}
                {"ok": true, "neural_engine": true, "previous": false}
                {"ok": true, "save_images": "/Users/me/Pictures/Siliconed", "previous": null}
                {"ok": true, "glass": false, "previous": true}
                {"ok": true, "updates": false, "previous": true}
                {"ok": true, "memory": {"budget_bytes": 12884901888, "chosen": true, …}, "previous": {…}}
                {"ok": true, "preview": false, "previous": true}
        FAILURES: usage (set alone or an unknown subcommand, the message gives every usage; an unknown Beta option, "hint"
        lists them; neural-engine, glass, updates or preview other than on, off; a memory out of this Mac's range), write
        (save-images: not a folder).
        """, fr: """
        silicontrol set beta [OPTION on|off]
        silicontrol set neural-engine [on|off]
        silicontrol set save-images [FOLDER|off]
        silicontrol set glass [on|off]
        silicontrol set updates [on|off]
        silicontrol set memory [Go|default]
        silicontrol set preview [on|off]

        Les interrupteurs des Réglages, et l'« Aperçu à chaque pas » de la boîte du DiT, depuis le terminal. Chacun passe par la fonction que son interrupteur appelle : le
        choix est écrit et la fenêtre suit, exactement comme un clic.

        BETA, ce sont les quatre interrupteurs de Réglages › Avancé › Bêta, éteints par défaut :
          more-solvers      les solveurs Euler a, DPM++ 2M et UniPC (--solver de « add »)
          steps             Choisir les pas : le nombre de pas dans la boîte DiT, pour les LoRA distillées en moins de
                            pas (--steps de « add » prend tout nombre que le modèle sait faire, allumé ou non)
          size-scales       Taille en deux échelles : le menu Taille en proportion, puis 1K ou 2K (3:4@2K, help add)
          negative-lora     forces de LoRA de −1,5 à 1,5, pour les LoRA « curseur » (--lora NOM:-0.5)
        « set beta » seul les lit ; « set beta OPTION on|off » en règle une (le nom du réglage marche aussi :
        size-in-two-scales, negative-lora-strengths, choose-steps). Les effets sont ceux de l'interrupteur : éteindre More
        Solvers remet un solveur sur Euler ; éteindre Choose Steps remet les pas du rack à la recette du modèle ; éteindre
        Negative LoRA Strengths ramène à 0 les forces déjà réglées. Comme les interrupteurs, elles
        n'attendent rien et valent pour ce qui se met en file ensuite. Ce sont les préférences de ce Mac ; quand la fenêtre
        calcule sur un autre, sa copie est réglée aussi. « status » les donne sous "beta".

        NEURAL-ENGINE, c'est l'interrupteur « Neural Engine » de Réglages › Général, allumé par défaut : le moteur confie au
        Neural Engine le travail qu'il fait plus vite sur ce Mac. Il décide seul de la part qu'il donne, d'après le temps
        mesuré de l'étage sur ce Mac, et n'en donne pas là où elle n'aide pas ; éteint, le GPU calcule seul. Allumé ou
        éteint, les valeurs restent aussi près de leurs références. « on » ou « off » ; seul, il est lu.
        Préférence de ce Mac, pour ce qu'il calcule ensuite : un rendu qu'il prête à un autre Mac suit son interrupteur, un
        rendu qu'il emprunte à un autre Mac suit celui de cet autre Mac. « status » le donne sous "neural_engine" ; la ligne
        de chaque image porte la part du Neural Engine dans le rendu, "neural_engine" (0 à 1), absente s'il n'a pas servi.

        SAVE-IMAGES, c'est « Enregistrer chaque image dans… » de Réglages › Général, éteint par défaut : avec un dossier
        (relatif au répertoire courant ; un dossier nommé « off » s'écrit ./off), chaque image nouvelle de l'historique — un
        rendu, la case d'une grille, un recadrage, un agrandissement — s'y écrit
        aussi en PNG avec son prompt, sa graine et son modèle, sous son nom de « save », jamais sur un autre fichier. « off »
        l'éteint ; seul, il est lu. Le dossier doit exister (sinon « write ») : il n'est jamais créé. Préférence de ce Mac,
        quel que soit le Mac qui calcule.

        GLASS, c'est « Verre » de Réglages › Général › Fenêtre, allumé par défaut : la colonne de la chaîne et celle de
        l'historique passent sur l'image en verre, et une image zoomée se voit au travers ; éteint, elles sont opaques, et rien
        ne se recompose sous elles quand l'image bouge (le verre coûte alors un peu de temps, plus sur un petit Mac). « on » ou
        « off » ; seul, il est lu. Préférence de ce Mac.

        UPDATES, c'est « Vérifier au lancement » de Réglages › Général › Mises à jour, allumé par défaut : chaque lancement
        demande à GitHub le numéro de la dernière version — une requête, rien de l'utilisateur dedans — et une version plus
        récente se dit dans un petit panneau, avec son image disque et le site. Siliconed › Rechercher les mises à jour…
        demande quoi qu'il dise. « on » ou « off » ; seul, il est lu.

        MEMORY, c'est « Limite de mémoire » de Réglages › Mac › Ce Mac : B, le plus que le moteur tient pendant un rendu,
        en gigaoctets (de 2³⁰ octets). Par défaut ¾ de la mémoire moins 2 Go ; il se règle de 4 Go à la mémoire moins 4 — un
        Mac de 8 Go n'a pas le choix, un de 16 va de 4 à 12. « default » rend le défaut ; seul, il est lu. Un rendu ne le
        dépasse jamais : plus laisse des images et des modèles plus grands tourner entiers, moins en laisse aux autres apps.
        Le processus suivant du moteur le prend ; un rendu en cours garde le B sous lequel il a commencé. « status » le donne
        en "memory".

        PREVIEW, c'est l'« Aperçu à chaque pas » de la boîte du DiT, allumé par défaut : le canevas montre l'image qui se forme
        après chaque pas ; il ne change pas l'image. « on » ou « off » ; seul, il est lu. Il est à la fenêtre, pour la session
        (ce Mac, ou le Mac sur lequel elle calcule), et vaut pour les rendus mis en file ensuite ; « add --preview » et
        « --no-preview » décident toujours pour un job.

        Réponse : {"ok": true, "option": "size-scales", "enabled": true, "previous": false,
                   "beta": {"more_solvers": false, "steps": false, "size_scales": true, "negative_lora": false}}
                  {"ok": true, "neural_engine": true, "previous": false}
                  {"ok": true, "save_images": "~/Pictures/Siliconed", "previous": null}
                  {"ok": true, "glass": false, "previous": true}
                  {"ok": true, "updates": false, "previous": true}
                  {"ok": true, "memory": {"budget_bytes": 12884901888, "chosen": true, …}, "previous": {…}}
                  {"ok": true, "preview": false, "previous": true}
        ÉCHECS : usage (set seul ou une sous-commande inconnue, le message donne tous les usages ; une option Bêta inconnue,
        "hint" les liste ; neural-engine, glass, updates ou preview autre que on, off ; une mémoire hors de la plage de ce
        Mac), write (save-images : pas un dossier).
        """),

        "settings": Page(en: """
        silicontrol settings [general|macs|lab|advanced]
        silicontrol settings lab [rewrite|describe] [--for ID]

        Siliconed › Settings… (⌘,), opened on one of its four panes: general (where the library is, Neural Engine, Save Every Image To…, Glass, Check at Launch),
        macs (this Mac — Memory Limit, Compiled for This Mac —, the other Macs), lab (Prompt Lab: the writer's instructions),
        advanced (Command Line, Beta).
        Without a pane, the one last shown. The window comes forward as the menu brings it; nothing in it changes.

          silicontrol settings macs
          silicontrol settings lab rewrite --for krea2      Prompt Lab on Krea 2's Rewrite instruction
          silicontrol settings lab describe                 the same model, its Describe

        lab with a gesture or --for: Prompt Lab on one instruction, its two menus set as a click on the Prompt Lab's
        « Your Instruction » sets them. --for ID is a model's id or a family, installed or not; without it, the model the
        Lab writes for. A menu not named stays as it is.

        Answer: {"ok": true, "settings": {"pane": "lab", "for": "krea2", "gesture": "rewrite"}}   ("for", "gesture": lab only)
        What the panes set, commands set: set (beta, neural-engine, save-images, glass, updates, memory), library, macs, compute, instructions.
        Allowed while the window computes on another Mac: Settings are this Mac's.
        FAILURES: usage (a pane that does not read; a gesture or --for on a pane other than lab), model (--for names no model).
        """, fr: """
        silicontrol settings [general|macs|lab|advanced]
        silicontrol settings lab [rewrite|describe] [--for ID]

        Siliconed › Réglages… (⌘,), ouvert sur l'un de ses quatre volets : general (Général : où est la bibliothèque,
        Neural Engine, Enregistrer chaque image dans…, Verre, Vérifier au lancement), macs (Mac : ce Mac — Limite de mémoire, Compilé pour ce Mac —, les autres Mac), lab (Atelier de prompts : les consignes du
        rédacteur), advanced (Avancé : Ligne de commande, Bêta). Sans volet, le dernier montré. La fenêtre
        passe devant comme le menu l'amène ; rien n'y change.

          silicontrol settings macs
          silicontrol settings lab rewrite --for krea2      l'Atelier de prompts sur la consigne Réécrire de Krea 2
          silicontrol settings lab describe                 le même modèle, son Décrire

        lab avec un geste ou --for : l'Atelier de prompts sur une consigne, ses deux menus posés comme les pose un clic sur
        « Votre consigne » dans l'Atelier. --for ID est l'id d'un modèle ou une famille, installé ou non ; sans lui, le modèle
        pour lequel écrit l'Atelier. Un menu qu'on ne nomme pas reste comme il est.

        Réponse : {"ok": true, "settings": {"pane": "lab", "for": "krea2", "gesture": "rewrite"}}   ("for", "gesture" : lab seulement)
        Ce que règlent les volets, des commandes le règlent : set (beta, neural-engine, save-images, glass, updates, memory), library, macs, compute, instructions.
        Permis quand la fenêtre calcule sur un autre Mac : les Réglages sont ceux de ce Mac.
        ÉCHECS : usage (un volet qui ne se lit pas ; un geste ou --for sur un autre volet que lab), model (--for ne nomme
        aucun modèle).
        """),

        "start-image": Page(en: """
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
        alone, with no image carried), ref (unreadable or missing image), unknown_image, usage.
        On a Mac followed (Compute On), the same: the image goes to the other Mac with each render, an image of its
        history (i7) by its number.
        """, fr: """
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
        sans image portée), ref (image illisible ou absente), unknown_image, usage.
        Sur un Mac suivi (Calculer sur), pareil : l'image part vers l'autre Mac avec chaque rendu, une image de son
        historique (i7) par son numéro.
        """),

        "bracket": Page(en: """
        silicontrol bracket seed|detail|solver|start|loras
        silicontrol bracket lora <name>
        silicontrol bracket close

        A click on a knob's yellow ± (or right-click › « Bracket… »): its bubble opens beside it, with the values the knob would
        take (three around the current one to begin with) — the window's own bubble, to be read or captured; nothing varies
        until it is committed there. seed: the Seed line's ± (how many seeds, the Seeds count, and on which axis); start: the starting image's strength; loras: the LoRA
        stack's « Compare LoRAs… »; lora <name>: one LoRA's strength, by its name in the stack as the rack's "loras" print it (start-image, reuse).
        close: the bubble closed, the values as they were. To make a series from the terminal: grid (its axes become the
        window's series, help grid).

          silicontrol bracket detail
          silicontrol bracket lora incase
          silicontrol bracket close

        Answer: {"ok": true, "bracketing": "detail"}   ("bracketing": null after close)
        FAILURES: usage, lora (no LoRA of that name in the stack: the names it holds are listed), bracket (no ± that would open
        under the current model and settings: Detail on a model without it, no starting image, nothing to compare, a LoRA off).
        """, fr: """
        silicontrol bracket seed|detail|solver|start|loras
        silicontrol bracket lora <nom>
        silicontrol bracket close

        Un clic sur le ± jaune d'un réglage (ou clic droit › « Bracketer… ») : sa bulle s'ouvre à côté, avec les valeurs que
        prendrait le réglage (trois autour de l'actuelle pour commencer) — la bulle de la fenêtre elle-même, à lire ou à
        capturer ; rien ne varie avant qu'on la valide. seed : le ± de la ligne Graine (combien de graines, le nombre des Graines, et sur quel axe) ; start : la force de l'image de
        départ ; loras : « Comparer des LoRA… » de la pile ; lora <nom> : la force d'une LoRA, par son nom dans la pile tel que
        l'écrivent les "loras" du rack (start-image, reuse). close : la bulle fermée, les valeurs comme avant. Pour faire une série depuis le terminal :
        grid (ses axes deviennent la série de la fenêtre, help grid).

          silicontrol bracket detail
          silicontrol bracket lora incase
          silicontrol bracket close

        Réponse : {"ok": true, "bracketing": "detail"}   ("bracketing" : null après close)
        ÉCHECS : usage, lora (aucune LoRA de ce nom dans la pile : les noms qu'elle tient sont listés), bracket (aucun ± qui
        s'ouvrirait sous le modèle et les réglages du moment : Détail sur un modèle qui n'en a pas, pas d'image de départ, rien
        à comparer, une LoRA éteinte).
        """),

        "rack": Page(en: """
        silicontrol rack top|dit|fold|unfold|detach|dock

        The window's one column — the Text and Images boxes, then the chain — scrolled as the hand scrolls it: top, to its head
        (the two boxes and the encoders); dit, to the DiT's box and its settings — the encoders then out of sight, the start of
        the text and the chain pinned in small at the top of the column, its stage at work lit, a click on it scrolling back.
        fold and unfold are the toolbar's button beside the window's lights, View › Show / Hide Chain (⌥⌘R): folded, the column
        is gone and the image takes its room; a scroll (top, dit), a problem to say, or a ⌘↩ that something stops unfolds it.
        Folding stops a Rewrite or a Describe under way. Nothing in the rack changes. With the chain detached (the toolbar's
        « Detach the Chain »: the column in a floating window of its own), unfold brings that window forward.
        detach and dock are that button, « Detach the Chain » and « Dock the Chain »: the column goes into its floating window,
        unfolded, or back into the main window, on the left; closing that window docks it too. The choice is kept across launches.

        Answer: {"ok": true, "rack": "dit", "folded": false, "detached": false}
        FAILURES: usage (a place that does not read).
        """, fr: """
        silicontrol rack top|dit|fold|unfold|detach|dock

        La colonne unique de la fenêtre — les boîtes Texte et Images, puis la chaîne — défilée comme la main la fait défiler :
        top, jusqu'à sa tête (les deux boîtes et les encodeurs) ; dit, jusqu'à la boîte du DiT et ses réglages — les encodeurs
        alors hors de vue, le début du texte et la chaîne épinglés en petit en haut de la colonne, l'étage qui travaille allumé,
        un clic dessus ramenant en haut. fold et unfold sont le bouton de la barre d'outils à côté des feux de la fenêtre,
        Présentation › Afficher / Masquer la chaîne (⌥⌘R) : repliée, la colonne disparaît et l'image prend sa place ; un
        défilement (top, dit), un problème à dire, ou un ⌘↩ que quelque chose arrête la déplie. Replier arrête un Réécrire ou
        un Décrire en cours. Rien ne change dans le rack. La chaîne détachée (« Détacher la chaîne » dans la barre d'outils :
        la colonne dans une fenêtre flottante à elle), unfold amène cette fenêtre devant.
        detach et dock sont ce bouton, « Détacher la chaîne » et « Ancrer la chaîne » : la colonne passe dans sa fenêtre flottante,
        dépliée, ou revient dans la fenêtre principale, à gauche ; fermer cette fenêtre l'ancre aussi. Le choix se garde d'un
        lancement à l'autre.

        Réponse : {"ok": true, "rack": "dit", "folded": false, "detached": false}
        ÉCHECS : usage (une place qui ne se lit pas).
        """),

        "zoom": Page(en: """
        silicontrol zoom in|out|fit

        View › Zoom In, Zoom Out, Zoom to Fit (⌘+ ⌘− ⌘0), on what the canvas shows. On a series' grid: its cells larger or
        smaller, or back to the size that fills the canvas, following the window again. Otherwise the chosen image: zoomed
        by 1.5 up to its pixels' limit, back towards fitted, or fitted. Where the menu item is greyed (nothing shown, the image
        already fitted, the cells already filling the canvas), nothing changes and the answer says what is shown.

        Answer: {"ok": true, "image": "i7", "zoom": 1.5, "fits": false}
                {"ok": true, "grid": "g2", "cell": 275, "fits": false}   (a series shown; cell: a cell's side, in points)
        FAILURES: usage (a step that does not read).
        """, fr: """
        silicontrol zoom in|out|fit

        Présentation › Zoom avant, Zoom arrière, Taille ajustée (⌘+ ⌘− ⌘0), sur ce que montre le canevas. Sur la grille d'une
        série : ses cases plus grandes ou plus petites, ou revenues à la taille qui remplit le canevas, qui suit de nouveau la
        fenêtre. Sinon l'image choisie : agrandie de 1,5 jusqu'à la limite de ses pixels, ramenée vers l'ajustée, ou ajustée.
        Là où l'article du menu est grisé (rien d'affiché, l'image déjà ajustée, les cases remplissant déjà le canevas), rien ne
        change et la réponse dit ce qui est affiché.

        Réponse : {"ok": true, "image": "i7", "zoom": 1.5, "fits": false}
                  {"ok": true, "grid": "g2", "cell": 275, "fits": false}   (une série affichée ; cell : le côté d'une case, en points)
        ÉCHECS : usage (un pas qui ne se lit pas).
        """),

        "snapshot": Page(en: """
        silicontrol snapshot <file.png> [--window WORD|N] [--appearance light|dark]

        One of the app's windows, drawn by the app itself into a PNG: its frame, title bar and toolbar included, at the screen's
        scale (a Retina screen gives twice the points). It is not a screen capture — only the app's own drawing is read —, so
        it asks for no Screen Recording permission and works from a session without a display (ssh), where screencapture
        cannot. The file is written where the command says, replacing one of the same name; .png is added when missing.

          silicontrol snapshot window.png                       # the main window
          silicontrol snapshot light.png --appearance light     # the same, in the light appearance, whatever the system's
          silicontrol snapshot chain.png --window Chain         # another window, by a word of its title (the Chain inspector, ⌥⌘I)
          silicontrol snapshot bubble.png --window 3            # or by its number in "windows" (a sheet, a popover: untitled)

        --appearance draws the window in that appearance and puts its own back: nothing stays changed.
        What the window server composes is not in the picture: glass and materials come as their plain fill, shadows not at all.
        Each window listed carries its "id", the window server's number: screencapture -l <id> captures it as the screen shows it.

        Answer: {"ok": true, "path": "/…/window.png", "window": "Siliconed", "pixels": [2960, 1880], "appearance": null,
                 "windows": [{"n": 1, "title": "Siliconed", "width": 1480, "height": 940, "main": true, "id": 412}, …]}
        FAILURES: usage, window (no window of that word or number is shown: the hint lists those that are), snapshot (the
        window could not be drawn), write (the file could not be written).
        """, fr: """
        silicontrol snapshot <fichier.png> [--window MOT|N] [--appearance light|dark]

        Une des fenêtres de l'app, dessinée par l'app elle-même dans un PNG : son cadre, sa barre de titre et sa barre d'outils
        compris, à l'échelle de l'écran (un écran Retina donne le double des points). Ce n'est pas une capture d'écran — seul
        le dessin de l'app est lu —, donc aucune autorisation d'Enregistrement de l'écran n'est demandée et la commande marche
        depuis une session sans écran (ssh), là où screencapture ne peut pas. Le fichier est écrit où la commande le dit, en
        remplaçant celui du même nom ; .png est ajouté s'il manque.

          silicontrol snapshot fenetre.png                      # la fenêtre principale
          silicontrol snapshot clair.png --appearance light     # la même, en apparence claire, quelle que soit celle du système
          silicontrol snapshot chaine.png --window Chaîne       # une autre fenêtre, par un mot de son titre (l'inspecteur de la chaîne, ⌥⌘I)
          silicontrol snapshot bulle.png --window 3             # ou par son numéro dans "windows" (une feuille, une bulle : sans titre)

        --appearance dessine la fenêtre dans cette apparence et lui rend la sienne : rien ne reste changé.
        Ce que le serveur de fenêtres compose n'est pas dans l'image : verre et matériaux viennent en aplat, les ombres pas du tout.
        Chaque fenêtre listée porte son "id", le numéro du serveur de fenêtres : screencapture -l <id> la capture telle que l'écran la montre.

        Réponse : {"ok": true, "path": "/…/fenetre.png", "window": "Siliconed", "pixels": [2960, 1880], "appearance": null,
                   "windows": [{"n": 1, "title": "Siliconed", "width": 1480, "height": 940, "main": true, "id": 412}, …]}
        ÉCHECS : usage, window (aucune fenêtre de ce mot ou de ce numéro n'est affichée : l'indice liste celles qui le sont),
        snapshot (la fenêtre n'a pas pu être dessinée), write (le fichier n'a pas pu être écrit).
        """),

        "measure": Page(en: """
        silicontrol measure [--model ID] [--size 512|1024|WxH] [--renders N] [--seed S] [--prompt TEXT]…
                            [--tsv FILE] [--out DIR] [--settle SECONDS] [--tag TEXT] [-- add options]

        Repeats our timings and peaks on your Mac, with this app alone: a series of renders, each timed
        from the request to the PNG, the app's peak memory sampled while it renders, the pixels hashed.
        Defaults: z-image, 1024x1024, 3 renders, seed 42, "a 30 year old woman posing in a library",
        15 s of rest first. --size is read as « add » reads a size (832x1216, 832 × 1216, 1024 a square); whether
        the model renders it, the app says at the first render.

        The series: the queue must be idle (else "busy"); a load render (the weights come in; its time
        is reported apart); a witness; the N renders; a witness again. The witness is the default prompt
        at seed 42; the machine drifts by a few percent over an hour, and the two witnesses bound that
        drift: a series whose witness moved by more than 13 % is not comparable to another. The Mac
        is kept awake for the series. Each render:
          wall_s       from the request to the PNG, on this command's clock (what we publish)
          engine_s     the app's own "seconds" for the image
          peak_GiB     the phys_footprint of the app and of its engine's process (silicore-worker) together,
                       read every 100 ms, the maximum (what we publish)
          start_GiB    the footprint just before the request: what the app and its engine held already
          swapouts     pages swapped out during the render (vm_stat's Swapouts); disk_read_GB the reads of both
          md5          of the decoded 8-bit RGB pixels: within a series with the Neural Engine off, the same model,
                       prompt, seed, size and settings must give the same md5 (a mismatch is a defect: report it); with
                       it on its share may move between renders, and the pixels with it; from one macOS to
                       the next the bits may change
        beside the model, variant, steps, seed, app version, chip, memory, macOS, power
        source and thermal state, and what the app said as the render was asked:
          neural_engine  its Neural Engine switch, on or off (help set)
          budget         its memory limit B, in GiB (help set)
        A render's prompt is encoded once and cached: with several renders on
        one prompt the encoder runs only the first time; give several --prompt to include it each time.

        Output: stdout, JSON lines — one {"type": "render", …} per render, then the verdict
        {"ok": true, "summary": {"median_seconds", "witness_head_seconds", "witness_tail_seconds",
        "witness_drift", "peak_gib", "swapouts", "bits_stable", "neural_engine", "budget_bytes", "notes", …},
        "tsv": "<path>"}. stderr: the same, for a person, the switch and B on its first line and in the summary. The TSV (default siliconed-measure-<date>-<--model, or default>-<size>.tsv in
        the current folder) has one line per render, the witnesses and the load included.
        --out keeps the PNGs there; otherwise they are deleted after they are hashed.
        --tag labels the series in the TSV. After --, options go to "add" as they are (--steps, --lora);
        not --wait, --out, --batch, --ref, --model, --preview, --no-preview.

        For a comparison to mean something: the adapter plugged in, no other job on the Mac, the app
        freshly opened for a memory figure (a peak read after other models ran includes what they left).

        FAILURES: usage (exit 2), app_not_running (exit 3), busy (a render or an installation is running), model (unknown
        or not installed), render (the render failed: its "error" and message), write. The rows measured before a failure stay in the TSV.
        """, fr: """
        silicontrol measure [--model ID] [--size 512|1024|WxH] [--renders N] [--seed S] [--prompt TEXTE]…
                            [--tsv FICHIER] [--out DOSSIER] [--settle SECONDES] [--tag TEXTE] [-- options de add]

        Refait nos temps et nos pics chez vous, avec cette seule app : une série de rendus, chacun chronométré
        de la demande au PNG, le pic mémoire de l'app échantillonné pendant qu'elle rend, les pixels hachés.
        Défauts : z-image, 1024x1024, 3 rendus, graine 42, « a 30 year old woman posing in a library »,
        15 s de repos d'abord. --size se lit comme « add » lit un format (832x1216, 832 × 1216, 1024 un carré) ;
        si le modèle le rend, l'app le dit au premier rendu.

        La série : la file doit être libre (sinon « busy ») ; un rendu de chargement (les poids entrent ; son temps
        est donné à part) ; un témoin ; les N rendus ; un témoin encore. Le témoin est le prompt par défaut à la
        graine 42 ; la machine dérive de quelques pour cent en une heure, et les deux témoins bornent cette
        dérive : une série dont le témoin a bougé de plus de 13 % ne se compare pas à une autre. Le Mac reste
        éveillé pendant la série. Pour chaque rendu :
          wall_s       de la demande au PNG, sur l'horloge de cette commande (ce que nous publions)
          engine_s     les « seconds » de l'app pour l'image
          peak_GiB     le phys_footprint de l'app et du processus de son moteur (silicore-worker) ensemble,
                       lu toutes les 100 ms, le maximum (ce que nous publions)
          start_GiB    l'empreinte juste avant la demande : ce que l'app et son moteur tenaient déjà
          swapouts     pages écrites en swap pendant le rendu (Swapouts de vm_stat) ; disk_read_GB les lectures des deux
          md5          des pixels décodés en RGB 8 bits : dans une série le Neural Engine éteint, même modèle, prompt,
                       graine, format et réglages doivent donner le même md5 (un écart est un défaut : le signaler) ;
                       allumé, sa part peut bouger d'un rendu à l'autre, et les pixels avec elle ;
                       d'un macOS au suivant les bits peuvent changer
        avec le modèle, la version, les pas, la graine, la version de l'app, la puce, la mémoire,
        macOS, la source d'alimentation et l'état thermique, et ce que l'app disait quand le rendu a été demandé :
          neural_engine  son interrupteur Neural Engine, on ou off (help set)
          budget         sa limite de mémoire B, en Gio (help set)
        Le prompt d'un rendu est encodé une fois puis gardé :
        avec plusieurs rendus du même prompt, l'encodeur ne tourne que la première fois ; plusieurs --prompt
        l'incluent à chaque fois.

        Sortie : stdout, des lignes JSON — une {"type": "render", …} par rendu, puis le verdict
        {"ok": true, "summary": {"median_seconds", "witness_head_seconds", "witness_tail_seconds",
        "witness_drift", "peak_gib", "swapouts", "bits_stable", "neural_engine", "budget_bytes", "notes", …},
        "tsv": "<chemin>"}. stderr : la même chose, pour une personne, l'interrupteur et B sur sa première ligne et dans le résumé. Le TSV (par défaut siliconed-measure-<date>-<--model, ou default>-<format>.tsv
        dans le dossier courant) a une ligne par rendu, témoins et chargement compris.
        --out garde les PNG là ; sinon ils sont effacés une fois hachés.
        --tag nomme la série dans le TSV. Après --, les options vont à « add » telles quelles (--steps, --lora) ;
        pas --wait, --out, --batch, --ref, --model, --preview, --no-preview.

        Pour qu'une comparaison veuille dire quelque chose : le secteur branché, aucun autre travail sur le Mac,
        l'app fraîchement ouverte pour un chiffre de mémoire (un pic lu après d'autres modèles inclut ce qu'ils ont laissé).

        ÉCHECS : usage (sortie 2), app_not_running (sortie 3), busy (un rendu ou une installation tourne), model (inconnu
        ou pas installé), render (le rendu a échoué : son « error » et son message), write. Les lignes déjà mesurées restent dans le TSV.
        """),
    ]
}
