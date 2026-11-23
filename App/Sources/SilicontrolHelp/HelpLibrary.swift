// The help's pages of the models and the library: install, import, uninstall, the library's place, the compiled programs, the other Macs.

extension RemoteControlHelp {
    static let libraryPages: [String: Page] = [
        "models": Page(en: """
        silicontrol models

        The models the app shows, installed or not: Z-Image Turbo, Anima Turbo, Krea 2 Turbo,
        MOSS-Image and Qwen-Image-2.1 Turbo (« add » refuses a model the app does not show). Read it
        before « add »: the ids and the formats come from here.

        Answer:
          {"ok": true, "models": [
            {"id": "z-image", "name": "Z-Image Turbo", "ready": true, "default": true, "default_steps": 8,
             "formats": ["1024x1024", "832x1216", …], "commercial_license": true,
             "license": "Apache 2.0", "license_accepted": true,
             "license_urls": ["https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/blob/<sha>/README.md"],
             "edit": false, "max_references": 0, "variant": "compact",
             "variants": [{"variant": "standard", "disk_bytes": 20357263360, "installed": false},
                          {"variant": "compact", "disk_bytes": 11749122176, "installed": true}],
             "loras": []},
            {"id": "krea2", "ready": false, "default": false, "edit": null, "max_references": null,
             "variant": null, "missing": "<the absent file>", "install": "silicontrol install krea2", …}, …]}

        id          → --model of « add »; "default": true = the one taken without --model
        variants    → the versions a model installs in, each with its space on disk and whether
                      it is whole there ("installed"): "standard" (the publisher's weights, bf16)
                      and "compact" (the default; 8-bit weights as published, by the publisher's
                      distributor or by third parties: smaller on disk, an image slightly
                      different). Both can be installed side by side; they share the small files
                      (tokenizer, VAE, and Anima's text encoder, which has no 8-bit publication)
        variant     → the version the model renders with (install --standard|--compact switches
                      between two installed versions); null while the model is not installed
        loras.lora  → --lora of « add » (only the LoRAs of THIS model)
        loras.triggers → the words that call the LoRA, to put in the prompt; only when its file
                      states them (many state none: see their publisher's page)
        edit        → the model accepts --ref (editing an image with a prompt); null while it is
                      not installed
        max_references → how many --ref it reads at most (0: no model here edits yet);
                      null while it is not installed
        start_image → "start_image": true when the model starts from an image (img2img: add --image,
                      start-image) — Z-Image, Anima, Krea 2, MOSS-Image; absent while it is not installed
        formats     → the suggested formats; any valid WxH is accepted (help add)
        license_urls → the license file(s) on Hugging Face, at the revision the installation takes:
                      the model's, then each component under a license of its own (a Compact: the
                      publisher's, then each third party's model card). Z-Image has none: its model card, which
                      declares Apache 2.0.
        commercial_license → false: the model is for non-commercial use (Anima: CircleStone Labs
                      Non-Commercial — its images may still be used commercially, the license says
                      so). Krea 2 Community is true with
                      conditions: under $1M of annual revenue, a content filter wherever the model
                      is deployed
        license_accepted → false: the first render of this model waits until the user accepts its
                      license in the app (a sheet shows it); declined, the job ends in "error"
                      with "code": "license_not_accepted"
        A model with "ready": false cannot render; its "install" field is the command that offers
        it in the app (help install) — the user reads the license and the size, and accepts there.
        """, fr: """
        silicontrol models

        Les modèles que l'app montre, installés ou non : Z-Image Turbo, Anima Turbo, Krea 2 Turbo,
        MOSS-Image et Qwen-Image-2.1 Turbo (« add » refuse un modèle que l'app ne montre pas). À lire
        avant « add » : les ids et les formats viennent d'ici.

        Réponse :
          {"ok": true, "models": [
            {"id": "z-image", "name": "Z-Image Turbo", "ready": true, "default": true, "default_steps": 8,
             "formats": ["1024x1024", "832x1216", …], "commercial_license": true,
             "license": "Apache 2.0", "license_accepted": true,
             "license_urls": ["https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/blob/<sha>/README.md"],
             "edit": false, "max_references": 0, "variant": "compact",
             "variants": [{"variant": "standard", "disk_bytes": 20357263360, "installed": false},
                          {"variant": "compact", "disk_bytes": 11749122176, "installed": true}],
             "loras": []},
            {"id": "krea2", "ready": false, "default": false, "edit": null, "max_references": null,
             "variant": null, "missing": "<le fichier absent>", "install": "silicontrol install krea2", …}, …]}

        id          → --model de « add » ; "default": true = celui pris sans --model
        variants    → les versions dans lesquelles s'installe un modèle, chacune avec sa place sur
                      le disque et si elle y est entière ("installed") : "standard" (les poids de
                      l'éditeur, bf16) et "compact" (le défaut ; poids 8 bits tels que publiés, par
                      le distributeur de l'éditeur ou par des tiers : plus léger sur le disque, une
                      image légèrement différente). Les deux s'installent côte à côte ; elles
                      partagent les petits fichiers (tokenizer, VAE, et l'encodeur de texte d'Anima,
                      qui n'a pas de publication 8 bits)
        variant     → la version avec laquelle le modèle rend (install --standard|--compact passe de
                      l'une à l'autre de deux versions installées) ; null tant que le modèle n'est
                      pas installé
        loras.lora  → --lora de « add » (seulement les LoRA de CE modèle)
        loras.triggers → les mots qui appellent le LoRA, à mettre dans le prompt ; seulement quand
                      son fichier les donne (beaucoup n'en donnent pas : voir la page de l'auteur)
        edit        → le modèle accepte --ref (édition d'une image par un prompt) ; null tant
                      qu'il n'est pas installé
        max_references → combien de --ref il lit au plus (0 : aucun modèle n'édite encore ici) ;
                      null tant qu'il n'est pas installé
        start_image → "start_image": true quand le modèle part d'une image (img2img : add --image,
                      start-image) — Z-Image, Anima, Krea 2, MOSS-Image ; absent tant qu'il n'est pas installé
        formats     → les formats conseillés ; tout WxH valide est accepté (help add)
        license_urls → le(s) fichier(s) de licence sur Hugging Face, à la révision que l'installation
                      prend : celui du modèle, puis chaque composant sous une licence propre
                      (une Compact : celui de l'éditeur, puis la fiche de chaque tiers). Z-Image n'en a pas : sa
                      fiche, qui déclare Apache 2.0.
        commercial_license → false : le modèle est d'usage non commercial (Anima : CircleStone Labs
                      Non-Commercial — ses images restent utilisables commercialement, la licence
                      le dit). Krea 2 Community est true
                      sous conditions : moins d'un million de dollars de chiffre d'affaires annuel,
                      un filtre de contenu partout où le modèle est déployé
        license_accepted → false : le premier rendu de ce modèle attend que l'utilisateur accepte
                      sa licence dans l'app (une feuille la montre) ; refusée, le travail finit en
                      "error" avec "code": "license_not_accepted"
        Un modèle "ready": false ne peut pas rendre ; son champ "install" est la commande qui le
        propose dans l'app (help install) — l'utilisateur y lit la licence et la place, et accepte.
        """),

        "install": Page(en: """
        silicontrol install ID [--standard|--compact]
        silicontrol install stop

        Opens the app's Models sheet on the model ID (an "id" of « models »), with its license and
        what is left to download. NOTHING is downloaded by the command: the user reads, then
        clicks « Accept and Install » in the app (tens of GB, several minutes). The answer comes at
        once:
          {"ok": true, "model": "krea2", "installed": false, "variant": "compact", "message": "…"}
          {"ok": true, "model": "z-image", "installed": true, "variant": "standard", "message": "already installed"}
        Then « silicontrol models » says "ready": true once the installation is done, and « status »
        follows it ("installing": the file, the bytes received and the total).

        Without a flag, the version the model renders with is kept (none installed: the Compact,
        preselected — the user sees both): a command that names no version never switches it.
        --standard  the Standard version (the publisher's weights);
        --compact   the Compact version (the "variants" of « models »).
                    Both versions of a model can be installed side by side. Naming one that is whole
                    on disk beside the one rendered switches to it at once, with no download and no
                    sheet ("installed": true). Naming one that is not offers its installation; when
                    the disk can take it only in place of the other, the sheet says so and the user
                    accepts the removal there — the model cannot render until the new one is whole.

        install stop is « Stop » in the Models sheet and the sidebar: the installation in progress stops; what was downloaded is kept,
        the file in progress included, and the next installation resumes from it. It works on a Mac followed too (this Mac's
        own upload of a file to import there, deleted on both sides; else the lender's installation). Answer: {"ok": true, "stopped": true | false}.

        FAILURES: usage (also: two version flags together, a version the model does not have), model
        (unknown), busy (a render or an installation is running: wait, then retry), remote (the window computes on
        another Mac: install acts on this one; install stop works there).
        """, fr: """
        silicontrol install ID [--standard|--compact]
        silicontrol install stop

        Ouvre la feuille Modèles de l'app sur le modèle ID (un "id" de « models »), avec sa licence
        et ce qui reste à télécharger. La commande ne télécharge RIEN : l'utilisateur lit, puis
        clique « Accepter et installer » dans l'app (des dizaines de Go, plusieurs minutes). La
        réponse vient tout de suite :
          {"ok": true, "model": "krea2", "installed": false, "variant": "compact", "message": "…"}
          {"ok": true, "model": "z-image", "installed": true, "variant": "standard", "message": "already installed"}
        Ensuite « silicontrol models » dit "ready": true quand l'installation est finie, et « status »
        la suit ("installing" : le fichier, les octets reçus et le total).

        Sans drapeau, la version avec laquelle le modèle rend est gardée (aucune installée : la
        Compact, présélectionnée — l'utilisateur voit les deux) : une commande qui ne nomme pas de
        version n'en change jamais.
        --standard  la version Standard (les poids de l'éditeur) ;
        --compact   la version Compact (les "variants" de « models »).
                    Les deux versions d'un modèle s'installent côte à côte. En nommer une qui est
                    entière sur le disque à côté de celle qui rend y passe aussitôt, sans
                    téléchargement ni feuille ("installed": true). En nommer une qui n'y est pas
                    propose son installation ; quand le disque ne peut la prendre qu'à la place de
                    l'autre, la feuille le dit et l'utilisateur y accepte le retrait — le modèle ne
                    peut plus rendre tant que la nouvelle n'est pas entière.

        install stop est « Arrêter » dans la feuille Modèles et la barre latérale : l'installation en cours s'arrête ; ce qui a été
        téléchargé est gardé, le fichier en cours compris, et la prochaine installation reprend de là. Il marche aussi sur un Mac suivi
        (l'envoi par ce Mac d'un fichier à y importer, supprimé des deux côtés ; sinon l'installation du prêteur). Réponse : {"ok": true, "stopped": true | false}.

        ÉCHECS : usage (aussi : deux drapeaux de version ensemble, une version que le modèle n'a pas), model
        (inconnu), busy (un rendu ou une installation tourne : attendre, puis réessayer), remote (la fenêtre calcule sur
        un autre Mac : install agit sur celui-ci ; install stop marche là-bas).
        """),

        "import": Page(en: """
        silicontrol import FILE.safetensors [FILE.safetensors…]

        Brings files into the library, as the app's « Import a Model or LoRA… » does: each is
        recognized by its tensors (their names and shapes, not what its metadata say) as a LoRA or
        as the image model of one family, then copied as it is. Nothing is converted, nothing is
        downloaded. Every file is recognized before any is copied: one refusal imports nothing.
        The answer comes at once, the copy goes on in the app:
          {"ok": true, "importing": [{"file": "inCase.safetensors", "kind": "lora", "model": "anima"},
                                     {"file": "myFineTune.safetensors", "kind": "model", "model": "z-image"}]}
        Then « silicontrol models » lists them: a LoRA under its model's "loras" (« add --lora NAME »),
        an imported model as a model of its own, with its family's license, steps and formats.

        A path is relative to the directory the command is run from.

        FAILURES: usage (no file, a file that is not a .safetensors), busy (a render or an
        installation is running: wait, then retry), remote (the window computes on another Mac),
        file_missing, import_refused (neither a LoRA nor a model of a supported family: the message
        says what is missing), import_needs_family (a model whose family is not installed: « install »
        it first), import_base_model (a publisher's base model, not distilled: it would not render in
        8 steps), import_deeper_model (more layers than its family: not supported yet, coming soon),
        unsupported_lora (LoKr, LoHa, DoRA, a text encoder's LoRA…).
        """, fr: """
        silicontrol import FICHIER.safetensors [FICHIER.safetensors…]

        Fait entrer des fichiers dans la bibliothèque, comme « Importer un modèle ou une LoRA… » dans
        l'app : chacun est reconnu à ses tenseurs (leurs noms et leurs formes, pas ce que disent ses
        métadonnées) comme un LoRA ou comme le modèle d'image d'une famille, puis copié tel quel.
        Rien n'est converti, rien n'est téléchargé. Tous les fichiers sont reconnus avant qu'aucun
        soit copié : un seul refus n'importe rien. La réponse vient tout de suite, la copie se
        poursuit dans l'app :
          {"ok": true, "importing": [{"file": "inCase.safetensors", "kind": "lora", "model": "anima"},
                                     {"file": "myFineTune.safetensors", "kind": "model", "model": "z-image"}]}
        Ensuite « silicontrol models » les liste : un LoRA sous les "loras" de son modèle
        (« add --lora NOM »), un modèle importé comme un modèle à part entière, avec la licence, les
        pas et les formats de sa famille.

        Un chemin est relatif au dossier d'où la commande est lancée.

        ÉCHECS : usage (aucun fichier, un fichier qui n'est pas un .safetensors), busy (un rendu ou
        une installation tourne : attendre, puis réessayer), remote (la fenêtre calcule sur un autre
        Mac), file_missing, import_refused (ni un LoRA ni un modèle d'une famille connue : le message
        dit ce qui manque), import_needs_family (un modèle dont la famille n'est pas installée :
        « install » d'abord), import_base_model (un modèle de base d'éditeur, non distillé : il ne
        rendrait pas en 8 pas), import_deeper_model (plus de couches que sa famille : pas encore pris
        en charge, bientôt), unsupported_lora (LoKr, LoHa, DoRA, LoRA d'encodeur de texte…).
        """),

        "compiled": Page(en: """
        silicontrol compiled
        silicontrol compiled erase

        Settings › Macs › This Mac's « Compiled for This Mac »: what Core AI compiled of the engine's functions for this
        Mac, kept so that the next render starts at once. Alone, it is weighed. erase erases it, as « Erase… » does: the
        engine does it between two pieces of work — after the render under way, before the ones asked after — and the
        answer comes once it is done. Each model's next render compiles again what it needs, from a few seconds to a
        minute. An update of the app erases it by itself, at launch.

        Answer: {"ok": true, "compiled_bytes": 1288490188, "erasing": false}
                {"ok": true, "freed_bytes": 1288490188, "compiled_bytes": 0}
        FAILURES: usage (another argument), busy (an erasure already under way), engine_stopped (the engine's process
        ended before it was done).
        """, fr: """
        silicontrol compiled
        silicontrol compiled erase

        « Compilé pour ce Mac », de Réglages › Mac › Ce Mac : ce que Core AI a compilé des fonctions du moteur pour ce
        Mac, gardé pour que le rendu suivant parte aussitôt. Seul, il est pesé. erase l'efface, comme « Effacer… » : le
        moteur le fait entre deux travaux — après le rendu en cours, avant ceux demandés ensuite — et la réponse vient
        une fois fini. Le rendu suivant de chaque modèle recompile ce qu'il lui faut, de quelques secondes à une minute.
        Une mise à jour de l'app l'efface d'elle-même, au lancement.

        Réponse : {"ok": true, "compiled_bytes": 1288490188, "erasing": false}
                  {"ok": true, "freed_bytes": 1288490188, "compiled_bytes": 0}
        ÉCHECS : usage (un autre argument), busy (un effacement déjà en cours), engine_stopped (le processus du moteur
        s'est arrêté avant la fin).
        """),

        "uninstall": Page(en: """
        silicontrol uninstall ID [--variant standard|compact] [--yes]
        silicontrol uninstall downloads|legacy [--yes]

        The Models sheet's removals. ID is a model of « models »: alone, « Uninstall… » — every file of the model goes, both versions;
        with --variant, « Remove the … Version… » — the files that version alone reads go, what the other version reads stays, and the
        model renders with the other one (refused when that version is not installed, or is the only one: uninstall the model).
        An imported model's ID (« z-image/my-model ») is its « Remove »: its file alone goes, its family stays ("kind": "imported";
        no --variant). downloads is « Discard » on the interrupted downloads. legacy is « Free… » on the files of version 0.21, which
        this version no longer reads: the converted maps and their companions, and the old cache ("kind": "legacy", with
        "files"). Reinstalling downloads again.
        **Without --yes nothing is removed**: the answer says what would go and the room it frees.
          {"ok": true, "dry_run": true, "target": "z-image", "kind": "family", "frees_bytes": 11749122176,
           "message": "nothing was removed: add --yes to remove it"}
          {"ok": true, "dry_run": true, "target": "z-image", "kind": "version", "variant": "standard", "frees_bytes": 20350000000, …}
        With --yes: the same answer with "removed": true instead of "dry_run".
        Refused as the sheet greys it out: a render running or waiting, or an installation under way (busy); the library's
        disk absent (library_absent). On a Mac followed, the sheet asks the other Mac and waits for a click on its screen: refused here
        (remote) — the removal belongs to whoever sits at the Mac whose models they are.
        FAILURES: usage, model (unknown, or nothing of it installed, or no interrupted download, or nothing of version 0.21 left),
        busy, library_absent, remote.
        """, fr: """
        silicontrol uninstall ID [--variant standard|compact] [--yes]
        silicontrol uninstall downloads|legacy [--yes]

        Les retraits de la feuille Modèles. ID est un modèle de « models » : seul, « Désinstaller… » — tous les fichiers du modèle partent,
        les deux versions ; avec --variant, « Retirer la version … » — les fichiers que seule cette version lit partent, ce que l'autre lit
        reste, et le modèle rend avec l'autre (refusé quand cette version n'est pas installée, ou qu'elle est la seule : désinstaller le
        modèle). L'ID d'un modèle importé (« z-image/mon-modele ») est son « Retirer » : son fichier seul part, sa famille reste
        ("kind": "imported" ; pas de --variant). downloads est « Jeter » sur les téléchargements interrompus. legacy est « Libérer… »
        sur les fichiers de la version 0.21, que cette version ne lit plus : les cartes converties et leurs compagnons, et l'ancien
        cache ("kind": "legacy", avec "files"). Réinstaller télécharge de nouveau.
        **Sans --yes rien n'est retiré** : la réponse dit ce qui partirait et la place libérée.
          {"ok": true, "dry_run": true, "target": "z-image", "kind": "family", "frees_bytes": 11749122176,
           "message": "nothing was removed: add --yes to remove it"}
          {"ok": true, "dry_run": true, "target": "z-image", "kind": "version", "variant": "standard", "frees_bytes": 20350000000, …}
        Avec --yes : la même réponse avec "removed": true au lieu de "dry_run".
        Refusé comme la feuille le grise : un rendu en cours ou en attente, ou une installation en route (busy) ; le disque de la
        bibliothèque absent (library_absent). Sur un Mac suivi, la feuille interroge l'autre Mac et attend un clic sur son écran : refusé ici
        (remote) — le retrait revient à qui est assis devant le Mac dont ce sont les modèles.
        ÉCHECS : usage, model (inconnu, ou rien d'installé, ou aucun téléchargement interrompu, ou plus rien de la version 0.21),
        busy, library_absent, remote.
        """),

        "macs": Page(en: """
        silicontrol macs
        silicontrol macs connect NAME | disconnect NAME
        silicontrol compute NAME|this

        Render › Compute On, from the terminal. macs lists the Macs paired to compute on, as the menu does, and their state; NAME is a Mac's
        name (case and accents aside) or its "id". compute NAME makes the window compute on that Mac — its queue and its history are shown,
        and every command follows (help contract) — and compute this brings it back to this Mac. A queue running here goes on. The switch
        is made as the menu makes it; the command answers once the session is up, or lost, or after 15 s ("link" says where it stands).
        macs connect and disconnect are Settings › Macs › Other Macs' buttons: the session kept, or dropped.
        Never a pairing, a loan or a forgetting: those are accepted by the person in front of the screens, in the window.

        Answer (macs): {"ok": true, "computing_on": null | "Mac mini", "pairings_read": true,
                        "macs": [{"name": "Mac mini", "id": "…", "state": "connected", "computing": false, "machine": "M4, 16 GB…"}],
                        "lending": false, "lent_to": [{"name": "MacBook", "connected": false}]}
          state: connected, connecting, lost (trying again), on_the_network, not_on_the_network, refused_forgotten (that Mac forgot this
          one: pair again, in the window), refused_version (the two must run the same version of Siliconed).
          "pairings_read" false: the app has not read the Keychain yet (it does at launch if a Mac was paired, never without the user): the
          list is empty until Settings › Macs › Other Macs has been opened.
        Answer (macs connect|disconnect): {"ok": true, "mac": {"name": "Mac mini", "id": "…", "state": "connecting", …}}
        Answer (compute): {"ok": true, "computing_on": "Mac mini", "changed": true, "link": "connected", "message": "…" (if not yet)}
        FAILURES: usage, unknown_mac (no such paired Mac, or two of that name: use the "id"), link (that Mac refused the session).
        """, fr: """
        silicontrol macs
        silicontrol macs connect NOM | disconnect NOM
        silicontrol compute NOM|this

        Rendu › Calculer sur, depuis le terminal. macs liste les Mac appairés pour calculer, comme le menu, et leur état ; NOM est le nom d'un
        Mac (casse et accents mis à part) ou son "id". compute NOM fait calculer la fenêtre sur ce Mac — sa file et son historique sont
        montrés, et toutes les commandes suivent (help contract) — et compute this la ramène sur ce Mac. Une file qui tourne ici continue.
        Le basculement est fait comme le menu le fait ; la commande répond quand la session est établie, ou perdue, ou après 15 s ("link" dit
        où elle en est). macs connect et disconnect sont les boutons de Réglages › Mac › Autres Mac : la session gardée, ou lâchée.
        Jamais un appairage, un prêt ni un oubli : ceux-là sont acceptés par la personne devant les écrans, dans la fenêtre.

        Réponse (macs) : {"ok": true, "computing_on": null | "Mac mini", "pairings_read": true,
                          "macs": [{"name": "Mac mini", "id": "…", "state": "connected", "computing": false, "machine": "M4, 16 GB…"}],
                          "lending": false, "lent_to": [{"name": "MacBook", "connected": false}]}
          state : connected, connecting, lost (nouvel essai), on_the_network, not_on_the_network, refused_forgotten (ce Mac a oublié celui-ci :
          appairer de nouveau, dans la fenêtre), refused_version (les deux doivent faire tourner la même version de Siliconed).
          "pairings_read" false : l'app n'a pas encore lu le Trousseau (elle le fait au lancement si un Mac a été appairé, jamais sans
          l'utilisateur) : la liste est vide tant que Réglages › Mac › Autres Mac n'a pas été ouvert.
        Réponse (macs connect|disconnect) : {"ok": true, "mac": {"name": "Mac mini", "id": "…", "state": "connecting", …}}
        Réponse (compute) : {"ok": true, "computing_on": "Mac mini", "changed": true, "link": "connected", "message": "…" (si pas encore)}
        ÉCHECS : usage, unknown_mac (aucun Mac appairé de ce nom, ou deux : prendre l'"id"), link (ce Mac a refusé la session).
        """),

        "library": Page(en: """
        silicontrol library
        silicontrol library move DIR [--yes]
        silicontrol library home [--yes]
        silicontrol library open DIR
        silicontrol library local

        Settings › General's library buttons. Alone: where the library is, its size and the room left.
        move is « Move Library… »: DIR is a folder on another disk (a « Siliconed » folder is made in it, unless DIR is one or already holds
        a library you are moving). The files are copied and read back one by one, the old place is emptied last; the licenses you accepted
        and this Mac's settings stay on this Mac. home is « Move Back to This Mac ».
        **Without --yes nothing is copied**: the answer says what would be and the room there is.
          {"ok": true, "dry_run": true, "destination": "/Volumes/X/Siliconed", "files": 14, "bytes": 11749122176, "free_bytes": 400000000000,
           "message": "nothing was moved: add --yes to move it"}
        With --yes: {"type": "started", …}, then {"type": "progress", "percent": 37} lines, then the verdict with "library".
        A move stopped in the app leaves the library where it was ("stopped").
        open is « Open Library… » (a folder that already holds a library: nothing is moved), local « Use This Mac’s Library » (nothing is moved).
        Refused as Settings greys them out: a render or an installation under way (busy), the disk absent (library_absent), and for
        move and home a Mac followed (remote: the library shown is the other Mac's).
        Answer: {"ok": true, "library": {"path": "…", "elsewhere": true, "reachable": true, "volume": "X", "free_bytes": …, "used_bytes": …,
                 "this_mac": "…"}}
        FAILURES: usage, busy, library_absent, remote, library (already on this Mac; or the engine refused the destination — the library
          itself, a disk that is full, one that cannot be written to, another library's models: "message" says which), stopped.
        """, fr: """
        silicontrol library
        silicontrol library move DOSSIER [--yes]
        silicontrol library home [--yes]
        silicontrol library open DOSSIER
        silicontrol library local

        Les boutons de bibliothèque de Réglages › Général. Seul : où est la bibliothèque, sa taille et la place qui reste.
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
        Refusés comme Réglages les grise : un rendu ou une installation en route (busy), le disque absent (library_absent), et pour
        move et home un Mac suivi (remote : la bibliothèque montrée est celle de l'autre Mac).
        Réponse : {"ok": true, "library": {"path": "…", "elsewhere": true, "reachable": true, "volume": "X", "free_bytes": …, "used_bytes": …,
                   "this_mac": "…"}}
        ÉCHECS : usage, busy, library_absent, remote, library (déjà sur ce Mac ; ou le moteur a refusé la destination — la bibliothèque
          elle-même, un disque plein, un dossier où l'on ne peut pas écrire, les modèles d'une autre bibliothèque : "message" dit laquelle), stopped.
        """),
    ]
}
