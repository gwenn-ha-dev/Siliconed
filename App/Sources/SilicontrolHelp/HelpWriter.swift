// The help's pages of the Prompt Lab: the writer's gestures, its instructions, the named subjects.

extension RemoteControlHelp {
    static let writerPages: [String: Page] = [
        "rewrite": Page(en: """
        silicontrol rewrite "<text>" [--for ID]

        The Prompt Lab's « Rewrite » (the Text box, at the head of the chain: View › Show Chain, ⌥⌘R), from the terminal: a language model rewrites a prompt the
        way the model it is aimed at reads best. That writer is Qwen-Image-2.1's text encoder (Qwen3-VL-8B), which writes
        as installed: it is there once that model is (silicontrol install qwen-image-2.1), with nothing else to download.
        The writer, its instructions for each model, the protection of the @names and the refusals are
        the Lab's own: a command and a click give the same text. Nothing is written to disk — the text comes back on
        the socket only (incognito). The text is one argument, in quotes, like « add »'s prompt.

          silicontrol rewrite "woman posing in a library"
          silicontrol rewrite --for anima "1girl, library, solo"
          silicontrol rewrite --for qwen-image-2.1 "@alice reading in a library"

        --for ID   the model to write for: an "id" of « models » (or its family) among the models installed. Without it,
                   what the Lab takes: the model in the rack, else the first installed.
        @names     where the model aimed at reads images (Qwen-Image-2.1), each @name is put aside and put back after the
                   writer: one it lost or deformed is told ("lost_mentions", "notice"), never dropped in silence. Elsewhere
                   @ is the model's own (Anima's artists) and the text goes as it is.

        LINES — the first text takes a few seconds, the weights being read; they go back when it ends:
          {"type": "started", "gesture": "rewrite", "for": "qwen-image-2.1"}
          {"type": "loading", "fraction": 0}          the weights start being read; then "fraction": 1, at the first token
                                                      (null before the engine has said anything)
          {"type": "text", "text": "A woman…"}        EVERYTHING written so far — cumulative, not a delta: the writer may revise its last word
        VERDICT:
          {"ok": true, "gesture": "rewrite", "for": "qwen-image-2.1", "text": "…", "tokens": 112, "lost_mentions": []}
          with "notice" (a sentence) when a mention was lost or deformed — the text is given anyway, as the Lab gives it.

        One gesture at a time: the writer is the Lab's too. An image that renders has priority, as in the Lab: a render
        that starts ends the gesture ("stopped"). A client that hangs up cancels it: the sampling stops, the weights go back.
        The engine plans the memory itself when the gesture runs: nothing is checked before.
        FAILURES: usage, model (--for names none of the models installed), busy (an image renders or waits), remote (the
        window computes on another Mac: the writer is this Mac's), writer_absent (no installed model carries a writer:
        install qwen-image-2.1), writer_busy (another Rewrite or Describe is under way, from the terminal or the Text
        box), stopped, prompt_too_long, writer_failed (it failed or wrote nothing; "message" says why), and the
        engine's codes (insufficient_memory: no plan fits this Mac's memory; render_already_running: another process
        renders; engine_stopped: the engine's process went past this Mac's budget, or crashed — the next gesture starts
        another).
        """, fr: """
        silicontrol rewrite "<texte>" [--for ID]

        « Réécrire » de l'Atelier de prompts (la boîte Texte, en tête de la chaîne : Présentation › Afficher la chaîne, ⌥⌘R), depuis le terminal : un modèle de langage
        réécrit un prompt comme le modèle visé le lit le mieux. Ce rédacteur est l'encodeur de texte de Qwen-Image-2.1
        (Qwen3-VL-8B), qui écrit tel qu'installé : il est là dès que ce modèle l'est (silicontrol install qwen-image-2.1), sans
        rien d'autre à télécharger. Le rédacteur, ses consignes pour chaque modèle, la protection des @noms et les refus sont
        ceux de l'Atelier : une commande et un clic donnent le même texte. Rien n'est écrit sur disque — le texte ne revient que
        par la socket (incognito). Le texte est un seul argument, entre guillemets, comme le prompt de « add ».

          silicontrol rewrite "woman posing in a library"
          silicontrol rewrite --for anima "1girl, library, solo"
          silicontrol rewrite --for qwen-image-2.1 "@alice reading in a library"

        --for ID   le modèle pour lequel écrire : un "id" de « models » (ou sa famille) parmi les modèles installés. Sans lui,
                   ce que prend l'Atelier : le modèle du rack, sinon le premier installé.
        @noms      là où le modèle visé lit des images (Qwen-Image-2.1), chaque @nom est mis de côté puis remis après le
                   rédacteur : un nom qu'il a perdu ou déformé est dit ("lost_mentions", "notice"), jamais supprimé en silence.
                   Ailleurs, @ est celui du modèle (les artistes d'Anima) et le texte passe tel quel.

        LIGNES — le premier texte prend quelques secondes, les poids se lisent ; ils repartent à la fin :
          {"type": "started", "gesture": "rewrite", "for": "qwen-image-2.1"}
          {"type": "loading", "fraction": 0}          la lecture des poids commence ; puis "fraction": 1, au premier jeton
                                                      (null tant que le moteur n'a rien dit)
          {"type": "text", "text": "A woman…"}        TOUT ce qui est écrit jusque-là — cumulatif, pas un delta : le rédacteur peut réviser son dernier mot
        VERDICT :
          {"ok": true, "gesture": "rewrite", "for": "qwen-image-2.1", "text": "…", "tokens": 112, "lost_mentions": []}
          avec "notice" (une phrase) quand une mention a été perdue ou déformée — le texte est donné quand même, comme le fait l'Atelier.

        Un geste à la fois : le rédacteur est aussi celui de l'Atelier. Une image qui se rend est prioritaire, comme dans
        l'Atelier : un rendu qui démarre met fin au geste ("stopped"). Un client qui raccroche l'annule : l'échantillonnage
        s'arrête, les poids repartent. Le moteur planifie lui-même la mémoire quand le geste tourne : rien n'est vérifié avant.
        ÉCHECS : usage, model (--for nomme aucun des modèles installés), busy (une image se rend ou attend), remote (la fenêtre
        calcule sur un autre Mac : le rédacteur est celui de ce Mac), writer_absent (aucun modèle installé n'a de rédacteur :
        install qwen-image-2.1), writer_busy (un autre Réécrire ou Décrire est en cours, du terminal ou de la boîte
        Texte), stopped, prompt_too_long, writer_failed (il a échoué ou n'a rien écrit ; "message" dit pourquoi), et les
        codes du moteur (insufficient_memory : aucun plan ne tient dans la mémoire de ce Mac ; render_already_running : un
        autre processus rend ; engine_stopped : le processus du moteur a dépassé le budget de ce Mac, ou planté — le geste
        suivant en lance un autre).
        """),

        "describe": Page(en: """
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
        """, fr: """
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
        """),

        "instructions": Page(en: """
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

        --for ID    a model's id or a family, installed or not. Without it, the model the Lab writes for (the one in the
                    rack, else the first installed).
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
        """, fr: """
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

        --for ID      l'id d'un modèle ou une famille, installé ou non. Sans lui, le modèle pour lequel écrit l'Atelier (celui
                      du rack, sinon le premier installé).
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
        """),

        "lab": Page(en: """
        silicontrol lab
        silicontrol lab text "<text>"
        silicontrol lab generate
        silicontrol lab add <image|i7> …
        silicontrol lab role <n> edited|reference|start|none
        silicontrol lab remove <n>
        silicontrol lab rewrite [--wait]
        silicontrol lab describe <n> [--wait]

        THE PROMPT LAB — the Text and Images boxes at the head of the window's one column, above the chain, where the
        next render's text and images are prepared: what a drop on the canvas, ⌘V, « Send to the Prompt Lab » and the
        role menus of its images do. Its text is the rack's prompt (add and reuse write it); its images are numbered as
        the Images box shows them, from 1. The column shows and hides with the chain (« rack fold|unfold », ⌥⌘R): the
        lab has no window or fold of its own.

          lab                          what it holds: the column shown or not, the text, the images and their roles
          lab text "<text>"            its text, as typed in the field (the rack's prompt); "" empties it
          lab generate                 a click on « Generate »: what the rack and the lab hold goes to the queue, a series
                                       whole — then « wait », « status », « history » as after a click
          lab add IMAGE|i7 …           files, or images of the history: each takes the role a newcomer
                                       takes — the next place under Qwen-Image-2.1 (edited, then references,
                                       three at most with the subjects the text cites), none elsewhere
          lab role N ROLE              edited · reference (Qwen-Image-2.1) · start (Z-Image, Anima, Krea 2,
                                       MOSS-Image: the render starts from it, the size takes the model's
                                       format nearest its proportions) · none. One image to edit and one
                                       start: choosing another sends the previous image to edit to the
                                       references, the previous start to none.
          lab remove N                 takes image N out
          lab rewrite                  the Text box's « Rewrite »: its text rewritten for the model it writes for (the
                                       rack's), IN THE TEXT, token by token with its marks (additions, removals struck) —
                                       ⌘Z brings the text back. « rewrite » answers with a text and leaves the box alone.
          lab describe N               image N's « Describe »: what it shows, written in the text where the caret is (at
                                       its end when the field has no caret), or as the whole text when it is empty
          --wait                       the answer once the text is written, after the lines of « rewrite » (help rewrite):
                                       started, loading, text — "text" being the field's whole text as it will read

        Both show the column first (the buttons are in it), then press the button: the writer, its instructions and the
        refusals are those of « rewrite ». Without --wait they answer as soon as the writer starts,
          {"ok": true, "gesture": "describe", "for": "z-image", "image": 2, "writing": true, "lab": {…}}
        and « lab » says where it stands ("writing": {"gesture": "describe", "phase": "loading"|"writing", "image": 2}, null
        once written). With --wait: {"ok": true, "gesture": "rewrite", "for": "z-image", "text": "<the field's text>",
        "lab": {…}}, and "notice" when a mention was lost. The gesture is the window's: a client that hangs up leaves it
        writing; the Text box's Stop or « rack fold » stops it, a render that starts too ("stopped").

        A CHANGE OF MODEL never gives a role, nor turns one into another: a reference keeps its role under
        Z-Image, without effect ("used": false), and works again back on Qwen-Image-2.1. "unused" counts
        what the next render will not read, as the line beside Generate does.

        ANSWER
          {"ok": true, "lab": {"unfolded": true, "model": "qwen-image-2.1", "text": "@alice on the quay",
            "images": [{"n": 1, "name": "quay.jpg", "width": 1536, "height": 1024, "role": "edited",
                        "used": true}], "unused": 0, "writing": null}}
          "unfolded": the column is shown (« rack unfold »).
        FAILURES: usage, image (no image N), role (a role the rack's model lacks), ref (image N cannot be read);
        lab generate: refused (the button is greyed; "message" is the reason said beside it);
        lab rewrite and lab describe: those of « rewrite » (remote, busy, writer_absent, writer_busy), empty_text
        (lab rewrite on an empty text); with --wait, stopped, writer_failed, prompt_too_long and the engine's codes
        (insufficient_memory, render_already_running).
        """, fr: """
        silicontrol lab
        silicontrol lab text "<text>"
        silicontrol lab generate
        silicontrol lab add <image|i7> …
        silicontrol lab role <n> edited|reference|start|none
        silicontrol lab remove <n>
        silicontrol lab rewrite [--wait]
        silicontrol lab describe <n> [--wait]

        L'ATELIER — les boîtes Texte et Images en tête de la colonne unique de la fenêtre, au-dessus de la chaîne, où se
        préparent le texte et les images du prochain rendu : ce que font un dépôt sur le canevas, ⌘V, « Envoyer à
        l'atelier » et les menus de rôle de ses images. Son texte est le prompt du rack (add et reuse l'écrivent) ; ses
        images sont numérotées comme la boîte Images les montre, à partir de 1. La colonne s'affiche et se masque avec la
        chaîne (« rack fold|unfold », ⌥⌘R) : l'atelier n'a ni fenêtre ni pli à lui.

          lab                          ce qu'il tient : la colonne affichée ou non, le texte, les images et leurs rôles
          lab text "<texte>"           son texte, comme tapé dans le champ (le prompt du rack) ; "" le vide
          lab generate                 un clic sur « Générer » : ce que tiennent le rack et l'atelier part en file, une
                                       série entière — puis « wait », « status », « history » comme après un clic
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
          lab rewrite                  « Réécrire » de la boîte Texte : son texte réécrit pour le modèle pour lequel elle
                                       écrit (celui du rack), DANS LE TEXTE, jeton par jeton avec ses marques (ajouts,
                                       retraits barrés) — ⌘Z ramène le texte. « rewrite » répond par un texte et laisse la
                                       boîte.
          lab describe N               « Décrire » de l'image N : ce qu'elle montre, écrit dans le texte au curseur (à la
                                       fin quand le champ n'a pas le curseur), ou comme tout le texte quand il est vide
          --wait                       la réponse une fois le texte écrit, après les lignes de « rewrite » (help rewrite) :
                                       started, loading, text — "text" étant tout le texte du champ tel qu'il se lira

        Les deux affichent d'abord la colonne (les boutons y sont), puis pressent le bouton : le
        rédacteur, ses consignes et les refus sont ceux de « rewrite ». Sans --wait, ils répondent dès que le rédacteur part,
          {"ok": true, "gesture": "describe", "for": "z-image", "image": 2, "writing": true, "lab": {…}}
        et « lab » dit où il en est ("writing" : {"gesture": "describe", "phase": "loading"|"writing", "image": 2}, null une
        fois écrit). Avec --wait : {"ok": true, "gesture": "rewrite", "for": "z-image", "text": "<le texte du champ>",
        "lab": {…}}, et "notice" quand une mention s'est perdue. Le geste est celui de la fenêtre : un client qui raccroche
        le laisse écrire ; Arrêter dans la boîte Texte ou « rack fold » l'arrête, un rendu qui démarre aussi ("stopped").

        UN CHANGEMENT DE MODÈLE ne donne jamais de rôle et n'en change aucun en un autre : une référence
        garde son rôle sous Z-Image, sans effet ("used": false), et resert de retour sous Qwen-Image-2.1.
        "unused" compte ce que le prochain rendu ne lira pas, comme la ligne à côté de Générer.

        RÉPONSE
          {"ok": true, "lab": {"unfolded": true, "model": "qwen-image-2.1", "text": "@alice on the quay",
            "images": [{"n": 1, "name": "quai.jpg", "width": 1536, "height": 1024, "role": "edited",
                        "used": true}], "unused": 0, "writing": null}}
          "unfolded" : la colonne est affichée (« rack unfold »).
        ÉCHECS : usage, image (pas d'image N), role (un rôle que le modèle du rack n'a pas), ref (l'image N ne se lit
        pas) ; lab generate : refused (le bouton est grisé ; "message" est la raison dite à côté) ; lab rewrite et lab describe : ceux de « rewrite » (remote, busy, writer_absent, writer_busy), empty_text
        (lab rewrite sur un texte vide) ; avec --wait, stopped, writer_failed, prompt_too_long et les codes du moteur
        (insufficient_memory, render_already_running).
        """),

        "subjects": Page(en: """
        NAMED SUBJECTS — silicontrol subjects [add … | remove NAME]

        The same library as the Prompt Lab's (its Images box, at the head of the chain). A subject is a picture (or two of the same subject) with a name, a kind and a frame; the
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
        """, fr: """
        SUJETS NOMMÉS — silicontrol subjects [add … | remove NOM]

        La même bibliothèque que celle de l'atelier (sa boîte Images, en tête de la chaîne). Un sujet est une image (ou deux du même sujet) avec un nom, un genre et
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
        """),
    ]
}
