[English](REMOTE-CONTROL.md) · Français

# silicontrol, la télécommande de Siliconed : le dedans

**La référence est dans l'app.** `silicontrol help` donne la vue d'ensemble, `silicontrol help <commande>`
ou `silicontrol <commande> --help` une page, `silicontrol help all` toute la référence (contrat, réponses
JSON, codes d'erreur, recettes). Pour mettre un agent au travail, il suffit de lui dire « regarde
`silicontrol help` ». Ce fichier n'explique que comment c'est fait ; il ne répète pas l'aide, pour
qu'elles ne puissent pas diverger.

## L'idée

L'app ouverte tient la seule file de rendus de la machine. `silicontrol …` y ajoute, attend, arrête,
relit l'historique : les mêmes fonctions que les boutons (`AppState.enqueue`, `cancel`, `remove`,
`stopAll`). Les demandes de la commande apparaissent dans la fenêtre ; celles de la souris, la
commande les voit. Il n'y a jamais deux moteurs dans les 16 Go. Quand la fenêtre calcule sur un autre Mac
(Rendu › Calculer sur), les commandes la suivent : elles ne parlent toujours qu'au socket de ce Mac, mais agissent
sur la file et l'historique de l'autre Mac, comme les boutons ; `install` (et `install writer`, `rewrite`, `describe` : le rédacteur est celui de ce Mac ; `import`, `uninstall`, `library move|home` : les fichiers et le clic sont ceux de ce Mac) y refuse, `diagnose` et
`set faster-rendering|accelerators` agissent sur l'autre Mac.

**Une licence ne s'accepte que dans la fenêtre.** Un travail dont la bibliothèque n'a pas
enregistré la licence du modèle est refusé par le moteur (`license_not_accepted`) et mis de côté,
hors de la file, pendant que la feuille de l'app montre la licence : `status` le liste sous
`"waiting_for_license"`, `wait` l'attend, `cancel` le retire. Acceptée, il revient en tête de file ;
refusée, il finit en `"error"`. Aucune commande n'accepte à la place de l'utilisateur.

**Les alternatives d'un prompt se développent des deux côtés de la même façon.**
`"woman posing in a {library|greenhouse}"` fait deux travaux, à la même graine, que le prompt vienne
du champ de l'app ou de `silicontrol add` : les deux passent par `PromptAlternatives` (bibliothèque,
`Sources/Siliconed/PromptAlternatives.swift`), puis par `AppState.enqueue`. Rien n'est refait côté serveur : il refuse ce
que le parseur refuse (erreur `prompt`), et chaque travail porte son prompt développé et ses
`choices`. La syntaxe et la réponse sont dans `silicontrol help add`.

**Une grille, ce sont des travaux, puis une image.** `silicontrol grid --x … [--y …]` et la fenêtre
« Exploration » de l'app (Rendu › Exploration, ⌥⌘G) vérifient toutes deux la grille avec la
bibliothèque (`RenderGrid`, `Sources/Siliconed/RenderGrid.swift`) et la mettent en file par
`AppState.enqueueGrid` : un travail ordinaire par case, dans l'ordre de lecture de la planche, plus
un `GridRun` qui retient seulement quel travail est quelle case (`Sources/SiliconedApp/Grid.swift`).
Les axes sont un groupe `{…}` du prompt, la graine, les pas, la force d'une LoRA, des LoRA à comparer
(`loras=none,flat,ink` : une ajoutée à la pile par case, noms résolus comme `--lora`), le modèle
(`model=z-image,qwen-image-2.1` : chaque case aux pas de son modèle, refusé avec une `--lora`) et les
images de l'édition (`image` : chaque `--ref` seule, avec la même instruction). `silicontrol grid
--sketch` met en file une exploration comme celle de la fenêtre (des esquisses, dans sa fenêtre,
sans planche) ; sinon chaque case est une image finie qui rejoint l'historique ; quand la dernière case finit —
faite, en échec, arrêtée ou retirée —, l'app dessine la planche hors du fil principal et l'ajoute à
l'historique ; le `Signal` `.sheet` prévient `wait g2`, `grid --wait` et `follow`
(`"type": "sheet"`). Une case non rendue est dessinée vide, avec la raison ; aucune case rendue, pas
de planche. Un groupe `{…}` sur aucun axe est refusé (`grid`), jamais multiplié. La syntaxe est dans
`silicontrol help grid`.

**L'exploration esquisse.** Dans la fenêtre (`Sources/SiliconedApp/Exploration.swift`), chaque case
s'arrête en esquisse (`Request.sketch`) à 512 sur le petit côté (512²,
512×768, 768×512) : environ les trois quarts du temps d'une image finie. Ses esquisses restent dans la
fenêtre, pas dans l'historique, jusqu'à l'exploration suivante, qui la remplace (ses cases en
attente quittent la file, celle en cours s'arrête). Sur une case, « Finir cette image » la rend
jusqu'au bout — même graine, même taille — dans l'historique, l'image que l'esquisse annonçait ;
« Reprendre ces réglages » donne au rack son modèle, son prompt, ses pas, ses forces de LoRA et sa
graine, pas sa taille. La planche ne rejoint l'historique que sur « Ajouter la planche à
l'historique ».

**Les variations sont celles du menu.** `silicontrol vary i7 subtle|strong [--batch N]` et le menu
« Variations » d'une image (le bouton sous l'image, son menu contextuel) appellent tous deux
`AppState.vary` : autant de travaux ordinaires que le champ « Lot » du rack (1 par défaut ;
`--batch` le remplace), avec exactement les réglages et la graine de cette
image, chacun ajoutant une graine de variation tirée au hasard (`Request.variations`,
`Sources/Siliconed/Variation.swift`) à la force de la famille (`Variation.Amount`), jamais un nombre que la
commande passerait. Une variation d'une variation s'ajoute à la chaîne, elle varie donc *cette*
image. Refusé sur la planche d'une grille, une esquisse, ou un modèle qui n'est pas prêt (`model`).
Le `variation` de chaque image donne ses graines de variation dans l'ordre ; la réponse est dans
`silicontrol help vary`.

**L'agrandissement aussi est celui du menu.** `silicontrol enlarge i7 [--download]` et « Agrandir ×4 »
appellent tous deux `AppState.enlarged` : la super-résolution de macOS (VideoToolbox, macOS 26),
sans prompt, hors de la file — refusé pendant un rendu —, avec la nouvelle image en réponse une
fois dans l'historique. Le modèle d'Apple n'est téléchargé par macOS qu'avec `--download` (sinon
`download_required`) ; le reste est dans `silicontrol help enlarge`.

**Une taille est `WxH` ou une proportion et une résolution.** `silicontrol add` et `grid` prennent `832x1216` comme toujours, ou `3:4@2K` : la proportion (1:1, 4:5, 3:4, 2:3, 9:16, 5:4, 4:3, 3:2, 16:9), `@`, puis `1K` ou `2K`. Aucun de ces caractères ne veut rien dire pour un shell, donc pas de guillemets. Elle se résout dans les pixels du Format en deux échelles du rack (`SizeScales`, une seule fonction pour les deux), que cette option Bêta soit allumée ou non, et elle est refusée (`format`) avec sa raison là où le rack la grise : la 2K d'Anima, par exemple, est le carré seul (2048 px de côté). La table est dans [la page de l'app](APP.fr.md#créer).

**Les solveurs suivent Plus de solveurs.** `--solver` n'est accepté qu'avec Plus de solveurs allumé (Réglages › Avancé › Bêta) et sur un modèle qui propose autre chose qu'Euler ; sinon la réponse est un refus (`solver`) qui nomme le réglage.

**Le rendu plus rapide est aussi un réglage de l'app, un par modèle, et une option Bêta.** `silicontrol set faster-rendering
on|off [--model ID]` passe par `AppState.setFasterRendering(_:for:)`, rangé par version installée comme le
choix des accélérateurs, et
l'interrupteur de la fenêtre suit, comme un clic ; sans `--model`, le modèle du rack. `models` donne la
valeur de chaque modèle installé, `status` celle du rack (`"faster_rendering"`). « on » est refusé (`faster_rendering`) tant que l'option Bêta (Réglages › Avancé › Bêta) est éteinte, comme `--solver` sans Plus de solveurs ; « off » passe toujours. La valeur que rendent `models` et `status` est celle qu'un rendu prend, donc `false` option éteinte. Refusé (`busy`) sauf si
rien ne tourne ni n'attend : un travail garde la valeur avec laquelle il est entré dans la file
(`RenderSettings.fasterRendering`), et un travail planifié sous une valeur et fini sous l'autre
mélangerait deux recettes. Il persiste comme l'interrupteur ; `silicontrol measure --faster-rendering
on|off` le pose pour le modèle mesuré, l'inscrit dans le TSV et rend la valeur d'avant à la fin, même si
la série échoue. Éteint, c'est la pleine taille : l'image contrôlée contre la référence.

**Rien ne dépend d'un clic, sauf l'appairage.** Ce que fait la fenêtre, une commande le fait aussi, **par les mêmes fonctions que les boutons** (`AppState`, le `LabModel` de l'Atelier de prompts), jamais par un chemin à part. Ce qui reste dans la fenêtre exprès : l'appairage d'un Mac, l'acceptation d'une licence, « Prêter ce Mac » et « Oublier ». Le rédacteur de l'Atelier de prompts et les interrupteurs des Réglages sont décrits ensuite, et après eux les gestes de l'historique, de l'exploration, des modèles, de Calculer sur et de la bibliothèque.

**Le rédacteur de l'Atelier de prompts est celui de la fenêtre.** `silicontrol rewrite "<texte>" [--for ID]` et `silicontrol describe <image|i7> [--for ID]` appellent ce qu'appellent les deux boutons de l'Atelier : `EngineLabWriter` sur `PromptWriter`, nos consignes par modèle et par geste (`PromptWriterInstructions`), l'image lue comme l'Atelier lit une image déposée (`PromptWriterImage`), et, pour un modèle qui lit les images, les `@noms` mis de côté puis remis (`LabInputs.MentionGuard`, partagé sous les noms `LabModel.rewriteInput` et `rewritten`). Le modèle pour lequel il écrit est celui que prendrait l'Atelier (`WriterTarget.pick` : le choix de l'Atelier, sinon celui du rack, sinon le premier installé) sauf si `--for` en nomme un. Les refus sont ceux de l'Atelier, dans son ordre (`WriterPause.decide`, une seule règle pour la phrase sous ses boutons et pour le code) : la fenêtre suit un autre Mac (`remote`), une image se rend (`busy`), pas de rédacteur, la tête à télécharger, un Mac trop petit (`writer_absent`, `writer_head`, `writer_memory`, `writer_preparing`) ; et un second geste pendant qu'un écrit (`writer_busy`), du terminal ou de la fenêtre. Les lignes sont `{"type": "started"}`, `{"type": "loading", "fraction"}` et `{"type": "text", "text"}`, qui porte tout ce qui est écrit jusque-là (cumulatif, comme le flux de l'Atelier : le détokeniseur peut réviser son dernier mot) ; le verdict a le `text` entier, les `tokens`, et la `notice` et `lost_mentions` quand le rédacteur a perdu un `@nom`. **Un rendu qui démarre arrête le geste** (`stopped`), comme il arrête celui de l'Atelier. **Un client qui raccroche l'annule** : `Connection.isGone` voit la fermeture sans écrire dans la socket, et `LabModel.cancelRemote` termine le flux, dont la fin arrête l'échantillonnage (`EngineLabWriter.write`, `PromptWriter.cancel`) avant que les poids repartent. Incognito : le texte sort par la socket et nulle part ailleurs. `silicontrol install writer` est « Installer le rédacteur » (`LabModel.installHead`) : la tête de l'encodeur de Qwen-Image-2.1, sans licence propre (c'est celle que l'installation de l'encodeur a fait accepter : `Installer.installWriterHead` la vérifie, et la commande le dit avant de commencer). Les trois agissent sur ce Mac : la fenêtre sur un autre, elles sont refusées (`remote`).

**Les accélérateurs et les options Bêta sont aussi des interrupteurs des Réglages.** `set accelerators on|off [--model ID]` (`neural-engine`, son ancien nom, encore lu) passe par `AppState.setAccelerators(_:for:)`, comme `set faster-rendering` par le sien, avec la même attente (`busy`) et les mêmes clés par version installée — celles du Neural Engine avant (clé `neuralEngine` des `UserDefaults`) : un choix fait alors est gardé ; il est refusé (`accelerators`) là où la mesure de ce Mac ne donne rien à faire aux accélérateurs, c'est-à-dire quand l'interrupteur est grisé (`DiTSwitches.acceleratorsUnavailable`). `models` et `status` le donnent (`"accelerators"`) ; `add --accelerators` / `--no-accelerators` (`--ane` / `--no-ane` avant) le règlent pour un travail, comme un clic sur l'interrupteur. `set beta [OPTION on|off]` écrit la propriété à laquelle est lié chacun des quatre interrupteurs de Réglages › Avancé › Bêta (`developerMode` pour Plus de solveurs, `fasterRenderingBeta`, `sizeScalesBeta`, `negativeLoRAStrengths`), si bien que leurs `didSet` font ce que fait un clic ; `status` les donne sous `"beta"`. Ce sont les préférences de ce Mac, et la copie du miroir que montre la fenêtre est réglée avec elles. `set save-images DOSSIER|off`, c'est Enregistrer chaque image dans… de Réglages › Général, éteint par défaut : chaque image nouvelle de l'historique (un rendu, un recadrage, un agrandissement, la planche d'une grille ; pas les esquisses d'une exploration) s'écrit aussi dans ce dossier en PNG avec ses métadonnées, sous son nom de `save` ; le dossier doit exister, il n'est jamais créé ; seul, `set save-images` le lit. Les arguments de `rewrite`, `describe` et `set` se lisent dans la bibliothèque (`WriterOrder`, `SetOrder`, `BetaOption`), où les tests les lisent.

**L'historique, l'exploration, les modèles et la bibliothèque sont aussi ceux de la fenêtre.** Chacun appelle la fonction qu'appelle le bouton ou l'élément de menu, avec la garde que la fenêtre y met : ce qui est grisé dans la fenêtre est un refus ici, avec la même phrase.
`remove i7 [i8 …]` est `AppState.delete` (⌫), `undo [--redo]` le `UndoManager` de la fenêtre (⌘Z, ⇧⌘Z : retraits, recadrages, un prompt porté au rack), `again <i7|fichier.png>` est `redo` (⌘R, Rendre à nouveau) et, depuis un PNG écrit par Siliconed, `restoreSettings(fromPNG:)` comme un dépôt le fait, puis Générer ; `reuse` est Reprendre ces réglages, sans le rendu. Un geste qui met des travaux en file répond avec eux (sur un Mac suivi chacun reçoit la réponse de ce Mac, comme pour `vary`). `crop i7 --box X,Y,W,H | --ratio P:Q` demande les règles de cadre de l'éditeur (`Crop.Frame`, `CropSpec` dans la bibliothèque) et le même `cropped` que Retour appelle ; `save` écrit PNG, JPEG et HEIC comme les panneaux Enregistrer et Exporter.
Dans l'exploration, une case se nomme `g2c3` (la grille, puis son rang dans l'ordre de lecture de la planche depuis 0, la même arithmétique que `column` et `row` du travail ; son travail `t5` la nomme aussi) : `finish g2c3` est `finishCell`, `sheet g2` est `addSheet`, tous deux sur l'exploration que montre la fenêtre et rien d'autre. `import FICHIER …` est `perform` (⌘O, un dépôt) : un modèle dont la famille n'est pas installée ouvre la feuille Modèles avec la licence de la famille, comme la fenêtre, et la commande le dit (`needs_family`) sans jamais l'accepter. `uninstall ID|lora NOM|downloads` est le Désinstaller de la feuille, la corbeille d'un modèle importé ou d'une LoRA, et Jeter ; `install stop` est son Arrêter. `macs` liste les Mac appairés et leur état, `compute NOM|this` est le menu Calculer sur (`MainWindow.compute(on:)`), `macs connect|disconnect` sont les boutons de Réglages › Autres Mac. `library move DOSSIER|home` est Déplacer la bibliothèque… et Ramener sur ce Mac, `open DOSSIER` et `local` les deux boutons voisins.
**Ce qui détruit ou déplace demande `--yes`** : `uninstall` et `library move|home`. Sans lui la réponse est un `dry_run` qui dit ce qui partirait, ou ce qui serait copié, et la place qu'il libère ou prend (`LibraryMove.plan`, la fonction même par laquelle le déplacement commence, donc les deux ne peuvent pas diverger). Les refus sont ceux de la feuille et des Réglages : un rendu, une installation ou la mesure en route (`busy`), le disque de la bibliothèque absent (`library_absent`). **Sur un Mac suivi** la feuille interroge l'autre Mac et attend un clic sur son écran ; les commandes n'attendent pas un clic qui n'est pas le leur : `import`, `uninstall` et `library move|home` y sont refusés (`remote`). Jamais un appairage, un prêt ni un oubli. Ce qui n'a pas de commande est de l'affichage, pas un geste : le rideau de comparaison, le plein écran, le zoom, les Statistiques, Partager et Copier, les champs du rack (`add` est le formulaire et Générer en un), et les feuilles de la fenêtre (elles ne montrent que ce qu'une commande répond).

## La commande

`silicontrol` est livrée dans l'app, `Siliconed.app/Contents/Helpers/silicontrol` (cible `Silicontrol`,
`Sources/Silicontrol/main.swift`). Réglages › Avancé › Ligne de commande (« Installer… ») la lie
dans `/usr/local/bin`. Ce n'est qu'un tuyau : Foundation et les frameworks du système seulement, elle ne lie pas le moteur. Si
l'app n'est pas lancée, toute commande sauf l'aide la lance et attend sa prise (30 s au plus) ;
`silicontrol open` ne fait que cela. L'aide (`help`, `help <sujet>`, `<commande> --help`, aucun
argument) est servie par la commande elle-même, sans lancer l'app.

## La prise et le protocole

- **Socket Unix** `silicontrol.sock` dans l'espace de l'app (par défaut
  `~/Library/Application Support/Siliconed/silicontrol.sock`, ou sous `SILICONED_ROOT`) — là même
  quand la bibliothèque a été déplacée sur un autre disque —, droits `0600` :
  seul le compte qui a lancé l'app y accède. Aucun port réseau. Elle disparaît quand l'app quitte ;
  une prise morte est remplacée au lancement suivant, et un second Siliconed sur la même
  bibliothèque n'écoute pas.
- **Une connexion par commande.** Le client écrit une ligne JSON,
  `{"argv": ["add", "--batch", "2", "…"], "cwd": "/dossier/courant"}` ; l'app répond par des lignes
  JSON puis ferme. Les lignes intermédiaires portent `"type"` (`added`, `image`, `sheet`, `start`, `stage`,
  `step`, `end` ; `started`, `loading`, `text` pour `rewrite` et `describe` ; `started`, `progress` pour `library move`) ; la dernière porte `"ok"` (`true` ou `false`, avec `"error"`, `"message"` et
  souvent `"hint"` en cas d'échec). Sans le client :
  `printf '{"argv":["status"]}\n' | nc -U ~/Library/Application\ Support/Siliconed/silicontrol.sock`.
- **Toute l'analyse est dans l'app** (`Sources/SiliconedApp/RemoteControl.swift`) : `argv` est la
  ligne de commande telle quelle et le client n'est qu'un tuyau. Seuls `open`, l'aide et `measure` sont
  traités côté client : les deux premiers doivent marcher app fermée ; `measure` est une série de
  rendus que le client pilote lui-même, chacun un `add --wait` sur la prise, chronométré, pendant
  qu'il échantillonne la mémoire de l'app (`Sources/Measurement`). Les chemins relatifs se résolvent contre `cwd`. Les
  images d'une édition (`--ref`, répétable, dans l'ordre : le premier est l'image 1, celle qu'on
  édite) voyagent de même, en chemins ou en noms d'historique (`i7`), jamais en pixels : l'app les
  lit elle-même.
- **L'aide a une seule source**, le module `SilicontrolHelp` (`Sources/SilicontrolHelp/RemoteControlHelp.swift`,
  pages anglaises et françaises, Foundation seule), lié des deux côtés : `silicontrol` y répond
  sans l'app, et l'app y répond sur sa prise (`nc -U` reçoit les mêmes pages). La langue suit
  l'ordre de l'utilisateur entre anglais et français. La réponse est
  `{"ok": true, "help": "<texte>"}` et le client imprime le texte tel quel ; un sujet inconnu est
  un échec d'usage (`"error": "usage"`, les sujets dans `"hint"`, sortie `2`). Commandes, drapeaux,
  clés JSON et codes d'erreur sont en anglais dans les deux langues. Quand une commande change
  dans l'app, son aide change dans le même commit.
- **Codes de sortie** du client : `0` la dernière ligne dit `"ok": true` ; `1` elle dit
  `"ok": false` ; `2` erreur d'usage ; `3` l'app est injoignable (pas lancée et non lançable, ou
  ne répond pas).
- **Les signaux.** L'`AppState` émet `Signal` (`started`, `stage`, `step`, `image`, `finished` avec
  son `Issue`, `sheet` quand la dernière case d'une grille a fini) à qui s'abonne ; `wait` et `follow` s'abonnent le temps de leur connexion et se
  désabonnent au premier envoi qui échoue (client parti).
- **SIGPIPE est ignoré** (`signal(SIGPIPE, SIG_IGN)`, plus `SO_NOSIGPIPE` sur chaque socket
  client), pour qu'un client qui raccroche ne puisse pas tuer l'app : mesuré, un client qui se
  connectait et fermait sans lire emportait l'app.
- **La plomberie ne touche pas le fil principal** : accepter et lire se font sur des files de fond
  (`Socket`, `Connection`), écrire sur la file de chaque client, si bien qu'un long `wait` ne bloque
  ni l'app ni les autres clients. Seules les commandes passent sur le fil principal, comme un clic.

## Ce qui a été mesuré

Pris sur le M1 Pro de référence, 16 Go, quand la télécommande a été écrite, avec une version ancienne de l'app et sans être refaits depuis :

- **Mêmes bits** : l'image de l'app = un rendu de la bibliothèque appelée directement, même requête
  (512², graine 42, octets décodés identiques). De même pour une variation de variation mise en file
  par `silicontrol vary` : refaite depuis la recette de son PNG, elle sort au bit près.
- **App Nap** : en arrière-plan, macOS freinait l'app : 64 s au lieu de 44 pour un 512², un pas
  à 85 s. L'app tient une activité « rendu » (`ProcessInfo.beginActivity`) du premier travail lancé
  jusqu'à la file vide : **45,9 s** en arrière-plan (44,3 s pour le même rendu hors de l'app).

## Un autre processus qui rend

Un rendu par machine : le moteur prend un verrou (`/private/tmp/siliconed-render.lock`) pour chaque
rendu, si bien qu'un autre processus qui se sert du moteur (une version compilée depuis les sources, par exemple) est refusé avec
`render_already_running` pendant que l'app rend, et inversement (deux modèles dans 16 Go : le
système swapperait et les deux chronos ne voudraient plus rien dire). Un tel programme peut mettre son
rendu dans la file de l'app avec `silicontrol add`.
