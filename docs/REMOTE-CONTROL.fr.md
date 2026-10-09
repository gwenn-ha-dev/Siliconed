[English](REMOTE-CONTROL.md) · Français

# silicontrol, la télécommande de Siliconed : le dedans

**La référence est dans l'app.** `silicontrol help` donne la vue d'ensemble, `silicontrol help <commande>`
ou `silicontrol <commande> --help` une page, `silicontrol help all` toute la référence (contrat, réponses
JSON, codes d'erreur, recettes). Pour mettre une IA au travail, il suffit de lui dire « regarde
`silicontrol help` ». Ce fichier n'explique que comment c'est fait ; il ne répète pas l'aide, pour
qu'elles ne puissent pas diverger.

## L'idée

L'app ouvert tient la seule file de rendus de la machine. `silicontrol …` y ajoute, attend, arrête,
relit l'historique : les mêmes fonctions que les boutons (`AppState.enqueue`, `cancel`, `remove`,
`stopAll`). Les demandes de la commande apparaissent dans la fenêtre ; celles de la souris, la
commande les voit. Il n'y a jamais deux moteurs dans les 16 Go.

**Une licence ne s'accepte que dans la fenêtre.** Un travail dont la bibliothèque n'a pas
enregistré la licence du modèle est refusé par le moteur (`license_not_accepted`) et mis de côté,
hors de la file, pendant que la feuille de l'app montre la licence : `status` le liste sous
`"waiting_for_license"`, `wait` l'attend, `cancel` le retire. Acceptée, il revient en tête de file ;
refusée, il finit en `"error"`. Aucune commande n'accepte à la place de l'utilisateur.

**Les alternatives d'un prompt se développent des deux côtés de la même façon.**
`"woman posing in a {library|greenhouse}"` fait deux travaux, à la même graine, que le prompt vienne
du champ de l'app ou de `silicontrol add` : les deux passent par `PromptAlternatives` (bibliothèque,
`docs/API.fr.md` §3.14), puis par `AppState.enqueue`. Rien n'est refait côté serveur : il refuse ce
que le parseur refuse (erreur `prompt`), et chaque travail porte son prompt développé et ses
`choices`. La syntaxe et la réponse sont dans `silicontrol help add`.

**Une grille, ce sont des travaux, puis une image.** `silicontrol grid --x … [--y …]` et la fenêtre
« Exploration » de l'app (Rendu > Exploration, ⌥⌘G) vérifient toutes deux la grille avec la
bibliothèque (`RenderGrid`, `docs/API.fr.md` §3.15) et la mettent en file par
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
s'arrête en esquisse (`Request.sketch`, `docs/API.fr.md` §3.16) à 512 sur le petit côté (512²,
512×768, 768×512) : environ la moitié du temps d'une image finie. Ses esquisses restent dans la
fenêtre, pas dans l'historique, jusqu'à l'exploration suivante, qui la remplace (ses cases en
attente quittent la file, celle en cours s'arrête). Sur une case, « Finir cette image » la rend
jusqu'au bout — même graine, même taille — dans l'historique, l'image que l'esquisse annonçait ;
« Reprendre ces réglages » donne au rack son modèle, son prompt, ses pas, ses forces de LoRA et sa
graine, pas sa taille. La planche ne rejoint l'historique que sur « Ajouter la planche à
l'historique ».

**Les variations sont celles du menu.** `silicontrol vary i7 subtle|strong [--batch N]` et le menu
« Variations » d'une image (le bouton sous l'image, son menu contextuel) appellent tous deux
`AppState.vary` : autant de travaux ordinaires que le champ « Images » du rack (1 par défaut ;
`--batch` le remplace), avec exactement les réglages et la graine de cette
image, chacun ajoutant une graine de variation tirée au hasard (`Request.variations`,
`docs/API.fr.md` §3.18) à la force de la famille (`Variation.Amount`), jamais un nombre que la
commande passerait. Une variation d'une variation s'ajoute à la chaîne, elle varie donc *cette*
image. Refusé sur la planche d'une grille, une esquisse, ou un modèle qui n'est pas prêt (`model`).
Le `variation` de chaque image donne ses graines de variation dans l'ordre ; la réponse est dans
`silicontrol help vary`.

## La commande

`silicontrol` est livrée dans l'app, `Siliconed.app/Contents/Helpers/silicontrol` (cible `Silicontrol`,
`Sources/Silicontrol/main.swift`). Le menu de l'app (« Install the silicontrol command… ») la lie
dans `/usr/local/bin`. Ce n'est qu'un tuyau : Foundation seule, elle ne lie pas le moteur. Si
l'app n'est pas lancée, toute commande sauf l'aide la lance et attend sa prise (30 s au plus) ;
`silicontrol open` ne fait que cela. L'aide (`help`, `help <sujet>`, `<commande> --help`, aucun
argument) est servie par la commande elle-même, sans lancer l'app.

## La prise et le protocole

- **Socket Unix** `<bibliothèque>/silicontrol.sock` (par défaut
  `~/Library/Application Support/Siliconed/silicontrol.sock`, ou sous `SILICONED_ROOT`), droits `0600` :
  seul le compte qui a lancé l'app y accède. Aucun port réseau. Elle disparaît quand l'app quitte ;
  une prise morte est remplacée au lancement suivant, et un second Siliconed sur la même
  bibliothèque n'écoute pas.
- **Une connexion par commande.** Le client écrit une ligne JSON,
  `{"argv": ["add", "--batch", "2", "…"], "cwd": "/dossier/courant"}` ; l'app répond par des lignes
  JSON puis ferme. Les lignes intermédiaires portent `"type"` (`added`, `image`, `sheet`, `start`, `stage`,
  `step`, `end`) ; la dernière porte `"ok"` (`true` ou `false`, avec `"error"`, `"message"` et
  souvent `"hint"` en cas d'échec). Sans le client :
  `printf '{"argv":["status"]}\n' | nc -U ~/Library/Application\ Support/Siliconed/silicontrol.sock`.
- **Toute l'analyse est dans l'app** (`Sources/SiliconedApp/RemoteControl.swift`) : `argv` est la
  ligne de commande telle quelle et le client n'est qu'un tuyau. Seuls `open` et l'aide sont
  traités côté client, car ils doivent marcher app fermée. Les chemins relatifs se résolvent contre `cwd`. Les
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
  `"ok": false` ; `2` erreur d'usage ; `3` l'app est injoignable (pas lancé et non lançable, ou
  ne répond pas).
- **Les signaux.** L'`AppState` émet `Signal` (`started`, `stage`, `step`, `image`, `sheet` quand la
  dernière case d'une grille a fini, `finished` avec
  son `Issue`) à qui s'abonne ; `wait` et `follow` s'abonnent le temps de leur connexion et se
  désabonnent au premier envoi qui échoue (client parti).
- **SIGPIPE est ignoré** (`signal(SIGPIPE, SIG_IGN)`, plus `SO_NOSIGPIPE` sur chaque socket
  client), pour qu'un client qui raccroche ne puisse pas tuer l'app : mesuré, un client qui se
  connectait et fermait sans lire emportait l'app.
- **La plomberie ne touche pas le fil principal** : accepter et lire se font sur des files de fond
  (`Socket`, `Connection`), écrire sur la file de chaque client, si bien qu'un long `wait` ne bloque
  ni l'app ni les autres clients. Seules les commandes passent sur le fil principal, comme un clic.

## Ce qui a été mesuré

- **Mêmes bits** : l'image de l'app = un rendu de la bibliothèque appelée directement, même requête
  (512², graine 42, octets décodés identiques). De même pour une variation de variation mise en file
  par `silicontrol vary` : refaite depuis la recette de son PNG, elle sort au bit près.
- **App Nap** : en arrière-plan, macOS freinait l'app : 64 s au lieu de 44 pour un 512², un pas
  à 85 s. L'app tient une activité « rendu » (`ProcessInfo.beginActivity`) du premier travail lancé
  jusqu'à la file vide : **45,9 s** en arrière-plan (44,3 s pour le même rendu hors de l'app).

## Un autre processus qui rend

Un rendu par machine : le moteur prend un verrou (`/private/tmp/siliconed-render.lock`) pour chaque
rendu, si bien qu'un autre programme bâti sur la bibliothèque est refusé avec
`render_already_running` pendant que l'app rend, et inversement (deux modèles dans 16 Go : le
système swapperait et les deux chronos ne voudraient plus rien dire). Un tel programme peut mettre son
rendu dans la file de l'app avec `silicontrol add`.
