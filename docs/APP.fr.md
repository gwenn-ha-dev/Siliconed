[English](APP.md) · Français

# L'app Siliconed

`Siliconed.app` transforme un prompt en image, retouche vos images quand vous lui dites quoi changer, et tient une file et un historique de tout ça. Pas de compte, pas de cloud, pas de roue qui tourne en attendant un serveur quelque part : juste votre Mac, qui travaille dur et vous dit ce qu'il fait. Elle suit la langue du système (anglais, français, allemand, espagnol, italien, japonais ou chinois simplifié).

![La fenêtre : le rack à gauche (prompt, encodeur de texte, débruiteur, décodeur, chacun avec ses réglages et son temps), une image Krea 2 Turbo en 1248×832 à droite — une Irlandaise rousse qui sort des vagues en riant, sa planche de surf sous le bras, sous un ciel ensoleillé —, l'historique en dessous](images/window.jpg)

*La fenêtre, avec la photo de Krea 2 Turbo de la section [Les modèles](#les-modèles) : le rack montre la chaîne, les réglages et le temps de chaque étage ; l'historique court sous le canevas.*

Le moteur derrière elle, avec les mesures : [Le moteur de Siliconed](ENGINE.fr.md).

## Créer

- **Cinq modèles.** Krea 2 Turbo, Z-Image Turbo, Qwen-Image-2.1 Turbo, Anima Turbo et MOSS-Image, chacun sous sa propre licence ([Les modèles](#les-modèles)). Tous cinq lisent le prompt avec un modèle de langage.
- **La chaîne, visible.** Pas de tuyauterie cachée : le rack montre le pipeline de haut en bas (prompt, encodeurs, débruiteur, décodeur), avec des fils typés et l'étage actif allumé ; chaque étage porte ses réglages et affiche son temps. Sous le titre, deux petites étiquettes disent ce qui est stocké et ce qui calcule : `poids int8` (ou `Q8_0`, `Q4_K_M`, `bf16`…, lu dans ce qui est installé) à côté de `calcule en fp32`. Sur le débruiteur, l'étiquette dit ce que font les accélérateurs de ce Mac : `calcule en fp32 · ANE` (une partie de ses produits sur le Neural Engine, en fp16), `calcule en fp32 · GPU bf16` (une partie sur les unités tensorielles du GPU, en bf16), les deux, ou un simple `calcule en fp32`. L'interrupteur **Accélérateurs** de la boîte du débruiteur les allume ou les éteint — éteint, le DiT est en fp32 de bout en bout —, et il est grisé, avec sa raison, sur un Mac dont la mesure ne leur donne rien à faire ([comment](ENGINE.fr.md#le-neural-engine)).
- **Plus de détail** avec Z-Image et Qwen-Image-2.1 : « Détail : Normal · Plus · Maximum » ajoute de la texture fine, sans temps en plus.
- **Rendu plus rapide** : dès 1024 px de chaque côté, les premières évaluations sur une grille en demi-taille, le reste en pleine taille. De 17 à 31 % de temps en moins sur la Compact en 1024² (7 % pour MOSS-Image), toujours 8 évaluations, une image un peu différente. C'est une option Bêta (Réglages › Avancé › Bêta), éteinte par défaut, qui ajoute un interrupteur à la boîte du débruiteur, à côté de celui du Neural Engine : un choix par modèle, éteint pour toutes les versions tant que vous ne l'allumez pas, de sorte que les images restent celles vérifiées contre la référence. Option éteinte, aucun rendu n'est plus rapide ; vos choix reviennent quand vous la rallumez. L'interrupteur est grisé, avec sa raison, quand il ne ferait rien pour la taille réglée (un côté sous 1024 px), une LoRA, le Détail ou une édition. Le choix est enregistré avec l'image, donc Rendre à nouveau redonne la même ([comment](ENGINE.fr.md#rendu-plus-rapide)).
- **Les variations** : des cousines d'une image, subtiles ou marquées, aux mêmes réglages (le bouton sous l'image, ou son menu contextuel). **Rendre à nouveau (même graine)**, ⌘R, va dans l'autre sens : la même image, pour vérifier qu'on ne rêve pas. Une graine est celle de diffusers sur les cinq modèles (le même bruit de départ, au bit près). Le nombre d'images d'un coup, c'est le **Lot** du rack.
- **Agrandir ×4** avec la super-résolution de macOS (macOS 26) : aucun prompt, rien d'ajouté, bords et textures affinés ; 1024² devient 4096² en une ou deux secondes une fois le modèle de macOS chaud. Elle prend les images jusqu'à 1920 px de côté, et une image agrandie ne s'agrandit pas une seconde fois. L'original reste dans l'historique.
- **Les LoRA** sur les cinq modèles.
- **Reproductible au bit près.** Même prompt, même graine, mêmes réglages, même version, sur le même Mac : la même image, octet pour octet. Deux choses peuvent changer les pixels : une nouvelle version de l'app (la 0.12 l'a fait, de quelques unités sur le dernier chiffre, quand son arithmétique s'est mise à arrondir comme la référence), et une nouvelle mesure de ce Mac, qui peut répartir le travail en d'autres lignes ([Mesurer ce Mac](#mesurer-ce-mac)).

Chaque côté doit être d'au moins 512 et multiple de 16 ; la surface n'a pas de plafond fixe : le budget mémoire décide. Ce qu'un format coûte se calcule sur les formes du modèle avant le premier calcul et se juge contre le budget mémoire du Mac ([comment](ENGINE.fr.md#la-mémoire-et-le-budget)) : un Mac qui ne peut pas le tenir le refuse, et dit lequel le peut. Le sélecteur de taille propose 1536² et 2048² à côté de 1024², et grise ce que le budget de ce Mac refuse ; toute autre taille se tape. Les formats au-delà de 1024×1536 sont chiffrés par les mêmes formules ; Anima Turbo en 2048² a été rendu sur le Mac de référence de 16 Go, avec un pic de 5,4 Go. Sous 512² les modèles sont hors de leur domaine. Qwen-Image-2.1 veut des multiples de 32 (son VAE divise par 16), et Anima au plus 2048 px de côté (les positions que sa référence calcule) ; l'un et l'autre sont refusés avant tout calcul, de même que tout côté au-delà de 8192 px.

**Format en deux échelles** (Réglages › Avancé › Bêta, éteint par défaut) sépare en deux le menu Format du rack : une proportion (1:1, 4:5, 3:4, 2:3, 9:16 et, couchés, 5:4, 4:3, 3:2, 16:9), puis une résolution, 1K (environ 1 mégapixel) ou 2K (environ 4) ; l'app calcule les pixels, toujours multiples de 32 pour que tous les modèles les prennent. Le crayon, le contrôle de mémoire et les limites de chaque modèle ne changent pas, et le menu par défaut reste le défaut.

| Proportion | 1K | 2K |
|---|---|---|
| 1:1 | 1024 × 1024 | 2048 × 2048 |
| 4:5 | 896 × 1120 | 1792 × 2240 |
| 3:4 | 864 × 1152 | 1728 × 2304 |
| 2:3 | 832 × 1248 | 1664 × 2496 |
| 9:16 | 768 × 1344 ≈ | 1440 × 2560 |

Les paysages sont les mêmes tailles couchées (le 3:2 fait 1248 × 832 en 1K). Le ≈ : en 1K, un 9:16 exact est hors d'atteinte : un 9:16 exact en multiples de 32 ne va que par pas de 288 × 512 (576 × 1024, 864 × 1536), donc la 1K garde 768 × 1344, qui est en réalité un 4:7 (1,6 % d'écart) et s'affiche « ≈ 9:16 » ; pareil pour le 16:9 couché. En 2K, 1440 × 2560 est un 9:16 exact. Anima s'arrête à 2048 px de côté : sa 2K est le carré seul, et les autres tailles 2K sont grisées avec cette raison. `silicontrol add` et `grid` prennent les mêmes tailles écrites `3:4@2K`, option Bêta allumée ou non ([Télécommande](REMOTE-CONTROL.fr.md)).

## Éditer

- **L'édition par instruction** avec Qwen-Image-2.1 Turbo : donnez-lui 1 à 3 images et une phrase (« remplace le ciel nuageux par un ciel bleu ») ; son encodeur voit les images. Dans l'app, une édition est plafonnée à une surface de 1024².
- **Trois façons de comparer** deux images, ou une édition ou un agrandissement et son original : Rideau, Alterner ou Côte à côte, à choisir dans la barre d'outils du canevas.
  - **Rideau** : faites glisser une ligne sur les images pour voir ce qui a changé. Zoomez vers le curseur pour regarder un détail : en pinçant, ou à la molette d'une souris (une molette horizontale déplace toujours l'image).
  - **Alterner** : les deux mêmes images, une à la fois, au même endroit et au même zoom ; cliquez sur l'image, ou appuyez sur ← pour la première et → pour la seconde.
- **Les sujets nommés** : nommez une personne, un objet ou un style à partir d'une photo, puis citez-le par `@nom` dans le texte de l'Atelier de prompts (voir Préparer, plus bas) ; l'atelier remet au rack le prompt traduit et les images numérotées.
- **Un historique comme dans Photos** : flèches, ⌫ et ⌘Z, sélection multiple, export, glisser-déposer, Partager. Depuis un terminal : `silicontrol remove`, `undo`, `crop`, `again`, `reuse`, `save` (PNG, JPEG, HEIC).

![Une édition par Qwen-Image-2.1 dans la fenêtre : la photo Krea 2 de la surfeuse en image 1 dans la boîte Images, l'instruction « Convert this photo to a western comic book style with cel shading », le résultat en cel shading dans le canevas](images/edit.jpg)

*Qwen-Image-2.1 Turbo, Standard : « Convert this photo to a western comic book style with cel shading », la photo Krea 2 en image 1, 1248×832, graine 42.*

![Une édition par Qwen-Image-2.1, sous le rideau qui glisse : la surfeuse de Krea 2 qui sort des vagues en combinaison, puis en chemisier de lin blanc, son visage, ses cheveux, sa planche et la mer inchangés](images/outfit.gif)

*Une édition qui ne change que ce qu'on lui dit : « Replace her wetsuit with a white linen blouse. Keep her face, hair, surfboard, pose and the beach unchanged. », la photo Krea 2 en image 1.*

## Préparer (et laissez un modèle taper)

L'**Atelier de prompts** (Fenêtre › Atelier de prompts, ⌥⌘L, ou la baguette à côté du Prompt du rack) est une fenêtre à part où l'on prépare les entrées d'un rendu : des images d'un côté, un texte de l'autre, et un modèle qui aide à écrire. Le rack reste ce qu'il est ; l'atelier remplit ses boîtes.

- **Images** : déposez-les, ouvrez-les (+), ou envoyez-en une depuis l'historique (Ouvrir dans l'atelier de prompts). Chaque image a un rôle, choisi dans son menu : **Image de départ** (l'image 1, celle que le modèle édite ; elle donne le format), **Référence** (ce que le modèle voit comme contexte) ou **Décrire seulement** (elle reste dans l'atelier). Avec Qwen-Image-2.1, la première image déposée est l'image de départ et les suivantes sont des références. Avec Z-Image, Anima, Krea 2 et MOSS-Image, une image ne devient l'image de départ que par son bouton **Partir de cette image** (img2img) : le rendu part d'elle au lieu du bruit et garde sa composition, et sa **force** (de 0 à 100 % ; pour commencer 65 % sur Krea 2, 75 % sur Z-Image et Anima, 80 % sur MOSS-Image) dit jusqu'où il s'en éloigne ; portée, elle paraît dans la boîte Images du rack avec sa vignette et sa force, à retoucher entre deux rendus, et la taille prend le format le plus proche de ses proportions (un autre reste à vous : le centre est gardé). Le PNG écrit la force, pas l'image. Ailleurs les images sont en Décrire seulement. Un rendu prend 3 images au plus.
- **Sujets** : Nommer… donne à une image un nom, un genre et un cadre, et le texte la cite par `@nom` (complété à la frappe). Un sujet ne part au rendu que si le texte le cite, après les images que les rôles envoient, dans l'ordre où le texte les nomme. Ils vivent en mémoire seulement (rien n'est écrit sur le disque, comme l'historique). `silicontrol subjects` agit sur les mêmes.
- **Texte** : un seul texte, toujours modifiable à la main. **Réécrire** en fait un texte écrit pour le modèle que vous choisissez (« Écrire pour ») et garde vos `@noms` (un nom perdu est signalé, jamais retiré en silence) ; les ajouts et les retraits sont marqués, et Modifier retire les marques. **Décrire**, sur une image, écrit dans le texte ce qu'elle montre : il remplit un texte vide, sinon il s'insère au curseur ou remplace la sélection (à la fin quand le texte n'est pas à l'écran). L'un et l'autre écrivent dans le texte au fil du rédacteur. ⌘Z annule tout geste du modèle. Un `@` n'est un sujet que pour un modèle qui lit les images ; pour Anima, il reste l'`@` de l'artiste. Un `@nom` ne peut pas se trouver dans un groupe `{a|b}`. `silicontrol rewrite` et `silicontrol describe` sont ces deux mêmes gestes pour un terminal (même rédacteur, mêmes consignes, même protection des `@noms`, mêmes attentes) ; le texte revient par la socket, jamais dans le champ.
- **Le rack recevra** dit, avant que rien n'arrive, le prompt traduit (chaque `@nom` devenu « image N »), les images numérotées, le format que donne l'image 1 et ce que **Porter dans le rack** remplace. Le prompt du rack revient par ⌘Z dans son champ (ou Édition › Annuler) ; ses images n'ont pas d'annulation. Un refus (sujet inconnu, plus de 3 images) s'y dit en une phrase.
- **En suivant un autre Mac** (voir plus bas) : Réécrire et Décrire sont éteints et le disent ; les images, les rôles, les sujets et Porter remplissent le rack que vous voyez, qui envoie ses références au Mac qui rend.
- **Réécrire et Décrire sont en bêta**, et ce sont les seuls gestes qui ont besoin du rédacteur (plus bas) : pendant un rendu, ou sans rédacteur, eux seuls attendent, et le reste de l'atelier marche.
- **Le rédacteur est un modèle que vous avez déjà** : l'encodeur de texte de Qwen-Image-2.1, Qwen3-VL-8B, mis à écrire. Il ne manque au modèle installé que sa couche de sortie, et « Installer le rédacteur » (ou `silicontrol install writer`) la télécharge dans le même dépôt (1,25 Go, par plage d'octets ; pas de licence propre, c'est celle de Qwen-Image-2.1). Il faut Qwen-Image-2.1 installé, et il lit ses couches selon un plan choisi dans le budget de mémoire avant le premier mot : sur un Mac de 16 Go, toutes celles de la Compact en mémoire ; sur un Mac de 8 Go, ou avec la Standard sur 16 Go, une partie depuis le disque à chaque jeton, plus lentement. La fenêtre dit ce qui manque.
- **Nos consignes, une par modèle**, tirées de ce que chaque éditeur dit de ses prompts. Chaque élément de votre prompt est gardé, le texte entre guillemets mot pour mot ; rien n'est retiré, adouci ni moralisé.
- **Jamais pendant un rendu** : une image qui démarre arrête le rédacteur, et Réécrire et Décrire attendent. Sur un M1 Pro avec la Compact, le premier mot vient après environ 10 s pour Réécrire et 25 s pour Décrire, qui lit d'abord l'image, puis environ 12 jetons par seconde, sur le GPU. Sa sortie gloutonne est celle de transformers, jeton pour jeton ([VERIFY](VERIFY.fr.md)).
- **Incognito** : textes, images et sujets restent en mémoire, sans fichier ni historique ; le rédacteur rend ses poids après chaque geste, si bien que la mémoire du Mac est libre entre deux.

## Explorer (esquissez d'abord, engagez-vous après)

Le panneau d'exploration construit une grille dans le canevas. Ses axes : les graines, les pas, la force d'une LoRA, des LoRA à essayer, les modèles, les variantes écrites entre accolades dans le prompt, chaque image d'une édition ou, avec Plus de solveurs allumé, les solveurs. Chaque case est une esquisse, arrêtée après quelques pas, à 512 sur le petit côté (avec Qwen-Image-2.1, une image finie) ; dès qu'un modèle a rendu à cette taille, le panneau dit ce que la grille coûtera sur ce Mac avant de la lancer. « Finir cette image » rend la case choisie jusqu'au bout avec la même graine : l'image que son esquisse annonçait. Depuis un terminal : `silicontrol finish g2c3` et `silicontrol sheet g2`.

![Une exploration avec Anima Turbo et la LoRA inCase qui se remplit dans le canevas : des esquisses d'une Irlandaise rousse qui rit sur une plage ensoleillée au pied des falaises, la couleur de la combinaison en colonnes (teal, coral red, mustard yellow, navy) et la force de la LoRA en lignes (0 ; 0,5 ; 1), les axes en étiquettes, la dernière ligne encore en cours](images/grid.jpg)

*Une exploration avec Anima Turbo et la LoRA inCase : la couleur de la combinaison écrite entre accolades dans le prompt (`{teal|coral red|mustard yellow|navy}`) en colonnes, inCase à 0, 0,5 et 1 en lignes ; des esquisses en 512×768.*

![La case finie dans la fenêtre : une illustration d'Anima Turbo avec la LoRA inCase, une jeune Irlandaise rousse qui rit, en combinaison bleu marine, en gros plan, des falaises et des vagues qui déferlent derrière elle](images/anima.jpg)

*La case choisie, finie : Anima Turbo avec la LoRA inCase à 1, « A close-up portrait of a young Irish woman with long wavy auburn red hair, pale freckled skin and green eyes, laughing, wearing a navy shorty wetsuit, on a sunny cliff-top beach in the west of Ireland. Big Atlantic waves breaking into white spray below green sea cliffs, deep blue sky, bright sunlight, wind in her hair. », 512×768, graine 4001.*

## La gestion des modèles

- **Rien n'est préinstallé.** Vous choisissez un modèle ; avant le premier octet téléchargé, l'app affiche sa licence et la place qu'il prend sur disque, et vous demande d'accepter. Chaque fichier est figé à une révision et vérifié par sha256.
- **Standard, Compact ou Légère.** Chaque modèle s'installe avec les poids de l'éditeur (Standard), avec des poids 8 bits publiés par des tiers (Compact, le défaut et celle que nous recommandons : 19,5 Go au lieu de 34,4 pour Krea 2, 11,7 Go au lieu de 20,4 pour Z-Image, 17,3 Go au lieu de 31,2 pour Qwen-Image-2.1, 4,10 Go au lieu de 5,65 pour Anima, 17,9 Go au lieu de 27,5 pour MOSS-Image ; une image légèrement différente), ou avec le transformeur en 4 à 6 bits tel qu'un tiers le publie (GGUF Q4_K_M ; Légère : 12,8 Go, 9,5 Go, 14,2 Go et 3,31 Go, le moins lu sur le disque à chaque pas ; un autre jeu de poids, une image qui peut se composer autrement). Pour Krea 2, Z-Image, Qwen-Image-2.1 et MOSS-Image, l'encodeur de texte de la Compact est lui aussi en 8 bits, et la Légère le reprend ; celui d'Anima reste le bf16 de l'éditeur dans les trois, faute de version 8 bits publiée. MOSS-Image n'a pas de Légère. La Légère est présélectionnée sur un Mac de 8 Go de mémoire, la Compact ailleurs ; vous choisissez avant d'installer. Une version à la fois : passer à l'autre la remplace ; l'installée continue de rendre jusqu'à ce que la nouvelle soit complète, puis libère sa place ; si le disque ne peut pas tenir les deux, l'app le dit avant l'acceptation.
- **Les imports Civitai et Hugging Face.** Les LoRA (formats Civitai, Comfy, kohya, diffusers) et le DiT d'un fine-tune (`.safetensors`, ou `.gguf` de Q4 à Q8_0), par glisser-déposer ou ⌘O ; un checkpoint Civitai d'une famille connue s'importe tel quel et emprunte l'encodeur et le VAE de sa famille. Déposé avant que sa famille soit installée, il ouvre l'installation de celle-ci — licence, encodeur et VAE, jamais le DiT de la famille — et s'importe une fois l'installation faite. Un fichier quantifié reste sur le disque tel que publié, jamais requantifié (Z-Image : 6,2 Go en 8 bits, environ 5 Go en GGUF Q4_K_M, au lieu de 12,3 Go ; chaque pas un peu plus lent, le calcul reste en fp32) ; sous 4 bits, le fichier est refusé. Une LoRA importée prend le nom de son fichier ; le nom donné par son auteur et ses mots déclencheurs s'affichent en infobulle.
- **Statistiques.** Une palette flottante (⌥⌘I, Fenêtre › Statistiques, ou le bouton de la barre d'outils) : où en est le rendu, ce que chaque étage a coûté, ce qui reste pour lui et pour la file, et ce que pesait le dernier rendu. Rien n'est estimé d'après une constante : un modèle ou un format jamais rendu ici n'a pas d'estimation, et la palette le dit.

### Mesurer ce Mac

Avant son premier rendu, puis après une mise à jour ou un nouveau macOS, Siliconed chronomètre ce que calcule chaque moteur de votre Mac. Une minute environ, cinq au plus, aucun modèle, et toujours une carte : rien n'est jamais refusé, et la confiance qu'on peut avoir dans chaque chiffre est écrite à côté. Vous pouvez l'ouvrir quand vous voulez : Aide › Mesurer ce Mac…, le titre Chaîne et l'icône du Mac en tête du rack (grisée tant que ce Mac n'est pas mesuré), ou Réglages › Avancé › Ce Mac. Elle n'est pas dans la fenêtre Modèles : installer un modèle et mesurer un Mac sont deux métiers différents.

La feuille montre, colonnes titrées : la vitesse de chaque route, sa part de la crête du GPU, si son résultat est égal au résultat exact (Exact ✓), ce que chaque moteur garde de sa vitesse quand ils travaillent ensemble, la mémoire que la mesure a tenue et sa durée. Ce qu'elle a trouvé est dit en phrases simples, chacune avec son effet sur le plan. Le partage est une option, jamais le bouton par défaut : Partager… ▸ Ouvrir un signalement GitHub, ou Copier le JSON.

Une mesure que vous demandez s'applique au **rendu suivant**, jamais à un rendu en cours : à partir de là, une même graine peut donner une image un peu différente de celle d'avant la mesure. Quand un rendu s'est écarté de plus de 10 % de ce que la mesure prévoyait, la barre latérale le dit, une fois, à côté du bouton qui mesure de nouveau.

Les modèles vivent dans `~/Library/Application Support/Siliconed` (modifiable avec `SILICONED_ROOT`). Pour libérer le disque interne, Réglages › Déplacer la bibliothèque… les déplace sur un autre — un disque externe, un partage réseau : chaque fichier est copié, vérifié, puis effacé du Mac. Siliconed ne copie alors sur le Mac que les modèles que vous rendez, le temps de la session, et les efface en quittant ; les licences acceptées et la mesure de ce Mac restent sur le Mac. Débranché, le disque est signalé : rien ne se rend avant son retour. Depuis un terminal : `silicontrol library move DOSSIER` (sans `--yes`, elle dit ce qui serait copié et la place qu'il y a).

![La fenêtre Modèles : les cinq modèles des éditeurs installés en Compacte, chacun avec sa licence et sa place sur le disque (Z-Image Turbo 11,74 Go, Qwen-Image-2.1 Turbo 17,3 Go, Anima Turbo 4,1 Go, Krea 2 Turbo 19,5 Go, MOSS-Image 17,88 Go, marqué bêta), les LoRA en dessous, la place occupée et libre en bas](images/models.jpg)

## Les modèles

Seulement des modèles récents dont l'encodeur de texte est un modèle de langage — ni Stable Diffusion, ni SDXL, ni Flux.1.

| Modèle | Éditeur | Ce qu'il fait | Licence |
|---|---|---|---|
| [Krea 2 Turbo](https://huggingface.co/krea/Krea-2-Turbo) | Krea | texte vers image (des images esthétiques, de la photographie à la peinture), LoRA | Krea 2 Community : usage commercial sous 1 M$ de chiffre d'affaires annuel, un filtre de contenu obligatoire au déploiement |
| [Z-Image Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) | Tongyi-MAI | texte vers image, LoRA | Apache 2.0 |
| [Qwen-Image-2.1 Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) | Qwen | texte vers image, édition par instruction avec 1 à 3 images, LoRA | Qwen Research (non commerciale) |
| [Anima Turbo v1.1](https://huggingface.co/circlestone-labs/Anima) | CircleStone Labs × Comfy Org | texte vers image (illustration, peinture, anime ; pas de photoréalisme, par choix de l'éditeur), LoRA | CircleStone Labs Non-Commercial ; ses images sont utilisables commercialement |
| [MOSS-Image](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT) | OpenMOSS, sous la LoRA 8 pas de lodestones | texte vers image, LoRA ; bêta | Apache 2.0 |

### Krea 2 Turbo, le plus gros des cinq

[Krea 2 Turbo](https://huggingface.co/krea/Krea-2-Turbo), de Krea, est le quatrième modèle, arrivé dans la 0.12 : un DiT de 12,9 milliards de paramètres, le plus gros des cinq, pour des images esthétiques, de la photographie à la peinture. Il lit le prompt avec Qwen3-VL-4B et fait toute sa recette de 8 évaluations en fp32 sur les poids bf16 de l'éditeur, avec le [rendu plus rapide](ENGINE.fr.md#rendu-plus-rapide) en option Bêta (temps et écarts : [Les chiffres](ENGINE.fr.md#les-chiffres)), ses poids lus depuis le disque dans le budget mémoire. Il s'installe en Standard, Compact ou Légère (tailles : [La gestion des modèles](#la-gestion-des-modèles)), depuis des copies ouvertes prouvées identiques aux fichiers de Krea (voir plus bas), sous la licence Krea 2 Community (tableau ci-dessus).

![Une photo de Krea 2 Turbo en 1248×832, en contre-plongée : une jeune Irlandaise aux longs cheveux auburn ondulés et aux taches de rousseur, en combinaison shorty à manches courtes, qui sort des vagues en riant, sa planche de surf sous le bras, à contre-jour d'un soleil doré bas avec un reflet d'objectif, des embruns lumineux, une eau turquoise et un ciel bleu profond](images/krea2.jpg)

*Krea 2 Turbo, Standard, 1248×832, 8 pas, graine 3001 : « Editorial surf photograph, low angle, of a young Irish woman with long wavy auburn red hair, pale freckled skin and green eyes, in a short-sleeved shorty wetsuit, running out of the surf laughing, surfboard under her arm, on a rare sunny day in the west of Ireland. Golden late-afternoon backlight, sun flare, spray glowing in the light, turquoise water, deep blue sky, high contrast, vivid colors, sharp focus, shot on a 50 mm lens. »*

Krea 2 Turbo ne télécharge rien de [krea/Krea-2-Turbo](https://huggingface.co/krea/Krea-2-Turbo) : ce dépôt demande un compte Hugging Face, et l'app n'a pas de connexion. Chacun de ses fichiers vient d'une copie ouverte prouvée identique à celle de Krea : le DiT et l'encodeur de texte Qwen3-VL-4B en bf16 de [Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2) (le DiT, le même fichier ; l'encodeur, égal tenseur pour tenseur), le VAE de [Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) (le VAE fp32 de Krea est ce bf16 élargi, valeur pour valeur), le tokenizer de [Qwen/Qwen-Image-2.1-Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) (le même fichier) ; les fichiers de configuration de Krea sont embarqués tels que publiés. Sa Compact prend le DiT et l'encodeur de texte en int8 convrot de [Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2) ; sa Légère, le GGUF Q4_K_M de [vantagewithai/Krea-2-Turbo-GGUF](https://huggingface.co/vantagewithai/Krea-2-Turbo-GGUF) avec l'encodeur de texte de la Compact.

### MOSS-Image, distillé par sa communauté

[MOSS-Image](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT), de l'équipe OpenMOSS, est le cinquième modèle, arrivé dans la 0.16 : un DiT mono-flux de 5,1 milliards de paramètres qui lit le prompt avec Qwen3.5-9B — un modèle de langage hybride, trois couches sur quatre en attention linéaire (Gated DeltaNet) — et décode avec le VAE de FLUX.2. Son éditeur n'a pas publié de version rapide ; la communauté, si : Siliconed fait tourner MOSS-Image-SFT sous la [LoRA DMD de lodestones](https://huggingface.co/lodestones/MOSS-Image-SFT-8-steps-dmd) à force 1, 8 évaluations, sans CFG, la LoRA gardée à côté des poids en fp32 (fusionnée dans les poids bf16, ses petits deltas se perdraient dans l'arrondi). Les deux sont vérifiés contre la référence, LoRA comprise ([VERIFY.fr.md](VERIFY.fr.md)). Texte vers image, et vos LoRA par-dessus celle des 8 pas — le `qkv` fusionné de ComfyUI et d'ai-toolkit est séparé exactement, et les LoRA entraînées sur MOSS-Image Base s'appliquent aussi, comme la communauté s'en sert ; ni Détail, ni édition. Le [rendu plus rapide](ENGINE.fr.md#rendu-plus-rapide) n'y prend qu'une évaluation en demi-taille : dès la deuxième, MOSS-Image perd du détail fin (la peau, les dos de livres) là où les autres modèles le gardent. Il est marqué **bêta** dans l'app, parce que sa recette en 8 pas est celle de la communauté, pas celle de l'éditeur ; il sort de bêta quand OpenMOSS publie sa propre version rapide. Il s'installe en Standard ou en Compact (tailles : [La gestion des modèles](#la-gestion-des-modèles)), sous Apache 2.0, LoRA comprise.

![La fenêtre avec MOSS-Image dans le rack : un portrait serré en 1248×832 d'une jeune Irlandaise aux longs cheveux auburn ondulés et aux taches de rousseur, en combinaison shorty, qui rit, des mèches mouillées soufflées sur le visage, des gouttes de pluie sur la peau, l'Atlantique qui déferle derrière elle sous un ciel nuageux](images/moss.jpg)

*MOSS-Image, Compacte, 1248×832, 8 pas, graine 6001 : « Close-up portrait of a young Irish woman with long wavy auburn red hair, pale freckled skin and green eyes, in a short-sleeved shorty wetsuit, the wind whipping wet strands of hair across her face, squinting and laughing. Hard low sunlight from the side just after a rain shower, dark storm clouds over the Atlantic behind her, raindrops on her skin, shot on Portra 400, shallow depth of field. »*

La Standard prend le DiT, l'encodeur de texte Qwen3.5-9B (son modèle de langage seul, sans la tour de vision), le VAE et le tokenizer d'[OpenMOSS-Team/MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT), et la LoRA de [lodestones/MOSS-Image-SFT-8-steps-dmd](https://huggingface.co/lodestones/MOSS-Image-SFT-8-steps-dmd). Sa Compact prend le DiT et l'encodeur de texte en int8 convrot de [Comfy-Org/MOSS-Image](https://huggingface.co/Comfy-Org/MOSS-Image), avec la même LoRA, le même VAE et le même tokenizer. Pas encore de Légère : un DiT en 4 bits est publié, mais pas encore vérifié ; sur un Mac de 8 Go, la Compact est présélectionnée, et son pic reste dans le budget de ce Mac.

### D'où viennent les fichiers

Les autres versions Compact utilisent des poids 8 bits publiés par des tiers : [unsloth/Z-Image-Turbo-GGUF](https://huggingface.co/unsloth/Z-Image-Turbo-GGUF), [Disty0/Z-Image-Turbo-SDNQ-int8](https://huggingface.co/Disty0/Z-Image-Turbo-SDNQ-int8), [unsloth/Qwen-Image-2.1-FP8](https://huggingface.co/unsloth/Qwen-Image-2.1-FP8), [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1) ; les versions Légères, le GGUF Q4_K_M de [unsloth/Z-Image-Turbo-GGUF](https://huggingface.co/unsloth/Z-Image-Turbo-GGUF) et d'[unsloth/Qwen-Image-2.1-Turbo-GGUF](https://huggingface.co/unsloth/Qwen-Image-2.1-Turbo-GGUF) avec l'encodeur de texte de la Compact. La Compact et la Légère d'Anima prennent le DiT en GGUF Q8_0 et Q4_K_M de [vanes430/Anima-Turbo-V1.1-GGUF](https://huggingface.co/vanes430/Anima-Turbo-V1.1-GGUF) et gardent l'encodeur de texte de l'éditeur en bf16. Les révisions et les sommes sha256 sont dans `Sources/Siliconed/Forge/Installation.swift`.

## Calculer sur un autre Mac

- **Un Mac rend pour un autre, sur le même réseau.** Le MacBook du salon calcule sur le Mac mini de la pièce d'à côté : **Rendu › Calculer sur** y fait passer toute la fenêtre — sa file, l'avancement, l'aperçu, l'historique, les modèles, la bibliothèque, les licences et la mesure sont ceux de l'autre Mac —, tandis que le formulaire, la sélection, les exports et le presse-papiers restent sur le vôtre. Un menu « Sur : » à côté de Chaîne dit où l'on calcule, et le rack porte un liseré bleu tant que la liaison tient ; un menu vous ramène. Jamais deux vues à la fois. Depuis un terminal : `silicontrol macs` et `silicontrol compute NOM|this`.
- **Appairer une fois.** Sur le Mac qui se prête : Réglages › Autres Mac › Prêter ce Mac, puis Appairer un Mac…. Sur le vôtre : Réglages › Autres Mac › Appairer avec un Mac…, et choisissez-le. Les deux écrans affichent six chiffres ; n'acceptez sur le Mac prêteur que s'ils sont les mêmes. Les deux Mac doivent avoir la même version de Siliconed. Rien ne se connecte avant qu'un appairage soit accepté, et Oublier en défait un d'un côté comme de l'autre.
- **Chiffré, rien en clair.** TLS 1.2 avec une clé par Mac appairé, gardée au Trousseau. macOS demande une fois l'accès au réseau local, la première fois que vous prêtez ce Mac ou que vous l'appairez ; un Mac qui ne fait ni l'un ni l'autre n'est jamais interrogé.
- **Tout marche là-bas** : rendus, éditions avec vos références, variations, grilles et explorations, agrandissement et recadrage, annulation ; l'Atelier de prompts remplit le rack que vous voyez (images, sujets et Porter ; Réécrire et Décrire restent éteints) ; installer un modèle ou importer une LoRA sur le Mac prêteur, le mesurer. En retirer un attend un clic sur ce Mac. Chaque Mac ne voit que ses travaux et ses images, et le Mac prêteur reste dans son propre budget mémoire. `silicontrol` suit Calculer sur ([REMOTE-CONTROL.fr.md](REMOTE-CONTROL.fr.md)).

## Réglages (peu, exprès)

Siliconed › Réglages… (⌘,), trois onglets.

- **Général** : où vivent les modèles, avec Déplacer la bibliothèque…, Ouvrir une bibliothèque… et Ramener sur ce Mac. Depuis un terminal : `silicontrol import`, `uninstall` (avec `--yes`) et `library`. Puis **Enregistrer chaque image dans…**, éteint par défaut : l'historique reste en mémoire, comme partout ailleurs dans l'app. Allumé, il demande un dossier, et chaque image nouvelle (un rendu, une variation, une édition, un recadrage, un agrandissement, la planche d'une grille ; pas les esquisses d'une exploration) s'y enregistre aussi en PNG avec son prompt, sa graine et son modèle, sous le nom que lui donne Enregistrer…, jamais sur un autre fichier. Un échec d'écriture se dit comme celui d'Enregistrer…. Depuis un terminal : `silicontrol set save-images DOSSIER|off`.
- **Autres Mac** : prêter ce Mac, appairer avec un autre ([Calculer sur un autre Mac](#calculer-sur-un-autre-mac)).
- **Avancé**, dans cet ordre : Ce Mac (la mesure) ; Ligne de commande (installe `silicontrol`, plus bas) ; et **Bêta**, éteint par défaut et susceptible de changer : Plus de solveurs (ajoute les solveurs Euler a, DPM++ 2M et UniPC à la boîte du débruiteur et aux axes de l'exploration ; Euler, le défaut, est celui pour lequel les modèles ont été distillés), le Rendu plus rapide (ajoute son interrupteur à la boîte du débruiteur), le Format en deux échelles (le Format du rack devient une proportion et une résolution, plus bas), les Forces de LoRA négatives (curseurs de −1,5 à 1,5, pour les LoRA « curseur » ; une LoRA ordinaire est abîmée par une force négative). `silicontrol set beta` les bascule comme les interrupteurs.

Quand vous calculez sur un autre Mac, la mesure est celle de cet autre Mac, et la fenêtre le dit.

## Clavier

Perdu ? Le menu Aide contient l'Aide de Siliconed (ce guide), l'Aide de la télécommande (celui de `silicontrol`), Signaler un problème… et Mesurer ce Mac… ; Fenêtre › Afficher Siliconed ramène la fenêtre si vous l'avez fermée. Les menus donnent la liste complète des raccourcis ; voici ceux qui valent le coup. ⌘ est Commande, ⌥ Option, ⌃ Contrôle.

| Touches | Fait |
|---|---|
| ⌘↩ | Générer (Explorer dans le panneau d'exploration, Réécrire dans l'Atelier de prompts) |
| ⌘R | Rendre à nouveau (même graine) |
| ⌘. · ⌥⌘. | Arrêter le rendu · Vider la file et arrêter |
| ⌥⌘G | Exploration |
| ⌘S · ⇧⌘E | Enregistrer l'image… · Exporter les images… |
| ⇧⌘C | Copier l'image |
| ⌘O · ⇧⌘M | Importer un modèle ou une LoRA… · Modèles… |
| ⌘+ · ⌘− · ⌘0 | Zoom avant · Zoom arrière · Taille ajustée |
| ⌃⌘F | Plein écran (Échap, Espace ou un clic le quitte ; ← → changent d'image) |
| ⌥⌘← · ⌥⌘→ | Image précédente · suivante, de partout |
| ⌥⌘A | Sélectionner toutes les images |
| ⌥⌘I · ⌥⌘L | Statistiques · Atelier de prompts |
| ⌘, · ⌘W | Réglages · Fermer la fenêtre |

Et sans ⌘, quand rien n'est en cours de saisie dans un champ et que l'historique n'est pas vide : **← →** parcourent l'historique (avec ⇧ ils étendent la sélection ; avec le rideau affiché ils déplacent sa ligne, avec Alterner ils montrent A ou B), **Début** et **Fin** sautent à l'image la plus récente et à la plus ancienne, **Espace** ouvre le plein écran, **⌫** retire les images sélectionnées de l'historique, **C** ouvre le recadrage (Retour recadre, Échap le quitte), **Échap** quitte une comparaison ou vide une sélection multiple, ⌘A sélectionne tout, ⌘V colle une image dans la boîte Images. Échap quitte aussi le champ où vous tapez. Un curseur, un stepper ou un bouton atteint par Tab garde ses propres touches.

## Privé par construction (personne ne regarde)

Le rendu se fait sur votre Mac. Le seul trafic réseau est le téléchargement du modèle que vous avez accepté, l'unique requête à GitHub pour connaître la dernière version quand vous choisissez Siliconed › Rechercher les mises à jour… et, si vous l'ouvrez (Réglages › Autres Mac), la liaison chiffrée avec un autre Mac du réseau ; le diagnostic copie son rapport et ouvre un formulaire d'issue GitHub que vous choisissez de soumettre ou non. Le cache de rendu ne dure qu'une session : il est vidé au lancement et à la fermeture. Avant chaque rendu, le moteur vérifie le modèle, sa licence, le format et la mémoire qu'il lui faut, et refuse d'emblée un rendu au-delà du budget de ce Mac. L'app ne censure aucun prompt et n'ajoute aucun filtre de contenu ; la licence de chaque modèle s'applique toujours, et celle de Krea 2 Community en demande un à qui déploie le modèle.

## Pour les scripts et les agents : silicontrol

Pilote l'app ouverte depuis un terminal. Elle est livrée dans `Siliconed.app/Contents/Helpers/silicontrol` ; Réglages › Avancé › Ligne de commande la lie dans `/usr/local/bin`. Ce que vous ajoutez apparaît dans la fenêtre de l'app, et il n'y a qu'une file, jamais deux rendus à la fois. Édition › Copier comme commande de Terminal met dans le presse-papiers la commande `silicontrol` du formulaire tel qu'il est : pratique pour transformer un rendu réussi en script.

```
silicontrol help                 la référence complète (toujours commencer par là)
silicontrol models
silicontrol add --wait --out out "a woman posing in a library"
silicontrol status
```

Elle fait ce que fait la fenêtre, par les mêmes fonctions que ses boutons : elle bascule le rendu plus rapide ou les accélérateurs d'un modèle (`silicontrol set faster-rendering on|off --model krea2`, `set accelerators …`) et les options Bêta (`silicontrol set beta`), réécrit et décrit avec le rédacteur de l'Atelier de prompts (`silicontrol rewrite`, `describe`, `install writer`), et refait chez vous une série chronométrée de rendus (`silicontrol measure`) ; elle retire et annule, recadre, refait une image à sa graine (même depuis un PNG d'un autre jour), finit la case d'une exploration et ajoute sa planche à l'historique (`silicontrol remove`, `undo`, `crop`, `again`, `finish`, `sheet`), importe et désinstalle des modèles et des LoRA, bascule Calculer sur vers un Mac appairé et déplace la bibliothèque (`silicontrol import`, `uninstall`, `compute`, `library` ; ce qui retire ou déplace demande `--yes`) ; elle fait aussi des grilles, des variations, des agrandissements, suit un rendu pas à pas, et plus : `silicontrol help all` donne toutes les commandes. La sortie est en lignes JSON ; la dernière porte `"ok"`. Le dedans : [REMOTE-CONTROL.fr.md](REMOTE-CONTROL.fr.md).
