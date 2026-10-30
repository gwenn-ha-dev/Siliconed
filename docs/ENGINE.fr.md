[English](ENGINE.md) · Français

# Le moteur de Siliconed

Comment Siliconed mesure votre Mac, choisit ses noyaux, reste dans son budget de mémoire et prouve ses résultats, chiffres à l'appui. Pour ce que fait l'app, voir [L'app Siliconed](APP.fr.md).

## Un moteur mesuré sur votre Mac

**Deux Mac ne calculent pas pareil, alors Siliconed ne devine pas, et ne cherche pas votre Mac dans une table.** Ce qu'il sait du vôtre, il le lit ou le mesure, et chaque décision d'un rendu se calcule à partir de ces nombres. Quatre temps :

1. **Lire.** En quelques millisecondes, ce que le silicium annonce : cœurs et fréquence du GPU, grappes du CPU, génération du Neural Engine, mémoire. Ces faits ne décident rien : ils disent quels moteurs existent, et ils sont le dénominateur de « atteint / crête ». Aucun nom de puce n'apparaît dans le moteur : un Mac qui n'existe pas encore est servi par le même code.
2. **Mesurer.** Avant le premier rendu qui suit chaque installation ou mise à jour — et dès que le Mac ou macOS change —, en moins d'une minute et sans aucun modèle, seulement ce que le système tait : chaque moteur seul — le GPU, l'unité matricielle du CPU (AMX), le Neural Engine — sur des produits matriciels aux formes des modèles (un coût fixe, un débit, et pour le Neural Engine ce que coûte le soulèvement du poids, pour l'AMX ce que la profondeur du produit fait au débit ; l'AMX et le Neural Engine en chaînes d'appels bout à bout, parce que le deuxième appel d'une paire est la grappe qui chauffe encore), les marches du Neural Engine par lignes, l'attention par ses noyaux, des produits lus directement sur les poids 8 bits publiés, des produits bf16 sur les unités tensorielles du GPU ; puis chaque ensemble de moteurs ensemble — des chaînes de l'ensemble aux lignes que le modèle équilibre, chaque côté chronométré par sa propre horloge, car les moteurs se partagent la puissance et la mémoire et leurs débits ne s'additionnent pas : ce que chacun garde de son débit se lit comme un recouvrement, le côté qui finit le premier ayant couru à côté de l'autre tout du long —, la chaleur, le débit mémoire et le coût de préparation d'un poids. Chaque résultat est vérifié contre le résultat exact, calculé en double précision sur le CPU, et une route qui a mal calculé est écartée. Les tailles des sondes suivent la machine, pour que le Mac le plus lent tienne aussi la minute, et leur mémoire est comptée avant le premier produit, sous le budget. **La mesure sait aussi quand le Mac n'est pas libre.** Après chaque étape, elle demande si autant de fils que de cœurs de performance ont encore chacun un cœur, si une soumission vide atteint encore le GPU sans attendre, et si des appels identiques prennent le même temps ; une étape qui échoue à l'un des trois est mesurée de nouveau après une attente, tant que la minute laisse des secondes, et ce qui reste occupé passe le verdict à l'orange avec *Autres programmes* — ce n'est pas une charge qu'on lit (un cœur de travail de fond déplace un chiffre de quelques pour cent, un programme qui utilise le GPU le divise par deux), c'est l'effet. La mesure tourne dans un processus à part, parce qu'un programme que le Neural Engine refuse peut tuer le processus qui le charge : quand cela arrive, la route est écartée, la mesure repart sans elle, et le rapport dit que la route est morte — aucun rendu ne charge une route que la mesure n'a pas vue se charger.
3. **Modéliser.** Les sondes s'ajustent en **modèle de ce Mac** : quelques nombres par moteur — pour chaque noyau un coût fixe et un débit, la marche du Neural Engine, ce que chaque moteur garde de son débit à côté des autres et ce que coûte l'ensemble, ce que la chaleur retire sur un long rendu, le débit mémoire, ce que coûte la préparation d'un poids. Le rapport garde le modèle et les mesures brutes sur lesquelles il a été ajusté : chacun peut rejouer l'ajustement.
4. **Résoudre, puis boucler.** Avant chaque produit matriciel et chaque attention d'un rendu, un solveur prend ce bloc — sa forme, la forme de son poids, sa phase —, le chronomètre par le modèle sur chaque ensemble de moteurs, et retient l'ensemble et les lignes entières que prend chaque moteur pour finir au plus tôt, sous le budget de mémoire et les règles de précision (la Standard calcule en fp32 sur le GPU et l'AMX ; le Neural Engine, qui calcule en fp16, ne s'y joint que là où vous le laissez allumé, ci-dessous). Cette réponse est le **plan** : des lignes exactes par moteur et le chemin du poids, résolus une fois par sorte de bloc et gardés. Chaque rendu compare ensuite le temps que le plan avait prévu à celui qu'il a mesuré, classe par classe, et en rapporte l'écart ; il ne change jamais le plan ni l'image — un modèle qui a dérivé est une raison de mesurer de nouveau, pas un facteur à appliquer. Quand un rendu s'écarte de plus de 10 % de sa prévision, l'app le dit sous le bouton Générer et propose de mesurer de nouveau, une fois pour une mesure donnée ; la nouvelle s'applique au lancement suivant, de sorte que deux rendus d'une même graine dans une même session sont une même image.

La mesure achète de la vitesse, jamais une autre image au-delà de l'arithmétique qu'elle a vérifiée. La même graine donne les mêmes pixels d'un rendu à l'autre sur un même Mac avec une même mesure ; une autre mesure, ou un autre Mac, peut planifier d'autres lignes, et d'autres lignes somment dans un autre ordre ([VERIFY](VERIFY.fr.md#durées-et-pic-de-mémoire)). Le fp16 du Neural Engine, c'est vous qui le choisissez, par modèle (ci-dessous). Aide → **Mesurer ce Mac** montre le verdict — vert, ou orange avec ce qui manque ou paraît faux, en une phrase —, ce que chaque moteur atteint contre la crête que le silicium annonce, la mémoire que la mesure a tenue et sa durée, et permet de partager le rapport ([diagnostic communautaire](#le-diagnostic-communautaire)).

### Le Neural Engine

Le Neural Engine du Mac calcule en fp16. Là où le plan de ce Mac lui donne des lignes, il prend une partie des produits matriciels du DiT à côté du GPU, bloc par bloc : chaque colonne de poids et chaque ligne d'activations est remontée d'une puissance de deux dans la plage du fp16 puis ramenée en fp32, et les lignes qui débordent retournent au GPU. Les encodeurs de texte et le VAE ne s'en servent jamais. Le tag **fp32** du module DiT du rack le bascule, par modèle : **éteint par défaut sur la Standard**, qui reste les poids de l'éditeur calculés en fp32, l'image vérifiée contre la référence ; **allumé par défaut sur la Compact et la Légère**, dont les poids en 8 et 4 bits font déjà une autre image que celle de la Standard. Le tag affiche **fp32 · ANE** quand il est allumé, et une image calculée avec lui porte `neural-engine` dans ses métadonnées. Sur un Mac dont la mesure n'offre pas le Neural Engine (aucun trouvé, ou ses résultats étaient faux), le tag est un simple fp32 et il n'y a rien à choisir.

Sur le M1 Pro, 1024², Standard, rendu plus rapide, graine 42, le prompt de la bibliothèque (« a 30 year old woman posing in a library »), le 23/10/2026, app 0.14 :

| | Krea 2 Turbo | Z-Image Turbo | Qwen-Image-2.1 Turbo |
|---|---|---|---|
| temps dans l'app, éteint / allumé | 144,6–144,9 s / 135,4–137,4 s | 86,8–93,9 s / 83,4–84,0 s | 103,6 s / 99,9–100,2 s |
| puissance de toute la machine, éteint / allumé | 43,9 W / 35,3 W | 38,0 W / 31,2 W | 45,4 W / 36,9 W |
| énergie par image, allumé contre éteint | −26 % | −16 % | −20 % |
| l'image, allumé contre éteint | PSNR 35,9 dB : la même scène, le même visage, la même lumière, les détails les plus fins déplacés (tranches de livres) | 55,9 dB, un dixième de niveau en moyenne | 56,3 dB, un dixième de niveau en moyenne |
| écart de la sortie du débruiteur à l'exact fp64, la moitié des lignes de chaque produit sur le Neural Engine | 4,8·10⁻³ (512²) | 3,6·10⁻³ (1024²) | 4,4·10⁻³ (512²) |
| …en part de l'écart du bf16 de l'éditeur | 0,14 | 0,06 | 0,31 |

Temps : des rendus mis en file dans l'app par `silicontrol`, chacun dans une app tout juste ouverte, 0 page swappée ; éteint, trois rendus de Z-Image d'affilée ont pris 93,9, 89,0 et 86,8 s aux mêmes bits, le premier juste après un déplacement de la bibliothèque. Ce sont les temps de la 0.14 et le prompt de la bibliothèque ; ceux de la 0.15, aux pics plus bas, sont dans [Les chiffres](#les-chiffres). Puissance et énergie : la même comparaison en ligne de commande, quatre rendus par modèle en alternant allumé et éteint, la consommation de toute la machine (écran compris) lue dans la télémétrie de l'adaptateur secteur. L'écart est celui d'une évaluation du débruiteur avec plus de Neural Engine qu'aucun plan n'en donne, la moitié des lignes de chaque produit, tel que l'imprime `siliconed check <modèle> dit --neural-engine` ([VERIFY.fr.md](VERIFY.fr.md)), le 24/10/2026 ; celui du bf16 est celui de diffusers sur les poids bf16 de l'éditeur, comme calculent la plupart des outils.

## Les chiffres

Machine de référence : MacBook Pro M1 Pro (8 cœurs, 14 cœurs GPU), 16 Go. App 0.15, le 25/10/2026 (MOSS-Image : app 0.16, le 26/10/2026), graine 42, 8 pas, rendu plus rapide allumé (par défaut), le prompt « Photo of a young Irish woman with wavy auburn hair and freckles, in a shorty wetsuit, holding her surfboard on a wild beach in the west of Ireland, Atlantic waves behind her, 35 mm photograph. »

| | Krea 2 Turbo | Z-Image Turbo | Qwen-Image-2.1 Turbo | Anima Turbo | MOSS-Image |
|---|---|---|---|---|---|
| 1024², Compact (8 bits, par défaut), Neural Engine allumé | **135,8 s** | **83,1 s** | **100,2 s** | **33,1 s** | **107,7 s** |
| 1024², Standard (les bf16 de l'éditeur, calculés en fp32) | 147,8 s | 82,4 s | 110,4 s | 34,3 s | 107,3 s |
| 1024², Légère (4 à 6 bits), Neural Engine allumé | 145,0 s | 87,5 s | 103,7 s | 34,7 s | — (pas de Légère) |
| 1024×1536, Compact / Standard | 207,7 s / 223,1 s | 130,2 s / 133,1 s | 161,1 s / 170,9 s | 53,6 s / 53,7 s | 174,2 s / 174,2 s |
| la première image d'un nouveau prompt (son encodage compris), 1024², Compact / Standard | 136,7 s / 150,2 s | 84,2 s / 84,9 s | 105,7 s / 118,4 s | 33,8 s / 33,9 s | 109,9 s / 109,7 s |
| pic mémoire de l'app, 1024² · 1024×1536, Compact / Standard | 2,1 · 2,7 / 2,1 · 2,6 Gio | 1,9 · 2,8 / 1,9 · 2,8 Gio | 1,5 · 1,7 / 1,5 · 1,7 Gio | 1,3 · 2,0 / 1,3 · 2,0 Gio | 2,4 · 2,8 / 2,4 · 2,8 Gio |
| lu sur le disque par image, 1024², Compact / Standard | 98 / 195 Go | 0 / 44 Go | 0,2 / 135 Go | 0 / 0 Go | 0,1 / 76 Go |
| 1024², Standard, toutes les évaluations en pleine taille (0.12) | 221,1 s | 117,7 s | 142,7 s | 44,4 s | 116,9 s (0.16) |
| autres, Standard | | | une édition à 1 image 211 s, à 2 images 297 s | 512² 12,1 s |  |
| écart de la sortie du débruiteur (`model_out`, pire canal) à la référence fp32 · à l'exact fp64 (celui de la référence) | 2,6·10⁻⁴ · 3,3·10⁻⁴ (3,1·10⁻⁴), 1024² | 9,8·10⁻⁶ · 1,2·10⁻⁵ (1,3·10⁻⁵), 1024² | 3,1·10⁻⁶ · 2,6·10⁻⁶ (2,5·10⁻⁶), 512² | 1,4·10⁻⁵ · par plan latent 2,9·10⁻⁶ (3,8·10⁻⁶), 1024² | par plan latent 1,2·10⁻⁶ · 1,3·10⁻⁶ (1,4·10⁻⁶), 512² |

**La précision complète ne coûte presque rien.** La Standard calcule en fp32 les poids bf16 de l'éditeur, la précision dans laquelle la référence est vérifiée ; la Compact lit des poids 8 bits publiés par des tiers et calcule elle aussi en fp32. Sur Z-Image et Anima, la Standard prend le temps de la Compact à 4 % près. Sur Krea 2 et Qwen-Image-2.1, elle prend 6 à 10 % de plus : en pleine précision, leurs poids ne restent pas dans la mémoire d'un Mac de 16 Go et sont relus sur le disque à chaque pas, 195 Go et 135 Go par image en 1024². La Standard de MOSS-Image lit 76 Go par image et prend pourtant le temps de la Compact : ses pas sont assez longs pour que le disque suive. C'est aussi pourquoi tous les pics ci-dessus restent sous 3 Gio : les poids projetés depuis le disque ne comptent pas dans `phys_footprint`, puisque le système peut les lâcher à tout moment. Ce que la Standard coûte vraiment, c'est la place sur le disque, le double de la Compact, et l'énergie, le Neural Engine y étant éteint par défaut ([Le Neural Engine](#le-neural-engine)).

Chaque ligne est une série `silicontrol measure` dans une app tout juste ouverte, la série que chacun peut refaire ([REMOTE-CONTROL.fr.md](REMOTE-CONTROL.fr.md)) : un rendu de chargement, un témoin (le prompt de la bibliothèque, « a 30 year old woman posing in a library »), deux rendus du prompt ci-dessus, un témoin encore. Un temps est celui du second rendu, de la demande au PNG, le prompt déjà encodé ; la première image d'un nouveau prompt y ajoute son encodage. Les deux témoins de chaque série sont restés à 1,4 % l'un de l'autre, 0 page swappée, et chaque image est sortie identique, au bit près, d'un rendu à l'autre. Le pic mémoire est le `phys_footprint` de l'app, relevé toutes les 100 ms. Toutes les évaluations en pleine taille : le 18/10/2026, app 0.12, 0 page swappée. Les autres temps : Qwen-Image-2.1 le 15/10/2026 avec la 0.10, le 512² d'Anima Turbo avec le moteur seul (14–15/10/2026). Les écarts sont ceux qu'a imprimés `siliconed check <modèle> dit --size <côté>` sur le même Mac le 24/10/2026, contre des goldens écrits par `tools/oracle/` ; chez vous, ils diffèrent sur les derniers chiffres ([VERIFY.fr.md](VERIFY.fr.md), [La vérification](#la-vérification)).

### Rendu plus rapide

Allumé par défaut, en tête des Réglages, pour les cinq modèles. Dès 1024 px de chaque côté (1024², 1024×1536, 1536×1024…), les premières évaluations se font sur la grille en demi-taille — 3 sur 8 pour Z-Image, Qwen-Image-2.1 et Anima, 4 pour Krea 2, dont le calendrier garde ses premiers niveaux de bruit plus hauts, 1 pour MOSS-Image, qui commence à perdre du détail fin dès la deuxième — et le reste en pleine taille. Toujours 8 évaluations ; une évaluation en demi-taille coûte environ un quart d'une pleine. Aucun format portrait ou paysage usuel n'en profite : la moitié de 832×1216 passerait sous 512 px d'un côté.

La petite grille reçoit l'état moyenné plus la basse bande du bruit de départ : elle tient une image propre, bruitée comme il faut ; la vitesse ramenée en pleine taille est celle qui mène l'état complet à l'estimation propre agrandie : elle emporte aussi le bruit fin qu'une demi-grille ne voit pas, au rythme d'un pas plein. Les pas en demi-taille retirés dans la 0.8.2 n'avaient ni l'un ni l'autre : leur vitesse agrandie laissait ce bruit fin en place (du sable sur la peau et les murs), et leur petite grille était une simple décimation, dont le repliement se lisait comme une texture (du craquelé). Le nombre d'évaluations en demi-taille de chaque modèle a été jugé à l'œil sur deux prompts, un portrait et la bibliothèque : ni sable ni craquelé, taches de rousseur, cheveux et dos des livres intacts ; un pas de plus, et le détail fin commence à partir.

Il donne une autre image que la recette en pleine taille à la même graine (la composition se décide dans les premiers pas : la même scène, parfois une autre personne), pas une copie dégradée. Pas avec une LoRA, Détail, une édition, un autre nombre de pas ni un autre calendrier. Éteint, toutes les évaluations sont en pleine taille et l'image est la même, au bit près, qu'avant : celle qui est vérifiée contre diffusers ; les écarts ci-dessus valent pour elle.

### À côté de Draw Things

[Draw Things](https://drawthings.ai) est la référence sur Mac ; ici, son moteur communautaire, [draw-things-community](https://github.com/drawthingsai/draw-things-community) à `bc3970c`, avec les réglages par défaut du modèle (UniPC trailing, CFG 1). La même machine, tout juste redémarrée, un seul moteur chargé à la fois ; Z-Image Turbo, 1024², 8 pas (8 évaluations de chaque côté), graine 42, trois rendus de chaque côté sur trois variantes du même prompt. Draw Things a été mesuré le 13/10/2026 (une séance de 70 minutes) et rien n'a changé de son côté depuis ; Siliconed en pleine taille le 14/10/2026, après le correctif, et avec le rendu plus rapide le 19/10/2026 (app 0.12, trois rendus du prompt de la bibliothèque chacun, chacun dans une app tout juste ouverte). Les pics de Siliconed dans ce tableau sont ceux de la 0.12 ; la 0.15 les a abaissés ([Les chiffres](#les-chiffres)).

| | poids | temps | pic mémoire (Gio) | lu sur le disque par rendu |
|---|---|---|---|---|
| Siliconed, Standard | les bf16 de l'éditeur, calculés en fp32 | 113–119 s | 2,6 | 51–58 Go |
| Siliconed, Légère | GGUF Q4_K_M | 118–120 s | 2,0 | 3,5–8,7 Go |
| Siliconed, Standard, rendu plus rapide (par défaut) | les bf16 de l'éditeur, calculés en fp32 | 85–86 s | 2,8 | 51–58 Go |
| Siliconed, Légère, rendu plus rapide (par défaut) | GGUF Q4_K_M | 90–91 s | 2,1 | 3,5–8,7 Go |
| Draw Things, Exact | f16 | 167–179 s | 5,3 | 99 Go |
| Draw Things, par défaut | 8 bits (`q8p`) | 142–164 s | 3,5–4,3 | 11 Go |
| Draw Things, 6 bits | 6 bits (`q6p`) | 136–138 s | 3,5 | 6–14 Go |

La comparaison porte sur le temps écoulé, pas sur une arithmétique identique : les solveurs et les poids diffèrent, les images aussi. Le pic mémoire ne compte pas les poids projetés depuis le disque, d'un côté comme de l'autre. Siliconed fait chauffer davantage la machine, ce qui découle du calcul en fp32. Les deux premières lignes de Siliconed sont en pleine taille, comme celles de Draw Things ; avec le rendu plus rapide, les 3 premières de ses 8 évaluations passent sur une grille de demi-taille, ce que Draw Things ne fait pas : moins de travail, une autre image ([comment](#rendu-plus-rapide)).

Huit bits contre huit bits, le 23/10/2026 (app 0.14) : la Compact de Siliconed (le GGUF Q8_0 du tiers, gardé tel que publié) contre le `q8p` par défaut de Draw Things, dans son moteur communautaire, mis à jour depuis à `6c1dd09` ; Z-Image Turbo, 1024², 8 pas, graine 42, le prompt de la bibliothèque ; chaque rendu dans un processus tout juste lancé de chaque côté (l'app, le serveur du moteur), trois de chaque côté, puis Siliconed encore en témoin.

| | poids | temps | pic mémoire (Gio) | lu sur le disque par rendu |
|---|---|---|---|---|
| Siliconed, Compact, rendu plus rapide et Neural Engine (tous deux par défaut) | GGUF Q8_0 | 81,1–83,2 s (témoin 81,3 s) | 2,6 | 3,7–11 Go |
| Draw Things, par défaut | 8 bits (`q8p`) | 135,0–144,2 s | 4,3 | 10–18 Go |

## La mémoire et le budget

Le plus gros modèle, Krea 2 Turbo en Standard (tailles : [La gestion des modèles](APP.fr.md#la-gestion-des-modèles)), tourne sur un Mac de 16 Go dans un budget fixe. Les poids sont lus depuis le disque au moment où ils servent. Siliconed ne prend jamais plus que les ¾ de la mémoire du Mac moins 2 Go (4 Go sur un Mac de 8 Go, 10 Go sur 16 Go, 22 Go sur 32 Go) : chaque rendu calcule ce qu'il lui faut avant de commencer, prend le plan le plus rapide qui y tient, et compare son propre pic à ce chiffre. Ce que les autres applications occupent leur appartient.

**Dans le budget, mesuré.** Le pic `phys_footprint` de chaque modèle et ses pages swappées (delta `vm_stat`) sont mesurés sur un Mac de 16 Go où rien d'autre ne tourne. Avec la 0.15 en 1024×1536, chaque modèle culmine entre 1,7 et 2,8 Gio, Standard ou Compact, 0 page swappée ([Les chiffres](#les-chiffres)) ; Anima Turbo en 2048² culmine à 5,4 Go ; Qwen-Image-2.1 en 1024×1536 avec 3 références, 0 page swappée (0.14). Le VAE de Z-Image décode les grandes images en bandes, aux mêmes bits.

## Face aux autres outils

- **[Draw Things](https://drawthings.ai)**, l'app native de référence sur Mac : son moteur est [mesuré plus haut](#à-côté-de-draw-things) sur la même machine. Siliconed calcule en fp32 et il est vérifié étage par étage contre l'implémentation de référence.
- **[ComfyUI](https://github.com/comfyanonymous/ComfyUI) ou [diffusers](https://github.com/huggingface/diffusers) en Python** : les mêmes modèles, sans environnement Python ni graphe de nœuds à câbler. Siliconed les implémente en natif et se vérifie contre diffusers et [transformers](https://github.com/huggingface/transformers) ; `silicontrol` couvre les scripts et l'automatisation.

## Le diagnostic communautaire

Une machine ne fait pas un banc. **Mesurer ce Mac** (menu Aide ; *Measure This Mac* en anglais) montre la mesure que ce Mac a prise avant son premier rendu, en moins d'une minute et sans aucun modèle. D'abord le verdict : vert quand chaque moteur que le silicium annonce a ses chiffres, chaque ensemble de moteurs son partage, que rien n'est aberrant, que la mémoire est restée sous son budget et que la minute a été tenue ; sinon orange, chaque chose qui manque ou paraît fausse dite en une phrase — un moteur sans chiffre, un chiffre qui paraît faux, d'autres programmes qui ont occupé le Mac pendant la mesure, un swap, une minute dépassée. Une route qui a mal calculé, ou dont le chargement a tué le processus de mesure, est écartée du plan et marquée ✗, mais ce n'est pas une raison d'orange. Ensuite ce que chaque moteur atteint — le GPU, l'unité matricielle du CPU et le Neural Engine, seuls, sur des produits matriciels de la taille de ceux des modèles et sur l'attention, des produits bf16, des produits sur les poids int8 et Q8_0 publiés — en barre contre la crête que le silicium annonce là où il en annonce une, avec ✓ là où le résultat a rejoint le résultat exact calculé en double précision ; la mémoire que la mesure a tenue, sa durée et, si vous avez rendu depuis, l'écart entre ce que le plan avait prévu et votre dernier rendu. **Ouvrir le signalement** copie le rapport JSON et ouvre un formulaire d'issue GitHub où le coller (**Copier le JSON** seul, ou vous l'envoyez autrement) : puce, cœurs CPU et GPU, ce que le registre dit du silicium (fréquence du GPU, grappes du CPU, génération du Neural Engine), mémoire, macOS, drapeaux du CPU, le modèle et les mesures brutes qui l'ont donné (pour chaque moteur et chaque noyau son coût fixe et son débit, la part que garde chaque moteur à côté des autres, la chaleur), si chacun a calculé juste, la mémoire tenue, les secondes de chaque étape, le verdict avec la liste de ce qui manque ou est aberrant, les temps prévus et mesurés du dernier rendu par classe, état thermique et alimentation avant et après. Rien n'est envoyé tant que vous ne soumettez pas l'issue.

Nous relisons les rapports, une ligne par sorte de Mac — le débit de chaque moteur et la crête annoncée, ce que chacun garde à côté des autres, la durée, le verdict —, pour vérifier qu'un seul modèle décrit tous les Mac, ceux que nous n'avons jamais eus compris. Un rapport n'ajoute jamais une règle pour une puce : il n'y en a pas.

Les temps de rendu de la machine de référence sont dans [Les chiffres](#les-chiffres).

## La vérification

Chaque étage (encodeur de texte, débruitage, VAE) est comparé à l'implémentation de référence, diffusers et transformers en fp32 sur CPU, sur la même entrée. L'écart se mesure canal par canal, chaque canal rapporté à sa propre norme, pour qu'une faute confinée à un canal ne se cache pas dans un chiffre global. Quand la référence fp32 porte sa propre erreur d'arrondi, l'une et l'autre sont jugées contre un calcul fp64 du même étage, et le moteur ne doit pas s'en écarter de plus que la référence plus 25 %. Depuis la 0.12, son arithmétique arrondit là où la référence arrondit. La comparaison à la référence est livrée : les scripts d'oracle de `tools/oracle/` et la commande `siliconed check` la refont sur votre Mac, étape par étape, avec les empreintes de nos goldens pour y comparer les vôtres ; [VERIFY.fr.md](VERIFY.fr.md) dit comment, et ce que ça coûte. Le journal des mesures ne fait pas partie de ce dépôt. Le diagnostic de l'app vérifie une machine contre des résultats exacts calculés sur son propre CPU.

## Construire depuis les sources

macOS 26 ou plus, Apple silicon, Xcode 26 ou plus.

```
swift build                 l'app et silicontrol
swift test                  les tests unitaires purs, sans aucun poids
tools/app.sh                construit Siliconed.app à la racine du dépôt et l'ouvre
tools/app.sh --no-open      construit seulement
tools/dmg.sh                emballe l'app construite en .build/Siliconed-<version>.dmg
```

`swift test` ne demande rien d'autre : ni poids, ni Python. Siliconed livre une app, pas une bibliothèque : les déclarations du moteur restent internes au paquet.

Par défaut, l'app est signée ad hoc : elle tourne sur le Mac qui l'a construite. Pour la distribuer, signez-la avec votre Developer ID :

```
tools/app.sh --sign "Developer ID Application: <nom> (<équipe>)"
```

Si `SILICONED_NOTARY_PROFILE` nomme un profil de trousseau `notarytool`, l'app est aussi notariée et agrafée ; `tools/dmg.sh` signe et notarie alors l'image disque avec la même identité.
