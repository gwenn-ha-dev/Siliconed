[English](RELEASE-NOTES.md) · Français

# Siliconed — notes de version

## 0.16

### Un cinquième modèle : MOSS-Image

- **[MOSS-Image](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT)**, de l'équipe OpenMOSS, rejoint
  Krea 2, Z-Image, Qwen-Image-2.1 et Anima. Un DiT mono-flux de 5,1 milliards de paramètres qui lit
  le prompt avec Qwen3.5-9B, un modèle de langage hybride (trois couches sur quatre en attention
  linéaire, Gated DeltaNet, nouvelle dans le moteur), et décode avec le VAE de FLUX.2.
- **MOSS-Image est en bêta**, et l'app le dit (un badge « Bêta » sur sa carte et dans Modèles) : sa
  recette en 8 pas est la LoRA de la communauté, pas celle de son éditeur. Il sort de bêta quand
  OpenMOSS publie sa propre version rapide.
- **Sa recette est celle de la communauté** : son éditeur n'a pas publié de version rapide, alors
  Siliconed fait tourner MOSS-Image-SFT sous la [LoRA DMD de lodestones](https://huggingface.co/lodestones/MOSS-Image-SFT-8-steps-dmd)
  à force 1 : 8 évaluations, sans CFG. La LoRA est gardée à côté des poids en fp32, jamais fusionnée :
  fusionnée dans les poids bf16, ses deltas se perdraient dans l'arrondi. Texte vers image ; ni
  Détail, ni édition.
- **Vos LoRA** s'empilent par-dessus celle des 8 pas : la disposition de ComfyUI et d'ai-toolkit (le
  `qkv` fusionné de l'attention, séparé exactement), celles de kohya et de diffusers. Les LoRA
  entraînées sur MOSS-Image Base s'appliquent comme la communauté s'en sert. Le ΔW de chaque module
  importé est égal à celui que PyTorch calcule depuis le fichier publié.
- **Temps** : sur un M1 Pro de 16 Go, graine 42, 8 évaluations, en file dans l'app : 1024² en 107,7 s
  en Compact, 107,3 s en Standard ; 1024×1536 en 174,2 s dans les deux.
- **Le rendu plus rapide n'y prend qu'une évaluation en demi-taille**, pas trois : dès la deuxième,
  MOSS-Image lisse la peau et adoucit le détail fin là où les autres modèles le gardent. 8 % de temps
  en moins à 1024² (116,9 s avec toutes les évaluations en pleine taille).
- **L'aperçu s'affiche pendant le rendu**, comme pour les autres modèles : une projection des
  128 canaux du latent FLUX.2 vers la couleur, ajustée sur douze rendus.
- **Vérifié contre la référence**, étage par étage : le tokenizer et le calendrier au bit près, les
  32 couches de l'encodeur de texte, le débruiteur avec et sans la LoRA, la trajectoire de 8 pas et
  le décodeur, chacun au témoin fp64 là où la référence porte sa propre erreur. En 512², la sortie du
  débruiteur est à 1,3·10⁻⁶ du résultat exact, celle de la référence à 1,4·10⁻⁶. Sous le Neural
  Engine, le débruiteur des deux versions reste dans l'erreur propre au bf16 plus 25 %, comme pour
  les autres modèles.
- **Standard ou Compact** : 27,5 Go (les bf16 de l'éditeur, la LoRA en fp32) ou 17,9 Go (le DiT et
  l'encodeur de texte en int8 convrot, tels que Comfy-Org les publie) ; la Compact est présélectionnée,
  comme pour les autres modèles, avec le Neural Engine allumé par défaut. Pas encore de Légère : un
  DiT en 4 bits est publié, pas encore vérifié.
- **Dans le budget** : son besoin en mémoire est calculé à partir des formes du modèle avant le
  premier calcul. Pic de l'app : 2,4 Gio à 1024², 2,8 Gio en 1024×1536, dans les deux versions.
- **Licence** : Apache 2.0, LoRA comprise. Usage commercial permis.

### Sur chaque Mac

- **Le premier lancement après une mise à jour ne va plus moins vite** : quand le premier geste était
  une exploration, une grille ou `silicontrol add`, la session ignorait la mesure de ce Mac et tournait
  sur les défauts — l'AMX éteint, environ 20 % par pas sur un M1 Pro. `silicontrol status` dit
  désormais la mesure qu'emploie la session.
- **Une mesure prise hors repos se refait** : mesurés sur un Mac chaud, en mode économie d'énergie ou
  avec un swap qui monte (une autre app qui presse la mémoire), les choix servent quand même, la puce
  de la Chaîne reste orange, et le prochain lancement au repos mesure à nouveau avant son premier
  rendu — une fois.
- **Une mise à jour qui ne touche pas le moteur garde la mesure de ce Mac** : la mesure suit les sources
  du moteur et son compilateur, plus le build de l'app ; une mise à jour du moteur fait mesurer à nouveau.
- **macOS 26 ou plus** : la seule version sur laquelle Siliconed a jamais été mesuré et contrôlé.
  L'app ne s'installe plus sur macOS 15.

## 0.15

### La Compact par défaut

- **La Compact est désormais présélectionnée, et dite recommandée,** quand vous installez un modèle
  sur un Mac de plus de 8 Go (la Légère reste présélectionnée sur 8 Go). Sur un M1 Pro, elle rend aussi vite que la Standard, ou plus, prend moitié moins de place sur le disque, et le Neural Engine y est
  allumé par défaut. Un modèle déjà installé garde sa version ; les trois restent à un clic.
- **La précision complète ne coûte presque rien.** La Standard, les poids bf16 de l’éditeur
  calculés en fp32, prend le temps de la Compact à 4 % près sur Z-Image et Anima, et 6 à 10 % de
  plus sur Krea 2 et Qwen-Image-2.1, dont les poids ne tiennent pas en mémoire en pleine précision
  et sont relus sur le disque à chaque pas. Tous les chiffres : [Les chiffres](ENGINE.fr.md#les-chiffres).

### Des images plus grandes, des pics plus bas

- **Plus de plafond d’aire : le budget mémoire de ce Mac décide.** Le sélecteur de format propose
  1536² et 2048², grisés avec le Mac qu’il leur faut là où le budget les refuse. Anima Turbo rend
  du 2048² sur un Mac de 16 Go, avec un pic de 5,4 Go.
- **Des pics plus bas, les mêmes pixels** : les poids sont lus sur le disque quelques tenseurs à la
  fois, et non plus une couche à la fois — 0,8 à 1,4 Go de moins pour Krea 2, 1,1 Go pour Z-Image,
  0,5 Go pour Qwen-Image-2.1.
- **Corrigé : la mémoire de Krea 2 montait pendant le débruitage** (des objets libérés seulement à
  la fin du rendu).

### Vérifiez nos nombres chez vous

- **`siliconed check <modèle>`** compare chaque étape d’un modèle à l’implémentation de référence,
  sur votre Mac, contre des goldens que vous écrivez avec les scripts de `tools/oracle/` et dont nous
  publions les empreintes. `--neural-engine` juge l’arithmétique du Neural Engine, `--variant
  compact|light` la Compact et la Légère contre la référence exécutée sur les poids publiés par le
  tiers. La commande est livrée dans l’app (`Contents/Helpers`) : elle vérifie le moteur même avec
  lequel l’app rend, sans Xcode. Comment, et ce que ça coûte : [VERIFY.fr.md](VERIFY.fr.md).
- **`silicontrol measure`** refait nos temps et nos pics avec la seule app : une série de rendus
  entre deux témoins, le pic mémoire échantillonné de l’app, les swap-outs, les lectures disque et
  l’empreinte de chaque image, en TSV. `silicontrol set faster-rendering on|off` bascule le rendu
  plus rapide comme les Réglages.

### Recadrer et enregistrer

- **Recadrer** une image rendue ou une référence d’édition : C pour entrer, poignées aux coins et
  aux côtés, règle des tiers pendant le geste, Retour pour accepter, Échap pour annuler ; jamais
  sous 512 px de côté. Le recadrage est une nouvelle entrée de l’historique ; l’original reste.
- **Enregistrer en PNG, JPEG ou HEIC** ; le dernier format est retenu. JPEG et HEIC ne portent
  aucune métadonnée.

### Le moteur

- **Les grands formats partagent leur attention** entre le GPU et le CPU à la part que la mesure de
  ce Mac trouve la plus rapide, le softmax des têtes du CPU en une passe NEON.
- **Un chemin à activations bf16** pour les produits du débruiteur, choisi par Mac par sa mesure,
  jugé contre diffusers en bf16 ; éteint sur le M1 Pro, où il ne gagne pas.
- **Ce Mac est mesuré à nouveau** avant le premier rendu après la mise à jour, environ une minute.

## 0.14

### Le Neural Engine

- **Le Neural Engine du Mac rejoint le GPU** dans les produits matriciels du modèle d’image, là où
  la mesure de ce Mac trouve la paire plus rapide que le GPU seul. À côté de l’unité matricielle du
  CPU, il prend la forme d’un produit partout où il garde 90 % de sa vitesse, pour l’énergie qu’il
  épargne. Il calcule en fp16 : chaque colonne de poids et chaque ligne d’activations est remontée
  d’une puissance de deux dans la plage du fp16, puis ramenée en fp32, et les lignes qui débordent
  retournent au GPU. Les encodeurs de texte et le VAE ne s’en servent jamais.
- **Votre choix, par modèle** : le tag **fp32** du module DiT du rack le bascule. **Éteint par
  défaut sur la Standard**, qui reste les poids de l’éditeur calculés en fp32, son image vérifiée
  contre la référence ; **allumé par défaut sur la Compact et la Légère**, dont les poids en 8 et
  4 bits font déjà une autre image. Le tag affiche **fp32 · ANE** quand il est allumé, et une image
  calculée avec lui porte `neural-engine` dans ses métadonnées. Là où la mesure a laissé le Neural
  Engine éteint, il n’y a rien à choisir.
- **Sur un M1 Pro**, 1024², Standard, rendu plus rapide : de 16 à 26 % d’énergie en moins par image,
  et aussi vite ou plus — Krea 2 135,4–137,4 s au lieu de 144,6–144,9 s, Z-Image 83,4–84,0 s au lieu
  de 86,8–93,9 s, Qwen-Image-2.1 99,9–100,2 s au lieu de 103,6 s. L’image bouge d’un dixième de
  niveau en moyenne sur Z-Image et Qwen-Image-2.1 ; Krea 2 garde sa scène, son visage et sa lumière,
  et ses plus fins détails bougent. Tous les chiffres : [Le Neural Engine](ENGINE.fr.md#le-neural-engine).
- **Huit bits contre huit bits** : Z-Image Compact en 1024², avec le rendu plus rapide et le Neural
  Engine (tous deux par défaut), rend en 81,1–83,2 s ; le `q8p` 8 bits par défaut de Draw Things,
  en 135,0–144,2 s sur le même Mac ([À côté de Draw Things](ENGINE.fr.md#à-côté-de-draw-things)).
- **Ce Mac est mesuré à nouveau** avant le premier rendu après la mise à jour, environ une minute,
  comme après chaque version : la mesure comprend désormais le Neural Engine, sur les produits
  propres à chaque modèle.
- **La télécommande** : `silicontrol add --ane` ou `--no-ane`, comme un clic sur le tag ;
  `neural_engine` dans `models`, dans les travaux et dans les images.

### Ouvrir un modèle de Hugging Face dans l’app

- **L’app ouvre les liens `siliconed://hf`**, relayés par une page du site qui propose le .dmg
  quand l’app n’est pas installée : ce qu’ouvre le bouton « Use this model » de Hugging Face, une
  fois que Hugging Face référence Siliconed. Le dépôt propre d’une famille ouvre son installation ;
  tout autre dépôt liste les fichiers `.safetensors` et `.gguf` que l’app pourrait importer — chacun
  jugé sur son en-tête avant tout téléchargement —, avec leur taille et leur licence. Le fichier
  choisi est téléchargé, vérifié, puis importé comme un fichier déposé. Rien du lien n’est cru sur
  parole.
- **Plus de LoRA s’importent** : un `alpha` entier (comme l’enregistre sd-scripts) est lu au lieu
  d’être refusé ; une LoRA qui ne touche que l’attention trouve sa famille d’après les formes des
  modèles installés, quand une seule la prend tout entière ; un fichier LyCORIS ou une LoRA
  d’encodeur de texte est refusé avec son propre message au lieu d’être pris pour autre chose.

### Moins de téléchargements

- **Un encodeur de texte ou un VAE que vous avez déjà est pris au lieu d’être téléchargé.** Déposé
  sur l’app, un fichier égal en taille et en SHA-256 à l’un de ceux qu’une installation récupère —
  les encodeurs 8 bits de Z-Image, Qwen-Image-2.1 et Krea 2, l’encodeur bf16 de Krea 2, les
  encodeurs de la Standard, le VAE de chaque famille — est copié là où le téléchargement l’aurait
  mis, puis vérifié comme un téléchargement.

### Des prompts plus longs

- **Chaque modèle lit jusqu’à 1 024 tokens de prompt**, et refuse un prompt plus long avec un
  message. Jusqu’à la 0.13, Z-Image, Anima et Krea 2 ignoraient sans rien dire tout ce qui dépassait
  512 tokens, comme les pipelines de référence par défaut, et Qwen-Image-2.1 refusait au-delà de 512.
  Un prompt de moins de 512 tokens donne la même image qu’avant.

### Corrigé

- **Une LoRA entraînée avec un rang différent par couche affichait « rang 0 »** dans la fenêtre
  Modèles et dans l’infobulle du rack ; elle affiche maintenant son étendue, « rangs 2–57 » pour
  inCase sur Anima.

## 0.13

### Votre bibliothèque sur un autre disque

- **Les Réglages déplacent les modèles** sur un disque externe ou un partage réseau, ouvrent une
  bibliothèque qui s’y trouve déjà, ou la ramènent sur le Mac. Chaque fichier est copié, relu et
  comparé (SHA-256) avant que rien ne bascule ; l’ancien emplacement est vidé en dernier. Arrêtée ou
  en échec, la bibliothèque reste telle quelle, et un nouvel essai garde les fichiers déjà vérifiés.
  Un disque FAT, un dossier en lecture seule ou une autre bibliothèque sont refusés avant le premier
  octet. ([#2](https://github.com/gwenn-ha-dev/Siliconed/issues/2))
- **Un rendu ne lit jamais l’autre disque.** Dès qu’un modèle est choisi, ce qu’il lit est copié sur
  le Mac le temps de la session (`~/Library/Caches/Siliconed` ; plusieurs modèles gardés, le moins
  récemment servi évincé, 5 Go toujours laissés libres), puis effacé quand l’app quitte — après un
  plantage, au lancement suivant. La même graine donne la même image que depuis le disque du Mac.
- **Ce qui appartient au Mac y reste** : les licences acceptées, la mesure de ce Mac, le cache de
  rendu. Débranché, le disque est signalé : rien ne se rend avant son retour, et jamais une
  bibliothèque vide ne prend sa place. La télécommande et la ligne de commande suivent une
  bibliothèque déplacée.

### Un modèle importé avant sa famille

- **Déposer un fine-tune avant que sa famille soit installée** (un checkpoint Civitai ou Hugging
  Face de Z-Image, Qwen-Image-2.1, Anima ou Krea 2) ouvre désormais l’installation de cette famille,
  le même panneau que dans la fenêtre Modèles : la licence en entier, la place, l’encodeur de texte et
  le VAE — jamais le DiT de la famille, puisque le fichier apporte le sien. Une fois l’installation
  faite, le fichier est importé, et la licence acceptée là le couvre : pas de seconde feuille à son
  premier rendu. Jusqu’à la 0.12, l’app disait seulement d’installer la famille d’abord.
  ([#5](https://github.com/gwenn-ha-dev/Siliconed/issues/5))
- **Un téléchargement refusé par Hugging Face** (HTTP 401 ou 403) ne renvoie plus à un terminal pour
  se connecter : aucun dépôt d’où l’app installe ne demande de compte. Le message désigne maintenant
  ce qui refuse — un proxy, un VPN, un pare-feu.

### Corrigé

- **Un rendu pouvait sortir entièrement non fini** (Z-Image en 1024² sous une LoRA, à certaines
  longueurs de prompt) : le produit matriciel fp32 rapide d’Apple écrit parfois des valeurs invalides
  dans certains tampons de sortie. Chaque tampon et chaque forme sont maintenant vérifiés une fois,
  et recalculés dans un tampon neuf en cas de faute ; les mêmes bits partout ailleurs, sans coût
  mesurable.
- **Détail sur Qwen-Image-2.1** : Plus et Maximum ne changent plus la personne. Calibrés sur la LoRA
  à 6 pas qu’employait la famille avant la 0.10, ils pesaient sur des pas où le Turbo décide encore
  le visage ; ils ne touchent plus que la dernière évaluation (Plus 1,3, Maximum 2) : la même
  personne, une texture plus fine. Normal ne change pas, au bit près.
- **La feuille de licence d’un modèle importé** dit que le modèle d’image est le fichier importé, et
  donne les licences de l’encodeur et du VAE de la version installée — ceux d’une Compact ou d’une
  Légère viennent de tiers.
- **Deux conversions du même modèle à la fois** (l’app et la ligne de commande, par exemple) ne
  vident plus le fichier en cours l’une de l’autre : la seconde est refusée.
- **Les métadonnées des rendus Krea 2** nomment la page Civitai qui porte le même fichier bf16 :
  Civitai reconnaît le modèle.
- Deux réglages du moteur qu’aucune mesure de Mac ne choisissait (attention en demi-précision,
  élargissement GPU) sont retirés ; les chemins par défaut ne changent pas, au bit près.

## 0.12

### Un quatrième modèle : Krea 2 Turbo

- **[Krea 2 Turbo](https://huggingface.co/krea/Krea-2-Turbo)**, de Krea, rejoint Z-Image Turbo,
  Qwen-Image-2.1 et Anima dans l’app, sans le mode développeur. Un DiT de 12,9 milliards de
  paramètres, le plus gros des quatre et le plus lent, qui lit le prompt avec Qwen3-VL-4B (douze de
  ses couches, fondues par une fusion apprise) et décode avec le VAE de Qwen-Image. Des images
  esthétiques, de la photographie à la peinture.
- **Sa recette** : 8 évaluations, sans CFG. Texte vers image, les formats rectangulaires, les quatre
  solveurs en mode développeur, chacun vérifié contre diffusers sur le calendrier de Krea ; les LoRA (format
  d’ai-toolkit, lues comme diffusers et PEFT les lisent ; ComfyUI donne la même mise à jour),
  appliquées entières sur la Compact et la Légère et jugées contre la référence, modules du temps
  compris ; ni édition par instruction ni Détail : sur ce modèle, sa cloche tombe là où l’image se
  décide encore, et la redessine au lieu de l’affiner. Les variations subtiles prennent 0,05 (0,1
  change la personne).
- **Les temps** : sur un M1 Pro de 16 Go, Standard, graine 42, 8 évaluations, mis en file dans l’app :
  1024² en 145,6 s, 1024×1536 en 216,9 s avec le rendu plus rapide (ci-dessous, allumé par défaut) ;
  221,1 s et 355,3 s avec toutes les évaluations en pleine taille.
- **Vérifié contre la référence** : à 1024², la sortie du débruiteur est à 1,44·10⁻⁴ de diffusers
  en fp32.
- **Standard, Compact ou Légère** : 34,4 Go (le bf16 de l’éditeur), 19,5 Go (le DiT et l’encodeur
  de texte en int8 convrot, tels que Comfy-Org les publie) ou 12,8 Go (le DiT en GGUF Q4_K_M, tel que
  vantagewithai le publie, avec l’encodeur de texte de la Compact). La Compact et la Légère sont
  vérifiées au bit contre leur déquantification de référence.
- **Rien ne vient du dépôt de Krea lui-même** : il demande un compte Hugging Face, et l’app n’a pas
  de connexion. Chaque fichier vient d’une copie ouverte prouvée identique : le DiT et l’encodeur de
  texte de Comfy-Org (le DiT, le même fichier ; l’encodeur, égal tenseur pour tenseur), le VAE de
  Qwen-Image (le VAE fp32 de Krea est son bf16 élargi, valeur pour valeur), le tokenizer de
  Qwen-Image-2.1-Turbo (le même fichier) ; les fichiers de configuration de Krea sont embarqués tels
  que publiés. La feuille de licence le dit, et montre la fiche de chaque dépôt d’où l’on télécharge.
- **Dans le budget** : son besoin de mémoire est calculé d’après les formes du modèle avant le
  premier calcul, comme pour les trois autres. Pics à 1024², le moteur seul : 3,07 Go en Standard
  (3,51 Go en 1024×1536), 2,20 Go en Compact, 1,35 Go en Légère.
- **Licence** : Krea 2 Community. Usage commercial sous 1 M$ de chiffre d’affaires annuel ; un
  filtre de contenu est obligatoire quand le modèle est déployé. L’app montre la licence en entier
  avant le premier octet téléchargé, avec l’attribution qu’elle demande.

### Rendu plus rapide, allumé par défaut

- **Un nouveau réglage, en tête des Réglages, allumé par défaut, pour les quatre modèles.** Dès 1024 px
  de chaque côté (1024², 1024×1536, 1536×1024…), les premières évaluations se font sur la grille en demi-taille — 3 sur 8
  pour Z-Image, Qwen-Image-2.1 et Anima, 4 pour Krea 2, dont le calendrier décalé garde ses premiers
  niveaux de bruit plus hauts —, puis le reste en pleine taille. Toujours 8 évaluations ; une
  évaluation en demi-taille coûte environ un quart d’une pleine. Aucun format portrait ou paysage
  usuel n’en profite : la moitié de 832×1216 passerait sous 512 px d’un côté.
- **Les temps**, sur un M1 Pro de 16 Go, Standard, graine 42, mis en file dans l’app (allumé /
  éteint) : en 1024², Z-Image 84,5 / 117,7 s, Qwen-Image-2.1 103,7 / 142,7 s, Anima 32,7 / 44,4 s,
  Krea 2 145,6 / 221,1 s ; en 1024×1536, allumé : 131,0 s, 161,7 s, 52,5 s et 216,9 s (Krea 2
  éteint : 355,3 s). En Légère, allumé, à 1024² : Z-Image 90–91 s, Qwen-Image-2.1 111,0 s, Anima
  35,4 s, Krea 2 153,8 s — la Légère économise le disque et la mémoire (pic de Krea 2 dans l’app :
  1,55 Go), pas le temps.
- **Pourquoi c’est propre cette fois.** Les pas en demi-taille retirés dans la 0.8.2 décimaient
  l’état et agrandissaient la vitesse : cette vitesse ne portait pas de haute bande, si bien que le
  bruit fin de l’état n’était jamais retiré pendant qu’ils tournaient (du sable sur la peau et les
  murs), et le repliement de la décimation arrivait au modèle comme une texture (du craquelé).
  Désormais la demi-grille reçoit la moyenne 2×2 de l’état plus la basse bande du bruit de départ
  — sans repliement, au bon niveau de bruit —, et la vitesse ramenée en pleine taille est celle qui
  mène l’état complet à l’estimation propre agrandie : elle retire le bruit fin au rythme d’un pas
  plein, et l’état reste exactement sur le chemin de la recette. Le nombre d’évaluations en demi-taille de chaque modèle a été
  jugé à l’œil sur deux prompts, un portrait et la bibliothèque, à la même graine : ni sable ni craquelé, taches de rousseur, cheveux et dos des
  livres intacts ; un pas de plus, et les taches de rousseur pâlissent, les étagères virent aux
  blocs.
- **Une autre image, pas une image dégradée.** À la même graine, l’image n’est pas celle de la
  recette en pleine taille : la composition se décide dans les premiers pas (la même scène, parfois
  une autre personne). Pas avec une LoRA, Détail, une édition, un autre nombre de pas ni un autre
  calendrier.
- **Éteint, rien ne bouge** : toutes les évaluations en pleine taille, les mêmes bits qu’avant,
  l’image vérifiée contre diffusers. Les écarts à la référence publiés pour chaque modèle valent pour
  cette image-là.

### Le moteur mesure votre Mac avant son premier rendu

- **Aucun rendu sans la carte de ce Mac.** Avant le tout premier rendu, et de nouveau dès que le Mac,
  macOS ou l’app change, l’app mesure la puce, en 30 à 40 secondes environ et sans aucun modèle, puis
  rend avec ce qu’elle a trouvé ; la barre latérale le dit pendant la mesure. Jusqu’à la 0.11, la
  carte était facultative, et les valeurs par défaut étaient celles d’un M1 Pro.
- **Ce qu’elle mesure** : les vrais produits matriciels des blocs des quatre DiT sur le GPU, sur
  l’unité matricielle du CPU (AMX) et sur les deux à la fois, à plusieurs profondeurs et largeurs ;
  l’attention par trois noyaux ; des produits lus directement sur les poids int8 et GGUF Q8_0
  publiés ; des produits bf16 sur les unités tensorielles du GPU (macOS 26, Metal 4). Chaque sortie
  est vérifiée contre le résultat exact, calculé en double précision.
- **Ce qu’elle décide** : un noyau ne s’allume que s’il a calculé juste et bat le défaut d’au moins
  5 %, ou de la dérive que la mesure elle-même a vue si elle est plus grande, aux tailles d’une image
  512² comme d’une image 1024². La co-exécution GPU + AMX, allumée d’après le nom de la puce jusqu’à
  la 0.11, est désormais éteinte partout sauf si la mesure la trouve plus rapide, et la part de
  chaque unité se calcule d’après la forme de chaque produit au lieu d’une part figée. Sur le M1 Pro
  (18/10/2026) : co-exécution allumée (×1,30), attention partagée par têtes entre le GPU et le CPU
  allumée (×1,49, désormais aussi pour les têtes groupées de Krea 2), attention flash éteinte (×1,05,
  dans la marge), produit int8 allumé, produit Q8_0 éteint (×0,43), produit tensoriel bf16 éteint
  (×0,53). Une autre puce décide d’après ses propres mesures.
- **Les poids 8 bits publiés, lus tels quels** : là où la carte les allume, les poids int8 des
  versions Compact (comme celles de Krea 2 et de Qwen-Image-2.1) et les poids GGUF Q8_0 (la Compact
  d’Anima) sont multipliés tels que publiés, au lieu d’être d’abord élargis en fp32 ; les activations
  et les sommes restent en fp32. Là où les unités tensorielles du GPU multiplient plus vite, les
  poids bf16 de l’éditeur se lisent de la même façon. Chaque chemin est vérifié contre la référence.
- **La carte change la vitesse, pas l’image** au-delà de l’arithmétique qu’elle a vérifiée ; le même
  Mac avec la même carte donne toujours les mêmes bits.
- **« Chaîne », en tête du rack, ouvre la mesure** : orange, avec sa puce, tant que ce Mac n’est pas
  mesuré, puce verte ensuite. **Mesurer ce Mac** (menu Aide) montre la carte et ce qu’elle
  a décidé ; « Ouvrir le signalement » copie désormais le rapport et ouvre un court formulaire GitHub où le
  coller, le rapport ne tenant plus dans un lien ; « Copier le JSON » sert à qui l’envoie autrement.

### Plus près de la référence, jusqu’à l’arrondi

- **L’arithmétique arrondit là où la référence arrondit** : les pas d’Euler, le correcteur d’UniPC
  (résolu comme LAPACK), les pas de temps entiers que diffusers donne au DiT sous DPM++ 2M et UniPC,
  les calendriers de Z-Image, d’Anima et de Krea 2, les tables de positions rotatives de chaque
  modèle, et les résidus à porte et modulations (a·b + c en deux arrondis, comme torch le calcule).
  **Chaque image change de quelques unités sur le dernier chiffre**, et chaque modèle reste à la
  distance de la référence elle-même d’un calcul fp64 : une image faite avec la 0.11 ne ressort plus
  au bit.
- **Z-Image en 832×1216 et 1216×832 change d’image** : ses jetons d’image sont maintenant complétés à
  un multiple de 32, comme le fait diffusers ; ces deux formats avaient 16 jetons de moins.
- **Les LoRA atteignent chaque module où la référence les applique** : sur Z-Image, les modules
  extérieurs (plongements des patchs et de la légende, plongement du temps, dernière couche), sur
  Krea 2, le plongement et la modulation du temps, qui étaient importés puis ignorés. Une LoRA qui
  vise un module que le moteur n’applique pas est refusée avant tout calcul, avec la raison.
- **Les éditions de Qwen-Image-2.1 gardent la transparence d’une référence**, comme diffusers : le
  VAE lit l’image avec son alpha, seule la copie que voit l’encodeur est posée sur du blanc.

### Plus solide

- **Un rendu raté n’est plus livré.** Si un rendu sort avec des valeurs non finies (une image
  noire), l’app le dit et ne garde rien, au lieu de l’ajouter à l’historique comme réussi ; dans un lot, seule
  cette image est écartée, les autres sont livrées. Rien de
  non fini ne passe plus d’un rendu au suivant : le cache de rendu ne garde ni ne relit une telle
  valeur, et une édition Qwen-Image-2.1 ne garde ses clés et valeurs d’attention que derrière un
  premier pas fini (une seule valeur fautive y noircissait toutes les éditions suivantes de la même
  image pour la session) ; une passe du VAE que le GPU refuse est signalée au lieu d’être ignorée.
  `silicontrol` le rend en `"error"` avec le code `non_finite_image`, et le verdict de `wait` (et la
  ligne `end` de `follow`) porte désormais les `"warnings"` du moteur.
- **Des formats par modèle** : Qwen-Image-2.1 veut des côtés multiples de 32, Anima au plus 2048 px ;
  l’un et l’autre sont refusés avant tout calcul, et le champ de taille dit la règle du modèle choisi.
- **Le cache de rendu ne dure qu’une session** : vidé au lancement et à la fermeture, il ne compte
  plus dans la place occupée de la fenêtre Modèles.
- **Importer ne télécharge jamais** : une LoRA ou un fine-tune dont le modèle n’est pas installé est
  refusé, et l’app dit quel modèle installer d’abord ; les vraies erreurs (un disque plein) ne sont
  plus avalées.
- **Les fichiers hostiles sont refusés** : une LoRA dont le motif d’alpha ferait revenir le filtre en
  arrière sans fin, des alphas ou des forces non finis, un rang 0, un bloc GGUF Q8_0 à l’échelle non
  finie.
- **Le besoin de mémoire compte ce qui lui échappait** : la mémoire de travail des poids int8
  tournés (la Compact de Krea 2 dépassait de 7 Mo à l’étage du texte), les jetons complétés de
  Z-Image. Un long prompt de Krea 2, au-delà de 170 jetons environ, ne s’arrête plus sur « arena
  exhausted ».
- **L’attention sans partage par têtes est découpée par requêtes** là où elle dépasserait la limite
  des matrices du GPU (Z-Image en 1024×1536 sur un Mac dont la carte éteint le partage).
- Le noyau d’attention flash calcule en mathématiques sûres et perd deux branches jamais utilisées
  qui pouvaient mal lire une clé ; les produits int8 et Q8_0 n’écrivent plus au-delà des colonnes du
  GPU ; un registre de téléchargement abîmé est ignoré ; la file attend la fin d’un agrandissement ou
  d’une planche d’exploration ; les requêtes de `silicontrol` ont une échéance.

### Interface

- **Une LoRA prend le nom de son fichier** : une LoRA importée s’appelle comme son fichier, et non
  plus du nom écrit dans ses métadonnées, souvent celui que l’outil d’entraînement met par défaut
  (« my first lora v1 copy copy »). Le nom donné par son auteur, et les mots déclencheurs quand le
  fichier les donne sans ambiguïté, s’affichent dans une infobulle sur la LoRA, dans le rack et la
  fenêtre Modèles. Les LoRA déjà importées gardent leur nom.
- La feuille de licence de Krea 2 montre la licence entière ; filets et fonds un peu plus marqués,
  texte secondaire là où il informe, un seul symbole d’export ; la fenêtre Réglages s’ouvre centrée ;
  la phrase de la fenêtre d’import est plus courte, la barre latérale du Finder garde sa largeur.
- Pour les développeurs : Siliconed livre une app, pas une bibliothèque. Le produit bibliothèque et
  la documentation de son API s’en vont ; `swift build` construit l’app et `silicontrol`.

Les temps sont mesurés sur un MacBook Pro M1 Pro de 16 Go.

## 0.11

- **Agrandir ×4** (le bouton sous l’image, son menu contextuel, le menu Image) : la super-résolution de
  macOS, le genre de réseau qu’utilise Pixelmator. Elle ne voit que l’image — aucun prompt —, elle
  affine donc bords, cheveux et textures sans rien ajouter : 1024 × 1024 devient 4096 × 4096 en une ou
  deux secondes, à côté de l’original, qui reste. Le modèle est celui d’Apple : macOS le télécharge une
  fois, après l’avoir demandé. Demande macOS 26 ; `silicontrol enlarge i7` fait de même.
- **Le rideau compare une image et son agrandissement**, et deux images marquées de mêmes
  proportions ; **le zoom va vers le curseur**, sur le canevas et sous le rideau, pour glisser la
  ligne sur un détail.
- **Coller une image** avec ⌘V (hors du prompt) : copiée dans Procreate, Photoshop, Aperçu ou le
  Finder, elle va là où un dépôt sur la case Images l’aurait mise — une image à éditer, ou un sujet à
  nommer.
- **Les LoRA à plat** : la pile est juste sous le modèle ; « Ajouter » est un menu des LoRA du
  modèle ; chacune montre le curseur de sa force, son chiffre (double-clic : retour à 0,8) et un
  bouton pour la retirer — plus de popover à ouvrir et refermer.
- **Explorer est à côté de Générer** : le même geste, beaucoup de petites images au lieu d’une.
- **Le rack en dit plus, à moins d’endroits** : chaque case montre ce qu’a pris son étage dès qu’il
  finit ; le survol du nom d’un étage dit ce qu’il fait (encodeur de texte, DiT, VAE…) ; la mémoire
  qu’il faut à un rendu est dite une fois, sous le rack ; ce qui grise Générer (« Écrivez un
  prompt »…) est dit au-dessus.
- **La file défile** au-delà de quelques rendus et montre en tête celui qui tourne : son image k/N
  dans un lot, le temps restant, un arrêt qui laisse le reste continuer. Le diagnostic quitte la
  barre d’outils (Réglages et le menu Aide le gardent).

### Corrigé

- Un plantage en marquant des images à comparer, surtout avec une grande image encore en lecture.
- Les métadonnées PNG écrivent toujours `CFG scale: 1` : le « Remix » de Civitai ne part plus de
  son échelle par défaut, qui brûlait l’image.
- L’icône du Dock ne rebondit plus quand la file se vide.

## 0.10.2

- **Une fenêtre Réglages** (Siliconed → Réglages…, ⌘,) remplace les interrupteurs du menu de l’app :
  le mode développeur, Expérimental (Sujets nommés, Forces de LoRA négatives), le rideau ou le côte à
  côte des comparaisons, la cartographie de ce Mac, la commande silicontrol (installée ou non, et où)
  et le dossier des modèles, montré dans le Finder ; chacun avec une phrase qui dit ce qu’il fait.
- **Forces de LoRA négatives** (Réglages → Expérimental) : les curseurs des LoRA vont
  de −1,5 à 1,5, dans le panneau et dans l’exploration. Une force négative éloigne l’image de ce que
  la LoRA a appris : c’est fait pour les LoRA « curseur » (âge, détail…), entraînées pour servir de
  −x à +x ; sur une LoRA ordinaire, elle abîme surtout l’image, d’où l’interrupteur. Éteint, une
  valeur négative est refusée, avec ce qu’il faut allumer. `silicontrol add --lora nom:-0.5` suit le
  même interrupteur.
- **Un axe « solveur » dans l’exploration** (mode développeur) : Euler, Euler a, DPM++ 2M et UniPC
  côte à côte, même graine, mêmes pas ; `silicontrol grid --x solver=euler,dpmpp-2m` fait de même. Le
  sous-titre de la planche nomme le solveur et le calendrier ComfyUI quand ce ne sont pas ceux par
  défaut.

### Corrigé

- **Les forces de LoRA tapées avec un point dans l’exploration** (`0.4, 0.8`) ne se lisaient pas sur un
  Mac réglé dans une langue qui écrit les décimales avec une virgule, et « Explorer » restait grisé.
  Les deux écritures se lisent.

## 0.10.1

- **Les images d’Anima créditent Anima sur Civitai.** Déposez un rendu d’Anima sur Civitai, il renvoie
  à la page du modèle, comme le faisaient déjà ceux de Z-Image et de Qwen-Image-2.1 : le PNG nomme
  Anima Turbo v1.1 (la page de circlestone_labs elle-même), dont le fichier est à l’octet près celui
  que l’app installe. Compact et Légère créditent la même page.

## 0.10

### Un troisième modèle : Anima Turbo v1.1

- **[Anima](https://huggingface.co/circlestone-labs/Anima) Turbo v1.1**, de CircleStone Labs et
  Comfy Org, rejoint Z-Image Turbo et Qwen-Image-2.1 dans l’app, sans le mode développeur. Un DiT de
  2 milliards de paramètres (Cosmos-Predict2) qui lit le prompt avec Qwen3-0.6B et un adaptateur, et
  décode avec le VAE de Qwen-Image. Il fait de l’illustration, de la peinture et de l’anime ; **pas
  de photoréalisme**, par choix de son éditeur. Il connaît les artistes par leur nom, écrit précédé
  d’un `@` (`@nom`), comme le dit sa fiche.
- **Sa recette** : 8 évaluations, CFG 1. Texte vers image, image vers image, LoRA (formats kohya et
  ComfyUI), les formats rectangulaires, les quatre solveurs en mode développeur ; ni édition par
  instruction ni Détail : sur ce modèle, il fait des blocs en mosaïque, ou rien de visible.
- **Les temps** : sur un M1 Pro de 16 Go, Standard, graine 42, 8 évaluations, 512² en 12,1 s, 1024² en
  43,7 s, 1024×1536 en 75,6 s.
- **Standard, Compact ou Légère** : 5,65 Go, 4,10 Go (le DiT en GGUF Q8_0) ou 3,31 Go (le DiT en
  GGUF Q4_K_M), les DiT tels que vanes430 les publie. L’encodeur de texte reste celui de l’éditeur,
  en bf16, dans les trois : aucune version 8 bits n’en est publiée.
- **Vérifié contre la référence**, étage par étage : à 1024², la sortie du débruiteur est à
  1,43·10⁻⁵ d’un calcul fp64 de la même évaluation, la référence fp32 à 1,47·10⁻⁵ ; le VAE plus près
  du fp64 que la référence ; une LoRA contre la fusion de ComfyUI elle-même.
- **Licence** : CircleStone Labs Non-Commercial pour le modèle, dérivé de Cosmos-Predict2 de NVIDIA
  (NVIDIA Open Model License). **Les images qu’il produit sont utilisables commercialement**, comme
  le dit la fiche de son éditeur. L’app montre la licence avant le premier octet téléchargé.

### Qwen-Image-2.1 : la recette telle que publiée

- **Tous les pas en pleine taille.** Jusqu’ici, une génération 1024² faisait ses deux premiers pas
  en demi-taille, un raccourci à nous : il disparaît, l’image est les 8 pas publiés de la Turbo et rien
  d’autre. Un 1024² prend 134 s dans l’app (142 s en ligne de commande) au lieu d’environ 120 s sur un M1 Pro 16 Go ;
  1024×1536 232 s au lieu de 190 s. Les éditions ne changent pas : elles n’ont jamais pris de pas en demi-taille.
- **Le calendrier de ComfyUI, pour comparer** (mode développeur) : un interrupteur « ComfyUI
  schedule » parcourt les niveaux de bruit que ComfyUI emploie pour ce modèle — ceux du modèle de
  base, pas ceux de la Turbo —, même graine, mêmes pas. `silicontrol add --schedule comfyui` fait de même.

### Corrigé

- **Un artiste écrit `@nom` dans un prompt d'Anima** était refusé par `silicontrol` quand les sujets nommés
  étaient activés (« No subject is called @nom ») : un modèle qui ne lit pas d'images garde maintenant son
  `@` comme du texte, comme le rack le faisait déjà.
- **Le temps de l'encodeur de texte** dans le rack pouvait afficher « ≈ 0,00 s » après un prompt relu du
  cache de rendu ; seul un prompt vraiment encodé l'apprend désormais.
- **Le temps restant** comptait le rendu en cours pour rien jusqu'à son premier pas complet : l'allure
  apprise pour ce modèle et ce format prend maintenant le relais.
- **La graine du rack** restait sur une ancienne graine fixe après un rendu de `silicontrol` à graine
  tirée au hasard ; elle dit maintenant « Au hasard ».

## 0.9

### « Signaler ma configuration » mesure maintenant la puce

- **Le diagnostic ne rend plus de modèle.** En une quinzaine de secondes, sans aucun modèle, il
  cartographie ce que calcule la puce de votre Mac : les produits de matrices des blocs des modèles sur
  le GPU, sur l’unité matricielle du CPU et sur les deux à la fois, les produits en fp16 et en bf16,
  l’attention par les trois noyaux du moteur, l’élargissement des poids sur le CPU et sur le GPU —
  chaque résultat vérifié contre le résultat exact, calculé en double précision. Le rapport JSON dit
  ce que la puce calcule, et si elle le calcule juste.
- **Corrigé : avec l’installation Compact, le diagnostic disait toujours « diffère de la
  référence ».** Il comparait la sortie des poids 8 bits à la référence en pleine précision : les poids
  différaient, pas la puce.

### Le moteur suit la cartographie

- **Une fois qu’elle a tourné, Siliconed utilise ce qu’elle a mesuré, dès le lancement suivant :** le
  GPU et l’unité matricielle du CPU ensemble sur les grands produits, et dans quelle proportion ; les
  têtes de l’attention partagées entre le GPU et le CPU ; le noyau d’attention maison — chacun
  seulement s’il est plus rapide sur ce Mac, au-delà du bruit de la mesure elle-même, et seulement
  s’il a calculé juste. Tant que vous ne la lancez pas, rien ne change. La feuille dit ce que ce Mac
  utilise, et si c’est maintenant ou au prochain lancement.
- Après une mise à jour de macOS ou de Siliconed, la cartographie est mise de côté jusqu’à ce que vous
  la relanciez.
- Avec une cartographie, la même graine peut donner des bits légèrement différents de ceux sans ; le
  même Mac avec la même cartographie redonne les mêmes bits.

## 0.8.2

### Corrigé : Z-Image Turbo suit sa recette

- **8 pas font maintenant 8 évaluations en pleine taille, comme le dit la fiche du modèle.** Jusqu’à
  la 0.8.1, Siliconed en faisait 7 (il suivait un ancien calendrier de diffusers dont le dernier pas
  ne bougeait rien), et à 1024² il calculait les deux premières à demi-taille. Les deux sont
  corrigés ; revérifié contre diffusers, étage par étage. Les images sortent plus propres, avec des
  détails plus fins.
- **Z-Image Turbo 1024² prend maintenant 113–119 s** (118–120 s en Légère), 512² ~34 s. Les
  temps publiés jusqu’à la 0.8.1 (82–85 s à 1024²) valaient pour le calendrier raccourci ; les chiffres
  de Siliconed dans le README sont remesurés ; sa comparaison avec Draw Things oppose désormais
  8 évaluations à 8, ceux de Draw Things étant ceux mesurés le 13/10/2026.
- Une image Z-Image faite avec la 0.8.1 ou avant ne ressort plus au bit ; à version égale et sur le
  même Mac, les mêmes bits restent garantis. Qwen-Image-2.1 ne change pas.

Les temps sont mesurés sur un MacBook Pro M1 Pro de 16 Go.

## 0.8.1

### Corrigé

- **Les dernières rangées du rack reprennent le clic.** Juste au-dessus du pied de la colonne, la
  rangée des LoRA, **+ LoRA** et **Importer…** étaient dessinés mais ne répondaient pas tant qu’on ne
  faisait pas défiler la colonne. Le pied est désormais sous le rack, plus par-dessus.

### Des métadonnées PNG que Civitai lit

- **Déposez une image sur Civitai, ses réglages s’affichent.** À côté de ses propres clés, chaque PNG
  porte désormais le texte que Civitai lit : le prompt avec une balise `<lora:nom:poids>` par LoRA,
  puis les pas, le solveur, la graine, la taille et le modèle. Les rendus de Z-Image Turbo et de
  Qwen-Image-2.1 créditent leur page Civitai sans rien saisir.
- **Les LoRA sont créditées aussi**, par le sha256 du fichier tel qu’il a été publié, gardé à
  l’import. Une LoRA importée avec une version antérieure est nommée sans être créditée : la
  réimporter suffit.

## 0.8

### Rechercher les mises à jour

- **Dans le menu de l’app, à la demande seulement.** Siliconed demande à GitHub sa dernière version
  et la compare à celle qui tourne ; s’il en existe une plus récente, il ouvre sa page. Rien n’est
  vérifié en arrière-plan, rien n’est installé à votre place.

### Solveurs (mode développeur)

- **Euler ancestral, DPM++ 2M et UniPC**, en plus d’Euler, sur Z-Image Turbo et Qwen-Image-2.1 : les
  mêmes pas, les mêmes σ, une autre marche entre eux. Chacun est vérifié contre le scheduler de
  diffusers du même nom. Euler, par défaut, donne les mêmes images qu’avant, au bit près.
- **Une autre image, le même temps.** UniPC peut recomposer une image de Qwen-Image-2.1 ; le solveur
  ancestral peut aplatir un fond. Les éditions gardent l’image 1 sous les quatre. Aperçus et
  esquisses montrent l’estimation propre à chaque solveur. Dans le rack, la légende de l’historique,
  le PNG et `silicontrol add|grid --solver`.

### Les chiffres

- **Remesurés par l’app**, chaque rendu mis en file par `silicontrol` : Z-Image Turbo 1024² en 82 à
  85 s, Qwen-Image-2.1 en 118 s, une édition à une image en 212 s, à deux en 285 s ; 0 page
  swappée. Le README les pose à côté du moteur de Draw Things sur le même Mac. *Note (0.8.2) : Z-Image faisait 7 des 8 évaluations de sa recette jusqu’à la 0.8.1 ; ces temps de Z-Image valent pour 7 (voir la 0.8.2).*

Temps et mémoire mesurés sur un MacBook Pro M1 Pro de 16 Go.

## 0.7.1

La 0.7 n’a jamais été publiée : la 0.7.1 la porte, avec le calcul de la mémoire rendu exact dans
les cas qu’il manquait.

### Mémoire : un budget fixe

- **Siliconed ne prend jamais plus que son budget** : les trois quarts de la mémoire du Mac moins
  2 Go (10 Go sur un Mac de 16 Go, 4 sur un Mac de 8). Il ne lit plus l’occupation du Mac : le même
  rendu sur le même Mac reçoit toujours la même réponse, quoi qu’il y ait d’autre d’ouvert. Ce que
  les autres applications occupent leur appartient.
- **Connu avant le premier pas.** Chaque rendu calcule la mémoire qu’il lui faut d’après son format,
  son prompt, ses images et ses LoRA, prend le plan le plus rapide qui tient dans le budget, et
  compare son propre pic à ce chiffre. Plus rien ne change de plan en cours de route. Ce chiffre est
  une borne haute : un long prompt sur Qwen-Image-2.1, un lot de plusieurs images et la mémoire de
  l’app elle-même sont comptés, et un rendu dont le besoin ne se calcule pas est refusé plutôt que
  lancé à l’aveugle.
- **Grisé d’avance.** Un format ou un modèle au-delà du budget de ce Mac est grisé avant le clic,
  avec le Mac qu’il demanderait : « demande un Mac de 24 Go ».
- **Les lots de Qwen-Image-2.1** ne gardent plus deux copies du modèle à la fois.
- Mêmes images, au bit près. Mesuré sur le M1 Pro : Z-Image Turbo en 1024² en 81 s,
  Qwen-Image-2.1 en 1024² en 116 s ; la plus grande édition (1024×1536, trois images) culmine à
  3,2 Go. *Note (0.8.2) : Z-Image faisait 7 des 8 évaluations de sa recette jusqu’à la 0.8.1 ; ces temps de Z-Image valent pour 7 (voir la 0.8.2).*
- Le rapport de diagnostic (schéma 4) donne le budget, le besoin de chaque étage et le pic mesuré en
  face.

### Sujets nommés (Expérimental)

- **Rien de gardé.** Les sujets vivent désormais en mémoire seulement, comme les images de la
  session : rien n’est écrit sur le disque, ils partent quand l’app se ferme. Un dossier de sujets
  écrit par la 0.6 est effacé au lancement.
- **Images ou Sujets.** La boîte au-dessus du prompt tient soit les images d’une édition (1 à 3),
  soit vos sujets (autant que vous voulez, le prompt en citant 3 au plus) — l’un ou l’autre, jamais
  les deux.
- **Le nom n’atteint jamais le modèle.** « @alice » devient « la personne de l’image 1 » : un nom lu
  comme du contenu donnait Alice au pays des merveilles.

Les temps et la mémoire sont mesurés sur un MacBook Pro M1 Pro de 16 Go.

## 0.6

### Une seule boîte Images, au-dessus du prompt

- **Les images d’abord.** Sur Qwen-Image-2.1, la boîte qui tient les images à éditer s’appelle
  désormais Images et se place au-dessus du prompt : on pose l’image, puis on dit ce qu’on en fait.
  Ses fils passent à côté du prompt jusqu’aux encodeurs qui les lisent.
- **Un seul budget, compté une fois.** Avec les sujets nommés (Expérimental), les sujets vivent dans
  la même boîte, sous les images de l’édition : pour le modèle, un sujet cité est une image de plus.
  Le titre de la boîte compte les deux face aux 3 images du modèle, et passe au rouge au-delà.

### Mémoire

- **Un Mac au repos ne refuse plus Qwen-Image-2.1 en 1024².** Une génération simple se voyait
  compter la mémoire d’une édition à 3 images, et la marge de sécurité ignorait ce que macOS libère
  en compressant la mémoire des autres apps. Désormais un rendu compte le pic mesuré pour son propre
  nombre d’images, et la marge tient compte de cette compression : sur le Mac du signalement, de
  2,9 à 4,5 Go disponibles pour le même bureau. La règle du zéro swap ne change pas : tout reste
  mesuré au pire cas.

### Sujets nommés (Expérimental)

- **Comparer avec le sujet.** « @ana en western comic » garde le cadrage de la photo, comme une
  édition : le rideau sous l’image compare maintenant le résultat avec son image 1, qu’elle vienne de
  la boîte Images ou d’un sujet (son cadre, tel qu’il a été envoyé).

### Sous le capot

- Des noyaux et des réglages dépassés sont retirés (un noyau d’attention scalaire, un mode en demi-
  précision) : une seule version de chaque calcul vit, celle qui est vérifiée. Rien ne change dans
  les images.

Temps et mémoire mesurés sur un MacBook Pro M1 Pro de 16 Go.

## 0.5

### Qwen-Image-2.1 est désormais le Turbo officiel de Qwen

- **Un seul modèle, de son éditeur.** Qwen-Image-2.1 fait maintenant tourner le modèle distillé
  de Qwen lui-même, [Qwen-Image-2.1 Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo).
  Jusqu'à la 0.4, l'app faisait tourner le modèle de base sous la LoRA turbo d'un tiers (celle de
  Viggle) ; cette LoRA et le modèle de base disparaissent. Texte vers image, édition par
  instruction avec 1 à 3 images et vos propres LoRA marchent comme avant.
- **8 pas, sur le calendrier que Qwen publie.** Le nombre de pas est fixé à 8 : le mode à 9 pas
  disparaît, et tout autre nombre est refusé.
- **La Compact et la Légère suivent.** La Compact prend le Turbo 8 bits (int8 convrot) de
  Comfy-Org, la Légère le Turbo GGUF Q4_K_M d'unsloth ; l'encodeur de texte 8 bits de la Compact
  ne change pas. Sur le disque : 31,2 Go en Standard, 17,3 Go en Compact, 14,2 Go en Légère
  (33,2, 19,4 et 16,2 Go avant : la LoRA turbo n'est plus stockée).
- **Une installation existante est reconnue.** La fenêtre Modèles propose de « Finir
  l’installation… » : l'ancien transformer et les petits fichiers sont effacés, ceux du Turbo
  sont téléchargés dans la même version (Standard, Compact ou Légère). L'encodeur de texte est
  gardé : celui du Turbo est celui du modèle de base, au bit près, il ne se retélécharge donc pas.
  Les fine-tunes de Qwen-Image-2.1 importés pour le modèle de base ne peuvent plus rendre et sont
  effacés.
- **Même licence** : Qwen Research, non commerciale — le Turbo porte le texte du modèle de base.
- **Vérifié comme le reste.** Chaque étage est comparé à l'implémentation de référence de Qwen ;
  le moteur reste à 3,0·10⁻⁶ du calcul exact (fp64).

### Plus rapide que la 0.4 pour une image, plus lent pour une édition

Le Turbo prend 8 pas là où la LoRA tierce en prenait 6. Nous l'avons choisi quand même : c'est la
qualité que Qwen publie et vérifie, sans pièce tierce entre les deux qui pourrait changer ou
disparaître. Puis nous avons rendu ses pas moins chers, sans baisser la précision : tout est
toujours calculé en fp32 et vérifié contre le calcul exact.

- **Les deux premiers pas à demi-taille.** À partir de 1024², une génération calcule ses deux
  premiers pas en 512², puis agrandit, comme Z-Image le fait déjà. L'image change (la même graine
  donne une autre composition) mais reste propre. Pas sous une LoRA ni Détail, ni en édition.
- **L'attention partagée entre le GPU et le CPU.** Dans chaque couche, 13 des 32 têtes d'attention
  vont à l'unité matricielle du CPU pendant que le GPU calcule les autres.
- **De plus petits gains**, aux mêmes bits : le modèle est lu d'un seul flux, et des normalisations
  sont fusionnées.

Sur le Mac de référence :

- génération 1024² : ~120 s (~134 s en 0.4, ~160 s pour le Turbo sans ces gains) ;
- une édition avec 1 référence : ~210 s (~190 s), avec 2 références ~285 s ;
- le pire cas, 1024×1536 avec 3 références : ~515 s, 4,7 Go de mémoire au pic, sans aucun swap ;
- la Légère prend le même temps que la Standard.

### Installation

- **Plus rapide, et plus légère en mémoire.** La préparation d'un modèle téléchargé prend maintenant
  quelques secondes au lieu de plusieurs minutes, et le fichier suivant se télécharge pendant que le
  précédent se prépare. Sur une connexion rapide (85 Mo/s), Qwen-Image-2.1 s'installe en ~6 min 30 en
  Standard (~8 min en 0.4), ~3 min 30 en Compact (~5 min 30) et ~3 min en Légère (~4 min) : c'est
  désormais le réseau qui fixe la durée. Le modèle installé est le même, à l'octet près.
- **Pas de swap pendant l'installation.** Le pic de mémoire passe de 4,6 Go à 2,2–2,5 Go, et les
  fichiers fraîchement téléchargés ne poussent plus le reste du Mac hors de la mémoire : une
  installation qui pouvait swapper sur un Mac chargé ne le fait plus.

### Lots

- **Les lots ne swappent plus.** Chaque image d'un lot reconstruisait le modèle, et sur un Mac
  chargé un lot pouvait finir par swapper. Désormais toutes les images se calculent avec le même
  modèle, puis toutes se décodent : les mêmes images, sans swap, sur Z-Image et Qwen-Image-2.1.
- **Y compris avec votre LoRA.** Sur Qwen-Image-2.1, un lot avec une LoRA à vous garde le
  modèle, LoRA appliquée, d'une image à l'autre.

Temps mesurés sur un MacBook Pro M1 Pro, 16 Go.

## 0.4

### Installation

- **Standard, Compact ou Légère.** Z-Image Turbo et Qwen-Image-2.1 s'installent avec les poids de
  l'éditeur (Standard), avec des poids 8 bits publiés par des tiers pour le transformer et
  l'encodeur de texte (Compact : 11,7 Go au lieu de 20,4 pour Z-Image, 19,4 Go au lieu de 33,2 pour
  Qwen-Image-2.1), ou avec le transformer en 4 à 6 bits (GGUF Q4_K_M) et l'encodeur de texte de la
  Compact (Légère : 9,5 Go et 16,2 Go). La Légère est présélectionnée sur un Mac de 8 Go de mémoire.
  La version installée continue de rendre jusqu'à ce que la nouvelle soit complète.
- **Des téléchargements qui reprennent.** Les gros fichiers arrivent par morceaux, plusieurs à la
  fois ; un morceau qui cale est redemandé, et un téléchargement coupé reprend là où il s'était
  arrêté. Chaque fichier reste vérifié par sha256.
- **Signée et notariée.** L'app et l'image disque sont signées avec un Developer ID et notariées par
  Apple : elles s'ouvrent d'un double clic.

### Édition

- **Éditer depuis n'importe où.** Clic droit sur une image de l'historique, « Éditer avec
  Qwen-Image-2.1 », même quand le rack est réglé sur un modèle qui n'édite pas : le rack passe à
  l'éditeur et l'image devient l'image 1. Une vignette de l'historique glissée sur la boîte Éditer
  devient une référence.
- **Rideau original / éditée.** Une image éditée a un bouton qui la pose sur son original, avec une
  ligne à faire glisser pour voir exactement ce qui a changé. Échap le quitte.

### Images

- **Détail à 1024².** « Détail : Normal · Plus · Maximum » donne maintenant un résultat propre à
  1024² aussi, sur Z-Image et Qwen-Image-2.1 : sa force suit la taille de l'image, et il ne laisse
  plus de texture granuleuse en haute résolution.
- **Les variations suivent « Images ».** Les variations subtiles ou marquées d'une image en font
  autant que le champ « Images » du rack (1 par défaut), au lieu de toujours quatre.

### Exploration

- **Le modèle se choisit dans le panneau.** Le panneau d'exploration a son propre menu « Modèle ».
- **Une seule ligne d'état.** Le panneau dit où il en est en une ligne : « Esquisse à n pas sur N »,
  ou « Images finies, N pas » pour Qwen-Image-2.1, qui explore désormais en images finies (ses
  premières esquisses étaient floues).

### Modèles importés

- **Un 8 bits reste 8 bits.** Un fine-tune importé en 8 bits — fp8, int8, GGUF Q8_0 ou int8
  convrot — est gardé en 8 bits sur le disque, environ la moitié du fichier 16 bits (Z-Image :
  6,2 Go au lieu de 12,3 Go). Le calcul reste en fp32, et chaque poids est vérifié au bit près contre
  la conversion de l'éditeur du fichier. L'app importe les `.gguf` comme les `.safetensors`.
- **Le GGUF jusqu'à 4 bits.** Un fine-tune `.gguf` en Q4_0 à Q6_K est gardé tel que publié, ses blocs
  copiés entiers (Z-Image : environ 5 Go en Q4_K_M au lieu de 12,3 Go). Les fichiers sous 4 bits, et
  les fichiers 4 bits hors GGUF, sont refusés, avec la raison.

### Vitesse et mémoire

- **Z-Image est plus rapide à 1024²** : environ 82 s au lieu d'environ 107 s, et 153 s au lieu de
  168 s à 1024×1536, toujours sans aucun swap. *Note (0.8.2) : Z-Image faisait 7 des 8 évaluations de sa recette jusqu’à la 0.8.1 ; ces temps de Z-Image valent pour 7 (voir la 0.8.2).*
- **Un contrôle de mémoire sur l'état réel de la machine** : avant chaque rendu, l'app lit la mémoire
  qu'elle peut vraiment avoir à ce moment-là (pas seulement le total du Mac) et refuse, chiffres à
  l'appui, un rendu qui ne tiendrait pas — au lieu de le laisser swapper. Si la mémoire vient à
  manquer pendant un rendu, le moteur passe à un plan plus économe : la même image, un peu plus
  lentement, plutôt que de swapper.

### Langues

- **Allemand, espagnol et italien**, en plus de l'anglais et du français : l'app suit la langue du
  système.

Temps mesurés sur un MacBook Pro M1 Pro, 16 Go.
