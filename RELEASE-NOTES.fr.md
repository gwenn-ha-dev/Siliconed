[English](RELEASE-NOTES.md) · Français

# Siliconed — notes de version

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
  168 s à 1024×1536, toujours sans aucun swap.
- **Un contrôle de mémoire sur l'état réel de la machine** : avant chaque rendu, l'app lit la mémoire
  qu'elle peut vraiment avoir à ce moment-là (pas seulement le total du Mac) et refuse, chiffres à
  l'appui, un rendu qui ne tiendrait pas — au lieu de le laisser swapper. Si la mémoire vient à
  manquer pendant un rendu, le moteur passe à un plan plus économe : la même image, un peu plus
  lentement, plutôt que de swapper.

### Langues

- **Allemand, espagnol et italien**, en plus de l'anglais et du français : l'app suit la langue du
  système.

Temps mesurés sur un MacBook Pro M1 Pro, 16 Go.
