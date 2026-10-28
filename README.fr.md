[English](README.md) · Français

# Siliconed

**Générer et éditer des images sur votre Mac, en local, avec des modèles récents aux poids ouverts : calculées en pleine précision, vérifiées contre le code d'origine, et sans jamais prendre plus de mémoire qu'un budget fixe.**

Siliconed est une app macOS libre et gratuite pour la **génération d'images à partir de texte et l'édition d'images par IA, en local**, sur les Mac Apple silicon (M1 et suivants). Elle fait tourner des modèles de diffusion récents — [Krea 2 Turbo](https://huggingface.co/krea/Krea-2-Turbo), [Z-Image Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo), [Qwen-Image-2.1 Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo), [Anima Turbo](https://huggingface.co/circlestone-labs/Anima) et [MOSS-Image](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT) (bêta : sa recette en 8 pas est celle de la communauté) — en natif, en Swift et Metal : ni Python, ni serveur, ni cloud, rien ne quitte votre Mac. Elle marche hors ligne une fois le modèle installé.

![Licence : Apache 2.0](https://img.shields.io/badge/licence-Apache%202.0-blue)
![macOS 26+](https://img.shields.io/badge/macOS-26%2B-black)
![Apple silicon](https://img.shields.io/badge/Apple%20silicon-M1%20et%20suivants-black)
[![tests unitaires](https://img.shields.io/github/actions/workflow/status/gwenn-ha-dev/Siliconed/test.yml?branch=main&label=tests%20unitaires)](https://github.com/gwenn-ha-dev/Siliconed/actions/workflows/test.yml)
[![noyaux vérifiés, étapes CPU](https://img.shields.io/github/actions/workflow/status/gwenn-ha-dev/Siliconed/kernels.yml?branch=main&label=noyaux%20v%C3%A9rifi%C3%A9s%2C%20%C3%A9tapes%20CPU)](https://github.com/gwenn-ha-dev/Siliconed/actions/workflows/kernels.yml)

![Une photo ensoleillée de Krea 2 Turbo — une Irlandaise rousse en combinaison shorty qui sort des vagues en riant, sa planche de surf sous le bras, à contre-jour d'un soleil bas — et son édition en bande dessinée western, en cel shading, par Qwen-Image-2.1, sous un rideau qui glisse](docs/images/curtain.gif)

*Une photo de Krea 2 Turbo, éditée par Qwen-Image-2.1 avec une phrase (« Convert this photo to a western comic book style with cel shading », graine 42, 215 s sur un M1 Pro), comparées sous le rideau.*

## Télécharger

L'app est distribuée en `.dmg` sur la page [Releases](https://github.com/gwenn-ha-dev/Siliconed/releases). Prérequis : macOS 26 ou plus, Apple silicon. Les modèles s'installent depuis l'app ; vous choisissez lesquels. Pour la construire vous-même : [Construire depuis les sources](docs/ENGINE.fr.md#construire-depuis-les-sources).

## Ce que fait l'app

- **Génère des images à partir de texte et les édite par instruction.** Qwen-Image-2.1 Turbo prend 1 à 3 images et une phrase (« remplace le ciel nuageux par un ciel bleu »). Quel modèle pour quoi, avec licences et tailles : [Les modèles](docs/APP.fr.md#les-modèles).
- **Explore avant de s'engager.** Une grille d'esquisses rapides à travers graines, pas, forces de LoRA ou modèles ; la case choisie devient l'image finie.
- **Prend vos propres modèles.** Importez des LoRA et des fine-tunes depuis Civitai ou Hugging Face.
- **Local et privé.** Le rendu se fait sur votre Mac, sans filtre de contenu ajouté par l'app. Elle suit la langue du système (anglais, français, allemand, espagnol, italien, japonais ou chinois simplifié).
- **Scriptable.** `silicontrol` pilote l'app ouverte depuis un terminal, pour les scripts et les agents IA ([détails](docs/REMOTE-CONTROL.fr.md)).

Pour en savoir plus sur l'app : [docs/APP.fr.md](docs/APP.fr.md).

## Ce qui la distingue

- **Rapide, sans rogner.** Z-Image Turbo en 1024² en 83 s sur un MacBook Pro M1 Pro de 16 Go, avec les 8 évaluations (passages du modèle) que demande sa recette, par défaut les trois premières en demi-taille, dans la Compact que l'app recommande (le modèle en poids 8 bits, moitié moins de disque ; [les trois versions](docs/APP.fr.md#la-gestion-des-modèles)). En précision complète, les poids bf16 de l'éditeur calculés en fp32 : 82 s. La précision complète ne coûte presque rien. [Tous les chiffres](docs/ENGINE.fr.md#les-chiffres).
- **De gros modèles sur un Mac de 16 Go.** 34 Go de modèle tournent sur un Mac de 16 Go : les poids sont lus sur le disque au moment où ils servent, et Siliconed ne prend jamais plus que les ¾ de la mémoire du Mac moins 2 Go. Chaque rendu calcule ce qu'il lui faut avant de commencer ; au-delà du budget, il est refusé d'emblée. [Comment c'est mesuré](docs/ENGINE.fr.md#la-mémoire-et-le-budget).
- **Vérifié contre les modèles d'origine.** Chaque étage est comparé, canal par canal, à l'implémentation de référence (diffusers / transformers en fp32 sur CPU), et ce qui ne se vérifie pas n'est pas livré. Refaites les contrôles chez vous : [VERIFY.fr.md](docs/VERIFY.fr.md). La CI passe les étages CPU des noyaux chaque fois que le moteur change ; son GPU est virtuel, les étages GPU se jugent sur votre Mac.
- **Réglé pour votre Mac.** Avant le premier rendu, en une minute environ et sans aucun modèle, il chronomètre ses noyaux sur votre puce et vérifie chaque résultat contre le résultat exact ; un noyau ne s'allume que s'il a calculé juste et battu le défaut. [Ce qu'il a décidé sur un M1 Pro](docs/ENGINE.fr.md#un-moteur-mesuré-sur-votre-mac).
- **Mesuré à côté du moteur de Draw Things** sur le même Mac, avec la méthode et tous les chiffres : [À côté de Draw Things](docs/ENGINE.fr.md#à-côté-de-draw-things).

## Documentation

- [docs/APP.fr.md](docs/APP.fr.md) : l'app en détail : créer, éditer, explorer, gérer les modèles, vie privée.
- [docs/ENGINE.fr.md](docs/ENGINE.fr.md) : le moteur : la mesure du Mac, le Neural Engine, les chiffres, le rendu plus rapide, Draw Things, la mémoire, le diagnostic communautaire, la vérification, la construction depuis les sources.
- [docs/VERIFY.fr.md](docs/VERIFY.fr.md) : vérifier les résultats contre la référence sur votre Mac.
- [docs/REMOTE-CONTROL.fr.md](docs/REMOTE-CONTROL.fr.md) : `silicontrol`, protocole et fonctionnement interne.
- [docs/RELEASE-NOTES.fr.md](docs/RELEASE-NOTES.fr.md) : ce qui change à chaque version.
- Les issues : [rapport de diagnostic, bug, idée](https://github.com/gwenn-ha-dev/Siliconed/issues/new/choose) (en anglais).

## Crédits

Les modèles appartiennent à leurs éditeurs (listés dans [Les modèles](docs/APP.fr.md#les-modèles)), chacun sous sa propre licence. Les implémentations de référence sont [diffusers](https://github.com/huggingface/diffusers) et [transformers](https://github.com/huggingface/transformers), de Hugging Face ; le rééchantillonnage des images reproduit le Lanczos de [Pillow](https://github.com/python-pillow/Pillow) au bit près. Les LoRA et les fine-tunes viennent de la communauté, sur [Civitai](https://civitai.com) et [Hugging Face](https://huggingface.co) ; la recette en 8 pas de MOSS-Image est la [LoRA DMD](https://huggingface.co/lodestones/MOSS-Image-SFT-8-steps-dmd) de lodestones.

## Licence

Le code est sous [licence Apache 2.0](LICENSE). Les poids des modèles ne font pas partie de ce dépôt : chacun est téléchargé par vous, sous sa propre licence, que l'app affiche avant de l'installer.
