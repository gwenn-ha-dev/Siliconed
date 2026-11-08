[English](VERIFY.md) · Français

# Vérifier Siliconed

Siliconed affirme que chaque étage de chaque modèle qu'il livre calcule ce que calcule l'implémentation de référence. **Cette page montre comment le vérifier soi-même**, sur son Mac, sans nous croire sur parole.

La référence est celle de l'éditeur : [diffusers](https://github.com/huggingface/diffusers) et [transformers](https://github.com/huggingface/transformers), en float32 sur le CPU, avec les poids publiés. Des scripts Python de ce dépôt l'exécutent et figent ses réponses en *goldens*. `siliconed check` fait tourner le moteur sur les mêmes entrées et le juge contre elles, étage par étage : encodeur de texte, débruiteur, échantillonneur, décodeur.

Ce qui n'est **pas** affirmé : les mêmes bits que la référence, ni qu'un autre Mac. Le moteur calcule dans un autre ordre sur un autre processeur ; il est jugé sur sa distance à la valeur exacte (la [règle](#la-règle) plus bas), pas sur ses bits.

Les chiffres d'ici, et les contrôles, sont ceux du rendu à pleine taille, Rendu plus rapide éteint. Le Rendu plus rapide produit, par construction, une autre image à graine égale, et rien ne compare cette image à la référence ([ENGINE](ENGINE.fr.md#rendu-plus-rapide)).

## Mise en place

Il faut l'app (0.18 ou suivante), Python 3.12, et ce dépôt à la version de l'app (Siliconed → À propos de Siliconed l'affiche). `siliconed` est livré dans l'app : c'est le moteur même avec lequel l'app rend, et il tourne sans Xcode.

```sh
alias siliconed=/Applications/Siliconed.app/Contents/Helpers/siliconed
git clone --branch v0.21 https://github.com/gwenn-ha-dev/Siliconed && cd Siliconed
python3.12 -m venv oracle-env && oracle-env/bin/pip install -r tools/oracle/requirements.txt
```

La première exécution d'un script d'oracle télécharge les poids de référence depuis le Hugging Face Hub, à des révisions épinglées (les tailles sont en [section 3](#3--tous-les-modèles)) ; `HF_HOME` ou `HF_HUB_CACHE` décide où ils vont. La référence de Krea 2, `krea/Krea-2-Turbo`, est à accès restreint : acceptez sa licence sur le Hub, puis `oracle-env/bin/hf auth login`, avant `krea2.py` ; faites de même pour tout autre dépôt qui le demande. Le code convrot int8 dont toutes les versions Compact ont besoin est récupéré une fois, à un commit épinglé, et son sha256 est vérifié.

Pour vérifier plutôt votre propre compilation : `swift build -c release --product siliconed` (Xcode 26), puis `.build/release/siliconed` à la place de l'alias.

Chaque étape affiche la carte qu'elle lit (une ligne `map`), ses mesures, puis une ligne de verdict : `✓`, `✗`, ou `—` (sautée, avec la raison). Le code de sortie dit la même chose : 0 réussi, 1 échoué, 2 sauté, 64 erreur d'usage. Les goldens vont dans `./goldens`, pour les scripts comme pour les contrôles, sauf si `--goldens DIR` ou `SILICONED_GOLDENS` en dit autrement.

## 1 · Les noyaux : cinq minutes, sans poids

Les noyaux numériques (tables RoPE, rééchantillonnage, opérations élément par élément du GPU, attention, partage d'un produit matriciel entre GPU et AMX) se vérifient sans aucun modèle.

```sh
oracle-env/bin/python tools/oracle/kernels.py all
siliconed check kernels
```

Sur notre MacBook Pro M1 Pro :

```
✓ kernels widen · 7 measures
✓ kernels arena · 6 measures
✓ kernels elementwise · 7 measures
✓ kernels gelu · 1 measure
✓ kernels sdpa S=4128 · 17 measures
✓ kernels sdpa-tiles S=4128 · 30 measures
✓ kernels gemm-split · 9 measures
✓ kernels rope · 13 measures
✓ kernels resample · 6 measures
```

Le [workflow kernels](https://github.com/gwenn-ha-dev/Siliconed/actions/workflows/kernels.yml) lance ces deux commandes sur les Mac de GitHub chaque fois que le moteur ou ses contrôles changent. Son GPU est virtuel : les étapes GPU y sont sautées, celles du CPU jugées.

## 2 · Un modèle entier : Anima, dix minutes et 5 Go

Anima est le modèle le plus léger à vérifier de bout en bout. L'installer d'abord dans l'app (Standard) : `siliconed check` lit les poids que l'app a installés, jamais ceux que télécharge l'oracle. Puis :

```sh
oracle-env/bin/python tools/oracle/anima.py all     # télécharge une fois 5 Go de poids de référence, puis 7 min
siliconed check anima                               # chaque étape à chaque taille : 3 min
```

Sur notre MacBook Pro M1 Pro (les mesures entre la ligne `map` et la ligne de verdict sont omises) :

```
✓ anima tokenizer · 1 measure
map ~/Library/Application Support/Siliconed/store/qwen3-0.6b-encoder.v0.silicon · 1.2 GB · bfloat16 (standard weights)
✓ anima text · 9 measures · nearest its threshold: prompt_embeds 1.33e-05 of 5.00e-04
map ~/Library/Application Support/Siliconed/store/anima-turbo-dit.v0.silicon · 3.9 GB · bfloat16 (standard weights)
✓ anima dit 512 · 12 measures · nearest its threshold: model_out per plane 1.93e-06 of 2.51e-06
map ~/Library/Application Support/Siliconed/store/anima-turbo-dit.v0.silicon · 3.9 GB · bfloat16 (standard weights)
✓ anima dit 1024 · 12 measures · nearest its threshold: model_out per plane 3.04e-06 of 4.74e-06
map ~/Library/Application Support/Siliconed/store/anima-turbo-dit.v0.silicon · 3.9 GB · bfloat16 (standard weights)
✓ anima dit 768x512 · 12 measures · nearest its threshold: temb 4.14e-05 of 5.00e-04
map ~/Library/Application Support/Siliconed/store/anima-turbo-dit.v0.silicon · 3.9 GB · bfloat16 (standard weights)
✓ anima trajectory 512 · 9 measures · nearest its threshold: v_3 9.79e-05 of 5.00e-04
map ~/Library/Application Support/Siliconed/store/anima-turbo-dit.v0.silicon · 3.9 GB · bfloat16 (standard weights)
✓ anima trajectory 1024 · 9 measures · nearest its threshold: v_7 2.17e-04 of 5.00e-04
✓ anima vae 512 · 1 measure · nearest its threshold: image RGB 6.33e-07 of 1.28e-06
✓ anima vae 1024 · 1 measure · nearest its threshold: image RGB 3.18e-07 of 1.05e-06
✓ anima vae 768x512 · 2 measures · nearest its threshold: image RGB 3.66e-07 of 1.28e-06
```

Vos chiffres différeront sur les derniers chiffres : la référence elle-même arrondit autrement sur un autre processeur. Les verdicts, non. Une étape se lance aussi seule, `siliconed check anima dit --size 1024` ; `siliconed check list` les nomme toutes.

## 3 · Tous les modèles

Les deux mêmes commandes, pour chaque modèle : `tools/oracle/<script> all`, puis `siliconed check <modèle>`.

| Modèle | Script | Téléchargement | Oracle, toutes les étapes | Son étape la plus lourde | Contrôle |
|---|---|---|---|---|---|
| Z-Image | `z-image.py` | 31 Go | 39 min | la trajectoire 1024² : 14 min, 13,3 Go | 7 min |
| Qwen-Image-2.1 | `qwen-image-2.1.py` | 30 Go | 27 min | le témoin fp64 du décodeur : 43 s, 15,5 Go | 10 min |
| Anima | `anima.py` | 5 Go | 7 min | le décodeur 1024² : 1 min, 20,8 Go | 3 min |
| Krea 2 | `krea2.py` | 33 Go | 1 h 45 min | les huit témoins fp64 des pas à 1024² : 44 min, 11 Go | 16 min |
| MOSS-Image | `moss-image.py` | 28 Go | 11 min | la trajectoire de 8 pas : 3,6 min, 10,6 Go | 3 min |

Sur un MacBook Pro M1 Pro, 16 Go, temps écoulé. Le téléchargement vient en plus des poids de l'app. L'oracle fait passer le débruiteur bloc par bloc en float32 (celui de Krea 2 pèse 48 Go à cette précision), si bien qu'un Mac de 16 Go exécute chaque script ; une étape dont le pic dépasse la mémoire de la machine (les décodeurs 1024²) tourne en swap et prend plus longtemps.

Les lignes de verdict des quatre autres modèles sur notre MacBook Pro M1 Pro, avec la ligne `map` que chaque étape affiche (celles d'Anima sont dans la section ci-dessus ; les mesures sont omises) :

<details>
<summary>Z-Image</summary>

```
✓ z-image tokenizer · 3 measures
map ~/Library/Application Support/Siliconed/store/qwen3-4b-encoder.v0.silicon · 7.8 GB · bfloat16 (standard weights)
✓ z-image text · 3 measures · nearest its threshold: layer34_out 1.45e-05 of 5.00e-04
map ~/Library/Application Support/Siliconed/store/z-image-turbo-dit.v1.silicon · 12.3 GB · bfloat16 ×514, float32 ×7 (standard weights)
✓ z-image block 512 · 19 measures · nearest its threshold: layer0_adaln_raw 6.63e-05 of 5.00e-04
map ~/Library/Application Support/Siliconed/store/z-image-turbo-dit.v1.silicon · 12.3 GB · bfloat16 ×514, float32 ×7 (standard weights)
✓ z-image block 1024 · 19 measures · nearest its threshold: layer0_adaln_raw 6.63e-05 of 5.00e-04
map ~/Library/Application Support/Siliconed/store/z-image-turbo-dit.v1.silicon · 12.3 GB · bfloat16 ×514, float32 ×7 (standard weights)
✓ z-image dit 512 · 11 measures · nearest its threshold: t_embedder_out 3.73e-04 of 5.00e-04
map ~/Library/Application Support/Siliconed/store/z-image-turbo-dit.v1.silicon · 12.3 GB · bfloat16 ×514, float32 ×7 (standard weights)
✓ z-image dit 1024 · 11 measures · nearest its threshold: t_embedder_out 3.73e-04 of 5.00e-04
map ~/Library/Application Support/Siliconed/store/z-image-turbo-dit.v1.silicon · 12.3 GB · bfloat16 ×514, float32 ×7 (standard weights)
✓ z-image dit 768x512 · 12 measures · nearest its threshold: t_embedder_out 3.73e-04 of 5.00e-04
map ~/Library/Application Support/Siliconed/store/z-image-turbo-dit.v1.silicon · 12.3 GB · bfloat16 ×514, float32 ×7 (standard weights)
✓ z-image dit 832x1216 · 12 measures · nearest its threshold: t_embedder_out 3.73e-04 of 5.00e-04
map ~/Library/Application Support/Siliconed/store/z-image-turbo-dit.v1.silicon · 12.3 GB · bfloat16 ×514, float32 ×7 (standard weights)
✓ z-image trajectory 512 · 16 measures · nearest its threshold: v_5 7.41e-05 of 5.00e-04
map ~/Library/Application Support/Siliconed/store/z-image-turbo-dit.v1.silicon · 12.3 GB · bfloat16 ×514, float32 ×7 (standard weights)
✓ z-image trajectory 1024 · 16 measures · nearest its threshold: v_3 1.65e-04 of 5.00e-04
✓ z-image vae 768x512 · 2 measures · nearest its threshold: image RGB 2.52e-07 of 8.50e-07
```

</details>

<details>
<summary>Qwen-Image-2.1</summary>

```
map ~/Library/Application Support/Siliconed/store/qwen3-vl-8b-encoder.v0.silicon · 16.3 GB · bfloat16 (standard weights)
✓ qwen-image-2.1 text · 11 measures · nearest its threshold: prompt_embeds 2.22e-05 of 5.00e-04
map ~/Library/Application Support/Siliconed/store/qwen3-vl-8b-encoder.v0.silicon · 16.3 GB · bfloat16 (standard weights)
✓ qwen-image-2.1 text --edit 1 · 16 measures · nearest its threshold: layer0_in 8.42e-05 of 5.00e-04
map ~/Library/Application Support/Siliconed/store/qwen3-vl-8b-encoder.v0.silicon · 16.3 GB · bfloat16 (standard weights)
✓ qwen-image-2.1 text --edit 2 · 16 measures · nearest its threshold: layer0_in 1.50e-04 of 5.00e-04
map ~/Library/Application Support/Siliconed/store/qwen3-vl-8b-encoder.v0.silicon · 16.3 GB · bfloat16 (standard weights)
✓ qwen-image-2.1 vision · 22 measures · nearest its threshold: vit_out_1 2.13e-04 of 5.00e-04
✓ qwen-image-2.1 references 512 · 2 measures
✓ qwen-image-2.1 references 443 · 2 measures
map ~/Library/Application Support/Siliconed/store/qwen-image-2.1-turbo-dit.v0.silicon · 14.2 GB · bfloat16 ×296, float32 ×1 (standard weights)
✓ qwen-image-2.1 dit 512 · 24 measures · nearest its threshold: step norm_out 1.74e-04 of 5.00e-04
map ~/Library/Application Support/Siliconed/store/qwen-image-2.1-turbo-dit.v0.silicon · 14.2 GB · bfloat16 ×296, float32 ×1 (standard weights)
✓ qwen-image-2.1 dit-edit1 512 · 24 measures · nearest its threshold: temb 2.11e-04 of 5.00e-04
map ~/Library/Application Support/Siliconed/store/qwen-image-2.1-turbo-dit.v0.silicon · 14.2 GB · bfloat16 ×296, float32 ×1 (standard weights)
✓ qwen-image-2.1 dit-edit2 512 · 24 measures · nearest its threshold: temb 2.11e-04 of 5.00e-04
map ~/Library/Application Support/Siliconed/store/qwen-image-2.1-turbo-dit.v0.silicon · 14.2 GB · bfloat16 ×296, float32 ×1 (standard weights)
✓ qwen-image-2.1 trajectory 512 · 11 measures · nearest its threshold: v_3 6.50e-06 of 5.00e-04
map ~/Library/Application Support/Siliconed/store/qwen-image-2.1-turbo-dit.v0.silicon · 14.2 GB · bfloat16 ×296, float32 ×1 (standard weights)
✓ qwen-image-2.1 trajectory-edit1 512 · 13 measures · nearest its threshold: v_1 2.02e-05 of 5.00e-04
map ~/Library/Application Support/Siliconed/store/qwen-image-2.1-turbo-dit.v0.silicon · 14.2 GB · bfloat16 ×296, float32 ×1 (standard weights)
✓ qwen-image-2.1 trajectory-edit2 512 · 14 measures · nearest its threshold: v_1 7.68e-06 of 5.00e-04
✓ qwen-image-2.1 vae 512 · 2 measures · nearest its threshold: image RGBA 4.51e-07 of 1.18e-06
✓ qwen-image-2.1 vae-edit1 512 · 3 measures · nearest its threshold: image RGBA 1.00e-06 of 5.00e-04
✓ qwen-image-2.1 vae-edit2 512 · 2 measures · nearest its threshold: image RGBA 9.64e-07 of 5.00e-04
✓ qwen-image-2.1 vae-encode 512 · 7 measures · nearest its threshold: picture 1 latent 4.27e-06 of 7.16e-06
```

</details>

<details>
<summary>Krea 2</summary>

```
map ~/Library/Application Support/Siliconed/store/qwen3-vl-4b-encoder.v0.silicon · 7.8 GB · bfloat16 (standard weights)
map ~/Library/Application Support/Siliconed/store/krea2-turbo-dit.v0.silicon · 26.3 GB · bfloat16 ×256, float32 ×174 (standard weights)
✓ krea2 text · 7 measures · nearest its threshold: tap 35 2.54e-05 of 5.00e-04
map ~/Library/Application Support/Siliconed/store/krea2-turbo-dit.v0.silicon · 26.3 GB · bfloat16 ×256, float32 ×174 (standard weights)
✓ krea2 dit 512 · 11 measures · nearest its threshold: model_out per plane 1.04e-04 of 1.14e-04
map ~/Library/Application Support/Siliconed/store/krea2-turbo-dit.v0.silicon · 26.3 GB · bfloat16 ×256, float32 ×174 (standard weights)
✓ krea2 dit 1024 · 11 measures · nearest its threshold: block0_attn_out 3.57e-03 of 3.59e-03
map ~/Library/Application Support/Siliconed/store/krea2-turbo-dit.v0.silicon · 26.3 GB · bfloat16 ×256, float32 ×174 (standard weights)
✓ krea2 dit 768x512 · 11 measures · nearest its threshold: block0_out 9.44e-04 of 1.20e-03
map ~/Library/Application Support/Siliconed/store/krea2-turbo-dit.v0.silicon · 26.3 GB · bfloat16 ×256, float32 ×174 (standard weights)
✓ krea2 trajectory 512 · 9 measures · nearest its threshold: v_2 5.08e-04 of 5.14e-04
map ~/Library/Application Support/Siliconed/store/krea2-turbo-dit.v0.silicon · 26.3 GB · bfloat16 ×256, float32 ×174 (standard weights)
✓ krea2 trajectory 1024 · 9 measures · nearest its threshold: v_1 3.43e-03 of 4.80e-03
✓ krea2 vae 512 · 1 measure · nearest its threshold: image RGB 5.20e-07 of 1.48e-06
✓ krea2 vae 1024 · 1 measure · nearest its threshold: image RGB 2.89e-07 of 1.03e-06
✓ krea2 vae 768x512 · 2 measures · nearest its threshold: image RGB 3.23e-07 of 9.63e-07
```

</details>

Deux mesures de Krea 2 passent de peu sur notre Mac : la vitesse `v_2` de la trajectoire en 512² (5,08e-4 pour une tolérance de 5,14e-4) et `block0_attn_out` du débruiteur en 1024² (3,57e-3 pour 3,59e-3, 0,6 % de marge). La tolérance se calcule sur l'écart de votre propre référence, elle suit donc les arrondis de votre processeur, et ces deux mesures peuvent échouer d'un cheveu sur un autre Mac. La règle n'est pas desserrée pour elles. Si elles échouent chez vous, une issue avec la sortie de `siliconed check` nous aide à trouver la couche qui porte l'écart.

<details>
<summary>MOSS-Image</summary>

```
✓ moss-image tokenizer · 11 measures
map ~/Library/Application Support/Siliconed/store/qwen3.5-9b-encoder.v0.silicon · 15.9 GB · bfloat16 ×249, float32 ×177 (standard weights)
✓ moss-image text-delta · 8 measures · nearest its threshold: layer 0 out 3.61e-06 of 5.00e-04
map ~/Library/Application Support/Siliconed/store/qwen3.5-9b-encoder.v0.silicon · 15.9 GB · bfloat16 ×249, float32 ×177 (standard weights)
✓ moss-image text-full · 15 measures · nearest its threshold: layer3 gated attention 2.39e-05 of 5.00e-04
map ~/Library/Application Support/Siliconed/store/qwen3.5-9b-encoder.v0.silicon · 15.9 GB · bfloat16 ×249, float32 ×177 (standard weights)
✓ moss-image text · 44 measures · nearest its threshold: layer3 gated attention 2.39e-05 of 5.00e-04
✓ moss-image schedule · 25 measures
map ~/Library/Application Support/Siliconed/store/moss-image-sft-dit.v0.silicon · 10.2 GB · bfloat16 (standard weights)
✓ moss-image dit 512 · 23 measures · nearest its threshold: model_out per plane 1.35e-06 of 1.75e-06
map ~/Library/Application Support/Siliconed/store/moss-image-sft-dit.v0.silicon · 10.2 GB · bfloat16 (standard weights)
✓ moss-image dit-lora 512 · 22 measures · nearest its threshold: model_out per plane 2.72e-06 of 4.17e-06
map ~/Library/Application Support/Siliconed/store/moss-image-sft-dit.v0.silicon · 10.2 GB · bfloat16 (standard weights)
✓ moss-image trajectory 512 · 11 measures · nearest its threshold: v_6 per plane 2.97e-05 of 5.00e-04
✓ moss-image vae 512 · 4 measures · nearest its threshold: image RGB (encoded test image) 2.34e-07 of 7.28e-07
```

</details>

## Vos goldens sont-ils les nôtres ?

Nous ne publions pas les goldens (plusieurs gigaoctets, et des poids dont les licences l'interdisent). Nous publions leurs empreintes : forme, type et statistiques de chaque tenseur.

```sh
oracle-env/bin/python tools/oracle/fingerprints.py compare
```

Les entiers (identifiants de tokens, masques) doivent concorder exactement. Les flottants se comparent avec une tolérance (1e-3 sur les normes), pas au bit près : PyTorch sur le CPU ne promet pas les mêmes bits float32 d'une machine à l'autre, et deux exécutions honnêtes de la référence diffèrent d'un arrondi. Cet arrondi est très en dessous de ce que jugent les contrôles.

## La règle

Le chiffre d'une étape est l'**écart relatif par canal** : pour chaque canal (le dernier axe d'un tenseur, ou chaque plan d'un latent), la norme L2 de la différence sur la norme L2 de la référence. On donne le pire canal, jamais une moyenne : une faute confinée à un canal disparaîtrait dans un chiffre global. La même règle juge chaque frontière de chaque modèle :

    sans témoin :   écart(moteur, référence)  ≤ 5e-4
    avec témoin :   écart(moteur, témoin)     ≤ max(plancher, 1,25 · écart(référence, témoin))

Le *témoin* est le même calcul en float64. Là où il existe, il tranche : le float32 a sa propre erreur d'arrondi, et sur une frontière mal conditionnée la référence elle-même peut être à 4e-3 de la valeur exacte ; le moteur peut alors en être aussi loin que la référence, plus 25 %.

La ligne de chaque étape dans `check list` donne sa règle. Une identité (tokens, plannings) se juge au bit près. Les étapes de tous les modèles tiennent le même plancher, 5e-4. Un rectangle se contrôle aussi contre sa transposée, qui doit échouer, pour qu'une hauteur et une largeur inversées ne passent pas.

Chaque décodeur, celui de Z-Image compris, se juge par plan de l'image contre un témoin float64 du décodeur de la référence (`vae-witness`).

**Le diagnostic de l'app applique la même règle.** **Mesurer ce Mac** ([ENGINE](ENGINE.fr.md#un-moteur-mesuré-sur-votre-mac)) vérifie chaque route qu'il chronomètre contre le résultat exact, calculé en double précision sur le CPU même du Mac, sur des opérandes synthétiques identiques sur tous les Mac, avec le même écart par canal : un produit ou une attention en fp32 peut en être à 10⁻⁵, le produit fp16 du Neural Engine à 1,25 fois l'arrondi propre du bf16. Une route au-delà de son seuil est écartée du plan — le moteur ne la propose jamais — et le rapport la nomme avec l'écart vu ; aucune vitesse ne rachète un résultat faux.

Le débruiteur de Krea 2 a quelques frontières si mal conditionnées qu'un changement d'une unité sur le dernier chiffre de l'entrée déplace la sortie de plus que la règle ne permet, référence comprise. Là, une mesure qui échoue au témoin est jugée par *nuages* : la référence et le moteur tournent tous deux sur des entrées décalées d'un ulp, et l'écart médian du moteur doit rester sous 1,25 fois le pire de la référence. Ce sont les étapes dont la règle dit « clouds when the oracle has draws ». `SILICONED_CHECK_CLOUDS=1` écrit chaque tirage des deux nuages sur stderr, une ligne par tirage séparée par des tabulations (`cloud`, la frontière, `oracle` ou `engine`, le genre du tirage, son écart), puis la médiane et le seuil.

## Au-delà de `all`

Deux contrôles demandent des goldens que `all` n'écrit pas, chacun par une étape de script de plus. `siliconed check <modèle>` les laisse de côté ; ils se lancent étape par étape.

**Les prompts longs.** L'app lit jusqu'à 1 024 tokens de prompt et refuse au-delà ; les pipelines de référence coupent à 512 par défaut. `text --long` vérifie l'entre-deux : le script lit `tools/oracle/inputs/long-prompt.txt` (844 tokens pour les tokenizers Qwen, 954 pour le T5 d'Anima), la longueur de séquence de la référence relevée pour que rien n'en soit coupé, et le contrôle lit le même prompt comme l'app l'admet.

```sh
oracle-env/bin/python tools/oracle/z-image.py text --long
siliconed check z-image text --long
```

**Le Neural Engine.** L'app peut partager les produits matriciels du débruiteur avec le Neural Engine du Mac, qui calcule en fp16 ([ENGINE](ENGINE.fr.md#le-neural-engine)). `--neural-engine` vérifie ce chemin : chaque produit élargi du DiT d'au moins 512 lignes donne la moitié de ses lignes à l'ANE, le même partage sur tous les Mac et plus qu'aucun plan n'en donne (un rendu laisse au GPU les produits de moins de 1 024 lignes), et l'étape est jugée contre la référence calculée en bf16 sur les mêmes entrées, le moteur ayant droit à l'écart propre au bf16 plus 25 %. Le verdict dit combien de produits l'ANE a calculés ; sur un Mac où il n'en calcule aucun, l'étape est sautée. Les étapes `dit` des cinq modèles l'acceptent.

```sh
oracle-env/bin/python tools/oracle/z-image.py bf16 --size 1024
siliconed check z-image dit --size 1024 --neural-engine
```

**Activations arrondies en bf16.** Là où le plan de ce Mac la retient, les produits du DiT prennent leur activation arrondie en bf16 juste avant le produit (les poids tels que publiés, les sommes et la sortie en fp32). `--bf16-activations` vérifie ce chemin sur n'importe quel Mac, pour chaque produit de 256 lignes ou plus, et juge l'étape comme celle du Neural Engine : contre la référence calculée en bf16, l'écart permis étant celui du bf16 plus 25 %. Le verdict dit combien de produits ont arrondi leur opérande. Les étapes `dit` des cinq modèles l'acceptent.

```sh
siliconed check z-image dit --size 512 --bf16-activations
```

**L'attention en demi-précision.** Là où le plan de ce Mac la retient, sur toutes les versions, la Standard comprise, l'attention du DiT arrondit une fois Q, K, V et les probabilités en bf16 ou en fp16, le softmax et toutes les sommes en fp32. `--bf16-attention` et `--fp16-attention` vérifient chacune sur n'importe quel Mac, pour chaque attention du DiT, sous la forme fusionnée à la première tuile candidate (`kernels sdpa-tiles` vérifie toutes les formes contre le float64), et jugent l'étape comme celle du Neural Engine : contre la référence calculée en bf16, l'écart permis étant celui du bf16 plus 25 %. Le verdict dit combien d'attentions ont tourné ainsi. Les étapes `dit` des cinq modèles les acceptent, une à la fois.

```sh
siliconed check anima dit --size 512 --fp16-attention
```

**Activations int8 sur une Compact (W8A8).** Les poids d'une Compact sont publiés en int8 avec une rotation de Hadamard (l'int8 convrot de Comfy-Org et de Bedovyy), et leur auteur les calcule en W8A8 : l'activation tournée par les mêmes blocs de 256, quantifiée en int8 par ligne, un produit int8 × int8 aux sommes int32, fois les deux échelles. `--int8-activations` vérifie cette route contre la recette de l'auteur elle-même, l'`int8_linear` par défaut de ComfyUI (comfy_kitchen 0.2.37), calculée sur les entiers publiés, le reste de la référence en bf16 : l'écart permis est celui de cette recette au témoin float64, plus 25 %. Les étapes `dit` de Z-Image, Qwen-Image-2.1, Anima, Krea 2 et MOSS-Image l'acceptent, avec `--variant compact` ; tant que la route du moteur n'est pas dans la version, l'étape est sautée plutôt que jugée.

```sh
siliconed check z-image dit --size 512 --variant compact --int8-activations
```

**Les accélérateurs éteints.** Un rendu dont l'interrupteur **Accélérateurs** de l'app est éteint calcule son DiT en fp32 de bout en bout sous la carte de ce Mac : les moteurs et les noyaux que la carte choisit, les poids lus tels que publiés à côté d'activations fp32, mais aucun produit sur le Neural Engine et aucun opérande arrondi sous le fp32. `--no-accelerators` vérifie ce chemin : avec `SILICONED_MAP` qui nomme le rapport de ce Mac (une mesure de ce Mac et de cette version), l'étape tourne sous le plan de cette carte interrupteur éteint, jugée par la règle du fp32 contre la référence fp32 ; sans lui, ce sont les défauts qui tournent, déjà en fp32. Les étapes `dit` des cinq modèles l'acceptent.

```sh
SILICONED_MAP=<rapport.json> siliconed check z-image dit --size 512 --no-accelerators
```

L'édition de Qwen-Image-2.1, avec une et deux images de référence, fait partie de son `all` et de `siliconed check qwen-image-2.1` ; une étape seule, c'est `siliconed check qwen-image-2.1 dit --edit 2`. Sa barre sous le Neural Engine s'écrit par `qwen-image-2.1.py bf16 --edit N`.

**La Compact et la Légère.** Le DiT de la Compact est un fichier de tiers (int8 convrot, de Comfy-Org et, pour Anima, de Bedovyy) ; celui de la Légère est le nôtre, dans le format 4 bits SQ, un fichier par famille sur Hugging Face (`gwenn-ha-dev/<modèle>-Siliconed-SQ4`, dont la fiche donne le format). L'app épingle chacun par dépôt, révision et sha256. Deux choses se vérifient, de l'extérieur.

*Le décodage*, valeur par valeur. La Compact : `swift test` lit de petits morceaux de chaque fichier publié (`Tests/SiliconedTests/Fixtures/quant`, lignes coupées par HTTP `Range` à la révision épinglée, octets identiques aux publiés) et compare l'élargissement du moteur à la bibliothèque de l'éditeur, au bit : `comfy_kitchen` de ComfyUI pour le convrot, torchao et SDNQ pour les fichiers float8 et int8 (`gguf-py` pour les fichiers GGUF que la forge importe encore). `tools/oracle/quant_fixtures.py` réécrit ces fixtures depuis les fichiers publiés (`python tools/oracle/quant_fixtures.py [slug ...]` ; les manifestes portent le sha256 de ce qu'il a écrit, et une relance doit les laisser inchangés). La Légère : aucune bibliothèque que la nôtre ne lit SQ, l'oracle porte donc un second décodeur, `sq_decode` dans `tools/oracle/quantized.py`, écrit d'après la seule spécification de la fiche (numpy, rien de commun avec le `SQ.swift` du moteur) : les codes deviennent des int8 exacts, multipliés par l'échelle float32 de la ligne en float64, tournés en arrière par R256 en float64, arrondis une fois en float32. `SQTests` vérifie le décodeur du moteur contre la même définition, réécrite dans le test.

*Le débruiteur*, en entier : le DiT de référence de diffusers, en float32 sur le CPU, avec les poids que le fichier contient (ceux de la Compact déquantés par la bibliothèque de l'éditeur, ceux de la Légère décodés par `sq_decode` ; `tools/oracle/quantized.py`), sur les entrées mêmes de la Standard (même latent, mêmes états du texte, même sigma). Les scripts de l'oracle prennent `--variant compact|light` pour `dit` et `bf16`, et, pour Z-Image et Krea 2, pour `witness` (Krea 2 aussi pour `witness-step --step 0` et `draw`) ; MOSS-Image n'a qu'une Compact, et son script prend `--variant compact` pour `dit`, `witness` et `dit-bf16` ; les variantes d'Anima et de Qwen-Image-2.1 (génération seule, sans `--edit`) n'ont pas de témoin fp64 et se jugent par la règle simple contre la référence. Le contrôle prend `--variant` pour `dit`, à 512, et lit la carte de cette version à côté de celle de la Standard (`--map` en désigne une autre). Une carte de l'autre version est refusée, et dit pourquoi : une Légère jugée contre le golden d'une Compact serait d'autres poids. Une carte est celle d'une Légère quand elle contient un tenseur SQ (`sq4`, `sq5`), celle d'une Compact quand elle contient de l'int8 et pas de SQ.

```sh
oracle-env/bin/python tools/oracle/z-image.py dit --size 512 --variant light      # télécharge le fichier de 5 Go une fois
oracle-env/bin/python tools/oracle/z-image.py bf16 --size 512 --variant light
oracle-env/bin/python tools/oracle/z-image.py witness --size 512 --variant light
siliconed check z-image dit --size 512 --variant light
siliconed check z-image dit --size 512 --variant light --neural-engine
```

L'oracle de la Compact demande aussi la bibliothèque de son éditeur : `oracle-env/bin/pip install -r tools/oracle/requirements-quant.txt` d'abord (il ajoute `gguf`, et `sdnq` et `torchao` pour `quant_fixtures.py`, aux dépendances de base). Celui de la Légère n'a besoin de rien de plus que les dépendances de base.

La Compact et la Légère calculent par défaut sur le Neural Engine (produits en fp16) : `--neural-engine` est donc la barre qui compte pour elles, jugée comme celle de la Standard : contre la référence en bf16 sur les mêmes poids (déquantés ou décodés), le moteur ayant droit à l'erreur propre du bf16 plus 25 %. Ces goldens demandent d'abord le golden `dit` de la Standard à 512 (ils en lisent les entrées) ; ils ne font partie ni de `all` ni de `siliconed check <modèle>`.

`dit --size 512` des cinq Compact et des quatre Légères a été lancé sur notre Mac mini M5 Pro, en fp32 puis avec `--neural-engine`, sur les fichiers que l'app installe (`siliconed install <modèle>`, avec `--light` pour une Légère ; les Légères contre des goldens refaits depuis les mêmes fichiers) : 18 exécutions, 17 passantes. La rouge est la Compact de MOSS-Image en fp32, dite sous la table. Ci-dessous, l'écart de `model_out` à la référence (pour Qwen-Image-2.1, `model_out (full)` ; pour Krea 2 et MOSS-Image, `model_out per plane`) dans chaque exécution, et la barre contre laquelle l'exécution sous le Neural Engine est jugée.

| Variante | `model_out` en fp32, contre la référence | Sous le Neural Engine, contre la référence | Sa barre (l'écart propre au bf16 + 25 %) |
|---|---|---|---|
| Z-Image, Compact | 1,058e-05 | 3,982e-03 | 6,79e-02 |
| Z-Image, Légère | 9,065e-06 | 2,708e-03 | 6,61e-02 |
| Qwen-Image-2.1, Compact | 3,942e-06 | 3,502e-04 | 1,59e-02 |
| Qwen-Image-2.1, Légère | 2,647e-06 | 2,804e-04 | 1,82e-02 |
| Anima, Compact | 5,916e-06 | 1,017e-03 | 6,01e-02 |
| Anima, Légère | 5,725e-06 | 1,145e-03 | 6,24e-02 |
| Krea 2, Compact | 4,008e-05 | 2,433e-04 | 1,67e-02 |
| Krea 2, Légère | 6,222e-05 | 2,198e-04 | 2,05e-02 |
| MOSS-Image, Compact | 1,871e-06 ✗ (2,142e-06 au témoin fp64, pour 1,91e-06) | 3,214e-04 | 2,89e-02 |

**La Compact de MOSS-Image en fp32 dépasse sa barre de 12 %.** Jugé au témoin fp64, son `model_out per plane` est à 2,142e-06 de la valeur exacte, là où la référence en fp32 en est à 1,53e-06 (barre 1,91e-06). La route qui la franchit est le noyau int8 du GPU sur les entiers publiés (252 produits, `matmul2d float × int8 → float`, l'échelle après la somme) ; la même exécution sur les poids élargis passe (1,638e-06), et le noyau seul est aussi près du produit exact qu'un produit fp32 (1,1–2,1e-6, comme MPS en fp32) : ce qui bouge est un ordre de sommation, à l'échelle de 1e-6. Sur notre MacBook Pro M1 Pro, où ce noyau ne prenait pas les produits, la même exécution passait (1,470e-06). Krea 2 tranche une telle frontière par nuages d'ordres de sommation (plus bas) ; MOSS-Image n'a pas encore de tirages, si bien que la ligne reste rouge jusque-là. L'app rend MOSS-Image sur le Neural Engine par défaut, et cette exécution passe (3,214e-04 pour 2,89e-02).

Avec `--bf16-activations` (l'activation arrondie en bf16, les poids lus tels que publiés), le `model_out` des Légères vaut 3,690e-02 pour Z-Image (barre 6,61e-02), 2,436e-03 pour Qwen-Image-2.1 (1,82e-02), 8,109e-03 pour Anima (6,24e-02) et 7,326e-04 pour Krea 2, par plan (2,05e-02) : les quatre passent.

La colonne fp32 est l'écart simple à la référence ; Z-Image, Krea 2 et MOSS-Image y sont jugés au témoin fp64, comme leurs Standards. Les lignes complètes :

<details>
<summary>Compact et Légère, les 18 exécutions</summary>

```
$ siliconed check z-image dit --size 512 --variant compact
map ~/Library/Application Support/Siliconed/store/z-image-turbo-dit.compact.v1.silicon · 6.2 GB · int8 ×270, bfloat16 ×244, float32 ×7 (8-bit weights)
✓ z-image dit compact 512 · 11 measures · nearest its threshold: t_embedder_out 2.95e-04 of 5.00e-04
$ siliconed check z-image dit --size 512 --variant compact --neural-engine
map ~/Library/Application Support/Siliconed/store/z-image-turbo-dit.compact.v1.silicon · 6.2 GB · int8 ×270, bfloat16 ×244, float32 ×7 (8-bit weights)
✓ z-image dit compact 512 · 12 measures · nearest its threshold: x_embedder_out 2.39e-04 of 3.21e-03
$ siliconed check z-image dit --size 512 --variant light
map ~/Library/Application Support/Siliconed/store/z-image-turbo-dit.light.v1.silicon · 5.0 GB · bfloat16 ×274, sq5 ×148, sq4 ×56, int8 ×36, float32 ×7 (packed weights)
✓ z-image dit light 512 · 11 measures · nearest its threshold: t_embedder_out 2.95e-04 of 5.00e-04
$ siliconed check z-image dit --size 512 --variant light --neural-engine
map ~/Library/Application Support/Siliconed/store/z-image-turbo-dit.light.v1.silicon · 5.0 GB · bfloat16 ×274, sq5 ×148, sq4 ×56, int8 ×36, float32 ×7 (packed weights)
✓ z-image dit light 512 · 12 measures · nearest its threshold: x_embedder_out 2.39e-04 of 3.21e-03
$ siliconed check qwen-image-2.1 dit --size 512 --variant compact
map ~/Library/Application Support/Siliconed/store/qwen-image-2.1-turbo-dit.compact.v0.silicon · 7.3 GB · int8 ×224, bfloat16 ×72, float32 ×1 (8-bit weights)
✓ qwen-image-2.1 dit compact 512 · 26 measures · nearest its threshold: temb 2.11e-04 of 5.00e-04
$ siliconed check qwen-image-2.1 dit --size 512 --variant compact --neural-engine
map ~/Library/Application Support/Siliconed/store/qwen-image-2.1-turbo-dit.compact.v0.silicon · 7.3 GB · int8 ×224, bfloat16 ×72, float32 ×1 (8-bit weights)
✓ qwen-image-2.1 dit compact 512 · 27 measures · nearest its threshold: temb 2.11e-04 of 5.00e-04
$ siliconed check qwen-image-2.1 dit --variant light
map ~/Library/Application Support/Siliconed/store/qwen-image-2.1-turbo-dit.light.v1.silicon · 4.4 GB · sq4 ×184, bfloat16 ×73, sq5 ×32, int8 ×7, float32 ×1 (packed weights)
✓ qwen-image-2.1 dit light 512 · 26 measures · nearest its threshold: temb 2.11e-04 of 5.00e-04
$ siliconed check qwen-image-2.1 dit --variant light --neural-engine
map ~/Library/Application Support/Siliconed/store/qwen-image-2.1-turbo-dit.light.v1.silicon · 4.4 GB · sq4 ×184, bfloat16 ×73, sq5 ×32, int8 ×7, float32 ×1 (packed weights)
✓ qwen-image-2.1 dit light 512 · 27 measures · nearest its threshold: temb 2.11e-04 of 5.00e-04
$ siliconed check anima dit --size 512 --variant compact
map ~/Library/Application Support/Siliconed/store/anima-turbo-dit.compact.v1.silicon · 2.4 GB · bfloat16 ×313, int8 ×254 (8-bit weights)
✓ anima dit compact 512 · 11 measures · nearest its threshold: block27_out 2.04e-05 of 5.00e-04
$ siliconed check anima dit --size 512 --variant compact --neural-engine
map ~/Library/Application Support/Siliconed/store/anima-turbo-dit.compact.v1.silicon · 2.4 GB · bfloat16 ×313, int8 ×254 (8-bit weights)
✓ anima dit compact 512 · 12 measures · nearest its threshold: block13_out 1.44e-03 of 5.81e-02
$ siliconed check anima dit --size 512 --variant light
map ~/Library/Application Support/Siliconed/store/anima-turbo-dit.light.v1.silicon · 1.4 GB · int8 ×188, bfloat16 ×152, sq4 ×148, sq5 ×79 (packed weights)
✓ anima dit light 512 · 11 measures · nearest its threshold: block27_out 1.85e-05 of 5.00e-04
$ siliconed check anima dit --size 512 --variant light --neural-engine
map ~/Library/Application Support/Siliconed/store/anima-turbo-dit.light.v1.silicon · 1.4 GB · int8 ×188, bfloat16 ×152, sq4 ×148, sq5 ×79 (packed weights)
✓ anima dit light 512 · 12 measures · nearest its threshold: block27_out 6.53e-03 of 2.21e-01
$ siliconed check krea2 dit --size 512 --variant compact
map ~/Library/Application Support/Siliconed/store/krea2-turbo-dit.compact.v0.silicon · 14.1 GB · int8 ×224, float32 ×174, bfloat16 ×32 (8-bit weights)
model_out per plane         4.008e-05   1.89e-05       15   1.072e-04    1.19e-04  ✓  oracle at fp64 9.48e-05
gemm_int8                   ✓  2016 products on the published integers
model_out per plane (widened)  3.016e-05   2.10e-05       15   1.032e-04    1.19e-04  ✓  oracle at fp64 9.48e-05
✓ krea2 dit compact 512 · 13 measures · nearest its threshold: model_out per plane 1.07e-04 of 1.19e-04
$ siliconed check krea2 dit --size 512 --variant compact --neural-engine
map ~/Library/Application Support/Siliconed/store/krea2-turbo-dit.compact.v0.silicon · 14.1 GB · int8 ×224, float32 ×174, bfloat16 ×32 (8-bit weights)
model_out per plane         2.433e-04   2.22e-04        2   2.486e-04    1.67e-02  ✓  oracle at fp64 1.34e-02
gemm_int8                   ✓  224 products on the published integers
model_out per plane (widened)  2.661e-04   2.50e-04       15   2.637e-04    1.67e-02  ✓  oracle at fp64 1.34e-02
✓ krea2 dit compact 512 · 14 measures · nearest its threshold: img_in_out 2.32e-04 of 3.19e-03
$ siliconed check krea2 dit --size 512 --variant light
map ~/Library/Application Support/Siliconed/store/krea2-turbo-dit.light.v1.silicon · 7.6 GB · sq4 ×196, float32 ×169, sq5 ×52, int8 ×13 (packed weights)
model_out per plane         6.222e-05   2.41e-05       13   8.839e-05    1.22e-04  ✓  clouds of summation orders: engine median of 9 (nominal + 8 orders), oracle worst of 12 (nominal + 3 ulp + 8 orders)
gemm_int8                   ✓  2016 products on the published integers
✓ krea2 dit light 512 · 13 measures · nearest its threshold: block27_out 4.53e-04 of 5.00e-04
$ siliconed check krea2 dit --size 512 --variant light --neural-engine
map ~/Library/Application Support/Siliconed/store/krea2-turbo-dit.light.v1.silicon · 7.6 GB · sq4 ×196, float32 ×169, sq5 ×52, int8 ×13 (packed weights)
model_out per plane         2.198e-04   2.09e-04        0   2.246e-04    2.05e-02  ✓  oracle at fp64 1.64e-02
gemm_int8                   ✓  224 products on the published integers
model_out per plane (widened)  2.735e-04   2.60e-04       11   2.785e-04    2.05e-02  ✓  oracle at fp64 1.64e-02
✓ krea2 dit light 512 · 14 measures · nearest its threshold: img_in_out 2.32e-04 of 4.54e-03
$ siliconed check moss-image dit --size 512 --variant compact
map ~/Library/Application Support/Siliconed/store/moss-image-sft-dit.compact.v0.silicon · 5.4 GB · int8 ×252, bfloat16 ×210 (8-bit weights)
model_out per plane         1.871e-06   1.29e-06       23   2.142e-06    1.91e-06  ✗  oracle at fp64 1.53e-06
gemm_int8                   ✓  252 products on the published integers
model_out per plane (widened)  1.240e-06   9.84e-07       82   1.638e-06    1.91e-06  ✓  oracle at fp64 1.53e-06
✗ moss-image dit compact 512 · 1 of 24 failing: model_out per plane
$ siliconed check moss-image dit --size 512 --variant compact --neural-engine
map ~/Library/Application Support/Siliconed/store/moss-image-sft-dit.compact.v0.silicon · 5.4 GB · int8 ×252, bfloat16 ×210 (8-bit weights)
model_out per plane         3.214e-04   2.86e-04      116   3.214e-04    2.89e-02  ✓  oracle at fp64 2.31e-02
gemm_int8                   ✓  252 products on the published integers
model_out per plane (widened)  3.314e-04   2.97e-04      120   3.314e-04    2.89e-02  ✓  oracle at fp64 2.31e-02
✓ moss-image dit compact 512 · 25 measures · nearest its threshold: video_embed 2.36e-04 of 3.20e-03
```

</details>

Les variantes de Krea 2 se jugent par nuages comme sa Standard, et le `model_out per plane` de la Légère se juge par eux : la médiane du moteur sur neuf ordres de sommation (le nominal et huit tirés des mêmes graines que ceux de l'oracle, sur une carte dont le flux résiduel est permuté, `Krea2PermutedMap` : un poids SQ se déplace par blocs entiers de 256 entrées, ses entiers copiés tels que stockés) vaut 6,222e-05 pour une tolérance de 1,22e-04, le pire des douze tirages de l'oracle (le nominal, trois déplacements d'un ulp, huit ordres) plus 25 %. Les tirages d'une variante sont `krea2.py draw --size 512 --step 0 --jitter 1 2 3 --variant compact|light` et, pour les ordres de sommation, `krea2.py draw --size 512 --step 0 --permute 1 2 3 4 5 6 7 8 --permute-kind group --variant light` (le pas 0 seul : les latents suivants appartiennent à la trajectoire de la Standard). `temb` et `block0_attn_out` de la Légère sont deux frontières où la référence en fp32 est elle-même loin de la valeur exacte (1,92e-03 et 2,56e-03 au témoin fp64) ; le moteur est à 1,16e-05 et 2,17e-03 du témoin, dans la tolérance que la règle leur donne (2,41e-03 et 3,20e-03) : en fp32 contre le seul oracle, elles vaudraient 1,93e-03 et 2,81e-03 pour 5,00e-4.

**Sous `--neural-engine`, la route int8 lit les entiers publiés sur le Neural Engine aussi.** Là où le Neural Engine partage un produit avec le GPU, ses rangées lisent les entiers à côté du noyau int8 du GPU, comme la carte d'un rendu peut les lui donner ; sans cela, tout produit partagé d'un grand modèle restait sur le poids élargi, et `gemm_int8`, qui exige que les produits sur les entiers publiés soient pris au moins une fois, comptait 0 pour Krea 2 (sa Légère et sa Compact) et MOSS-Image sur notre Mac mini M5 Pro, toutes les lignes de précision passant. La Légère de Krea 2 lit maintenant 224 produits sur les entiers (`model_out per plane` 2,198e-04 pour 2,05e-02), et la route élargie est jugée aussi (2,735e-04). Les exécutions des Compact ci-dessus les lisent aussi : 891 produits pour Qwen-Image-2.1, 224 pour Krea 2, 252 pour MOSS-Image.

Les LoRA ne sont pas couvertes ici : ce sont des fichiers que vous choisissez, pas une part de ce que Siliconed livre.

## Durées et pic de mémoire

`siliconed check` juge la précision et rien d'autre. Les durées et le pic de mémoire se mesurent avec l'app seule : `silicontrol measure` ([REMOTE-CONTROL](REMOTE-CONTROL.fr.md#la-prise-et-le-protocole) ; `silicontrol help measure` en liste les options) rend une série dans l'app, chronomètre chaque rendu, échantillonne la mémoire de l'app et écrit un TSV et son résumé. Les durées et les pics de [ENGINE](ENGINE.fr.md) se prennent de la même façon (des rendus mis en file dans l'app avec `silicontrol`) : vous pouvez les refaire sur votre Mac avec l'app seule, sans Python ni oracle.

La même graine donne les mêmes pixels (le même md5) d'un rendu à l'autre sur un même Mac, avec une même version de l'app et une même mesure de la machine. Elle ne les donne pas d'un Mac à l'autre, ni d'une mesure à l'autre : le plan tire d'autres lignes d'autres nombres, et d'autres lignes somment dans un autre ordre. Ce qui passe d'un Mac à l'autre, c'est la règle ci-dessus, pas les bits.

**Une graine est la graine de diffusers, sur les cinq modèles.** Le bruit de départ est celui de PyTorch : `torch.randn` sur le CPU en float32 (`torch.Generator("cpu").manual_seed(graine)`, Mersenne Twister puis Box-Muller par blocs de 16), reproduit au bit près, tiré dans la forme que tire le `prepare_latents` de chaque pipeline — le latent plan par plan, comme le moteur le range. `swift test` garde les premières valeurs, celles du milieu et les dernières de ce bruit pour chaque modèle, à 512² et sur un rectangle, relevées une fois avec diffusers ; l'étage de trajectoire de chaque modèle le compare aussi au `latent_init` de l'oracle, à la graine que nomme son manifeste. La même graine et le même prompt partent donc du bruit dont part diffusers ; l'image qui suit se juge par les règles ci-dessus, étage par étage. Une variation tourne ce bruit vers celui d'une seconde graine, tiré de la même façon. Le rendu plus rapide et le bruit frais des solveurs (Euler a) ne sont pas la recette de diffusers et ne le prétendent pas.

## Toutes les étapes

La sortie de `siliconed check list --markdown`, telle que la commande l'écrit, en anglais. `{size}` vaut 512, 1024 ou un rectangle comme 768x512 ; les étapes sans taille se contrôlent à celle de la référence.

| Stage | Proves | Rule | Oracle | Weights |
|---|---|---|---|---|
| `kernels widen` | bf16 → fp32 widening and its inverse are exact, on the values a naive shift gets wrong | bit for bit | — | — |
| `kernels arena` | the arena hands out distinct page-aligned slices, refuses duplicates and growth | each property holds | — | — |
| `kernels elementwise` | the GPU's SwiGLU, gate and gated residual are as close to double precision as the CPU's; the residual rounds twice, as torch | GPU worst ulp ≤ max(CPU's, 1), mean ≤ CPU's + 5 %; residual bit for bit | — | — |
| `kernels gelu` | the GPU's GELU (erf) is as close to double precision as the CPU's, over one bit pattern in 64 of [−12, 12] | GPU worst deviation ≤ the CPU's, in ulp of the operand | — | — |
| `kernels sdpa` | the attention cut by heads (GPU + CPU), by query blocks, past maxMatrix and the matrix kernel return the whole graph's result; a head width padded to 128 (the vision tower's 72) returns a float64 witness's | against the whole graph: query blocks ≤ 1e-6 per channel; head cut ≤ 1e-5 with the GPU heads bit for bit; matrix kernel ≤ 1e-5; padded width ≤ 1e-5 against float64 | — | — |
| `kernels sdpa-tiles` | the attention by tiles (Metal 4 matmul2d) returns the float64 attention — in three stages in fp32, and with fp16 or bf16 operands and fp32 sums in three stages and fused at every candidate tile — on a self-attention, a cross-attention over few keys, grouped k/v heads and a length that tiles nothing | against float64, per channel: fp32 ≤ 1e-5 and ≤ the graph's own error + 25 % on the golden tensors' amplitudes (saturated softmax); fp16 operands ≤ 1.25 × the error of rounding the same exact output to bf16 (invariant 3, the ANE's bar); bf16 operands ≤ 1.25 × bf16's own error (the float64 attention of the operands rounded to bf16, its output rounded too); both on the map's amplitudes | — | — |
| `kernels gemm-split` | the token-wise split of a GEMM between the GPU and the AMX stitches back cblas_sgemm's product | against cblas_sgemm, per row: the GPU's rows bit for bit, the AMX's ≤ 1e-5 (another summation order over k = 3 840) | — | — |
| `kernels rope` | every RoPE table of the engine is the reference's, value by value (13 tables, 5 bit for bit) | max \|Δ\| per table within its bound, in units of 2⁻²⁴: 0 where the reference is bit-reproducible, 1 to 64 where torch uses SLEEF's cos, sin, pow | `tools/oracle/kernels.py rope` | — |
| `kernels resample` | the spectral schedule's nearest-exact decimation and bilinear upsampling are torch's, on a square and a rectangle | decimation bit for bit (and the even decimation, its counter-test, must differ); bilinear within one ulp of the input's scale | `tools/oracle/kernels.py resample` | — |
| `z-image tokenizer` | the chat template and the tokenizer give transformers' 512 identifiers and mask, padding included, and the app's admission the same real tokens (with --long, past 512: 1024) | bit for bit | `tools/oracle/z-image.py text [--long]` | tokenizer |
| `z-image text` | the Qwen3-4B encoder gives the reference's hidden states, layer by layer, and its cap_feats (with --long, on a prompt past 512 tokens) | the rule, floor 5e-4 | `tools/oracle/z-image.py text [--long]` | text encoder |
| `z-image block` | block 0 matches the reference at its 19 internal boundaries: norms, projections, attention, SwiGLU | the rule, floor 5e-4 | `tools/oracle/z-image.py dit --size {size}` | DiT |
| `z-image dit` | one evaluation of the DiT, latent to velocity: embedders, refiners, 30 blocks, head, unpatchify | the rule, floor 5e-4; on a rectangle, the transposed golden must deviate; with --neural-engine, --bf16-activations, --bf16-attention or --fp16-attention, against bf16's own error + 25 %; with --int8-activations, against ComfyUI's W8A8 error + 25 % | `tools/oracle/z-image.py dit --size {size}; witness --size {size}; with --neural-engine, --bf16-activations, --bf16-attention or --fp16-attention, bf16 --size {size}; with --variant compact or light, the same three with --variant, at 512 (the Compact's published weights, dequantized by their publisher's library; the Light's own SQ file, decoded by the oracle's reference decoder of the published format); with --int8-activations, w8a8 --size {size} --variant compact (ComfyUI's own int8 products)` | DiT |
| `z-image trajectory` | the seed's noise bit for bit, then the 8-step sampler: each step's velocity and latent, from the reference's own latent at that step | the rule on the guided pass, floor 5e-4; the free pass is shown with its gain, not judged | `tools/oracle/z-image.py trajectory --size {size}` | DiT |
| `z-image vae` | Flux's decoder on a rectangular latent (the centered crop of the final 1024² latent), plane by plane: what a square cannot see | the rule per plane, floor 5e-4, at the fp64 witness (no floor there); a decoder in bf16 no further from it than the reference in bf16, + 25 %; on a rectangle, the transposed image must deviate | `tools/oracle/z-image.py vae --size {size}; vae-witness --size {size}; vae-bf16 --size {size}` | VAE |
| `z-image img2img` | the start of an img2img at the oracle's strength (0.6): the seed's noise and the schedule, t_start and σ_s through the product's admission, the start's mix bit for bit, the encoder's z₀ and x_s, each step's velocity from the oracle's latent, then the product's denoiser from its own start over exactly the steps diffusers runs | noise and the mix from the oracle's z₀ bit for bit; t_start equal, σ_s within 1e-6; z₀ and x_s per latent plane, floor 5e-4; each guided velocity by the family's trajectory rule (floor 5e-4, at the step's fp64 witness where it exists; Krea 2: clouds of summation orders when the oracle drew the failing step's); the product's steps exactly σ[t_start] … σ[N − 1]; its free final latent shown, not judged | `tools/oracle/z-image.py img2img --size {size}` | VAE, DiT |
| `qwen-image-2.1 text` | Qwen3-VL-8B reads the prompt (and with --edit N, the N pictures; with --long, a prompt past 512 tokens): template, identifiers, positions, 36 layers, prompt_embeds | identifiers and masks bit for bit; the rule, floor 5e-4 (with images, the deep layers judged on their text rows) | `tools/oracle/qwen-image-2.1.py text [--edit N]; text --long` | text encoder |
| `qwen-image-2.1 vision` | the processor (Lanczos, patches, grids), the 3D positions of both edit templates, and the vision tower with its deepstack, image by image | pixels, grids, identifiers and positions bit for bit; the rule, floor 5e-4 | `tools/oracle/qwen-image-2.1.py vision` | text encoder |
| `qwen-image-2.1 references` | a reference with transparency, prepared as the pipeline does: the VAE's RGBA input and the vision copy over white (at 512, Pillow premultiplies around its Lanczos; at 443 it copies) | bit for bit; the vision copy on its bytes (the Pil and Fast processors differ by 6e-8 in float) | `tools/oracle/qwen-image-2.1.py references [--side 443]` | — |
| `qwen-image-2.1 dit` | one evaluation three ways — prefill of the conditions, the step from their K/V cache, the full pass under the block-causal mask — plus the layout and the modulation; on published int8 weights, along the int8 kernel a render's map may pick (gemm_int8), and the step once on the widened weights | layout bit for bit; the rule, floor 5e-4, at the fp64 witness where one exists; cache and full pass ≤ 1e-5 apart; a second step the same bits; with --neural-engine, --bf16-activations, --bf16-attention or --fp16-attention, against bf16's own error + 25 %, cache and full pass too; with --int8-activations, against ComfyUI's W8A8 error + 25 %; on int8 weights, every route by the same rule | `tools/oracle/qwen-image-2.1.py dit [--edit N]; witness; with --neural-engine, --bf16-activations, --bf16-attention or --fp16-attention, bf16 [--edit N]; with --variant compact or light (no --edit), dit and bf16 with --variant (the Compact's published weights, dequantized by their publisher's library; the Light's own SQ file, decoded by the oracle's reference decoder of the published format); with --int8-activations, w8a8 --variant compact (ComfyUI's own int8 products)` | DiT |
| `qwen-image-2.1 trajectory` | the 8 steps: σ, timesteps and noise to the bit, then each step's velocity from the oracle's latent and encoder states; the product's free run from the real inputs is shown | schedule and noise bit for bit; the rule on each guided velocity, floor 5e-4; the free run's final latent and image are shown, not judged | `tools/oracle/qwen-image-2.1.py trajectory [--edit N]` | DiT, text encoder, VAE |
| `qwen-image-2.1 vae` | the decoder on the trajectory's final latent (with --edit N, the edit's): unpacking and de-normalization to the bit, then the RGBA image plane by plane | vae_in bit for bit; the rule per plane, floor 5e-4, at the fp64 witness where one exists; a decoder in bf16 no further from it than the reference in bf16, + 25 %; on a rectangle, the transposed image must deviate | `tools/oracle/qwen-image-2.1.py vae [--edit N]; vae-witness; vae-bf16` | VAE |
| `qwen-image-2.1 vae-encode` | the encoder on both edit pictures (RGBA): the raw mean and the normalized latent, plane by plane; the product's module (RGB in, alpha added) gives the same bits | the rule per plane, floor 5e-4, at the fp64 witness where one exists; the module bit for bit | `tools/oracle/qwen-image-2.1.py vae-encode; vae-witness` | VAE |
| `qwen-image-2.1 writer` | Qwen3-VL-8B writes: the head map is the published lm_head and final norm; the chat prompt (with or without a system turn, with a picture); greedy decoding with a K/V cache, teacher-forced on the reference's tokens — the tokens and the logits at every step | head, identifiers and tokens bit for bit (a step whose gap is within the rule is a tie); the logits by the rule, floor 5e-4, a vocabulary entry a channel | `tools/oracle/qwen-image-2.1-writer.py [CASE]` | text encoder, writer head |
| `anima tokenizer` | the T5 tokenizer against `tokenizers` on a corpus of edge cases: normalization, whitespace, special tokens, truncation | every case bit for bit | `tools/oracle/anima.py tokenizer` | tokenizer |
| `anima text` | the prompt through both tokenizers, Qwen3-0.6B (final norm included) and the conditioner, to the DiT's [512, 1024] context (with --long, a prompt past 512 tokens: more rows) | identifiers bit for bit; the rule, floor 5e-4 | `tools/oracle/anima.py text [--long]` | text encoder, DiT (the conditioner) |
| `anima dit` | one evaluation of the DiT at σ = 1, boundary by boundary, and its output per latent plane against the fp64 witness | the rule, floor 5e-4; model_out per latent plane at the fp64 witness; on a rectangle, the transposed golden must deviate; with --neural-engine, --bf16-activations, --bf16-attention or --fp16-attention, against bf16's own error + 25 %; with --int8-activations, against ComfyUI's W8A8 error + 25 % | `tools/oracle/anima.py dit --size {size}; witness --size {size} --step 0; with --neural-engine, --bf16-activations, --bf16-attention or --fp16-attention, bf16 --size {size}; with --variant compact or light, dit and bf16 with --variant, at 512 (the Compact's published weights, dequantized by their publisher's library; the Light's own SQ file, decoded by the oracle's reference decoder of the published format); with --int8-activations, w8a8 --size {size} --variant compact (ComfyUI's own int8 products)` | DiT |
| `anima trajectory` | the seed's noise bit for bit, the schedule, then the 8 Euler steps: each step's velocity from the oracle's latent at that step | σ within 1e-6; the rule on each guided velocity, floor 5e-4; the free pass is shown with its gain, not judged | `tools/oracle/anima.py trajectory --size {size}` | DiT |
| `anima vae` | the Qwen-Image decoder on the final latent of the oracle's trajectory, plane by plane | the rule per plane, floor 5e-4, at the fp64 witness (no floor there); a decoder in bf16 no further from it than the reference in bf16, + 25 %; on a rectangle, the transposed image must deviate | `tools/oracle/anima.py vae --size {size}; vae-witness --size {size}; vae-bf16 --size {size}` | VAE |
| `anima img2img` | the start of an img2img at the oracle's strength (0.6): the seed's noise and the schedule, t_start and σ_s through the product's admission, the start's mix bit for bit, the encoder's z₀ and x_s, each step's velocity from the oracle's latent, then the product's denoiser from its own start over exactly the steps diffusers runs | noise and the mix from the oracle's z₀ bit for bit; t_start equal, σ_s within 1e-6; z₀ and x_s per latent plane, floor 5e-4; each guided velocity by the family's trajectory rule (floor 5e-4, at the step's fp64 witness where it exists; Krea 2: clouds of summation orders when the oracle drew the failing step's); the product's steps exactly σ[t_start] … σ[N − 1]; its free final latent shown, not judged | `tools/oracle/anima.py img2img --size {size}` | VAE, DiT |
| `krea2 text` | the prompt's identifiers, the 12 Qwen3-VL taps (padding removed, judged against states computed with it), then the fusion and txt_in on the oracle's taps (with --long, a prompt past 512 tokens read whole, as the app admits it: no fusion golden for it) | identifiers bit for bit; the rule, floor 5e-4 | `tools/oracle/krea2.py text [--long]` | text encoder, DiT (the fusion) |
| `krea2 dit` | one evaluation of the DiT at σ = 1, boundary by boundary, judged at the fp64 witness where it exists | the rule, floor 5e-4, at the fp64 witness where it exists, clouds of summation orders when the oracle has order draws (the same seeds drawn by the engine), of one-ulp moves when it has only those; model_out per latent plane at the witness; on a rectangle, the transposed golden must deviate; with --neural-engine, --bf16-activations, --bf16-attention or --fp16-attention, against bf16's own error + 25 %; with --int8-activations, against ComfyUI's W8A8 error + 25 % | `tools/oracle/krea2.py dit --size {size}; witness --size {size}; witness-step --size {size} --step 0; draw --size {size} --step 0 --jitter 1 2 3; with --neural-engine, --bf16-activations, --bf16-attention or --fp16-attention, bf16 --size {size}; with --variant compact or light, dit, bf16, witness, witness-step --step 0, draw --step 0 --jitter 1 2 3 and draw --size {size} --step 0 --permute 1 2 3 4 5 6 7 8 --permute-kind group with --variant, at 512 (the Compact's published weights, dequantized by their publisher's library; the Light's own SQ file, decoded by the oracle's reference decoder of the published format); with --int8-activations, w8a8 --size {size} --variant compact (ComfyUI's own int8 products)` | DiT |
| `krea2 trajectory` | the seed's noise bit for bit, the schedule, then the 8 Euler steps: each step's velocity from the oracle's latent at that step, at the step's fp64 witness where it exists | σ within 1e-6; the rule on each guided velocity, floor 5e-4, at the step's fp64 witness where it exists, clouds of summation orders when the oracle has order draws of the step (the same seeds drawn by the engine), of one-ulp moves when it has only those; the free pass is shown, not judged; --free-orders 1,2,3: the free pass again under each seed's summation order (residual stream permuted by groups of 256), shown step by step against the nominal pass and the oracle, beside the oracle's own passes under the same orders where it drew them, not judged | `tools/oracle/krea2.py trajectory --size {size}; witness-step --size {size} --step 0 1 2 3 4 5 6 7; draw --size {size} --step i --permute 1 2 3 4 5 6 7 8 --permute-kind group, one run per step i from 0 to 7; with --free-orders, trajectory --size {size} --permute 0 and the same seeds` | DiT |
| `krea2 vae` | the Qwen-Image decoder with Krea's weights on the final latent of the oracle's trajectory, plane by plane | the rule per plane, floor 5e-4, at the fp64 witness (no floor there); a decoder in bf16 no further from it than the reference in bf16, + 25 %; on a rectangle, the transposed image must deviate | `tools/oracle/krea2.py vae --size {size}; vae-witness --size {size}; vae-bf16 --size {size}` | VAE |
| `krea2 img2img` | the start of an img2img at the oracle's strength (0.6): the seed's noise and the schedule, t_start and σ_s through the product's admission, the start's mix bit for bit, the encoder's z₀ and x_s, each step's velocity from the oracle's latent, then the product's denoiser from its own start over exactly the steps diffusers runs | noise and the mix from the oracle's z₀ bit for bit; t_start equal, σ_s within 1e-6; z₀ and x_s per latent plane, floor 5e-4; each guided velocity by the family's trajectory rule (floor 5e-4, at the step's fp64 witness where it exists; Krea 2: clouds of summation orders when the oracle drew the failing step's); the product's steps exactly σ[t_start] … σ[N − 1]; its free final latent shown, not judged; with --variant, the Standard's start through the version's DiT, along the int8 route when it publishes int8 and once widened | `tools/oracle/krea2.py img2img --size {size}; img2img-witness --size {size} --step 3 4 5 6 7; img2img-draw --size {size} --step i --permute 1 2 3 4 5 6 7 8 for a step i failing at its witness; with --variant compact, the same with --variant (the Compact's published weights, dequantized by their publisher's library; the Light's own SQ file, decoded by the oracle's reference decoder of the published format)` | VAE, DiT |
| `moss-image tokenizer` | the chat template (no system turn, thinking block open) and the Qwen2 tokenizer against `tokenizers` on six prompts: accents, a newline and tab, JSON, digits and punctuation, a long text; cut at 4096 | the template string and every identifier bit for bit | `tools/oracle/moss-image.py tokenizer` | tokenizer |
| `moss-image text-delta` | the embedding rows and layer 0, a Gated DeltaNet layer, part by part: input norm, fused projection, delta rule, gated norm, mixer, MLP, output | the rule, floor 5e-4 | `tools/oracle/moss-image.py text` | text encoder (layer 0) |
| `moss-image text-full` | layers 0 to 3: three Gated DeltaNet layers and the first full-attention layer (gated queries, partial RoPE, output gate), part by part | the rule, floor 5e-4 | `tools/oracle/moss-image.py text` | text encoder (layers 0–3) |
| `moss-image text` | the prompt through Qwen3.5-9B: all 32 layers' outputs and the final state, the DiT's `prompt_embeds` | the rule, floor 5e-4; the layers 0, 3, 15, 31 and the final state at the fp64 witness | `tools/oracle/moss-image.py text; text-witness` | text encoder |
| `moss-image schedule` | the sigmas of MOSSImageScheduler (shift 5 on the fp32 grid, snapped to the 1000-step training grid) and the timesteps the DiT receives, for 1 to 10, 20 and 50 steps | bit for bit | `tools/oracle/moss-image.py schedule` | — |
| `moss-image dit` | one evaluation of the DiT at σ = 1, boundary by boundary — time, modulations, RoPE, text path, block 0 inside out, blocks 1, 13, 27, 41, the head —, judged at the fp64 witness | the rule, floor 5e-4, at the fp64 witness where it exists; model_out per latent plane at the witness; with --neural-engine, --bf16-activations, --bf16-attention or --fp16-attention, against bf16's own error + 25 %; with --int8-activations, against ComfyUI's W8A8 error + 25 % | `tools/oracle/moss-image.py dit --size {size}; witness --size {size}; with --neural-engine, --bf16-activations, --bf16-attention or --fp16-attention, dit-bf16 --size {size}; with --variant compact, dit, witness and dit-bf16 with --variant at 512 (the Compact's published weights, dequantized by their publisher's library; the Light's own SQ file, decoded by the oracle's reference decoder of the published format); with --int8-activations, w8a8 --variant compact (ComfyUI's own int8 products)` | DiT |
| `moss-image dit-lora` | the same evaluation with the DMD 8-step LoRA at strength 1 (kohya `lora.safetensors`, rank 128, fp32), the golden having it merged in fp32 | the rule, floor 5e-4, at the fp64 witness where it exists; model_out per latent plane at the witness | `tools/oracle/moss-image.py dit --lora; witness --lora` | DiT, LoRA |
| `moss-image trajectory` | the 8 steps of the recipe: the seed's noise bit for bit, the σ, each step's velocity from the oracle's latent at that step (the DMD LoRA at strength 1), then the product's denoiser run free from the noise on the oracle's text — its latents shown against the oracle's, and against the free loop the stage runs itself | noise and σ bit for bit; each guided velocity per latent plane, floor 5e-4; the product's free run against the stage's own loop, 1e-5 (the same arithmetic); the free latents against the oracle's are shown, not judged (eight steps amplify the evaluation's rounding) | `tools/oracle/moss-image.py trajectory` | DiT, LoRA |
| `moss-image vae` | FLUX.2's decoder: the pipeline's de-normalization and 2×2 de-patchification, then the image, plane by plane — on an encoded test image's latent and on the final latent of the reference pipeline | the de-patchified latent per channel, floor 5e-4; the image per plane, floor 5e-4, at the fp64 witness (no floor there) | `tools/oracle/moss-image.py vae --size {size}; vae-witness --size {size}; vae-final --size {size}` | VAE |
| `moss-image img2img` | the start of an img2img at the oracle's strength (0.6): the seed's noise and the schedule, t_start and σ_s through the product's admission, the start's mix bit for bit, the encoder's z₀ and x_s, each step's velocity from the oracle's latent, then the product's denoiser from its own start over exactly the steps diffusers runs | noise and the mix from the oracle's z₀ bit for bit; t_start equal, σ_s within 1e-6; z₀ and x_s per latent plane, floor 5e-4; each guided velocity by the family's trajectory rule (floor 5e-4, at the step's fp64 witness where it exists; Krea 2: clouds of summation orders when the oracle drew the failing step's); the product's steps exactly σ[t_start] … σ[N − 1]; its free final latent shown, not judged; with --variant, the Standard's start through the version's DiT, along the int8 route when it publishes int8 and once widened | `tools/oracle/moss-image.py img2img --size {size}; with --variant compact, the same with --variant (the Compact's published weights, dequantized by their publisher's library; the Light's own SQ file, decoded by the oracle's reference decoder of the published format)` | VAE, DiT |
