[English](API.md) · Français

# L'API de Siliconed

Pour qui écrit une app macOS (SwiftUI, « à la Draw Things ») par-dessus la bibliothèque
`Siliconed`. État au 09/10/2026. Une app SwiftUI complète, **Siliconed.app**, est dans [`Sources/SiliconedApp/`](../Sources/SiliconedApp/) (`App.swift` : l'app,
ses menus et sa fenêtre ; `AppState.swift` : l'état et tous les appels à la bibliothèque ;
`MainView.swift` : la vue ; `Statistics.swift` : la palette des statistiques ;
`ModelsSheet.swift` : la feuille « Modèles » ; `RemoteControl.swift` : la socket que
pilote `silicontrol …`, voir [`REMOTE-CONTROL.fr.md`](REMOTE-CONTROL.fr.md) ;
`Localizable.xcstrings` : ses phrases en anglais, français, allemand, espagnol et italien). Elle est **compilée par le
paquet** (cible exécutable `SiliconedApp`, en client pur, sans `@testable`), donc elle ne peut pas
dériver de l'API : en cas de doute, c'est l'exemple de référence. `tools/app.sh` la construit en `Siliconed.app` (traductions et icône
comprises) et l'ouvre ; `SILICONED_ROOT=<folder> swift run SiliconedApp` aussi, dans la langue du
système.

Paquet : `Package.swift`, produit `.library(name: "Siliconed")`, **macOS 15+**, Swift 6.

---

## 1. En 30 secondes

```swift
import Foundation
import Siliconed

let library = Library(root: URL(fileURLWithPath: "/Users/me/Siliconed"))
try EngineSettings.load(from: library)   // at launch, before the first render (see §2.1)

let model = try Model.zImage(in: library)     // checks every file, computes nothing
let request = Request("a 30 year old woman posing in a library", resolution: 1024, seed: 42)

var progress = Engine.Progress()
for try await event in Engine().events(request, model: model) {
    progress.receive(event)
    switch event {
    case let .step(_, index, total, _, seconds, _, _):
        print("step \(index)/\(total) in \(seconds) s — \(Int(progress.fraction * 100)) %"
              + (progress.estimatedRemaining.map { ", ~\(Int($0)) s left" } ?? ""))
    case .warning(let message):
        print("⚠ \(message)")
    case .image(_, let renderResult):
        try PNG.write(try renderResult.png(), to: "/Users/me/library.png")   // sRGB PNG + metadata
    default:
        break
    }
}
```

Le rendu tourne sur la file du moteur, pas sur le pool coopératif. Si la `Task` qui itère est
annulée, ou si la boucle s'arrête, le rendu est annulé et sa mémoire rendue.

---

## 2. Concepts

**Conventions**, une seule façon de faire chaque chose :

- **Noms** : ce que le moteur *émet* est imbriqué dans `Engine` (`Render`, `Event`, `Plan`,
  `Stage`, `Preview`, `Progress`, `Timings`, `Footprints`) ; ce que l'app *construit ou branche*
  est au premier niveau (`Request`, `Model`, `Chain`, les modules, `ImageRGB`, `Library`,
  `Cancellation`). **Une seule erreur pour tout : `EngineError`**, une énumération fermée (§5) ;
  toute fonction publique qui lève est `throws(EngineError)`.
- **Étiquettes** : la bibliothèque sous `in:` ; `render`, `renderBatch`, `events` prennent
  `(request, [seeds:], model:, …)` dans cet ordre ; une force est un `Double`.
- **Surface** : ce que seuls les outils et les tests du paquet emploient est `package` —
  invisible d'une app.

### 2.1 `Library` : le dossier des modèles

```swift
public struct Library: Sendable {
    public let root: URL
    public init(root: URL)
}
```

Tous les chemins se déduisent de `root`. (Les noms sur disque `composants`, `importes`,
`telechargements` et `profil.json` sont ceux de la disposition d'origine, conservés : les
bibliothèques existantes restent valides.)

| dossier | contenu | taille (installation Standard, `Family.installedSize`) |
|---|---|---|
| `<root>/store/` | cartes forgées (`*.silicon` : DiT, encodeurs de texte), LoRA forgées (`*.lora.silicon`), `profil.json` | Z-Image : 12,3 Go de DiT + 7,8 Go d'encodeur ; Qwen-Image-2.1 : 14,2 Go de DiT + 16,3 Go d'encodeur (Compact : 11,7 et 19,4 Go en tout ; Légère : 9,5 et 16,2 Go, §3.11) |
| `<root>/store/composants/<famille>/` | les petits fichiers publiés d'une famille, lus tels quels : tokenizers, VAE, `transformer.json`, `dit.json` (noms et formes du DiT de l'éditeur, contre lesquels un import se vérifie) | 0,18 Go pour Z-Image, 2,7 Go pour Qwen-Image-2.1 (sa LoRA turbo, 1,36 Go, comprise) |
| `<root>/store/importes/` | les DiT importés (`<nom>.silicon`) | la taille d'un DiT de la famille |
| `<root>/telechargements/` | les sources d'une installation, le temps de la forger | vide au repos |
| `<root>/cache/` | **le cache de rendu** : ce que les rendus gardent d'un rendu à l'autre pour ne pas le refaire (ci-dessous) | jusqu'à ~0,5 Go, plus les K/V d'une édition Qwen-Image-2.1 (2 à 6,4 Go) |

**Rien d'autre n'est lu au rendu, et tout se refait** : `store/` se reconstruit depuis une racine
vide (`library.install(_:)`, §3.11), qui télécharge chez chaque éditeur, à une révision
figée et vérifiée par sha256, puis forge les mêmes cartes au bit près. Un fichier importé peut
être jeté une fois importé. `cache/` ne fait gagner que du temps : l'effacer ne change aucune image.

**Le cache de rendu** (`library.cacheFolder`). Refaire le même prompt sous une autre graine, ou la
même édition (même instruction, mêmes images), ne refait pas ce qui ne dépend pas de la graine :

- `cache/conditioning/` : la sortie de l'encodeur de texte, pour tous les modèles, et les latents
  que le VAE encode (l'image de départ d'un img2img, les références d'une édition). Borné à 512 Mo ;
  les entrées les moins récemment servies partent d'abord. L'étage texte de Z-Image passe de 2,8 s
  à 0,01 s ;
- `cache/kv/` : les conditions de Qwen-Image-2.1 (instruction et références) telles que son DiT les
  relit à chaque pas — les clés et valeurs que calcule son premier pas. **Une seule édition** (la
  dernière, ses deux phases en mode 9 pas), 2 à 6,4 Go sur disque (6,4 Go à 1024×1536 avec trois
  références) ; elle n'est pas gardée si elle laissait moins de 8 Go libres sur le volume. Une
  génération (sans référence) ne garde rien ici : ses conditions ne coûtent presque rien.

Ce qui revient, ce sont **les flottants mêmes** qui ont été calculés : l'image est la même, au bit
près, que sans le cache. La clé de chaque entrée nomme tout ce qui la détermine : les fichiers du
modèle (chemin, taille, date : une réinstallation ou un nouvel import font une autre clé), le
prompt, chaque octet de chaque image, la pile de LoRA pour les K/V, les réglages qui changent des
bits, et le binaire du moteur lui-même (une nouvelle version de l'app part d'un cache vide). Le
cache sert aux modèles construits depuis une bibliothèque (`Model.named(_:in:)` et les autres) ; un
`Model(card:chain:)` composé à la main n'en a pas.

- `library.occupancy().cache` le compte (et `total` l'inclut) ; `try library.emptyCache()` le
  rend — un rendu en cours garde le fichier qu'il a ouvert.
- `SILICONED_CACHE=0` dans l'environnement, ou `"cache": false` dans les réglages du profil,
  l'éteint (`EngineSettings.effective.cache`).

- **Où la mettre dans une app** : la bibliothèque ne cherche rien toute seule. L'app construit
  `Library(root:)`, soit sur `~/Library/Application Support/Siliconed/` (`Library.standard`), soit
  sur un dossier choisi par l'utilisateur (dans une app sandboxée : signet à portée de sécurité).
  Les cartes sont projetées en mémoire (mmap) et relues à chaque évaluation : il faut les mettre
  sur le SSD interne. Aucun disque externe n'a été mesuré.
- **Ce qu'on en montre** : `library.displayPath` (`~/Library/Application Support/Siliconed`), jamais
  `root.path` — une capture porterait le nom du compte. `Library.withoutHome(_:)` écrit le dossier
  personnel `~` partout dans un texte (un message d'erreur qui cite un chemin) ; l'`error` du
  diagnostic y passe avant de pouvoir être publiée.
- `Library.processWide` est le repli du paquet quand aucune bibliothèque n'est désignée
  (`SILICONED_ROOT`, puis le répertoire courant s'il a un `store/`, puis le dépôt déduit du binaire,
  puis `Library.standard`). Il est `package` : une app n'a pas à l'utiliser.
- `try EngineSettings.load(from: library)` désigne le `store/profil.json` de la bibliothèque,
  qui contient les réglages de la machine (co-exécution AMX, tranches du VAE). Les réglages se
  lisent **une seule fois par processus**, et **rien ne les lit avant un rendu** : ni construire
  une `Request` (son `reproducible` vaut `nil`, « le réglage de la machine », résolu au rendu),
  ni un `Model`, ni `library.models()`. Seuls un rendu, `EngineSettings.effective` et `load`
  lui-même les lisent. On appelle donc `load` au lancement (au plus tard avant le premier
  rendu) ; **trop tard, il lève `EngineError.settingsAlreadyLoaded`** au lieu de ne rien faire
  en silence (sauf si le profil déjà lu est le même). Un profil mesuré sur une autre machine est
  ignoré, avec un avertissement dans `EngineSettings.effective.warnings`, et les défauts
  s'appliquent.
- Accès bas niveau : `map(_ name: String) -> String` (`<root>/store/<nom>`), `profile`. Le
  reste (composants, référence du DiT, téléchargements) est interne au paquet.

### 2.2 `Model` et le catalogue

Un **modèle** regroupe une chaîne préréglée, une fiche et une licence. Le construire vérifie que
chacun de ses fichiers existe. S'il en manque un, l'erreur donne son chemin complet. Aucun calcul
n'est lancé.

```swift
public static func zImage(in b: Library, denoising: ZImageDenoisingModule? = nil) throws(EngineError) -> Model
public static func qwenImage21(in b: Library) throws(EngineError) -> Model
public static func named(_ identifier: String, in b: Library) throws(EngineError) -> Model
public init(card: ModelCard, chain: Chain)        // a hand-composed chain (§2.4)
// model.identifier ("z-image" | "qwen-image-2.1" | "<family>/<name>" imported), .name, .family, .license, .card, .chain
```

La bibliothèque se passe toujours sous l'étiquette `in:`, comme pour `card.missing(in:)`.

Le **catalogue** décrit les modèles sans rien ouvrir de lourd, et il ne lève jamais :

```swift
let readySet: [ModelCard] = library.models()    // the ready models (all files present)
let allCards: [ModelCard] = library.cards()   // the repository's models, then the imported ones, ready or not
let missingList: String? = allCards[1].missing(in: library)   // what is missing, or nil
let model: Model = try Model.named(readySet[0].id, in: library)   // a card's model
let loras: [LoRACard] = library.loras(for: "z-image")  // store/*.lora.silicon, read from their header, sorted by name
```

`ModelCard` (`Identifiable`, `Hashable` : utilisable telle quelle en `tag` d'un `Picker`) :
`id`, `name`, `family: Family`, `isImported`, `license: License` (`text`, `commercial`,
`requiredFilter`, `urls`), `defaultSteps` (8 pour Z-Image, 6 pour Qwen-Image-2.1), `space`,
`formats: [RecommendedFormat]`. Un modèle importé hérite de sa famille l'architecture, les
formats et le texte de licence de l'éditeur (à vérifier sur la page du modèle).
`LoRACard` : `path`, `name` (lisible), `rawName`, `target` (la famille visée), `rank`,
`resolution?`, `trainedOn?`, `compatible(with:)`, `entry(strength:) -> LoRAEntry`.

| identifiant | nom | licence | débruiteur | VAE |
|---|---|---|---|---|
| `z-image` | Z-Image Turbo | Apache 2.0 | DiT S3, 8 pas, schedule spectral à 1024² | Flux |
| `qwen-image-2.1` | Qwen-Image-2.1 Turbo (§3.12) | **non commerciale** (Qwen Research) | DiT Qwen-Image-2.1 (32 couches à un flux, attention bloc-causale) + LoRA turbo de Viggle, 6 pas ; édition avec 1 à 3 images (§3.12) | Qwen-Image-2.1 (16×, 64 canaux) |

### 2.3 `Request` : tout ce qui distingue un rendu

```swift
public init(_ prompt: String, width: Int, height: Int, seed: UInt64 = 42, steps: Int? = nil,
            loras: [LoRAEntry] = [], image: ImageRGB? = nil, strength: Double = Strength.defaultValue,
            previews: Bool = false, reproducible: Bool? = nil)
public init(_ prompt: String, resolution: Int = 1024, seed: UInt64 = 42, steps: Int? = nil, …)  // the square
```

Construire une requête **ne lit rien** (ni fichier, ni profil) : elle peut vivre dans l'état
d'une app avant tout le reste.

- **Prompt** : un prompt vide ou blanc est refusé par tous les modèles, avant tout calcul
  (`EngineError.emptyPrompt`).

- **Format** : `width` × `height` en pixels (dans cet ordre, comme Draw Things :
  `832x1216` est un portrait). Les règles sont au §4.
- **Graine** : `UInt64`. La valeur par défaut est 42 ; pour une graine au hasard, l'app tire
  `UInt64.random(in:)` elle-même.
- **Pas** : `nil` donne le défaut du débruiteur (8 pour Z-Image Turbo, 6 pour Qwen-Image-2.1
  Turbo ; tous deux sont distillés pour ce nombre). Une autre valeur est acceptée (Qwen-Image-2.1 :
  seulement 5, 7 ou 9, §3.12), mais sa qualité n'a pas été jugée. Un nombre hors de
  `1...Request.maximumSteps` (50) est refusé avant tout calcul (`EngineError.invalidSteps`,
  `allowed` vide : « N steps: a render takes between 1 and 50. »).
- **LoRA** : `[LoRAEntry]`, une pile. `LoRAEntry(_ path: String, strength: Double = 1)`, ou
  `loraCard.entry(strength: 0.8)`. Toute la pile doit viser le modèle, sinon le rendu est refusé.
  Une force est un `Double` partout dans l'API (LoRA comme img2img).
- **Image et force** (img2img) : `image: ImageRGB?` de taille quelconque. **La requête la
  remplit et la recadre au centre à son format dès qu'elle la reçoit** (à l'init, à
  l'affectation, et quand `width`/`height` changent) : elle ne garde jamais une photo pleine
  taille. Changer le format ensuite réajuste l'image *déjà ajustée* ; pour ne pas perdre de
  bords, reposer l'image source après le format. Le moteur ne l'étire pas. `strength` a
  exactement le sens du `strength` de diffusers, dans `]0 ; 1]`, 0,6 par défaut
  (`Strength.defaultValue`). Elle est **quantifiée par pas de 1/N** et vaut du txt2img au-delà
  de `1 − 1/N`. Sans `image`, `strength` est ignorée : le txt2img et l'img2img passent par la
  même API.
- **Références** (l'édition) : `var references: [ImageRGB]`, vide par défaut. Contrairement à
  `image`, une référence **n'est pas rebruitée** : le débruiteur la lit à chaque couche et l'image
  générée part du bruit ; le prompt dit ce qui change (« change her jacket to bright red »). Chaque
  référence garde **ses** proportions, à la taille que lui donne le débruiteur
  (`DenoisingModule.preparedReferences` : Qwen-Image-2.1 à ~1024² aux multiples de 32), quel que
  soit le format de la requête — donner à la requête le format de l'image 1 (`editFormat`, §3.12)
  garde le cadre. Seuls les débruiteurs qui l'annoncent les lisent
  (`DenoisingModule.maxReferences` : 3 pour Qwen-Image-2.1, 0 pour Z-Image) ; au-delà, ou sur un
  autre modèle, le rendu est refusé avant tout calcul (`EngineError.tooManyReferences`). Coût : la
  séquence s'allonge d'autant de jetons (Qwen-Image-2.1 : une génération 1024² 132–136 s, une
  édition en 1248×832 avec une référence ~190 s). **L'édition par instruction** — l'ordre
  des images, comment le prompt les nomme, la taille de sortie : §3.12.

```swift
let qwen = try Model.qwenImage21(in: library)
let photo = try ImageRGB(contentsOf: chosenURL)
let (w, h) = qwen.chain.denoising.editFormat(referenceWidth: photo.width, referenceHeight: photo.height)
var request = Request("change her jacket to bright red", width: w, height: h)
request.references = [photo]                          // image 1, the one edited
let edited = try await Engine().render(request, model: qwen)
```
- **`reproducible`** : coupe GPU/AMX figée, c'est-à-dire mêmes bits pour une même graine (§4).
  `nil` par défaut : le réglage de la machine (`frozenCut`, vrai par défaut), lu au rendu.
- **`previews`** : une petite image par pas (§3.6). Faux par défaut ; quand il est faux, rien
  n'est calculé.
- **`sketch`** : `Int?`, `nil` par défaut — s'arrêter après ce nombre d'évaluations et décoder
  l'estimation de l'image finale par le modèle au lieu de l'image (§3.16). Se règle en propriété,
  pas dans `init`.
- **`detail`** : `Detail` (`.normal`, `.more`, `.most`), `.normal` par défaut — une texture plus
  fine au même coût ; `.normal` est le rendu sans elle, au bit près (§3.17). Se règle en propriété.
- **`variations`** : `[Variation]`, vide par défaut — le bruit de départ de la graine tourné vers
  celui d'autres graines, dans l'ordre : des cousines de l'image que donne la graine (§3.18). Vide :
  le bruit de la graine, au bit près. Dans un lot, chaque graine est tournée par les mêmes
  variations. Se règle en propriété.

### 2.4 `Chain` et ses câbles

Un rendu enchaîne toujours les mêmes étages :

```
TextModule ─ Conditioning ─┐
                           ├─ DenoisingModule ─ Latent ─ DecodingModule ─ ImageRGB
[ImageEncodingModule: ImageRGB → Latent] ─┘   (img2img only)
```

Les étages se relient par des câbles typés : `TextFormat` (`.zImage`, `.qwenImage21`) et
`LatentSpace` (`.flux` — 16 canaux, une case pour 8 pixels —, `.qwenImage21` — 64 canaux, une
case pour 16 pixels).
`Chain` vérifie les branchements **à la construction**, et un mauvais branchement lève
`EngineError.incompatibleChain`. Un format qui **lit les images** (`TextFormat.readsImages` : celui de
Qwen-Image-2.1) doit venir d'un `ImageTextModule`, qui reçoit les références avec le prompt —
sinon `EngineError.incompatibleChain`.

```swift
public init(text: any TextModule, denoising: any DenoisingModule,
            decoding: any DecodingModule, encoding: (any ImageEncodingModule)? = nil) throws(EngineError)
```

| modèle | texte | image (img2img) | débruitage | décodage |
|---|---|---|---|---|
| Z-Image | `ZImageTextModule` | `FluxEncodingModule` | `ZImageDenoisingModule` | `FluxDecodingModule` |
| Qwen-Image-2.1 Turbo | `QwenImage21TextModule` (un `ImageTextModule`) | `QwenImage21EncodingModule` (les références seulement : l'img2img est refusé) | `QwenImage21DenoisingModule` | `QwenImage21DecodingModule` |

Le système est **fermé** : les protocoles sont publics pour qu'on puisse composer une chaîne
avec les modules du dépôt, pas pour en écrire de nouveaux (leur `Context` n'a rien d'utilisable
hors du paquet). Une app n'a presque jamais besoin d'une `Chain` explicite : `model.chain`
suffit. Le seul réglage utile est celui du débruiteur de Z-Image :

```swift
let withoutSpectral = try Model.zImage(in: library, denoising: ZImageDenoisingModule(
    map: library.map("z-image-turbo-dit.v1.silicon"), spectral: 0))
```

Une chaîne composée à la main se rend comme les autres, sous une fiche :
`Model(card: .zImage, chain: myChain)` — `render`, `renderBatch` et `events` ne prennent
qu'un `Model`.

Un mauvais branchement est refusé avant le moindre calcul :

```swift
let z = try Model.zImage(in: b), q = try Model.qwenImage21(in: b)
_ = try Chain(text: z.chain.text, denoising: q.chain.denoising, decoding: q.chain.decoding)
// ✗ EngineError.incompatibleChain(output: "qwen3-4b · …", input: "qwen3-vl-8b · prompt + images · …")
```

Chaque étage construit son modèle, calcule, puis **rend sa mémoire avant l'étage suivant** :
c'est ce qui permet de tenir sur une machine de 16 Go. La contrepartie : les poids sont relus à
chaque rendu, et aucun DiT ne reste chargé entre deux rendus.

### 2.5 `Engine` : un rendu à la fois

```swift
public final class Engine: Sendable { public init() }
```

Un `Engine` ne contient rien. Tous les rendus du processus passent par **une file statique
unique** (`siliconed.engine`, QoS `.userInitiated`), et cela quel que soit le nombre de `Engine()`
créés : deux vues qui lancent chacune un rendu les verront s'exécuter l'un après l'autre. La
règle a deux raisons :

1. le moteur n'est pas réentrant (mémoïsations de `GEMM`, état du `Conductor`, arènes). Deux
   rendus simultanés ne planteraient pas, ils corrompraient leurs données sans rien signaler ;
2. il n'y a qu'un GPU, deux blocs AMX et 16 Go. Deux rendus en parallèle doubleraient le pic
   mémoire et ne gagneraient rien.

Il y a deux façons d'appeler le moteur, avec **les mêmes paramètres dans le même ordre**
(`request`, [`seeds:`], `model:`, [`onProgress:`]). Elles exécutent le même code (`execute`).

```swift
// async: what an app calls
public func render(_ request: Request, model: Model,
                   onProgress: (@Sendable (Event) -> Void)? = nil) async throws(EngineError) -> Render
public func renderBatch(_ request: Request, seeds: [UInt64], model: Model,
                      onProgress: (@Sendable (Event) -> Void)? = nil) async throws(EngineError) -> [Render]
public func events(_ request: Request, seeds: [UInt64]? = nil,
                       model: Model) -> AsyncThrowingStream<Event, Error>

// blocking: a command-line tool (cancellation goes through an explicit token)
public func render(_ request: Request, model: Model, cancellation: Cancellation? = nil,
                   onProgress: (@Sendable (Event) -> Void)? = nil) throws(EngineError) -> Render
public func renderBatch(_ request: Request, seeds: [UInt64], model: Model, cancellation: Cancellation? = nil,
                      onProgress: (@Sendable (Event) -> Void)? = nil) throws(EngineError) -> [Render]
```

**Lequel préférer** : une UI prend le flux `events`, qui livre les évènements dans l'ordre,
là où l'on itère. La fermeture `onProgress` est appelée **sur le fil du rendu**, de façon
synchrone ; elle doit rendre la main vite. Sans `onProgress`, les avertissements d'un rendu vont
à `Warnings.outsideRender` (ils ne sont plus perdus).

**Ordre des évènements** d'un rendu réussi : `start(Plan)`, `stage(.text)`, `text`,
[`stage(.image)`, `encoding`], puis pour chaque image : `stage(.denoising)`, `step` × N (chacun
suivi de son `preview` s'il est demandé), `stage(.decoding)`, `decoding`, `image(index:, Render)`.
Un `warning` peut arriver n'importe quand.

| évènement | usage dans l'UI |
|---|---|
| `start(Plan)` | `plan.denoising.evaluations`, `plan.seeds`, `plan.stages`, `plan.denoising.startStep` / `startSigma` (img2img), `plan.denoising.reduced` (spectral) |
| `stage(Stage, image:)` | libellé « texte… / image… / débruitage… / décodage… » ; émis *avant* l'étage |
| `text(tokens:tokenizer:encoder:)`, `encoding(seconds:)`, `decoding(seconds:)` | chronos |
| `step(image:index:total:sigma:seconds:latentGridHeight:latentGridWidth:)` | `index` va de 1 à `total`, numéroté de la même façon pour tous les modèles |
| `preview(Preview)` | vignette du pas (si `request.previews`) |
| `image(index:, Render)` | le résultat (le dernier évènement de chaque image) |
| `warning(String)` | à journaliser ou afficher discrètement |

`Engine.Progress` tire de ces évènements une `fraction` (dans `[0, 1]`, qui ne recule jamais
pendant un rendu) et un `estimatedRemaining` en secondes (`nil` avant le premier pas), calculé
sur la durée des pas déjà faits. L'unité est l'évaluation du DiT. Chaque `start` le remet à
zéro : un seul `Progress` peut servir toute la vie de l'app.

### 2.6 `Engine.Render` : ce qui sort

Un `Render` est une image avec sa trace. **Le moteur n'écrit aucun fichier.**

- ce qui l'a produite : `model`, `prompt`, `seed`, `steps`, `loras`, `loraSummary` (la pile en
  une ligne), `strength?` (`nil` en txt2img), `startStep`, `startSigma`, `reproducible` (résolu),
  `detail`, `variations` (sans celles de force nulle) ;
- l'image : `image: ImageRGB` — `pixels` (`[3, height, width]`, planaire, fp32 autour de
  `[-1, 1]`, non écrêtés : le décodeur sort jusqu'à ±1,14 ; `rgba8()` écrête), `height`, `width` ;
- le coût : `evaluations` (pour une esquisse, celles après lesquelles elle s'est arrêtée), `sketch`
  (`nil` sauf si le rendu s'est arrêté en esquisse, §3.16), `spectral` (le `k` appliqué),
  `timings` (`text`, `encoding`, `denoising`, `decoding` qui partitionnent `total`), `footprints`
  (mémoire aux points de passage), `tokens` ;
- les sorties : `metadata: [String: String]` et `png() throws(EngineError) -> Data`.

À 1024², `renderResult.image.pixels` pèse 12 Mo. Pour un historique, il vaut mieux garder
`renderResult.image.cgImage()` ou `rgba8()`.

---

## 3. Recettes

### 3.1 txt2img

```swift
let renderResult = try await Engine().render(Request("woman posing in a library", width: 832, height: 1216,
                                              seed: 7), model: try Model.qwenImage21(in: library))
```

### 3.2 img2img depuis un fichier

```swift
let entry = try ImageRGB(contentsOf: chosenURL)   // PNG, JPEG, HEIC, TIFF… ; EXIF applied ; sRGB ; alpha over white
let request = Request("a watercolor painting of a woman reading in a library", resolution: 1024, seed: 42,
                      image: entry, strength: 0.6)       // the image is fitted to 1024² right here
let renderResult = try await Engine().render(request, model: try Model.zImage(in: library))
// renderResult.startStep, renderResult.startSigma: what the strength gave FOR THIS MODEL
```

- Le recadrage que le moteur va appliquer peut être montré à l'avance :
  `ImageRGB.crop(source:target:)` calcule la géométrie sans toucher aux pixels.
- Le décodage d'un gros fichier est **sous-échantillonné par ImageIO** :
  `ImageRGB(contentsOf:maxSide:)`, 3072 px au plus sur le grand côté par défaut
  (`ImageRGB.maximumDecodedSide`, le plus long côté qu'un format accepté puisse avoir). Une photo
  de 48 Mpx ne vit jamais pleine taille.
- Une image **en mémoire** (glisser-déposer, presse-papiers) : `try ImageRGB(cgImage: cg)` (sRGB,
  alpha sur blanc ; pas d'orientation EXIF, un `CGImage` n'en a pas), bornée de même : un côté plus
  long est dessiné directement à `maxSide`, jamais gardé pleine taille.
- Bornes du curseur de force : entre 0,2 et 0,8 en pratique. Pour Z-Image, la force doit dépasser
  1/N, soit 1/8 à 8 pas (`EngineError.strengthTooLow` donne le `floor`).
  `Strength.startStep(steps:strength:)` donne le pas de départ, pour l'afficher avant le rendu.
- Z-Image ne change pas de *style* en img2img, même à 0,85.
- Qwen-Image-2.1 ne fait pas d'img2img (`EngineError.imageToImageUnsupported`) : passer l'image
  en image 1 d'une édition (§3.12).

### 3.3 LoRA, et le refus d'une LoRA d'un autre modèle

```swift
let model = try Model.zImage(in: library)
let compatible = library.loras(for: model.identifier)     // show only those
let request = Request("portrait, studio light", resolution: 1024,
                      loras: compatible.prefix(1).map { $0.entry(strength: 0.8) })
let renderResult = try await Engine().render(request, model: model)
```

Chaque LoRA forgée déclare sa famille (`cible.modele` dans son en-tête — clé sur disque conservée
de la disposition d'origine — écrite par la forge d'après le DiT de la famille contre lequel elle
a vérifié chaque module et chaque forme). Si une LoRA de la pile vise un autre modèle, le rendu
lève `EngineError.loraForOtherModel` **avant tout calcul** (~0,3 s). Sans ce refus, la LoRA ne
toucherait aucun module et l'image sortirait sans elle, sans aucun message. Une carte forgée sans
cible lève `.unsupportedLoRA(file:reason: .noTarget)` : il faut la reforger.

Une LoRA s'**importe** telle qu'elle est publiée (§3.11) : ai-toolkit, PEFT, diffusers, kohya
(`lora_unet_…` et `alpha`), ComfyUI. Elle vise une famille : elle s'applique aussi aux modèles
importés de cette famille.

### 3.4 Lot de graines

```swift
let renderResults = try await Engine().renderBatch(request, seeds: [42, 43, 44, 45], model: model)
// or: Engine().events(request, seeds: [42, 43, 44, 45], model: model)
```

`request.seed` est alors ignorée. Le texte (et l'image d'entrée) sont encodés **une seule
fois**, et l'image n du lot est **identique au bit** au rendu isolé de sa graine (vérifié).
Le gain se limite au texte, ~3 s par image, parce que le DiT est reconstruit pour chaque image.
Dans les évènements, `image` / `index` donnent la position dans le lot.

### 3.5 Annulation (`Task.cancel`)

```swift
let task = Task { try await Engine().render(request, model: model) }
// … Stop button:
task.cancel()
// → `try await task.value` throws EngineError.cancelled, 0.03 to 0.41 s later, memory returned
```

- Avec `render` / `renderBatch` async : `EngineError.cancelled` est levée.
- Avec `events` : annuler la `Task` qui itère, ou sortir de la boucle, annule le rendu.
  **La boucle se termine alors sans lever d'erreur** (c'est le comportement
  d'`AsyncThrowingStream`) : il faut tester `Task.isCancelled` après la boucle.
- Avec l'API bloquante : `let a = Cancellation()`, passer `cancellation: a`, puis `a.cancel()`
  depuis n'importe quel fil — depuis un gestionnaire de `SIGINT`, pour le Ctrl-C d'un outil en
  ligne de commande.
- Le jeton est vérifié entre les étages, entre les pas et entre les couches des DiT et des
  encodeurs, jamais au milieu d'un GEMM. Un rendu lancé juste après une annulation attend donc
  sur la file que le précédent ait atteint la couche suivante.

### 3.6 Aperçu par pas

```swift
let request = Request(prompt, resolution: 1024, previews: true)
// in the stream: case .preview(let p): thumbnail = p.cgImage()   // p.width × p.height = the latent grid
```

L'aperçu est x̂₀ = x − σ·v, la prédiction de l'image finale (pas l'état bruité), projetée de 16
canaux vers RGB. Il fait un pixel par case de latent : 64×64 à 512², 128×128 à 1024². Il faut
l'agrandir à l'affichage (`.interpolation(.none)` ou lissé). Qualité : 13 à 19 dB au premier pas,
25 à 30 dB au dernier. Coût : négligeable devant un pas.

### 3.7 Barre de progression SwiftUI : `AsyncThrowingStream`

```swift
@MainActor @Observable final class ProgressIndicator {
    // One property per line: `@Observable` refuses `var a = 0, b = 1`.
    var fraction = 0.0
    var remaining: Double?
    var stage: Engine.Stage?
    var preview: CGImage?
    var image: CGImage?
    var errorMessage: String?
    private var task: Task<Void, Never>?

    func launch(_ request: Request, _ model: Model) {
        task?.cancel()
        task = Task {
            var progress = Engine.Progress()
            do {
                for try await item in Engine().events(request, model: model) {
                    progress.receive(item)
                    fraction = progress.fraction; remaining = progress.estimatedRemaining
                    switch item {
                    case .stage(let e, _): stage = e
                    case .preview(let a): preview = a.cgImage()
                    case .image(_, let renderResult): image = renderResult.image.cgImage()
                    default: break
                    }
                }
                if Task.isCancelled { return }            // cancelled: the loop ended without throwing
            } catch EngineError.cancelled {
            } catch {
                errorMessage = error.localizedDescription        // an EngineError; its `code` keys the translation
            }
        }
    }
    func cancel() { task?.cancel() }
}
// View: ProgressView(value: p.fraction) ; Image(decorative: cg, scale: 1)
```

La version complète est l'app ([`Sources/SiliconedApp/`](../Sources/SiliconedApp/)).

**Le rack, à gauche** : la chaîne de haut en bas, dans le sens du signal, **chaque module avec ses
réglages dans sa boîte** — il n'y a pas d'autre formulaire. D'abord les entrées, telles que la
chaîne les déclare (`chain.entries` : le texte, obligatoire, tapé dans la boîte du prompt ; « Éditer », les références, si le
débruiteur en lit — des vignettes numérotées jusqu'à `maxReferences`, choisies, glissées sur la
boîte ou la toile, ou « Éditer cette image » depuis l'historique, qui en fait l'image 1 ; la
sortie prend le format de l'image 1, §3.12). **L'image de départ d'un img2img (`.image`) n'y est
pas** : rebruitée, elle ne suit pas un prompt-instruction ; `Request.image` reste
dans l'API. Puis le DiT (modèle, licence et précision en étiquettes,
orientation et format conseillé, graine fixe ou tirée, nombre d'images, pas, pile de trois LoRA
compatibles au plus, import ; en marche, ses pas), puis le VAE — le latent
n'est pas une boîte, l'image non plus : c'est le résultat, sur la toile. La colonne suit la fenêtre : plus étroite, ses boîtes rétrécissent et
leurs lignes de réglages passent sur deux lignes. **Entre deux modules, le câble** qui les
relie, de son vrai type : `chain.text.output` (le `TextFormat`) vers le DiT, un latent de plus
quand des images sont posées pour l'édition, `chain.denoising.space` (le `LatentSpace`, ses canaux)
vers le VAE ; il est en surbrillance pendant que le module qu'il nourrit travaille. Le nom de chaque module
vient de la chaîne ; le voyant de la boîte qui travaille s'allume (évènements `.stage`), rouge si
l'erreur y est tombée. Sous le rack : « Générer » avec **son coût** dès que ce modèle a
rendu à ce format dans la session (la vitesse apprise × `plan(…).evaluations`, jamais une
constante), et **la file** : « Générer » pendant un rendu devient « Ajouter à la file » — le
travail prend les réglages du moment ; chaque attente se retire, « Tout arrêter » vide la file.

**La toile, à droite** : pendant le rendu, le cadre a déjà la forme de l'image à venir et
l'aperçu de chaque pas s'y dessine, avec l'étage, un segment par évaluation, le temps restant et
« Arrêter » ; puis l'image (pincer pour zoomer, deux doigts pour la déplacer dans sa
zone), son prompt et sa légende, « Refaire (même graine) », et **ses détails** : réglages, temps et pic mémoire de chaque étage
(`Render.timings`, `Render.footprints`). L'historique en bande de vignettes (menu : reprendre les
réglages, éditer, retirer), **parcourable pendant un rendu** : le rendu en cours ouvre la bande, son
aperçu et sa progression sur une vignette en direct ; une image choisie entre-temps garde la toile,
la vignette en direct l'y ramène. **Un PNG écrit par Siliconed, déposé sur la toile, reprend ses
réglages** (modèle, prompt, graine fixée, pas, taille, LoRA retrouvées par leur nom de fichier) par
le même chemin que « Reprendre ces réglages » ; ce qui ne peut pas revenir (modèle non installé,
LoRA introuvable, image source d'un img2img) est dit, le reste est repris. Toute autre image déposée
là garde son sens (une référence d'édition). **Les menus** portent tous les raccourcis : Générer ⌘↩, Refaire ⌘R,
Arrêter ⌘., Tout arrêter ⌥⌘., Enregistrer ⌘S (PNG avec métadonnées ou JPEG), copier l'image
⇧⌘C, plein écran ⌘F, images voisines ⌥⌘← →, importer ⌘O, statistiques ⌥⌘I.

**La palette des statistiques** (⌥⌘I, ou son bouton de la barre d'outils) flotte au-dessus : le rendu en cours
(écoulé, reste, heure de fin), chaque étage (jetons, tokenizer et encodeur, la durée de chaque
pas du DiT en barres, le VAE), la file (fin estimée au rythme appris dans la session pour chaque
modèle et format — une attente jamais mesurée le dit au lieu d'inventer), la dernière image
(temps et pics mémoire par étage) et la session.

**Le fil principal ne calcule pas** : décoder l'image de référence, encoder le PNG et réduire la
vignette se font sur une tâche détachée ; l'historique garde le PNG et une vignette, et seule
l'image choisie est décodée en plein format ; le catalogue (LoRA, ce qui manque) se relit quand
le modèle change, pas à chaque dessin. **Cinq langues** : anglais, français, allemand, espagnol et
italien, selon le système ; `tools/translations.sh` extrait les phrases du code, les synchronise
dans `Localizable.xcstrings` et refuse une phrase qui manque dans l'une d'elles. **Incognito** : l'historique vit en mémoire, et rien n'est
écrit sans un geste explicite — ni fichier temporaire, ni position de fenêtre, ni état restauré.
**Pour un script** : aucun argument de lancement ne remplit le formulaire ni ne rend ; `silicontrol add` met un rendu en
file dans l'app ouverte (`docs/REMOTE-CONTROL.fr.md`, `silicontrol help add`).

### 3.8 Écrire un PNG avec métadonnées, obtenir un `CGImage`

```swift
let cg: CGImage = renderResult.image.cgImage()           // sRGB, 8 bits
let bytes: [UInt8] = renderResult.image.rgba8()         // RGBA row by row, opaque alpha
let png: Data = try renderResult.png()                   // sRGB + iTXt chunks, no date: same bytes for the same render
try PNG.write(png, to: url.path)               // atomic; throws EngineError.imageWriteFailed
let custom = try renderResult.image.png(metadata: renderResult.metadata.merging(["note": "favorite"]) { $1 })
let reread: [String: String] = PNG.text(try Data(contentsOf: url))   // read the parameters back
```

Les clés écrites sont `prompt`, `seed`, `model`, `steps`, `format` (`LxH`), `reproducible`,
`Software`, plus `strength` et `image` (`source not included`) en img2img, et `lora`
(`fichier.lora.silicon:0.80,…`, **nom de fichier seul**) quand il y en a, et `sketch` (les
évaluations après lesquelles elle s'est arrêtée) pour une esquisse (§3.16), `detail` (`more` ou
`most` ; **absente pour `normal`**, comme sur tout rendu d'avant, §3.17) et `variation`
(`graine:force,…`, dans l'ordre appliqué ; `seed` reste celle de l'origine, §3.18). Elles suffisent à
refaire un **txt2img** (mêmes bits si `reproducible`, même machine, même version) ; **pas un
img2img** : l'image source n'est pas dans le PNG, et la clé `image` le dit. Une clé doit être en
ASCII et faire au plus 79 octets, sinon elle est ignorée.

`Engine.Render.Recipe(metadata: PNG.text(data))` les relit en valeurs (`model`, `prompt`, `seed`,
`steps`, `width`/`height`, `loras`, `strength`, `sourceImageMissing`, `detail` (`normal` si la
clé est absente), `variations` (vide si absente) — pas `sketch` : la recette
d'une esquisse refait l'image finie), `nil` si `Software` n'est
pas `Siliconed` ; une clé présente mais illisible vaut `nil` et figure dans `unreadable`. Pure :
elle ne dit pas si le modèle ou une LoRA est installé, et `model` est la famille (`z-image`), pas la
carte d'un fine-tune importé.

### 3.9 Lister les modèles et les LoRA compatibles

```swift
let readySet = library.models()                                   // [ModelCard]
for card in ModelCard.allCards where !readySet.contains(where: { $0.id == card.id }) {
    print(card.name, "unavailable:", card.missing(in: library) ?? "")
}
let loras = library.loras(for: "qwen-image-2.1")                // [LoRACard], sorted by name
```

`library.models()` construit chaque modèle pour vérifier ses fichiers : c'est rapide (pas
de poids lus), et cela ne lit pas le profil de la machine (§2.1).

### 3.10 Formats conseillés

`ModelCard.recommendedFormats` (et `card.formats`) sont les formats de Draw Things qui tiennent
sous le plafond : **1024², 896×1152, 832×1216, 768×1344**, leurs transposées, et **512²** pour
itérer vite. `RecommendedFormat` a `width`, `height`, `orientation` (`.square`, `.portrait`,
`.landscape`) et un `description` affichable (`832×1216 (portrait)`). 1024×1536 est accepté mais
pas conseillé (sur Z-Image il prend ~153 s contre ~82 s pour un 1024²). Pour un champ libre,
`Format.parse("832x1216")` puis `Format.check(width:height:)`.
Quand la mémoire refuse un format (`insufficientMemory`), `RecommendedFormat.largestFitting(card.formats,
available:current:need:)` donne le plus grand qui tient dans un budget lu une fois — l'orientation
choisie d'abord, jamais sous 512 de côté — ou `nil` : ce que l'app propose au lieu d'une impasse.

---

### 3.11 Importer un `.safetensors` ou un `.gguf`, installer une famille

Un utilisateur apporte un `.safetensors` tel que Civitai ou Hugging Face le publie : une LoRA, ou
le DiT d'un modèle, **sans encodeur ni VAE** — ou un DiT en `.gguf`. L'import reconnaît ce que
c'est et pour quelle famille, installe chez l'éditeur ce qui manque à la famille (tokenizers,
encodeur de texte, VAE), et forge. Les noms d'origine (ComfyUI, `net.`, qkv fusionné) comme
diffusers sont lus ; bf16, fp16, fp32, les formats 8 bits et les types GGUF de 4 à 6 bits
ci-dessous aussi. Chaque nom et chaque
forme sont vérifiés contre le DiT de l'éditeur : un fichier d'une autre architecture lève, il n'est
pas deviné.

**Une carte n'est jamais plus large ni plus étroite que le fichier donné.** Les octets publiés et
leurs échelles sont recopiés tels quels, seulement transposés ; un fp16 reste fp16, et une valeur
fp32 que le bf16 ne tient pas exactement reste fp32 (un fine-tune publié en fp32 peut donc donner
une carte fp32 entière, ~24 Go pour Z-Image). Les formats 8 bits gardés tels que publiés :

| Format | Poids | Tel que publié par |
|---|---|---|
| fp8 E4M3, une échelle par tenseur | `w = f8 · s` | ComfyUI « scaled » (vérifié sur fichiers synthétiques seulement) |
| fp8 E4M3, une échelle par ligne | `w = f8 · s[n]` | torchao, SDNQ |
| int8, une échelle par ligne | `w = q · s[n]` | SDNQ, torchao, ComfyUI `int8_tensorwise` |
| int8 convrot | `w = (q · s[n]) · R`, R une rotation de Hadamard par groupes de 256 entrées | ComfyUI |
| GGUF Q8_0 (`.gguf`) | blocs de 32 entrées, `w = d · q`, d fp16 | unsloth |

**Sous 8 bits, en GGUF seulement, jusqu'à 4 bits.** Les `.gguf` qu'unsloth, jayn7, leejet
(stable-diffusion.cpp) et Abiray publient pour Z-Image Turbo et Qwen-Image-2.1 sont lus dans ces
types, mêlés tenseur par tenseur comme l'éditeur l'a choisi (un fichier `Q4_K_M` porte des tenseurs
Q4_K, Q5_K, Q6_K et parfois Q8_0) :

| Type | Bits par poids | Poids |
|---|---|---|
| Q6_K | 6,56 | super-blocs de 256 entrées, `w = (d · sc) · q`, d fp16, sc int8 par 16 |
| Q5_K, Q4_K | 5,5 ; 4,5 | super-blocs de 256 entrées, `w = (d · sc) · q − dmin · m`, sc et m sur 6 bits par 32 |
| Q5_0, Q4_0 | 5,5 ; 4,5 | blocs de 32 entrées, `w = d · (q − 16)` ou `d · (q − 8)`, d fp16 |
| Q5_1, Q4_1 | 6 ; 5 | blocs de 32 entrées, `w = d · q + m`, d et m fp16 |

**Gardés tels que publiés** veut dire que les blocs sont recopiés entiers dans la carte, octets
inchangés : jamais dépaquetés en 8 bits (la carte grossirait), jamais requantifiés par nous. **Le
plancher est 4 bits** : Q3_K, Q2_K, les I-quants (IQ…), TQ, MXFP4 et Q8_1 font refuser le fichier en
entier, même quand un seul tenseur en porte — de même un `Q4_K_S` qui mêle des tenseurs Q3_K (celui
d'unsloth pour Qwen-Image-2.1), avec un message qui nomme le type.

Le moteur déquantifie chaque poids en le chargeant et **calcule en fp32**, comme pour toute carte :
pas de produit matriciel int8, les activations ne descendent jamais à 8 bits (`input_scale` est
ignorée). Chaque poids fp8, int8 et GGUF, une fois déquantifié, est **identique au bit à la
déquantification fp32 de l'éditeur** (torchao, SDNQ, le code GGUF de diffusers, et gguf-py pour les
types sous 8 bits), vérifié sur des morceaux de fichiers Z-Image et Qwen-Image-2.1 publiés et sur
des cartes entières (Z-Image Q8_0, Q6_K, Q4_K_M ; Qwen-Image-2.1 Q4_K_M), sur CPU et GPU. L'int8 convrot, dont la référence arrondit deux fois, est exact à l'arrondi fp32 du
résultat fp64, dix fois plus près de lui que le fp32 de ComfyUI.

Mesuré sur Z-Image Turbo (M1 Pro, 16 Go) :

| | carte bf16 | SDNQ int8 | int8 convrot | GGUF Q8_0 |
|---|---|---|---|---|
| carte sur disque | 12,3 Go | 6,2 Go | 6,2 Go | 7,25 Go (raffineurs publiés en bf16) |
| import | — | 11 s | 31 s | 18 s |
| temps par pas, 512² | 3,9–4,3 s | +15 à +19 % | non chronométré | +8 à +17 % |
| temps par pas, 1024² | 15,5–16,1 s | +2 à +5 % | non chronométré | +3 à +6 % |
| 1024×1536 | 0 swap | 0 swap | 0 swap | 0 swap |

Un fichier 8 bits est un autre jeu de poids : même composition, une autre image que celle du bf16
(PSNR de 23 à 31 dB à 512²).

Les GGUF sous 8 bits achètent une carte plus petite et moins d'octets relus sur le disque à chaque
pas — ce qui compte sur un Mac de 8 Go, où la carte du DiT ne peut pas rester en mémoire d'un pas à
l'autre. Mesuré sur Z-Image Turbo (M1 Pro, 16 Go) :

| | carte bf16 | GGUF Q8_0 | GGUF Q6_K | GGUF Q4_K_M |
|---|---|---|---|---|
| carte sur disque | 12,3 Go | 7,25 Go | 5,93 Go | 5,04 Go |
| relu sur le disque par pas | 5,43 Go | 2,89 Go | 2,23 Go | 1,78 Go |
| 1024×1536 | 0 swap | 0 swap | 0 swap, pic 3,35 Go | 0 swap, pic 3,35 Go |

Sur cette machine, un rendu est aussi un peu plus rapide qu'en Q8_0. Un fichier sous 8 bits est
encore un autre jeu de poids : pas dégradé à l'œil, mais pas la même image non plus — à 1024², un
fichier Q4_K_M peut composer une autre image à la même graine.

**Refusés**, le fichier entier, avec un message qui dit pourquoi : le GGUF sous 4 bits (ci-dessus) ;
le 4 bits hors GGUF (nvfp4, SDNQ uint4, Nunchaku) — un fichier qui mêle une seule telle couche à des
couches 8 bits est refusé en entier ; le mxfp8 ; le fp8 E5M2 ; l'int8 asymétrique (un point zéro
non nul).

```swift
// In a detached task: an import reads and writes GBs, an installation downloads some.
let r = try ModelImport.importFile("/Users/me/Downloads/my-model.safetensors",
                                 name: "My model", in: library) { rowLine in print(rowLine) }
// r.kind (.lora | .model), r.family, r.name, r.path, r.journal
if let id = r.identifier {                           // "z-image/my-model"
    let model = try Model.named(id, in: library)
    _ = try await Engine().render(Request("a 30 year old woman posing in a library"), model: model)
}

// An absent family, or redoing everything from an empty root:
try library.install(.qwenImage21) { print($0) } // baseDiT: false → without the publisher's DiT
let ready: [Family] = library.readyFamilies() // enough to render an imported model

// Z-Image and Qwen-Image-2.1 also install in a Compact or a Light version (below):
try library.install(.zImage, variant: .compact) { print($0) }
library.variant(of: .zImage)                   // .compact — what the render reads
Family.zImage.preselectedVariant()             // .light on a Mac of 8 GB, .standard elsewhere
```

**Standard, Compact ou Légère.** `Family.variants` liste les versions dans lesquelles une famille
s'installe : chacune a sa **Standard** (`Variant.standard`, les poids de l'éditeur tels que publiés —
le défaut) ; Z-Image et Qwen-Image-2.1 ont aussi une **Compact** (`Variant.compact`) : des poids
8 bits publiés par des tiers, pour le DiT **et** l'encodeur de texte, gardés 8 bits dans les cartes —
environ 40 % de moins sur le disque, une image légèrement différente de celle de la Standard, à peu
près aussi rapide (les poids sont élargis à chaque évaluation, et moins lus sur le disque).
Tokenizers, VAE et la LoRA turbo de Qwen-Image-2.1 sont ceux de la Standard. Elles ont aussi une
**Légère** (`Variant.light`) : le DiT tel qu'unsloth le publie en GGUF Q4_K_M — des blocs de 4 à
6 bits gardés entiers, comme tout import GGUF (plus bas), et quatre tenseurs Q8_0 pour Qwen — et
**l'encodeur de texte de la Compact**, la même carte (aucun encodeur n'est publié sous 8 bits sous
une forme que la forge lit, et Siliconed n'en quantifie jamais un lui-même). La plus petite sur le
disque et la moins relue à chaque pas (Z-Image : 1,78 Go par pas au lieu de 2,89) ; un autre jeu de
poids, donc une image qui peut se composer autrement. Sur un M1 Pro de 16 Go, elle rend dans
le temps de la Compact (Z-Image 512² −5 à −7 %, 1024² et Qwen-Image-2.1 à ±1,5 %). Passer de la
Compact à la Légère ne remplace que le DiT. `family.preselectedVariant()` est la version qu'une app
propose d'abord : la Standard, la Légère sur un Mac de 8 Go de mémoire — présélectionnée, jamais
imposée.

| famille | DiT Compact | encodeur de texte Compact | disque, Standard → Compact |
|---|---|---|---|
| Z-Image | `unsloth/Z-Image-Turbo-GGUF` `z-image-turbo-Q8_0.gguf` (GGUF Q8_0) | `Disty0/Z-Image-Turbo-SDNQ-int8` `text_encoder/model.safetensors` (SDNQ int8), avec le `config.json` de l'éditeur | 20,4 → 11,7 Go |
| Qwen-Image-2.1 | `Comfy-Org/Qwen-Image-2.1` `diffusion_models/qwen_image_2.1_int8_convrot.safetensors` (int8 convrot) | `unsloth/Qwen-Image-2.1-FP8` `Qwen-Image-2.1-text_encoder-INT8-ConvRot.safetensors` (int8 convrot), avec le `config.json` de l'éditeur | 33,2 → 19,4 Go |

| famille | DiT Légère | encodeur de texte Légère | disque, Standard → Légère |
|---|---|---|---|
| Z-Image | `unsloth/Z-Image-Turbo-GGUF` `z-image-turbo-Q4_K_M.gguf` (Q4_K, Q5_K, Q6_K) | celui de la Compact | 20,4 → 9,5 Go |
| Qwen-Image-2.1 | `unsloth/Qwen-Image-2.1-GGUF` `qwen-image-2.1-Q4_K_M.gguf` (Q4_K, Q5_K, Q6_K, Q8_0) | celui de la Compact | 33,2 → 16,2 Go |

Chaque fichier est figé (révision et sha256). **Une seule version d'une famille est installée à la
fois, et jamais aucune** : `install(_:variant:)` avec une autre version forge la nouvelle à côté de
l'installée, qui continue de rendre — `library.variant(of:)` est la version dont les cartes sont
complètes — et retire le DiT et l'encodeur de l'ancienne **une fois la nouvelle complète**
(l'encodeur seulement si ni la nouvelle version ni une autre famille ne le lit). Un disque plein, une coupure, un arrêt ou un
fichier refusé laissent la version installée entière. Seulement quand le disque ne peut pas tenir
les deux, l'ancienne part avant le premier octet, après un contrôle qui compte la place qu'elle
libère : `library.installPlan(_:variant:baseDiT:)` le dit d'avance (`removesFirst`), avec ce que
l'installation ajoute (`added`), libère (`freed`) et demande à son pic (`peak`) — à montrer avant
l'acceptation. `variant: nil` (le défaut) garde la version déjà là, la Standard s'il n'y en a pas :
un appel qui ne nomme pas de version n'en change jamais. `baseDiT: false` (ce qui manque à un
modèle importé) n'en change jamais non plus : une autre version que l'installée est refusée.
`library.installedVariant(of:)` vaut `nil` tant que rien de la famille n'est là. Le rendu n'a
besoin de rien de plus : `Model.named` lit les cartes de `library.variant(of:)`.
`family.licenseURLs(variant)` liste les fichiers de licence qu'une installation de cette version
montre — `ModelCard.license.urls`, puis la fiche de chaque tiers.

```swift
let plan = try library.installPlan(.zImage, variant: .compact)
if plan.removesFirst { print("not enough room for both: the \(plan.replacing!) version goes first") }
```

Un dépôt à accès restreint demande d'accepter sa licence sur huggingface.co, puis de poser
`HF_TOKEN` (ou `hf auth login`, dont le jeton est relu) ; Z-Image et Qwen-Image-2.1 sont ouverts. Pour montrer ce
qu'une installation coûte et ce qui est installé : `library.occupancy()` (le `variant` de chaque
famille), `Family.installedSize(_:)`, `library.uninstall(_:)` (toutes les versions).

### 3.12 L'édition par instruction

Le prompt est une **instruction** (« replace the cloudy sky with a blue sky », « remove the second
person from image 1 and put the dog of image 2 in her place »), et les images dont il parle
voyagent dans `Request.references`. Un modèle qui édite le dit dans sa chaîne : `chain.entries`
porte une entrée `.reference`, et `chain.denoising.maxReferences` dit combien d'images il lit
(Qwen-Image-2.1 : 3 ; 0 pour un modèle sans édition, Z-Image). Rien n'est rebruité :
les images sont lues, la nouvelle image part du bruit. Ni masque, ni inpainting : le prompt seul
dit ce qui change.

**Qwen-Image-2.1**, chaque étage vérifié contre la référence, rend son pire cas (1024×1536 avec
trois références) sans un octet de swap (pic de 4,58 Go). Il prend 6 pas (les 5 et 7 de
Viggle, et un mode à 9 pas, sont acceptés ; tout autre nombre est refusé avant tout calcul), et ne
fait **pas d'img2img** : une `Request.image` est refusée — passer l'image en image 1. Demander
`chain.denoising.check(steps:startImage:)` pour le savoir d'avance.

**L'ordre est la numérotation.** `references[0]` est **l'image 1, celle qu'on édite** ;
`references[1]` est l'image 2, `references[2]` l'image 3 — ce où le prompt puise. L'encodeur de
Qwen-Image-2.1 (Qwen3-VL-8B) **voit** les images : son gabarit place `<image1>`, `<image2>`… avec
les jetons de vision de chaque image devant le prompt, si bien que le prompt peut les nommer
(« image 1 », « image 2 »). `chain.text.output.readsImages` (`TextFormat.readsImages`) dit que
l'encodeur voit les images : c'est ce qu'une app lit avant de suggérer « image 1 », « image 2 »
dans le prompt.

**Chaque référence est redimensionnée par le moteur** (`chain.denoising.preparedReferences`), quel
que soit le format de la requête, ses proportions gardées. Qwen-Image-2.1 : à environ
`output_resolution²` (1024², ~1 MP), chaque côté au multiple de 32 le plus proche (égalité vers le
pair), par le Lanczos de Pillow au bit ; la même image redimensionnée nourrit l'encodeur et le VAE.

**La taille de sortie suit l'image 1.** Le `width` × `height` de la requête est celui de la
sortie ; pour une édition, lui donner le format de l'image 1 selon cette règle :

- la surface de `output_resolution²` (1024²) aux proportions de l'image 1,
- chaque côté **multiple de 32**, et **d'au moins 512** (`Format.minimumSide`) — le petit côté relevé
  en gardant les proportions,
- la surface **sous `Format.maxSurface`** (1024×1536) : le plus grand côté cède.

Le pipeline de diffusers, sans `height`/`width`, prend la taille de la *dernière* image ;
Siliconed prend celle de l'image 1, celle qu'on édite, pour que le cadre soit gardé. La règle
appartient au débruiteur : `chain.denoising.editFormat(referenceWidth:referenceHeight:)`, que
l'app et `silicontrol add --ref` appellent tous deux sans format tapé — une
commande et un clic donnent la même image.

```swift
let photoURL = URL(fileURLWithPath: "photo.jpg"), dogURL = URL(fileURLWithPath: "dog.jpg")
let photo = try ImageRGB(contentsOf: photoURL), dog = try ImageRGB(contentsOf: dogURL)
let qwen = try Model.named("qwen-image-2.1", in: library)
let (w, h) = qwen.chain.denoising.editFormat(referenceWidth: photo.width, referenceHeight: photo.height)
var request = Request("remove the second person from image 1 and put the dog of image 2 in her place",
                      width: w, height: h)
request.references = [photo, dog]                      // image 1 (edited), image 2
let edited = try await Engine().render(request, model: qwen)
```

**Coût** : chaque référence allonge le rendu — ses jetons rejoignent la séquence. Les conditions
de Qwen-Image-2.1 (texte et références) ne voient pas l'image générée : leurs clés et valeurs se
calculent **une fois par rendu**, puis se relisent à chaque pas. Mesuré en série : une
génération 1024² prend 132–136 s, une édition en 1248×832 ~190 s avec une référence, 262–264 s
avec deux ; une évaluation du DiT à 1024² ~20 s sous le turbo. Une app apprend son propre coût par modèle, format **et
nombre de références** (l'app le fait), jamais d'une constante.

Au-delà de `maxReferences`, ou sur un modèle sans édition, le rendu est refusé avant tout calcul
(`EngineError.tooManyReferences`).

**Dans l'app**, la boîte « Éditer » montre les références en vignettes numérotées (1, 2, 3),
au plus `maxReferences` du modèle : ajoutées par son bouton, en déposant des images sur la boîte
ou la toile, ou par « Éditer cette image » sur une image de l'historique (qui devient l'image 1) ;
retirées et réordonnées en glissant ou depuis le menu d'une vignette. L'image 1 est marquée comme
celle qu'on édite, la ligne de taille montre le format de sortie (« suit l'image 1 »), et une ligne
discrète rappelle que le prompt peut dire « image 1 / image 2 ».

### 3.13 Le diagnostic : ce que fait cette machine, dans un JSON

`Diagnostic.run` mesure les modèles installés sur **ce** Mac et rend un rapport `Codable` — ce
qu'envoie le bouton « Signaler ma configuration » de l'app, et ce qui remplit le tableau par puce du
README. Pour chaque modèle passé qui est installé (`ModelCard.missing(in:) == nil` ; les autres sont
sautés), à 512², prompt « a 30 year old woman posing in a library », graine 42 : l'encodeur de texte
(à froid, sans le cache de rendu), **deux évaluations du DiT** (la première paie la construction et
l'échauffement, la seconde est le pas en régime), le décodeur VAE. Un rendu entier est **dérivé, pas
exécuté** : `estimatedRenderSeconds = encodeur + première + (E − 1) · régime + décodeur`, `E` les
évaluations des pas par défaut du modèle. À côté des temps : la machine (puce, cœurs P/E, cœurs GPU,
mémoire, macOS, version), le pic `phys_footprint` échantillonné toutes les 10 ms, les swapouts
`vm_stat` avant et après, et — pour un modèle dont une petite sortie de référence est embarquée dans la bibliothèque
(Z-Image et Qwen-Image-2.1) — l'écart du second `model_out` à la référence fp32, canal par canal, avec
son seuil. Un modèle qui n'en a pas rend `"golden": null, "deviation": null` ; un modèle qui échoue rend son
`error`, et les autres tournent quand même : `run` ne lève jamais.

Chaque modèle tourne sous le plan mémoire qu'un rendu prendrait (`memory` : le budget au lancement, le
plancher, `lean`, et les `decisions` du plan ; rempli même quand le contrôle préalable refuse), et dit
les octets qu'il a lus sur le disque (`diskReadBytes` : une carte lue depuis le cache de pages ne compte
pas). Le rapport ajoute l'état de la machine `before`/`after` (état thermique, mode économie
d'énergie, batterie, mémoire récupérable, marge du compresseur, swap occupé), un témoin GEMM fp32
`witnessBefore`/`witnessAfter` sur le GPU et par `cblas_sgemm` (TFLOP/s, et la pire ligne du GPU contre
`cblas`), les `settings` qui ne sont pas aux défauts (et `amx`, toujours), le `profile` lu, et le nom
Metal, la famille et l'ensemble de travail du GPU dans `machine`. `schema` vaut 2.

```swift
let cards = library.models().filter { ["z-image", "qwen-image-2.1"].contains($0.id) }   // les visibles
let report = await Task.detached {
    Diagnostic.run(cards, in: library) { card, stage in   // stage : .textEncoder, .denoiser, .decoder
        print("\(card.name) · \(stage)")                  // sur le fil du diagnostic : passer à celui de l'UI
    }
}.value                                                   // ~15 s par modèle
let json = try report.json()                  // stable : clés triées, null explicites
let issue = try report.issueURL()            // une issue GitHub préremplie : NSWorkspace.shared.open(issue)
let form = try report.issueURL(includingReport: false)   // le même formulaire sans le JSON, à coller
```

Un lien au-delà de ~8 000 caractères est coupé par les navigateurs et GitHub : l'app copie alors
`json()` dans le presse-papiers et ouvre `issueURL(includingReport: false)`. `Diagnostic.Stage` est
ce que `onProgress` rapporte, en anglais par `description`, pour qu'une app le dise dans sa langue.

Il est synchrone et lourd (les poids d'un modèle à la fois, 2,9 Go au pic pour Qwen-Image-2.1) : le
lancer hors du fil principal, et **jamais pendant un rendu** — les chiffres mesureraient les deux.
Sa `cancellation` est consultée entre les étages.

### 3.14 Les alternatives dans un prompt : `{a|b}`

`PromptAlternatives` fait d'un prompt à groupes d'alternatives les prompts qu'il représente —
**combinatoire, jamais aléatoire** : plusieurs groupes font leur produit cartésien, dans l'ordre de
lecture, le premier groupe variant le plus lentement. Le moteur, lui, ne voit jamais d'accolade : une
`Request` prend un prompt développé, et une app met en file un rendu par prompt, **à la même graine**
(une série compare des prompts, rien d'autre ne doit changer). C'est ce que font « Générer » dans
l'app et `silicontrol add`.

```swift
let series = try PromptAlternatives("woman posing in a {library|greenhouse} at {dawn|dusk}")
series.count                          // 4 — connu avant de rien construire
series.groups                         // [["library", "greenhouse"], ["dawn", "dusk"]]
for v in series.variants {            // library·dawn, library·dusk, greenhouse·dawn, greenhouse·dusk
    let r = try await Engine().render(Request(v.prompt, resolution: 512, seed: 42), model: model)
    print(v.choices, r.metadata["prompt"] ?? "")   // ["library", "dawn"] woman posing in a library at dawn
}
```

- **La syntaxe.** `{` ouvre un groupe, `|` sépare ses alternatives, `}` le ferme. Une alternative
  vide est permise (`{|red }dress` → `dress`, `red dress`) et rien n'est rogné : les espaces
  appartiennent à l'alternative qui les porte. `\{`, `\}`, `\|` sont les caractères eux-mêmes ; hors
  d'un groupe, `|` est un caractère ordinaire : un prompt sans accolades ni barres obliques inverses
  revient tel quel, au caractère près.
- **Les refus**, avant de rien construire : `promptSyntax(position:reason:)` pour une `{` jamais
  fermée, une `}` qui ne ferme rien, un groupe dans un groupe (`PromptRefusal` : `unclosed_group`,
  `unmatched_close`, `nested_group` ; `position` compte les caractères depuis 1), et
  `tooManyPromptVariants(count:max:)` au-delà de `PromptAlternatives.maxCombinations` (64) — ou du
  `limit:` passé à `init`.
- **Pour une grille** : `groups` sont les étiquettes des axes, et `variants[i].choices` l'étiquette
  prise dans chaque groupe par le prompt `i`, telle que tapée (échappements résolus).
- `PromptAlternatives.escaping(_:)` réécrit un prompt développé pour qu'il se relise lui-même (ses
  accolades échappées) : ce qu'un formulaire montre quand il rouvre une image de l'historique, pour
  que « Refaire » rende ce prompt-là et pas de nouveau la série.

### 3.15 Une grille XY : `RenderGrid`

`RenderGrid` balaie un ou deux réglages — un groupe `{…}` du prompt, la graine, le nombre de pas, la
force d'une LoRA, le modèle, les images de l'édition — et énumère **un rendu ordinaire par case**, rangée par rangée, X variant le plus
vite (l'ordre où se lit une planche, et celui où les mettre en file). Le moteur n'entend jamais parler
de grille : chaque case est une `Request`. Puis `sheet(imageWidth:imageHeight:)` dit où tout va sur la
planche de comparaison — de la géométrie pure, le dessin est à l'app. C'est ce que font la fenêtre
« Exploration » de l'app (ses cases sont des esquisses, §3.16) et `silicontrol grid` (des images
finies).

```swift
let grid = try RenderGrid(.init(prompt: "woman posing in a {library|greenhouse}", seed: 42, steps: 8),
                          x: .prompt(group: 0), y: .consecutiveSeeds(count: 2))
grid.count                            // 4 = grid.columns × grid.rows — connu avant tout rendu
for cell in grid.cells {              // library·42, greenhouse·42, library·43, greenhouse·43
    let r = try await Engine().render(Request(cell.prompt, resolution: 512, seed: cell.seed, steps: cell.steps),
                                      model: model)
    print(cell.column, cell.row, r.seed)
}
let sheet = grid.sheet(imageWidth: 512, imageHeight: 512)
print(sheet.width, sheet.height, sheet.cell(column: 1, row: 0))   // origine en haut à gauche, en pixels
let axis = try RenderGrid.axis("lora:flat=0.4,0.8", loras: ["flat"])   // .loraStrength(slot: 0, values: [0.4, 0.8])
_ = axis
```

- **Les axes** (`RenderGrid.Axis`) : `.prompt(group:)` (ses alternatives, telles que tapées),
  `.seeds([…])`, `.consecutiveSeeds(count:)` (depuis la graine de base, comme un lot),
  `.steps([…])`, `.loraStrength(slot:values:)` (les autres LoRA gardent la leur), `.models([…])`
  (par identifiant, `z-image`, `qwen-image-2.1`… : `Cell.model`), `.images` (chacune des `images`
  images d'édition de la base, seule, avec la même instruction : `Cell.image`, son indice depuis 0 —
  sans lui, une édition lit toutes ses images ensemble), `.addedLoRAs([…], strength:)` (une LoRA
  ajoutée à la pile par case, toutes à une même force — `""` pour la case sans : `Cell.addedLoRA`,
  `Cell.addedStrength` ; les chaînes sont ce que garde l'appelant, des chemins dans l'app).
  `values(of:)` donne
  les valeurs d'un axe en `RenderGrid.Value`, qu'une app étiquette dans sa langue ; `title` est le
  prompt dont les groupes sont réécrits `{a|b}`.
- **Un axe « modèle » donne à chaque case le nombre de pas de son modèle** : `Cell.steps` vaut `nil`
  (le défaut du modèle — 8 pour Z-Image, 6 pour Qwen-Image-2.1) sauf si les
  pas sont l'autre axe ; sans axe « modèle », c'est celui de la base ou de l'axe. `Cell.model` vaut
  `nil` sans axe « modèle » (le modèle de la base), `Cell.image` sans axe « image » (toutes les
  images).
- **Les refus**, avant tout rendu : `gridRefused(reason:)` (`GridRefusal` : `empty_axis` — un axe
  « image » sans image d'édition en est un —, `same_axis_twice` — la graine est un seul réglage,
  quelle que soit sa liste —, `no_such_group`, `no_such_lora`, `group_not_on_axis`, `invalid_value`,
  `unreadable_axis`, `lora_across_models` — un axe « modèle » avec une pile de LoRA ou un axe
  « LoRA » : une LoRA est faite pour un seul modèle) et
  `tooManyGridCells(count:max:)` au-delà de `RenderGrid.maxCells` (64, la file de l'app) ou de
  `limit:`. **Tout groupe du prompt doit être un axe** : un groupe oublié empilerait plusieurs images
  dans une case — refusé plutôt que multiplié comme le ferait `PromptAlternatives` seul.
- **La planche** (`RenderGrid.Sheet`) : chaque case réduite à 512 px au plus sur son grand côté,
  toutes les cases ensemble à 12 Mpx (64 cases de 512² sortent à 432 px : une planche de ~14 Mpx,
  ~55 Mo en RGBA pendant qu'on la dessine), des tailles de police qui suivent les cases, une colonne
  d'étiquettes de rangée seulement avec un axe Y.
- `RenderGrid.axis(_:loras:)` lit un axe écrit `sorte[:lequel][=valeurs]` (`prompt:2`,
  `seed=42,7`, `seeds=4`, `steps=6,8`, `lora[:NOM]=0.4,0.8`, `loras[:FORCE]=none,flat,ink`
  (force 1 si absente), `model=z-image,qwen-image-2.1`, `image`, décimales avec un point) : la syntaxe de `silicontrol grid`.

### 3.16 Une esquisse : s'arrêter là où l'image se lit

`Request.sketch = n` arrête le rendu après `n` évaluations et décode l'estimation de l'image finale
par le débruiteur, x̂₀ = x − σ·v, au lieu de l'image. L'estimation ne coûte rien (l'évaluation a
calculé `v` de toute façon) et parle tôt. Les évaluations faites sont **les premières du rendu
complet, au bit près** : une esquisse est un préfixe, pas un autre rendu — la même requête sans
`sketch` finit l'image qu'elle a montrée. Sous 1, une évaluation ; au nombre du plan ou au-delà, le
rendu complet (`Render.sketch` vaut alors `nil`).

```swift
let model = try Model.zImage(in: library)
var request = Request("woman posing in a library", resolution: 512, seed: 42)
request.sketch = model.sketchEvaluations(steps: nil, width: 512, height: 512)   // 3 (sur 7)
let rough = try await Engine().render(request, model: model)                    // rough.sketch == 3
request.sketch = nil
let finished = try await Engine().render(request, model: model)                 // l'image annoncée
```

- **Où s'arrêter** (`Sketch`) : après la première évaluation qui amène σ à `Sketch.threshold` (0,8)
  ou dessous — un σ, pas un nombre de pas : le calendrier de Qwen-Image-2.1 garde σ haut plus
  longtemps. `Sketch.evaluations(sigmas:)` le lit sur un calendrier (un pas nul n'est pas une
  évaluation), `Model.sketchEvaluations(steps:width:height:)` sur un modèle à un format (`steps`
  nil : son défaut), et `DenoisingPlan.sketched(_:)` coupe un plan à ce compte, pour estimer la
  durée d'une esquisse avant qu'elle tourne.
- **Ce qu'elle vaut** (512², estimation contre l'image finale réduite ×4, comme sur une
  planche) : Z-Image s'arrête après 3 évaluations sur 7 — ~16,7 s au lieu de ~32,5 s, ~21 dB, pose,
  vêtements et lumière déjà ceux de l'image finale ; Qwen-Image-2.1 après 4 sur 6 — ~20 dB, la
  composition, floue, d'où il n'esquisse plus (`DenoisingModule.sketches` : ses cases sont des
  images finies). Une esquisse montre où va l'image ; le rendu fini juge (sous une LoRA, la pose
  de Z-Image bougeait encore entre 3 et 4).
- **Les événements** : `start` porte le plan du rendu complet ; les événements `step` comptent
  jusqu'à l'esquisse (`total` = `n`), puis décodage et `image` comme d'habitude. Le PNG porte une clé
  `sketch` (§3.8).

### 3.17 Plus de détail : `Detail`

`Request.detail = .more` (ou `.most`) dit au débruiteur, au milieu de la trajectoire, qu'il reste
un peu moins de bruit qu'en réalité : il en retire moins, et ce qu'il laisse est de la texture
fine. Le pas d'Euler garde le vrai σ, la trajectoire arrive donc toujours à σ = 0. C'est le Detail
Daemon de ComfyUI (Jonseed, MIT), porté ligne à ligne et réduit à un choix nommé : aucun de ses
dix réglages n'est exposé. **Mêmes évaluations, même coût** ; `.normal` est le rendu sans lui, au
bit près.

```swift
var request = Request("woman posing in a library", resolution: 512, seed: 42)
request.detail = .more                                       // .normal · .more · .most
let detailed = try await Engine().render(request, model: try Model.zImage(in: library))
print(detailed.detail, detailed.metadata["detail"] ?? "")   // more more
```

- **Ce qu'il vaut** (512², nombre de pas par défaut, à l'œil et par le |laplacien| moyen de
  la luminance) : `.more` ≈ +10 % de détail fin et la même image, `.most` ≈ +20 %, le plus fort
  resté propre. Z-Image prend la cloche du node ; sur Qwen-Image-2.1, dont le calendrier décalé
  décide la composition tôt, cette cloche changeait l'image : il a une cloche tardive, sur son seul
  5ᵉ pas.
- **La cloche est posée sur les indices de pas**, pas sur σ : un autre nombre de pas que celui par
  défaut, ou le 1024², n'a pas été jugé. En img2img, elle couvre les pas qui tournent.
- Le PNG porte `detail` sauf pour `.normal` (§3.8) ; l'app l'offre en « Détail : Normal · Plus ·
  Maximum » dans la boîte du DiT.

### 3.18 Variations : `Variation`

`Request.variations` tourne le bruit de départ de la graine vers celui d'une autre graine : l'image
garde sa composition à proportion de `strength` — la « variation seed » de Draw Things et
d'AUTOMATIC1111. Seul le bruit change : le texte, les images et les clés du cache de rendu ne la
voient pas.

```swift
let model = try Model.zImage(in: library)
var request = Request("woman posing in a library", resolution: 512, seed: 42)
let subtle = Variation.Amount.subtle.strength(for: model.family)    // 0.1 (Qwen-Image-2.1 : 0.05)
request.variations = [Variation(seed: UInt64.random(in: 0...UInt64.max), strength: subtle)]
let cousin = try await Engine().render(request, model: model)       // graine 42, son bruit tourné
request.variations.append(Variation(seed: 7, strength: 0.5))        // une variation de CETTE image
```

- **Le mélange est une rotation** : bruit ← cos θ·a + sin θ·b, θ = `strength`·π/2, en fp64 arrondi
  une fois en fp32. Pour deux tirages normaux standard indépendants, c'est *exactement* encore une
  normale standard — le bruit sur lequel le débruiteur a été entraîné —, là où le slerp
  d'AUTOMATIC1111 ne l'est qu'en moyenne. `strength` dans `[0, 1]` (écrêtée au-delà, 0 si non
  finie) : 0 est le bruit de la graine au bit près, 1 celui de la graine de variation seul (un
  nouveau tirage).
- **Les variations s'enchaînent**, appliquées dans l'ordre : une variation d'une variation tourne
  le bruit de *cette* image, ses cousines restent donc proches d'elle, pas de son parent.
- **Le second bruit vient du même générateur, à la même forme** : tous les modèles, édition et
  img2img compris. Une variation se définit sur le latent : la même paire de graines à un autre
  format est une autre image.
- **`Variation.Amount`**, les deux choix de l'app, jamais montrés en chiffres : `.strong` 0,5
  partout (la même idée ; pose, tenue et cadrage qui bougent) ; `.subtle` 0,1 (même cadrage, même
  pose, même personne), **0,05 sur Qwen-Image-2.1**, dont les premiers pas décident qui est dans
  l'image. Calibrées sur Z-Image et Qwen-Image-2.1 à 512² seulement.
  `Variation.Amount.named(_:)` nomme une force relue dans un PNG.
- Le PNG porte `variation` (§3.8), et `Recipe.variations` la relit : un rendu refait depuis
  son PNG donne les mêmes bits.

## 4. Limites et contrats

| règle | valeur | dans le code |
|---|---|---|
| plancher | **chaque côté ≥ 512 px** (latent 64). Aucune évaluation n'a lieu sous 512 : en dessous, les modèles sont hors de leur domaine | `Format.minimumSide` |
| multiple | chaque côté multiple de **16** (VAE ×8, patch ×2) | `Format.multiple` |
| plafond | surface ≤ **1024×1536** (1 572 864 px), le dernier format mesuré | `Format.maxSurface` |
| mémoire | machine de **16 Go** (MacBookPro18,3). Pics mesurés (`phys_footprint`) : 1,7 Go pour Z-Image et 2,9 Go pour Qwen-Image-2.1 à 512², 2,9 Go pour Z-Image et 3,6 Go pour Qwen-Image-2.1 à 1024², **3,8 Go** pour Z-Image en 1024×1536, **4,58 Go** pour Qwen-Image-2.1 en 1024×1536 avec trois références ; 0 swapout à chaque pire cas. Avant un rendu, le préflight juge le besoin contre la mémoire réellement disponible (§5) | la portée par étage (`Chain.swift`), `MemoryBudget` |
| précision | calcul **fp32** partout ; les poids peuvent être stockés en bf16, fp16, 8 bits ou GGUF 4 à 6 bits (§3.11), puis élargis. Aucun calcul en fp16/bf16 | `Widen` (les poids élargis en fp32) |
| déterminisme | avec `reproducible` (vrai par défaut, coupe GPU/AMX figée) : **même graine, même requête, même machine et même version → mêmes bits**, et la non-régression se juge au md5. L'image n d'un lot = le rendu isolé de sa graine. `Noise` n'est **pas** `torch.randn` : une graine ne redonne pas l'image de diffusers ni celle de Draw Things. D'une version à l'autre, les pixels peuvent changer, et d'une machine à l'autre, rien n'a été mesuré | `Request.reproducible` (`nil` : `EngineSettings.effective.frozenCut`) |
| schedule spectral (Z-Image) | seulement quand **les deux côtés font ≥ 1024** : 1024², 1024×1536 et sa transposée (`k = 2`). Les portraits et paysages usuels paient leurs 7 évaluations pleines. Il est éteint en img2img ; forcé sous le plancher, il est refusé avant tout calcul | `ZImageDenoisingModule.plan` |

**Temps indicatifs** : MacBookPro18,3 (M1 Pro, 16 Go), profil de la machine chargé, pas par défaut
(8 pour Z-Image, 6 pour Qwen-Image-2.1), graine 42, « a 30 year old woman posing in a library »,
cache de rendu éteint, rendu complet (texte + débruitage + décodage). Ils dépendent de la chaleur et du cache de pages (le premier rendu
relit les cartes) : l'UI doit se fier à `Progress.estimatedRemaining`, pas à ce tableau.

| modèle | 512² | 1024² | 1216×832 | 1024×1536 |
|---|---|---|---|---|
| Z-Image Turbo (8 pas) | **~29 s** | **~82 s** (spectral k = 2) | 110 s | ~153 s |
| Qwen-Image-2.1 Turbo (6 pas) | **~38 s** | **132–136 s** | — | — |

Édition avec Qwen-Image-2.1 : en 1248×832, ~190 s avec une référence, 262–264 s avec
deux ; 1024×1536 avec trois références ~523 s. Une LoRA fusionnée ajoute 0,2 à 0,4 s par pas
à une édition. Sur Z-Image, le cache d'une LoRA ajoute environ +340 Mo au pic du débruitage.

**Licences** : `model.license` est affichée à l'utilisateur, qui l'accepte : **le rendu d'un modèle
dont la licence n'est pas acceptée dans sa bibliothèque est refusé** (`EngineError.licenseNotAccepted`,
§5). L'app enregistre l'acceptation par `library.acceptLicense(card)` (`<racine>/accepted-licenses.json`,
le texte accepté gardé : une licence qui change redemande). Siliconed ne fournit aucun poids : il est
seulement compatible, et chacun lit la licence du modèle qu'il importe.

`license.urls` dit où la lire : le(s) fichier(s) de licence sur Hugging Face **à la révision que
l'installation télécharge**, celui du modèle d'abord, puis chaque composant sous une licence propre.
Un modèle importé reçoit ceux de sa famille.

```swift
let licenseURLs: [URL] = model.license.urls   // https://huggingface.co/<dépôt>/blob/<sha de 40 hex>/<fichier>
```

| modèle | `urls` (chacune à sa révision figée) |
|---|---|
| Z-Image Turbo | `Tongyi-MAI/Z-Image-Turbo` `README.md` — pas de fichier de licence ; la fiche déclare Apache 2.0 |
| Qwen-Image-2.1 Turbo | `Qwen/Qwen-Image-2.1` `LICENSE`, puis `Viggle/Qwen-Image-2.1-viggle-turbo` `LICENSE` (la LoRA turbo, même texte Qwen Research) |

Une Compact (§3.11) est sous la même licence ; son installation montre aussi la fiche de chaque
tiers, qui la déclare (`family.licenseURLs(.compact)`) : `unsloth/Z-Image-Turbo-GGUF` et
`Disty0/Z-Image-Turbo-SDNQ-int8` (Apache 2.0), `Comfy-Org/Qwen-Image-2.1` et
`unsloth/Qwen-Image-2.1-FP8` (Qwen Research). Une Légère montre la fiche de son DiT, puis celle de
son encodeur : `unsloth/Z-Image-Turbo-GGUF` et `Disty0/Z-Image-Turbo-SDNQ-int8`,
`unsloth/Qwen-Image-2.1-GGUF` et `unsloth/Qwen-Image-2.1-FP8`.

- **Z-Image Turbo** : Apache 2.0.
- **Qwen-Image-2.1 Turbo** : Qwen Research, non commerciale.
- Chaque LoRA a sa propre licence, que la bibliothèque ne lit pas.

---

## 5. Erreurs

**Une seule erreur publique : `EngineError`**, une énumération fermée (`Sources/Siliconed/Errors.swift`).
Toute fonction publique qui lève est `throws(EngineError)` ; `events` termine son flux par une
`EngineError`. Chaque cas porte ses valeurs, jamais une phrase toute faite :

- `code` — une clé stable (`"disk_full"`, `"license_not_accepted"`…), celle par laquelle l'app
  retrouve sa phrase traduite (son `Localizable.xcstrings`). Jamais renommée une fois publiée.
- `errorDescription` — la phrase source anglaise ; `recoverySuggestion` — ce que l'utilisateur
  peut faire.
- `FormatRefusal`, `LoRARefusal`, `PromptRefusal` et `GridRefusal` précisent `format_refused`,
  `unsupported_lora`, `prompt_syntax` et `grid_refused` (`format_refused.too_large`…).
- `internalFailure(component:detail:)` et `importRefused(file:detail:)` portent un diagnostic
  technique en anglais, à montrer sous la phrase, jamais dedans.

Les lecteurs et les noyaux gardent leurs propres erreurs dans le paquet ; `EngineError(_:)` replie
chacune dans son cas à la porte. Les protocoles de modules (`TextModule.encoder`,
`DecodingModule.decode`…) ne sont publics que de nom (§2.4) et lèvent sans type ; `render` replie
ce qu'ils lèvent.

**Les préflights.** Avant tout calcul, un rendu vérifie dans cet ordre et lève le premier qui échoue :
le modèle est installé (`card.missing(in:)`) → sa licence est acceptée
(`library.isLicenseAccepted(card)`) → le format (`Format.check`) → la mémoire
(`card.checkMemory(width:height:)` : le plancher du modèle à ce format, `card.memoryNeed(width:height:)`,
contre `MemoryBudget.current().available` — l'état de la machine lu au lancement du rendu : pages
libres, purgeables et de fichier, plus l'empreinte du processus, moins une réserve ; ni sa mémoire
physique, ni sa seule mémoire « libre », qui ne veut rien dire sous le cache de pages) → aucun autre
processus ne rend sur cette machine, quels que soient sa bibliothèque ou son utilisateur (`/private/tmp/siliconed-render.lock`). Viennent ensuite les refus propres à la
requête, toujours avant tout calcul. Une installation vérifie le disque **avant son premier octet**
(`diskFull`). Une chaîne composée à la main (`Model(card:chain:)`) n'a pas de bibliothèque : les
deux premiers et le verrou ne s'y appliquent pas.

| cas | `code` | quand | ce que l'app en fait |
|---|---|---|---|
| `modelNotInstalled(model:)` | `model_not_installed` | `Model.named`, un rendu après une désinstallation | proposer `library.install(card.family)` |
| `unknownModel(model:)` | `unknown_model` | `Model.named` | identifiant hors de `library.cards()` |
| `licenseNotAccepted(model:)` | `license_not_accepted` | rendu | montrer `card.license`, puis `library.acceptLicense(card)` |
| `fileMissing(path:)`, `fileUnreadable(path:)` | `file_missing`, `file_unreadable` | une LoRA ou une image déplacée, des droits | rechoisir le fichier |
| `corruptMap(file:)` | `corrupt_map` | une carte ou un fichier publié qui ne se lit pas (signature, taille, en-tête, autre version) ; `file` peut être `nil` | réinstaller le modèle, réimporter |
| `diskFull(needed:available:)` | `disk_full` | `install`, `importFile`, le cache K/V d'une édition | faire de la place (`occupancy()`, `uninstall`, `emptyCache`) ; les téléchargements reprennent |
| `downloadInterrupted(file:)`, `downloadRefused(file:status:)`, `downloadCorrupt(file:)` | `download_interrupted`, `download_refused`, `download_corrupt` | `install` | relancer ; 401/403 : dépôt à accès restreint, `HF_TOKEN` |
| `importRefused(file:detail:)`, `notAnImportedModel(model:)` | `import_refused`, `not_an_imported_model` | `ModelImport` | ni LoRA ni DiT d'une famille connue, forme en désaccord, sous 4 bits ou autre format non lu |
| `emptyPrompt` | `empty_prompt` | rendu | griser « Générer » si le prompt est blanc |
| `invalidSteps(steps:allowed:)` | `invalid_steps` | rendu | `allowed` (Qwen-Image-2.1 : 5, 6, 7, 9), ou de 1 à `Request.maximumSteps` (50) s'il est vide |
| `tooManyReferences(count:max:)` | `too_many_references` | rendu | `max` 0 : le modèle n'édite pas par référence |
| `imageTooSmall(side:minimum:)` | `image_too_small` | `Format.check`, rendu | proposer `card.formats` |
| `formatRefused(width:height:reason:)`, `formatUnreadable(text:)` | `format_refused`, `format_unreadable` | `Format.check`, rendu | proposer `card.formats` |
| `promptSyntax(position:reason:)`, `tooManyPromptVariants(count:max:)` | `prompt_syntax`, `too_many_prompt_variants` | `PromptAlternatives(_:)` (§3.14) | griser « Générer » et dire où ; le nombre avant le clic |
| `promptReservedText(text:)`, `promptTooLong(tokens:max:)` | `prompt_reserved_text`, `prompt_too_long` | rendu, avant tout poids lu (Qwen-Image-2.1 : `<\|image_pad\|>` tapé dans le prompt ; plus de 512 jetons) | dire quoi retirer ; `tokens` et `max` donnent le compte et la limite |
| `gridRefused(reason:)`, `tooManyGridCells(count:max:)` | `grid_refused`, `too_many_grid_cells` | `RenderGrid(_:x:y:limit:)`, `RenderGrid.axis(_:loras:)` (§3.15) | griser « Générer » ; les cases avant le clic |
| `strengthOutOfRange(strength:)`, `strengthTooLow(strength:steps:floor:)` | `strength_out_of_range`, `strength_too_low` | rendu | borner le curseur à `]floor ; 1]` |
| `imageToImageUnsupported(model:)` | `image_to_image_unsupported` | rendu | Qwen-Image-2.1 : passer l'image en référence 1 |
| `imageUnreadable(file:)`, `imageEncodingFailed`, `imageWriteFailed(file:)` | `image_unreadable`, `image_encoding_failed`, `image_write_failed` | `ImageRGB(contentsOf:)`, `png()`, `PNG.write` | refuser l'image ; enregistrer ailleurs |
| `unsupportedLoRA(file:reason:)`, `loraForOtherModel(lora:target:model:)` | `unsupported_lora`, `lora_for_other_model` | rendu (~0,3 s) | ne proposer que `loras(for:)` ; sinon réimporter |
| `incompatibleChain(output:input:)` | `incompatible_chain` | `Chain(…)` | bug de programmation : n'arrive pas avec `model.chain` |
| `insufficientMemory(needed:available:)` | `insufficient_memory` | rendu ; `card.checkMemory` avant le clic | un format plus petit, un modèle plus léger, quitter d'autres apps |
| `renderAlreadyRunning` | `render_already_running` | rendu, pendant qu'un autre processus rend sur la machine | attendre, ou mettre en file |
| `cancelled` | `cancelled` | l'utilisateur a annulé (une `Task` annulée aussi) | rien : revenir à l'état prêt |
| `settingsAlreadyLoaded(loaded:requested:)` | `settings_already_loaded` | `EngineSettings.load(from:)` après un rendu | appeler `load` au lancement |
| `internalFailure(component:detail:)` | `internal_failure` | une panne au fond du moteur (GPU, noyaux, arènes, forge) | afficher, journaliser, relancer ; une issue si elle persiste |

Le moteur **n'imprime rien**. Ses avertissements arrivent en évènements `.warning` pendant un rendu,
ou par `Warnings.outsideRender` en dehors d'un rendu. Au démarrage, l'app doit aussi afficher
`EngineSettings.effective.warnings` (un profil ignoré, par exemple). `SILICONED_MEMORY_AVAILABLE_GB`
(Gio) remplace la lecture de l'état de la machine (ce qu'elle trouverait avant la réserve : `8` laisse
7,0 Go disponibles, la réserve étant de 1,6 Go), dans le moteur comme dans `checkMemory` : un outil de mesure et de test, pour
atteindre `insufficientMemory` ou rendre sous le budget d'une autre machine — l'image reste la même,
seul le temps change.

---

## 6. Ce que l'API ne fait pas encore

- **CFG et prompt négatif** : les deux modèles sont distillés (Z-Image : une seule évaluation par
  pas ; Qwen-Image-2.1 : sa LoRA turbo, sans CFG). Il n'y a pas de champ `negative`.
- **Inpainting / outpainting / masque**, **ControlNet**, IP-Adapter, image de référence.
- **Agrandissement** (upscaler, *hires fix*) ; formats au-delà de 1024×1536.
- **Choix de l'échantillonneur ou du schedule** : chaque modèle a le sien (Euler flow matching ;
  `Sampler` pour Z-Image).
- Pondération de prompt, textual inversion. Un lot fait varier la graine seulement ; des **prompts**
  différents sont une série de rendus (`PromptAlternatives`, §3.14), pas un lot.
- **Garder le DiT chargé** entre deux images ou deux rendus : il est reconstruit à chaque fois,
  pour tenir en 16 Go et garantir l'identité au bit.
- **Importer un checkpoint non distillé** et le rendre bien : le moteur n'a ni CFG ni prompt
  négatif. Un checkpoint de base (non Turbo) d'une famille connue s'importe et rend, mais mal ; les
  fine-tunes des Turbo sont la cible.
- **Le GGUF sous 4 bits** (Q3_K, Q2_K, I-quants), **le 4 bits hors GGUF** (nvfp4, SDNQ uint4,
  Nunchaku), mxfp8, fp8 E5M2, int8 asymétrique, et LoRA LyCORIS / DoRA / `diff` : refusés à l'import,
  avec le message qui dit pourquoi (§3.11). Un fichier quantifié reste sur le disque tel que publié,
  mais le calcul est en fp32 : pas de produit matriciel int8 ni 4 bits.
- Changer de profil de machine en cours de processus (`EngineSettings` se lit une seule fois ;
  `load` le dit en levant).
- Relire un PNG pour **rejouer** un img2img : l'image source n'est pas dans les métadonnées (la
  clé `image` le dit), et les LoRA n'y figurent que par leur nom de fichier.
- Annuler **au milieu** d'un VAE (encodage, décodage) : l'annulation attend la fin de l'étage (quelques secondes au plus).
- Pas encore vérifié contre la référence : une trajectoire entière dans un format rectangulaire ;
  Qwen-Image-2.1 à 1024² et avec deux références.
