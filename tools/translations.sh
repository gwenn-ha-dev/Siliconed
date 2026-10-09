#!/bin/zsh
# **The app's phrases, in English and every language of `languages` below.** The catalog is
# `Sources/SiliconedApp/Localizable.xcstrings` (English source, one translation per language); this
# script keeps it up to date:
#
#   1. it compiles the app's files alone (3 s, against the already-built `Siliconed`
#      module) asking the compiler for the phrases they display — the keys of `Text`,
#      `Button`, `.help`, `String(localized:)`: exactly those the app will look up;
#   2. it synchronizes them into the catalog (`xcstringstool sync`) and removes those the code
#      no longer displays — only one version lives;
#   3. it fails, listing, language by language, every phrase with no translation;
#   4. it fails, listing every phrase whose typography is wrong for its language — only rules that
#      cannot fire on correct text:
#      fr  an ordinary space before `: ; ? !` or inside « » (U+00A0 before `:`, U+202F before
#          `; ? !` and inside « »), or a straight apostrophe ' (the French one is ’; JSON and code
#          quote with ");
#      de  « or » (German quotes „…“), or more straight " than the French value has (those quote
#          code, and the French keeps them too);
#      es  a space of any kind after « or before » (Spanish «…» are tight);
#      it  the same, or a straight apostrophe ' (the Italian one is ’).
#
# `tools/app.sh` calls it before packaging the app, reads `languages` from here for the bundle's
# `CFBundleLocalizations`, then compiles the catalog into the bundle.
set -e
cd "${0:A:h}/.."
# The languages the app ships besides English, the source: one list, read by `tools/app.sh` too.
languages=(fr de es it)
catalog=Sources/SiliconedApp/Localizable.xcstrings
module=""
for d in .build/out/Products/Release .build/out/Products/Debug .build/release/Modules .build/debug/Modules; do
  # Both modules the app imports.
  [[ -e $d/Siliconed.swiftmodule && -e $d/SilicontrolHelp.swiftmodule ]] && { module=$d; break }
done
[[ -n $module ]] || { echo "translations: build the module first (swift build --product SiliconedApp)" >&2; exit 1 }

folder=.build/translations
rm -rf $folder && mkdir -p $folder
xcrun swiftc -c -wmo -Onone -parse-as-library -swift-version 6 -target arm64-apple-macosx15.0 \
  -I $module -I Sources/AMXLegacy/include -module-name SiliconedApp -o $folder/app.o \
  -emit-localized-strings -emit-localized-strings-path $folder Sources/SiliconedApp/*.swift
data=()
for f in $folder/*.stringsdata; do data+=(--stringsdata $f); done
xcrun xcstringstool sync $catalog $data

python3 - "$catalog" $languages <<'PY'
import json, re, sys
path, languages = sys.argv[1], sys.argv[2:]
c = json.load(open(path))
stale = [k for k, v in c["strings"].items() if v.get("extractionState") == "stale"]
for k in stale:
    del c["strings"][k]
def translated(v, lang):
    unit = v.get("localizations", {}).get(lang)
    if not unit: return False
    if "stringUnit" in unit: return unit["stringUnit"].get("state") == "translated"
    return "variations" in unit
wanted = {k: v for k, v in c["strings"].items() if v.get("shouldTranslate", True) is not False}
missing = {lang: sorted(k for k, v in wanted.items() if not translated(v, lang)) for lang in languages}
json.dump(c, open(path, "w"), ensure_ascii=False, indent=2, sort_keys=True, separators=(",", " : "))
open(path, "a").write("\n")
n = len(c["strings"])
if stale: print(f"translations: {len(stale)} phrase(s) removed (the code no longer displays them)")
def values(node):
    """Every value of a language's entry, its plural and device variations included."""
    if isinstance(node, dict):
        if "stringUnit" in node: yield node["stringUnit"].get("value", "")
        for key, child in node.items():
            if key != "stringUnit": yield from values(child)
def french_quotes(v):
    return max((s.count('"') for s in values(v.get("localizations", {}).get("fr", {}))), default=0)
tight = re.compile(r"«\s|\s»")
untypeset = {
    "fr": (lambda s, v: re.search(r" [:;?!]|« | »|'", s),
           "an ordinary space before : ; ? ! or inside « », or a straight apostrophe"),
    "de": (lambda s, v: re.search(r"[«»]", s) or s.count('"') > french_quotes(v),
           "« » instead of „…“, or a straight \" the French value does not have"),
    "es": (lambda s, v: tight.search(s), "a space after « or before »"),
    "it": (lambda s, v: tight.search(s) or "'" in s,
           "a space after « or before », or a straight apostrophe"),
}
wrong = {lang: sorted(s for k, v in c["strings"].items()
                      for s in values(v.get("localizations", {}).get(lang, {}))
                      if lang in untypeset and untypeset[lang][0](s, v))
         for lang in languages}
for lang in languages:
    if missing[lang]:
        print(f"translations: {len(missing[lang])} phrase(s) out of {n} with no {lang}:", file=sys.stderr)
        for k in missing[lang]: print(f"  {k!r}", file=sys.stderr)
for lang in languages:
    if wrong[lang]:
        print(f"translations: {len(wrong[lang])} {lang} phrase(s) with {untypeset[lang][1]}:", file=sys.stderr)
        for s in wrong[lang]: print(f"  {s!r}", file=sys.stderr)
if any(missing.values()) or any(wrong.values()):
    sys.exit(1)
print(f"translations: {n} phrases, all in English and {', '.join(languages)}")
PY
