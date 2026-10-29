#!/bin/zsh
# **Siliconed.app — the product, one artifact.** Builds the products in release and
# packages them into `Siliconed.app` at the repository root (ignored by git), then opens it:
#
#     Contents/MacOS/Siliconed          the app (target `SiliconedApp`)
#     Contents/Helpers/silicontrol      the remote control (target `Silicontrol`: Foundation, plus Measurement for IOKit and ImageIO);
#                                       the app's menu links it into the PATH; `silicontrol help`
#                                       is the complete reference
#     Contents/Helpers/siliconed        `siliconed check`: the engine checked against the reference's goldens
#                                       (docs/VERIFY.md) — the very binary the app renders with, no Xcode needed
#     Contents/Helpers/siliconed-dev    the maintainer's command line, when its sources are present
#     Contents/Resources/               the phrases (en, fr, de, es, it) and the icon
#
# The models live in the app's space (`~/Library/Application Support/Siliconed/`), which it fills
# itself: nothing from the repository is read. The phrases: `tools/translations.sh` synchronizes
# them and refuses a phrase missing in any of its `languages`; the app follows the system language. The icon:
# `tools/icon.svg`, rendered by `tools/icon.swift`, redrawn only when its source changes.
#
#     tools/app.sh [--no-open]                       ad hoc signature (the default)
#     tools/app.sh --sign "Developer ID Application: <name> (<team>)" [--no-open]
#
# **Two signatures.** By default the bundle is signed ad hoc (`codesign --sign -`): it runs on this
# machine, but a copy downloaded on another Mac is refused by Gatekeeper — the user has to go
# through System Settings → Privacy & Security → "Open Anyway". With `--sign "<identity>"`, every
# binary is signed with that Developer ID, hardened runtime and a secure timestamp; if
# `SILICONED_NOTARY_PROFILE` names a `notarytool` keychain profile
# (`xcrun notarytool store-credentials <profile> …`, once), the app is submitted to Apple's notary,
# waited for, and the ticket stapled — then it opens with a double click anywhere. Without the
# profile, the signed app is not notarized and says so. `tools/dmg.sh` packages the result.
set -e
cd "${0:A:h}/.."
root=$PWD
identity=-
open_app=1
while (( $# )); do
  case $1 in
    --no-open) open_app=0 ;;
    --sign) shift; identity=${1:?"--sign needs an identity"} ;;
    *) echo "usage: tools/app.sh [--sign \"Developer ID Application: …\"] [--no-open]" >&2; exit 64 ;;
  esac
  shift
done

swift build -c release --product SiliconedApp
swift build -c release --product silicontrol
swift build -c release --product siliconed
# The maintainer's command line, when its sources are present (Package.swift), ships beside it.
dev=0; [[ -d Sources/CLI ]] && dev=1 && swift build -c release --product siliconed-dev
tools/translations.sh

app=$root/Siliconed.app
rm -rf "$app"
mkdir -p "$app/Contents/MacOS" "$app/Contents/Helpers" "$app/Contents/Resources"
cp .build/release/SiliconedApp "$app/Contents/MacOS/Siliconed"
cp .build/release/silicontrol "$app/Contents/Helpers/silicontrol"
cp .build/release/siliconed "$app/Contents/Helpers/siliconed"
(( dev )) && cp .build/release/siliconed-dev "$app/Contents/Helpers/siliconed-dev"
# Local symbols serve only a debugger: without them each binary is about half its size (the app 20.9 → 10.3 MB),
# the same code. Before the signature, which covers the stripped bytes.
strip -x "$app/Contents/MacOS/Siliconed" "$app"/Contents/Helpers/*
xcrun xcstringstool compile Sources/SiliconedApp/Localizable.xcstrings --output-directory "$app/Contents/Resources" >/dev/null
xcrun xcstringstool compile Sources/SiliconedApp/InfoPlist.xcstrings --output-directory "$app/Contents/Resources" >/dev/null
# English is the language of the keys: the compiler emits only what differs (the translations, and
# the plurals), so its table may be empty, but `en.lproj` must exist for a system in English to pick
# it rather than another language present. The other languages: the one list of `translations.sh`.
languages=(en ${=$(sed -n 's/^languages=(\(.*\))$/\1/p' tools/translations.sh)})
(( ${#languages} > 1 )) || { echo "app: no languages=(…) line in tools/translations.sh" >&2; exit 1 }
localizations=""
for l in $languages; do
  mkdir -p "$app/Contents/Resources/$l.lproj"
  localizations+="<string>$l</string>"
done
[[ -e $app/Contents/Resources/en.lproj/Localizable.strings ]] || : > "$app/Contents/Resources/en.lproj/Localizable.strings"

# The icon: redrawn only when its source (or its renderer) changes. Each size is rendered from the
# vector, not scaled down from the 1024 px one: the 16 px stays sharp.
icon=.build/Siliconed.icns
# Under 64 px the five tracks blur into a moiré: those sizes come from `icon-small.svg`, three thicker ones.
if [[ ! -e $icon || tools/icon.svg -nt $icon || tools/icon-small.svg -nt $icon || tools/icon.swift -nt $icon ]]; then
  iconset=.build/Siliconed.iconset
  rm -rf $iconset && mkdir -p $iconset
  for t in 16 32 128 256 512; do
    for scale in 1 2; do
      px=$((t * scale)) svg=tools/icon.svg
      (( px < 64 )) && svg=tools/icon-small.svg
      suffix=; (( scale == 2 )) && suffix=@2x
      swift tools/icon.swift $svg $iconset/icon_${t}x${t}$suffix.png $px
    done
  done
  iconutil -c icns $iconset -o $icon
fi
cp $icon "$app/Contents/Resources/Siliconed.icns"

build=$(git rev-list --count HEAD 2>/dev/null || echo 1)
# **The engine's digest** (`SiliconedEngine`, read by `Diagnostic.Fingerprint`): what this Mac's capacity
# map measured — the engine's sources, each with its path, `Package.swift` and the compiler. An update that
# moves only the app, its phrases or its docs keeps the map; one that moves the engine measures again.
engine=$({ cat Package.swift; xcrun swift --version 2>&1
  find Sources/Siliconed Sources/AMXLegacy -type f ! -name .DS_Store | LC_ALL=C sort | while read -r f; do print -r -- "== $f"; cat "$f"; done
} | shasum -a 256 | cut -c1-12)
cat > "$app/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>Siliconed</string>
  <key>CFBundleDisplayName</key><string>Siliconed</string>
  <key>CFBundleIdentifier</key><string>dev.gwenn-ha.siliconed</string>
  <key>CFBundleExecutable</key><string>Siliconed</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleIconFile</key><string>Siliconed</string>
  <key>CFBundleShortVersionString</key><string>0.17</string>
  <key>CFBundleVersion</key><string>$build</string>
  <key>SiliconedEngine</key><string>$engine</string>
  <key>CFBundleDevelopmentRegion</key><string>en</string>
  <key>CFBundleLocalizations</key><array>$localizations</array>
  <key>LSApplicationCategoryType</key><string>public.app-category.graphics-design</string>
  <key>LSMinimumSystemVersion</key><string>26.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSHumanReadableCopyright</key><string>Siliconed</string>
  <key>NSLocalNetworkUsageDescription</key><string>Siliconed finds the other Macs of this network that run it, to render on one of them or to lend this Mac to them.</string>
  <key>NSBonjourServices</key><array><string>_siliconed._tcp</string><string>_siliconed-pair._tcp</string></array>
  <key>CFBundleURLTypes</key><array><dict>
    <key>CFBundleURLName</key><string>dev.gwenn-ha.siliconed.hub</string>
    <key>CFBundleURLSchemes</key><array><string>siliconed</string></array>
  </dict></array>
</dict></plist>
PLIST

if [[ $identity == - ]]; then
  codesign --force --deep --sign - "$app"
  echo "→ $app (signed ad hoc: on another Mac, System Settings → Privacy & Security → Open Anyway)"
else
  # Inside out: the helpers first, then the bundle — `--deep` is not used for a real identity.
  for binary in "$app"/Contents/Helpers/*(N) "$app"; do
    codesign --force --options runtime --timestamp --sign "$identity" "$binary"
  done
  codesign --verify --strict --verbose=2 "$app"
  if [[ -n ${SILICONED_NOTARY_PROFILE:-} ]]; then
    archive=.build/Siliconed-notarize.zip
    rm -f $archive
    ditto -c -k --keepParent "$app" $archive
    xcrun notarytool submit $archive --keychain-profile "$SILICONED_NOTARY_PROFILE" --wait
    xcrun stapler staple "$app"
    rm -f $archive
    echo "→ $app (Developer ID, notarized)"
  else
    echo "→ $app (Developer ID, NOT notarized: set SILICONED_NOTARY_PROFILE)"
  fi
fi
(( open_app )) && open "$app"
exit 0
