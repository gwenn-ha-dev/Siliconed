#!/bin/zsh
# **Siliconed.app — the product, one artifact.** Builds the two products in release and
# packages them into `Siliconed.app` at the repository root (ignored by git), then opens it:
#
#     Contents/MacOS/Siliconed          the app (target `SiliconedApp`)
#     Contents/Helpers/silicontrol      the remote control (target `Silicontrol`, Foundation only);
#                                       the app's menu links it into the PATH; `silicontrol help`
#                                       is the complete reference
#     Contents/Resources/               the phrases (en, fr, de, es, it), the icon, the library's resource bundle
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
tools/translations.sh

app=$root/Siliconed.app
rm -rf "$app"
mkdir -p "$app/Contents/MacOS" "$app/Contents/Helpers" "$app/Contents/Resources"
cp .build/release/SiliconedApp "$app/Contents/MacOS/Siliconed"
cp .build/release/silicontrol "$app/Contents/Helpers/silicontrol"
# The library's resources (the diagnostic's reduced goldens), shared by the app and its helper.
cp -R .build/release/Siliconed_Siliconed.bundle "$app/Contents/Resources/"
xcrun xcstringstool compile Sources/SiliconedApp/Localizable.xcstrings --output-directory "$app/Contents/Resources" >/dev/null
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
if [[ ! -e $icon || tools/icon.svg -nt $icon || tools/icon.swift -nt $icon ]]; then
  iconset=.build/Siliconed.iconset
  rm -rf $iconset && mkdir -p $iconset
  for t in 16 32 128 256 512; do
    swift tools/icon.swift tools/icon.svg $iconset/icon_${t}x${t}.png $t
    swift tools/icon.swift tools/icon.svg $iconset/icon_${t}x${t}@2x.png $((t * 2))
  done
  iconutil -c icns $iconset -o $icon
fi
cp $icon "$app/Contents/Resources/Siliconed.icns"

build=$(git rev-list --count HEAD 2>/dev/null || echo 1)
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
  <key>CFBundleShortVersionString</key><string>0.4</string>
  <key>CFBundleVersion</key><string>$build</string>
  <key>CFBundleDevelopmentRegion</key><string>en</string>
  <key>CFBundleLocalizations</key><array>$localizations</array>
  <key>LSApplicationCategoryType</key><string>public.app-category.graphics-design</string>
  <key>LSMinimumSystemVersion</key><string>15.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSHumanReadableCopyright</key><string>Siliconed</string>
</dict></plist>
PLIST

if [[ $identity == - ]]; then
  codesign --force --deep --sign - "$app"
  echo "→ $app (signed ad hoc: on another Mac, System Settings → Privacy & Security → Open Anyway)"
else
  # Inside out: the helpers first, then the bundle — `--deep` is not used for a real identity.
  for binary in "$app/Contents/Helpers/silicontrol" "$app"; do
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
