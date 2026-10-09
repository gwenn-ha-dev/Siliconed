#!/bin/zsh
# **The disk image that is distributed**: `.build/Siliconed-<version>.dmg` (UDZO, compressed), the
# app and a link to `/Applications` side by side — drag one onto the other.
#
#     tools/dmg.sh            packages the `Siliconed.app` that `tools/app.sh` built
#
# The version is the app's (`CFBundleShortVersionString`), so the file name cannot drift from the
# bundle. The image follows the app's signature: an ad hoc app gives an unsigned image; a Developer
# ID app gives an image signed with the same identity and, when `SILICONED_NOTARY_PROFILE` names a
# `notarytool` profile (see `tools/app.sh`), notarized and stapled itself — Apple's advice for a
# downloaded `.dmg`, so Gatekeeper never stops at the image before reaching the app's own ticket.
set -e
cd "${0:A:h}/.."
app=$PWD/Siliconed.app
[[ -d $app ]] || { echo "dmg: build the app first (tools/app.sh --no-open)" >&2; exit 1 }
version=$(/usr/libexec/PlistBuddy -c "Print :CFBundleShortVersionString" "$app/Contents/Info.plist")
dmg=$PWD/.build/Siliconed-$version.dmg
staging=$PWD/.build/dmg
rm -rf $staging "$dmg"
mkdir -p $staging
ditto "$app" $staging/Siliconed.app
ln -s /Applications $staging/Applications
hdiutil create -volname "Siliconed $version" -srcfolder $staging -format UDZO -ov "$dmg" >/dev/null
rm -rf $staging
identity=$(codesign -dv --verbose=2 "$app" 2>&1 | sed -n 's/^Authority=\(Developer ID Application: .*\)$/\1/p')
if [[ -n $identity ]]; then
  codesign --force --timestamp --sign "$identity" "$dmg"
  if [[ -n ${SILICONED_NOTARY_PROFILE:-} ]]; then
    xcrun notarytool submit "$dmg" --keychain-profile "$SILICONED_NOTARY_PROFILE" --wait
    xcrun stapler staple "$dmg"
  else
    echo "dmg: signed, NOT notarized (set SILICONED_NOTARY_PROFILE)"
  fi
fi
echo "→ $dmg ($(du -h "$dmg" | cut -f1))"
