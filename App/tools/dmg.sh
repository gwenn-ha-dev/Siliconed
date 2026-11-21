#!/bin/zsh
# **The disk image that is distributed**: `.build/Siliconed-<version>.dmg` (UDZO, compressed), the
# app and a link to `/Applications` side by side on `dmg-background.svg` — drag one onto the other.
#
#     tools/dmg.sh            packages the `.build/Siliconed.app` that `tools/app.sh` built
#
# The window's look (icon view, positions, background) lives in the volume's `.DS_Store`, which only
# Finder writes: the image is built read-write, Finder lays it out by AppleScript, then it is
# compressed. So it runs in a logged-in session (a Terminal, not ssh), with Automation allowed to
# drive Finder; the signature asks the same session already (the keychain stays shut over ssh).
#
# The version is the app's (`CFBundleShortVersionString`), so the file name cannot drift from the
# bundle. The image follows the app's signature: an ad hoc app gives an unsigned image; a Developer
# ID app gives an image signed with the same identity and, when `SILICONED_NOTARY_PROFILE` names a
# `notarytool` profile (see `tools/app.sh`), notarized and stapled itself — Apple's advice for a
# downloaded `.dmg`, so Gatekeeper never stops at the image before reaching the app's own ticket.
set -e
cd "${0:A:h}/.."
app=$PWD/.build/Siliconed.app
[[ -d $app ]] || { echo "dmg: build the app first (tools/app.sh --no-open)" >&2; exit 1 }
version=$(/usr/libexec/PlistBuddy -c "Print :CFBundleShortVersionString" "$app/Contents/Info.plist")
dmg=$PWD/.build/Siliconed-$version.dmg
staging=$PWD/.build/dmg rw=$PWD/.build/dmg-rw.dmg
rm -rf $staging "$dmg" "$rw"
mkdir -p $staging/.background
ditto "$app" $staging/Siliconed.app
ln -s /Applications $staging/Applications
# The background at 1× and 2× in one TIFF: Finder picks the one of the screen.
swift tools/icon.swift tools/dmg-background.svg $staging/.background/1x.png 640x400
swift tools/icon.swift tools/dmg-background.svg $staging/.background/2x.png 1280x800
tiffutil -cathidpicheck $staging/.background/1x.png $staging/.background/2x.png -out $staging/.background/background.tiff 2>/dev/null
rm $staging/.background/{1x,2x}.png
hdiutil create -volname "Siliconed $version" -srcfolder $staging -format UDRW -ov "$rw" >/dev/null
rm -rf $staging
mnt=$(hdiutil attach -readwrite -noverify -noautoopen "$rw" | awk -F'\t' '$NF ~ /^\/Volumes\// { print $NF }')
[[ -d $mnt ]] || { echo "dmg: $rw did not mount" >&2; exit 1 }
# The icons on the background's marks (`dmg-background.svg`: the app at 160, Applications at 480,
# their centres on y = 200); the window's bounds are its frame, the title bar included (28 pt). The
# background's bottom 24 pt hold nothing to read: Finder's path bar, a preference of the reader's, covers them.
osascript - "${mnt:t}" <<'APPLESCRIPT'
on run {volume}
  tell application "Finder"
    tell disk volume
      open
      set current view of container window to icon view
      set toolbar visible of container window to false
      set statusbar visible of container window to false
      set bounds of container window to {200, 120, 840, 548}
      set options to icon view options of container window
      set arrangement of options to not arranged
      set icon size of options to 128
      set text size of options to 13
      set background picture of options to file ".background:background.tiff"
      set position of item "Siliconed.app" of container window to {160, 200}
      set position of item "Applications" of container window to {480, 200}
      close
      open
      update without registering applications
      delay 1
      close
    end tell
  end tell
end run
APPLESCRIPT
rm -rf "$mnt/.fseventsd"
sync
hdiutil detach -quiet "$mnt"
hdiutil convert "$rw" -format UDZO -imagekey zlib-level=9 -o "$dmg" >/dev/null
rm -f "$rw"
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
