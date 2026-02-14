#!/usr/bin/env bash
# Build a release APK for sideload install (no Play Store).
# Output: build/app/outputs/flutter-apk/app-release.apk
# Install: adb install build/app/outputs/flutter-apk/app-release.apk
# Or copy the APK to the device and open it (enable "Install from unknown sources" if needed).

set -e
cd "$(dirname "$0")"

echo "Building release APK for sideload..."
flutter pub get
flutter build apk --release

APK="build/app/outputs/flutter-apk/app-release.apk"
if [[ -f "$APK" ]]; then
  echo ""
  echo "Done. APK: $APK"
  echo "Install: adb install $APK"
  echo "Or copy $APK to your device and tap to install (allow unknown sources if prompted)."
else
  echo "Build failed or APK not found at $APK" >&2
  exit 1
fi
