# Build Android APK for sideload install

Install the app on Android devices without the Play Store (sideload).

## Prerequisites

- Flutter SDK installed and on your PATH
- Android SDK (Android Studio or command-line tools)

## Build

From the `market_news_app` directory:

```bash
./build_android_sideload.sh
```

Or manually:

```bash
flutter pub get
flutter build apk --release
```

The APK is written to:

**`build/app/outputs/flutter-apk/app-release.apk`**

## Install on a device

**Option A – USB (ADB)**

1. Enable Developer options and USB debugging on the device.
2. Connect via USB and run:
   ```bash
   adb install build/app/outputs/flutter-apk/app-release.apk
   ```

**Option B – Copy and tap**

1. Copy `app-release.apk` to the device (email, cloud, USB file copy, etc.).
2. Open the APK on the device and tap Install.
3. If prompted, allow installation from unknown sources (Settings → Security or App permissions).

## Signing (optional)

The release build currently uses the **debug** keystore so it runs and installs without extra setup. For distribution or updates, create a release keystore and configure signing in `android/app/build.gradle.kts` (see [Flutter doc: Android signing](https://docs.flutter.dev/deployment/android#signing-the-app)).

## App ID

Application ID: `com.example.market_news_app` (in `android/app/build.gradle.kts`). Change it before publishing or if you need a unique package name.
