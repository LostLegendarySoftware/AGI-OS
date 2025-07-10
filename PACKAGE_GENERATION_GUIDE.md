# Package Generation Guide

## Overview

The AGI OS Build System provides comprehensive package generation capabilities for multiple deployment platforms. This guide covers the automated generation of ISO images, Windows deployment packages, and Android APK structures using the integrated template scripts.

## Package Generation Templates

### Available Templates

The build system includes the following package generation templates:

**Template Scripts:**
- `templates/iso_generation.sh`: Bootable ISO image creation
- `templates/windows_exe_packaging.sh`: Windows deployment package generation
- `templates/android_apk_build.sh`: Android APK structure creation

**Generated Packages:**
- **ISO**: `agi_os_[timestamp].iso` - Bootable UEFI-compatible ISO image
- **Windows**: `agi_os_windows_[timestamp].zip` - Windows deployment package with QEMU launcher
- **Android**: `agi_os_android_[timestamp].apk` - Android application package structure

## ISO Image Generation

### ISO Generation Script (`templates/iso_generation.sh`)

**Features:**
- UEFI-compatible ISO creation
- Bootloader configuration
- File system optimization
- Multi-boot support
- Hybrid MBR/GPT compatibility

**Usage:**
```bash
# Make script executable
chmod +x templates/iso_generation.sh

# Generate ISO with default settings
./templates/iso_generation.sh

# Generate ISO with custom options
./templates/iso_generation.sh --output custom_agi_os.iso --label "AGI_OS_CUSTOM"

# Generate ISO with verification
./templates/iso_generation.sh --verify --checksum
```

**Script Parameters:**
- `--output`: Custom output filename
- `--label`: Volume label for ISO
- `--verify`: Enable integrity verification
- `--checksum`: Generate checksums
- `--hybrid`: Create hybrid MBR/GPT ISO
- `--uefi-only`: UEFI-only boot (no legacy support)

### ISO Structure

**Directory Layout:**
```
ISO Root/
├── EFI/
│   └── BOOT/
│       └── BOOTX64.EFI (AGI OS Kernel)
├── startup.nsh (UEFI shell startup script)
├── README.txt (Boot instructions)
└── [BOOT]/ (Legacy boot support - optional)
```

**Boot Configuration:**
- **Primary Boot**: UEFI x86_64 via BOOTX64.EFI
- **Fallback**: UEFI shell with startup.nsh
- **Legacy Support**: Optional MBR boot sector
- **Boot Options**: Configurable kernel parameters

### ISO Creation Process

**Automated Process:**
1. **Environment Setup**: Create temporary ISO directory structure
2. **File Copying**: Copy kernel and boot files to ISO structure
3. **Boot Configuration**: Generate startup scripts and boot entries
4. **ISO Generation**: Create ISO using xorriso or genisoimage
5. **Verification**: Validate ISO integrity and boot capability
6. **Cleanup**: Remove temporary files and directories

**Manual ISO Creation:**
```bash
# 1. Create ISO directory structure
mkdir -p iso_temp/EFI/BOOT

# 2. Copy kernel to ISO structure
cp final/agi_os_kernel.efi iso_temp/EFI/BOOT/BOOTX64.EFI

# 3. Create startup script
cat > iso_temp/startup.nsh << 'EOF'
@echo -off
echo AGI OS - MachineGod Kernel Loading...
echo Ternary CPU Architecture with 150 AI Innovations
\EFI\BOOT\BOOTX64.EFI
EOF

# 4. Generate ISO
xorriso -as mkisofs \
    -R -f -e EFI/BOOT/BOOTX64.EFI \
    -no-emul-boot \
    -o agi_os.iso \
    iso_temp/

# 5. Verify ISO
file agi_os.iso
isoinfo -d -i agi_os.iso
```

## Windows Package Generation

### Windows Packaging Script (`templates/windows_exe_packaging.sh`)

**Features:**
- QEMU launcher integration
- Installation script generation
- Registry configuration
- Dependency management
- Uninstaller creation

**Usage:**
```bash
# Make script executable
chmod +x templates/windows_exe_packaging.sh

# Generate Windows package
./templates/windows_exe_packaging.sh

# Generate with custom options
./templates/windows_exe_packaging.sh --installer --shortcuts --registry

# Generate portable version
./templates/windows_exe_packaging.sh --portable --no-installer
```

**Script Parameters:**
- `--installer`: Create installation package
- `--portable`: Create portable version
- `--shortcuts`: Generate desktop shortcuts
- `--registry`: Configure registry entries
- `--qemu-path`: Custom QEMU installation path
- `--no-installer`: Skip installer generation

### Windows Package Structure

**Package Contents:**
```
agi_os_windows_[timestamp].zip
├── agi_os_kernel.efi (Main kernel file)
├── agi_os_launcher.bat (QEMU launcher script)
├── install_agi_os.bat (Installation script)
├── uninstall_agi_os.bat (Uninstaller script)
├── README.txt (Usage instructions)
├── QEMU/ (Optional QEMU binaries)
└── config/ (Configuration files)
```

**Launcher Script Features:**
- QEMU availability checking
- Automatic QEMU download (optional)
- Memory configuration
- Network setup
- Display options
- Error handling and logging

### Windows Installation Process

**Installation Steps:**
1. **Extract Package**: Unzip to desired location
2. **Run Installer**: Execute `install_agi_os.bat` as Administrator
3. **Configure QEMU**: Install QEMU if not present
4. **Create Shortcuts**: Generate desktop and start menu shortcuts
5. **Registry Setup**: Configure Windows registry entries
6. **Verification**: Test launcher functionality

**Manual Installation:**
```cmd
REM 1. Extract package
unzip agi_os_windows_[timestamp].zip -d C:\AGI_OS\

REM 2. Install QEMU (if needed)
choco install qemu
REM or download from https://www.qemu.org/download/#windows

REM 3. Create shortcuts
mklink "%USERPROFILE%\Desktop\AGI OS.lnk" "C:\AGI_OS\agi_os_launcher.bat"

REM 4. Test launcher
cd C:\AGI_OS
agi_os_launcher.bat
```

## Android APK Generation

### Android APK Build Script (`templates/android_apk_build.sh`)

**Features:**
- APK structure creation
- Manifest configuration
- Asset packaging
- Resource compilation
- Signing preparation

**Usage:**
```bash
# Make script executable
chmod +x templates/android_apk_build.sh

# Generate APK structure
./templates/android_apk_build.sh

# Generate with custom options
./templates/android_apk_build.sh --package com.custom.agios --version 2.0

# Generate signed APK (requires keystore)
./templates/android_apk_build.sh --sign --keystore my_keystore.jks
```

**Script Parameters:**
- `--package`: Custom package name
- `--version`: Version code and name
- `--sign`: Sign APK with keystore
- `--keystore`: Path to signing keystore
- `--assets`: Custom assets directory
- `--resources`: Custom resources directory

### APK Structure

**APK Contents:**
```
agi_os_android_[timestamp].apk
├── AndroidManifest.xml (Application manifest)
├── assets/
│   └── agi_os_kernel.efi (Kernel as asset)
├── res/
│   ├── values/
│   │   └── strings.xml (String resources)
│   ├── layout/ (UI layouts)
│   └── drawable/ (Icons and images)
├── META-INF/ (Signing information)
└── classes.dex (Compiled Java code)
```

**Manifest Configuration:**
```xml
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="com.machinegod.agios"
    android:versionCode="1"
    android:versionName="1.0">
    
    <uses-permission android:name="android.permission.INTERNET" />
    <uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" />
    
    <application
        android:allowBackup="true"
        android:label="@string/app_name">
        
        <activity android:name=".MainActivity">
            <intent-filter>
                <action android:name="android.intent.action.MAIN" />
                <category android:name="android.intent.category.LAUNCHER" />
            </intent-filter>
        </activity>
    </application>
</manifest>
```

### Android Development Setup

**Prerequisites:**
```bash
# Install Android SDK tools
sudo apt install android-sdk
export ANDROID_HOME=/usr/lib/android-sdk
export PATH=$PATH:$ANDROID_HOME/tools:$ANDROID_HOME/platform-tools

# Install build tools
sdkmanager "build-tools;30.0.3"
sdkmanager "platforms;android-30"
```

**APK Building Process:**
1. **Structure Creation**: Generate APK directory structure
2. **Asset Copying**: Copy kernel and resources to APK
3. **Manifest Generation**: Create Android manifest file
4. **Resource Compilation**: Compile resources using aapt
5. **DEX Creation**: Compile Java code to DEX format
6. **APK Assembly**: Package all components into APK
7. **Signing**: Sign APK with debug or release key

## Build Integration

### Makefile Integration

**Package Targets:**
```makefile
# ISO generation
iso: kernel $(OUTPUT_DIR)/$(ISO_TARGET)
	$(call log_success,Bootable ISO created)

# Windows package
windows: kernel $(OUTPUT_DIR)/$(WINDOWS_TARGET)
	$(call log_success,Windows package created)

# Android APK
android: kernel $(OUTPUT_DIR)/$(ANDROID_TARGET)
	$(call log_success,Android APK structure created)
```

**Template Script Execution:**
```makefile
$(OUTPUT_DIR)/$(ISO_TARGET): $(OUTPUT_DIR)/$(KERNEL_EFI)
	@./templates/iso_generation.sh --output $@ --verify

$(OUTPUT_DIR)/$(WINDOWS_TARGET): $(OUTPUT_DIR)/$(KERNEL_EFI)
	@./templates/windows_exe_packaging.sh --output $@

$(OUTPUT_DIR)/$(ANDROID_TARGET): $(OUTPUT_DIR)/$(KERNEL_EFI)
	@./templates/android_apk_build.sh --output $@
```

### Build Script Integration

**Package Generation in build.sh:**
```bash
# Create ISO
create_iso() {
    log_info "Creating bootable ISO image..."
    ./templates/iso_generation.sh --output "${OUTPUT_DIR}/agi_os_${BUILD_TIMESTAMP}.iso"
    log_success "ISO image created"
}

# Create Windows package
create_windows_exe() {
    log_info "Creating Windows executable wrapper..."
    ./templates/windows_exe_packaging.sh --output "${OUTPUT_DIR}/agi_os_windows_${BUILD_TIMESTAMP}.zip"
    log_success "Windows package created"
}

# Create Android APK
create_android_apk() {
    log_info "Creating Android APK structure..."
    ./templates/android_apk_build.sh --output "${OUTPUT_DIR}/agi_os_android_${BUILD_TIMESTAMP}.apk"
    log_success "Android APK structure created"
}
```

## Advanced Configuration

### Custom Package Options

**ISO Customization:**
```bash
# Custom ISO with additional files
./templates/iso_generation.sh \
    --output custom_agi_os.iso \
    --label "AGI_OS_CUSTOM" \
    --add-files "docs/,configs/" \
    --boot-message "Custom AGI OS Build"
```

**Windows Customization:**
```bash
# Windows package with custom QEMU
./templates/windows_exe_packaging.sh \
    --output custom_windows.zip \
    --qemu-bundle \
    --memory 512 \
    --network user \
    --display gtk
```

**Android Customization:**
```bash
# Android APK with custom package
./templates/android_apk_build.sh \
    --package com.custom.agios \
    --version-code 2 \
    --version-name "2.0.0" \
    --target-sdk 30
```

### Signing and Security

**ISO Signing:**
```bash
# Sign ISO for secure boot
gpg --detach-sign --armor agi_os.iso
sha256sum agi_os.iso > agi_os.iso.sha256
```

**Windows Code Signing:**
```bash
# Sign Windows executables
signtool sign /f certificate.p12 /p password /t http://timestamp.server agi_os_launcher.exe
```

**Android APK Signing:**
```bash
# Generate keystore
keytool -genkey -v -keystore agi_os.keystore -alias agi_os_key -keyalg RSA -keysize 2048 -validity 10000

# Sign APK
jarsigner -verbose -sigalg SHA1withRSA -digestalg SHA1 -keystore agi_os.keystore agi_os.apk agi_os_key

# Align APK
zipalign -v 4 agi_os.apk agi_os_aligned.apk
```

## Testing and Verification

### Package Testing

**ISO Testing:**
```bash
# Test ISO boot with QEMU
qemu-system-x86_64 -bios /usr/share/ovmf/OVMF.fd -cdrom agi_os.iso -m 512

# Verify ISO integrity
isoinfo -d -i agi_os.iso
md5sum agi_os.iso
```

**Windows Package Testing:**
```cmd
REM Test Windows launcher
cd extracted_package
agi_os_launcher.bat

REM Verify package contents
dir /s
```

**Android APK Testing:**
```bash
# Install APK on device/emulator
adb install agi_os_android.apk

# Verify APK structure
aapt dump badging agi_os_android.apk
unzip -l agi_os_android.apk
```

### Automated Testing

**Package Verification Script:**
```bash
#!/bin/bash
# verify_packages.sh

# Test ISO
echo "Testing ISO package..."
file final/agi_os_*.iso
isoinfo -d -i final/agi_os_*.iso

# Test Windows package
echo "Testing Windows package..."
unzip -t final/agi_os_windows_*.zip

# Test Android APK
echo "Testing Android APK..."
aapt dump badging final/agi_os_android_*.apk
```

## Troubleshooting

### Common Issues

**ISO Generation Issues:**
- Missing xorriso/genisoimage tools
- Insufficient disk space
- Invalid file permissions
- UEFI boot configuration errors

**Windows Package Issues:**
- QEMU installation problems
- Path configuration errors
- Permission restrictions
- Antivirus false positives

**Android APK Issues:**
- Missing Android SDK tools
- Signing key problems
- Manifest validation errors
- Resource compilation failures

### Debug Procedures

**ISO Debug:**
```bash
# Check ISO contents
isoinfo -l -i agi_os.iso
mount -o loop agi_os.iso /mnt/iso
ls -la /mnt/iso/EFI/BOOT/
```

**Windows Debug:**
```cmd
REM Check package contents
unzip -l agi_os_windows.zip
REM Test QEMU availability
where qemu-system-x86_64
```

**Android Debug:**
```bash
# Validate APK
aapt dump xmltree agi_os.apk AndroidManifest.xml
aapt dump resources agi_os.apk
```

## Performance Optimization

### Package Size Optimization

**ISO Optimization:**
- Remove unnecessary files
- Compress boot files
- Optimize file system layout
- Use efficient compression

**Windows Optimization:**
- Bundle minimal QEMU components
- Compress executable files
- Remove debug information
- Optimize batch scripts

**Android Optimization:**
- Minimize APK size
- Compress assets
- Optimize resources
- Remove unused code

### Build Performance

**Parallel Generation:**
```bash
# Generate packages in parallel
./templates/iso_generation.sh &
./templates/windows_exe_packaging.sh &
./templates/android_apk_build.sh &
wait
```

**Caching Strategy:**
- Cache intermediate files
- Reuse compiled resources
- Optimize file copying
- Minimize redundant operations

---

**Package Formats**: ISO, Windows ZIP, Android APK  
**Target Platforms**: UEFI x86_64, Windows 10+, Android 7.0+  
**Build Tools**: xorriso, QEMU, Android SDK  
**Security**: Code signing, integrity verification, secure boot support