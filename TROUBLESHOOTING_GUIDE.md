# Troubleshooting Guide

## Overview

This comprehensive troubleshooting guide addresses common issues encountered when building, deploying, and running the AGI OS MachineGod Ternary CPU Kernel. The guide covers build system problems, UEFI boot issues, package generation errors, and runtime problems.

## Build System Issues

### Dependency Problems

**Missing GNU-EFI Development Files**
```
Error: GNU-EFI headers not found at /usr/include/efi
```

**Solution:**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install gnu-efi-dev

# CentOS/RHEL/Fedora
sudo yum install gnu-efi-devel
# or
sudo dnf install gnu-efi-devel

# Arch Linux
sudo pacman -S gnu-efi

# Verify installation
ls -la /usr/include/efi/
ls -la /usr/lib/crt0-efi-x86_64.o
```

**Missing Build Tools**
```
Error: gcc not found
Error: ld not found
Error: objcopy not found
```

**Solution:**
```bash
# Install complete build environment
sudo apt install build-essential binutils gcc make

# Verify tools
gcc --version
ld --version
objcopy --version
make --version
```

**QEMU Not Available**
```
Error: qemu-system-x86_64 not found
```

**Solution:**
```bash
# Install QEMU
sudo apt install qemu-system-x86

# Verify QEMU installation
qemu-system-x86_64 --version
ls -la /usr/share/ovmf/OVMF.fd
```

### Compilation Errors

**EFI Linking Errors**
```
Error: cannot find -lgnuefi
Error: cannot find -lefi
```

**Solution:**
```bash
# Check GNU-EFI library installation
ls -la /usr/lib/libgnuefi.a
ls -la /usr/lib/libefi.a

# If missing, reinstall GNU-EFI
sudo apt remove gnu-efi-dev
sudo apt install gnu-efi-dev

# Verify linker script
ls -la /usr/lib/elf_x86_64_efi.lds
```

**Header File Issues**
```
Error: efi.h: No such file or directory
Error: efilib.h: No such file or directory
```

**Solution:**
```bash
# Check header file locations
find /usr -name "efi.h" 2>/dev/null
find /usr -name "efilib.h" 2>/dev/null

# Update include paths in Makefile if necessary
EFI_INC := /usr/include/efi
EFI_INC_ARCH := /usr/include/efi/x86_64
```

**Kernel Source Issues**
```
Error: main.c: No such file or directory
Error: ternary.h: No such file or directory
```

**Solution:**
```bash
# Verify extracted kernel source
ls -la project/extracted/project/final/src/
ls -la project/extracted/project/final/src/main.c
ls -la project/extracted/project/final/src/ternary.h
ls -la project/extracted/project/final/src/ipc.h

# Check build script paths
grep -n "KERNEL_SRC" scripts/build.sh
grep -n "KERNEL_SRC_DIR" Makefile
```

### Memory and Resource Issues

**Insufficient Disk Space**
```
Error: No space left on device
```

**Solution:**
```bash
# Check available space
df -h
du -sh project/

# Clean temporary files
make clean
rm -rf project/build_system/temp/*
rm -rf project/build_system/logs/*

# Check for large log files
find project/ -name "*.log" -size +10M
```

**Memory Allocation Errors**
```
Error: virtual memory exhausted
Error: cannot allocate memory
```

**Solution:**
```bash
# Check system memory
free -h
cat /proc/meminfo

# Reduce parallel build jobs
make -j1 all

# Increase swap space if needed
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

## UEFI Boot Issues

### Boot Failure Problems

**System Hangs at UEFI Logo**

**Diagnosis:**
- UEFI firmware compatibility issue
- Kernel initialization problem
- Memory allocation failure

**Solution:**
```bash
# Test with QEMU first
qemu-system-x86_64 -bios /usr/share/ovmf/OVMF.fd \
    -drive format=raw,file=fat:rw:./final \
    -nographic -serial stdio -m 256

# Enable verbose boot logging
# Modify startup.nsh:
echo Debug mode enabled
echo Memory limit: 256MB
echo Loading kernel with debug output...
\EFI\BOOT\BOOTX64.EFI debug=verbose
```

**"No Bootable Device" Error**

**Diagnosis:**
- USB not properly created
- UEFI boot mode not enabled
- Boot order configuration issue

**Solution:**
```bash
# Verify USB creation
sudo fdisk -l /dev/sdX
sudo parted /dev/sdX print

# Check UEFI settings in BIOS:
# - Enable UEFI boot mode
# - Disable Legacy/CSM mode
# - Set USB as first boot device
# - Disable Secure Boot (temporarily)

# Recreate USB with verification
sudo ./configs/create_bootable_usb.sh /dev/sdX --verify
```

**Kernel Panic During Boot**

**Diagnosis:**
- Memory constraint violation
- Hardware incompatibility
- Kernel corruption

**Solution:**
```bash
# Verify kernel integrity
file final/agi_os_kernel.efi
hexdump -C final/agi_os_kernel.efi | head

# Test with increased memory
# Modify kernel parameters:
memory_limit=512M
debug_memory=enabled

# Check hardware compatibility
./configs/hardware_detection.sh --report
```

### UEFI Configuration Issues

**Secure Boot Problems**
```
Error: Verification failed
Error: Image not signed
```

**Solution:**
```bash
# Temporarily disable Secure Boot in UEFI settings
# Or sign the kernel:
sbsign --key db.key --cert db.crt \
    --output agi_os_kernel_signed.efi \
    agi_os_kernel.efi

# Verify signature
sbverify --cert db.crt agi_os_kernel_signed.efi
```

**Boot Entry Management Issues**
```
Error: Boot entry not found
Error: Invalid boot order
```

**Solution:**
```bash
# List current boot entries
sudo efibootmgr -v

# Remove invalid entries
sudo efibootmgr -b 0001 -B

# Add AGI OS boot entry
sudo efibootmgr -c -d /dev/sdX -p 1 \
    -L "AGI OS" -l "\EFI\BOOT\BOOTX64.EFI"

# Set boot order
sudo efibootmgr -o 0001,0000,0002
```

## Package Generation Issues

### ISO Creation Problems

**xorriso/genisoimage Not Found**
```
Error: xorriso: command not found
Error: genisoimage: command not found
```

**Solution:**
```bash
# Install ISO creation tools
sudo apt install xorriso genisoimage

# Verify installation
xorriso --version
genisoimage --version

# Alternative: use cdrtools
sudo apt install cdrtools
```

**ISO Boot Failure**
```
Error: ISO not bootable
Error: Invalid boot sector
```

**Solution:**
```bash
# Verify ISO structure
isoinfo -d -i agi_os.iso
isoinfo -l -i agi_os.iso | grep -i boot

# Test ISO with QEMU
qemu-system-x86_64 -bios /usr/share/ovmf/OVMF.fd \
    -cdrom agi_os.iso -m 512

# Recreate ISO with proper options
xorriso -as mkisofs -R -f \
    -e EFI/BOOT/BOOTX64.EFI -no-emul-boot \
    -o agi_os.iso iso_temp/
```

### Windows Package Issues

**QEMU Integration Problems**
```
Error: QEMU not found in PATH
Error: OVMF.fd not found
```

**Solution:**
```cmd
REM Install QEMU for Windows
REM Download from: https://www.qemu.org/download/#windows
REM Or use Chocolatey:
choco install qemu

REM Add QEMU to PATH
set PATH=%PATH%;C:\Program Files\qemu

REM Download OVMF firmware
REM Place OVMF.fd in QEMU directory or specify path:
qemu-system-x86_64 -bios "C:\Program Files\qemu\share\ovmf\OVMF.fd"
```

**Batch Script Execution Issues**
```
Error: Access denied
Error: Script not recognized
```

**Solution:**
```cmd
REM Run as Administrator
REM Right-click Command Prompt -> Run as Administrator

REM Enable script execution
powershell Set-ExecutionPolicy RemoteSigned

REM Check file associations
assoc .bat
ftype batfile

REM Test script manually
type agi_os_launcher.bat
```

### Android APK Issues

**Android SDK Problems**
```
Error: aapt: command not found
Error: Android SDK not found
```

**Solution:**
```bash
# Install Android SDK
sudo apt install android-sdk

# Set environment variables
export ANDROID_HOME=/usr/lib/android-sdk
export PATH=$PATH:$ANDROID_HOME/tools:$ANDROID_HOME/platform-tools

# Install build tools
sdkmanager "build-tools;30.0.3"
sdkmanager "platforms;android-30"

# Verify installation
aapt version
```

**APK Signing Issues**
```
Error: jarsigner not found
Error: Keystore not found
```

**Solution:**
```bash
# Install Java development tools
sudo apt install openjdk-11-jdk

# Generate keystore
keytool -genkey -v -keystore agi_os.keystore \
    -alias agi_os_key -keyalg RSA -keysize 2048 \
    -validity 10000

# Sign APK
jarsigner -verbose -sigalg SHA1withRSA -digestalg SHA1 \
    -keystore agi_os.keystore agi_os.apk agi_os_key
```

## Runtime Issues

### Kernel Runtime Problems

**Memory Allocation Failures**
```
Error: Out of memory
Error: Memory allocation failed
```

**Diagnosis:**
- Kernel exceeding 256MB limit
- Memory fragmentation
- Memory leak in kernel code

**Solution:**
```bash
# Monitor memory usage
# Add debug output to kernel:
printf("Memory usage: %d MB\n", current_memory_usage);
printf("Available memory: %d MB\n", available_memory);

# Adjust memory limits
# Modify kernel configuration:
#define KERNEL_MEMORY_LIMIT (512 * 1024 * 1024)  // Increase to 512MB

# Test with QEMU memory options
qemu-system-x86_64 -m 1024 -bios /usr/share/ovmf/OVMF.fd
```

**Ternary CPU Issues**
```
Error: Ternary CPU initialization failed
Error: Invalid ternary operation
```

**Diagnosis:**
- Ternary CPU register corruption
- Invalid instruction execution
- Arithmetic operation errors

**Solution:**
```c
// Add debug output to ternary.h:
void debug_ternary_state(TernaryCPU *cpu) {
    printf("Ternary CPU State:\n");
    for (int i = 0; i < TERNARY_REGISTERS; i++) {
        printf("R%d: %d\n", i, cpu->registers[i]);
    }
    printf("PC: %d\n", cpu->program_counter);
}

// Validate ternary operations:
int validate_ternary_value(int value) {
    return (value >= -1 && value <= 1);
}
```

**IPC System Problems**
```
Error: IPC buffer overflow
Error: Message queue full
```

**Diagnosis:**
- Ring buffer overflow
- Message size exceeding limits
- Producer/consumer synchronization issues

**Solution:**
```c
// Add IPC debugging to ipc.h:
void debug_ipc_state(IPCRingBuffer *buffer) {
    printf("IPC Buffer State:\n");
    printf("Head: %d, Tail: %d\n", buffer->head, buffer->tail);
    printf("Count: %d, Capacity: %d\n", buffer->count, buffer->capacity);
    printf("Overflow count: %d\n", buffer->overflow_count);
}

// Increase buffer size if needed:
#define IPC_BUFFER_SIZE (4 * 1024 * 1024)  // Increase to 4MB
```

### Hardware Compatibility Issues

**CPU Architecture Problems**
```
Error: Unsupported CPU architecture
Error: x86_64 required
```

**Solution:**
```bash
# Check CPU architecture
uname -m
cat /proc/cpuinfo | grep -i "model name"
lscpu

# Verify 64-bit support
grep -o -w 'lm' /proc/cpuinfo | head -1
```

**UEFI Firmware Issues**
```
Error: UEFI not supported
Error: Legacy BIOS detected
```

**Solution:**
```bash
# Check UEFI support
ls /sys/firmware/efi/
efibootmgr -v

# Enable UEFI in BIOS settings:
# - Boot Mode: UEFI
# - CSM: Disabled
# - Secure Boot: Disabled (for testing)
```

**Memory Hardware Issues**
```
Error: Insufficient memory
Error: Memory test failed
```

**Solution:**
```bash
# Test system memory
memtest86+
# or
sudo apt install memtester
sudo memtester 1024M 1

# Check memory configuration
sudo dmidecode --type memory
cat /proc/meminfo
```

## CI/CD Pipeline Issues

### GitHub Actions Problems

**Workflow Execution Failures**
```
Error: Workflow failed
Error: Job cancelled
```

**Solution:**
```yaml
# Check workflow syntax
# Use GitHub Actions validator
# Review workflow logs in Actions tab

# Add debug steps:
- name: Debug Environment
  run: |
    echo "Runner OS: ${{ runner.os }}"
    echo "Working directory: $(pwd)"
    ls -la
    env | sort
```

**Dependency Installation Issues**
```
Error: Package not found
Error: Permission denied
```

**Solution:**
```yaml
# Update package lists first
- name: Update packages
  run: sudo apt update

# Install with specific versions
- name: Install dependencies
  run: |
    sudo apt install -y build-essential=12.9ubuntu3
    sudo apt install -y gnu-efi-dev=3.0.14-2
```

### GitLab CI Problems

**Runner Configuration Issues**
```
Error: No runners available
Error: Runner offline
```

**Solution:**
```bash
# Check runner status
gitlab-runner status

# Restart runner
gitlab-runner restart

# Re-register runner
gitlab-runner register \
    --url https://gitlab.com/ \
    --registration-token $TOKEN
```

**Docker Image Issues**
```
Error: Image not found
Error: Container failed to start
```

**Solution:**
```yaml
# Use specific image versions
image: ubuntu:22.04

# Add image pull policy
variables:
  DOCKER_PULL_POLICY: always

# Test image locally
docker run -it ubuntu:22.04 /bin/bash
```

## Performance Issues

### Build Performance Problems

**Slow Compilation**
```
Issue: Build takes too long
Issue: High CPU usage
```

**Solution:**
```bash
# Use parallel builds
make -j$(nproc) all

# Optimize compiler flags
CFLAGS += -O2 -pipe

# Use ccache for faster rebuilds
sudo apt install ccache
export CC="ccache gcc"
```

**Large Package Sizes**
```
Issue: ISO too large
Issue: Package exceeds limits
```

**Solution:**
```bash
# Strip debug information
strip final/agi_os_kernel.efi

# Compress packages
gzip -9 final/*.iso
zip -9 final/*.zip

# Remove unnecessary files
rm -rf temp/
rm -rf logs/
```

### Runtime Performance Issues

**Slow Boot Times**
```
Issue: Kernel takes long to boot
Issue: UEFI initialization slow
```

**Solution:**
```c
// Optimize kernel initialization
// Remove unnecessary debug output
// Reduce memory allocation overhead
// Optimize ternary CPU operations

// Profile boot time
uint64_t start_time = get_timestamp();
// ... initialization code ...
uint64_t end_time = get_timestamp();
printf("Boot time: %llu ms\n", end_time - start_time);
```

## Debug Tools and Techniques

### Build System Debugging

**Verbose Build Output**
```bash
# Enable verbose make
make V=1 all

# Enable bash debug mode
bash -x scripts/build.sh

# Check build logs
tail -f logs/compile_*.log
tail -f logs/link_*.log
```

**Dependency Tracing**
```bash
# Trace library dependencies
ldd final/agi_os_kernel.so

# Check symbol resolution
nm final/agi_os_kernel.so | grep -i ternary
objdump -t final/agi_os_kernel.so
```

### Kernel Debugging

**QEMU Debugging**
```bash
# Enable QEMU monitor
qemu-system-x86_64 -monitor stdio \
    -bios /usr/share/ovmf/OVMF.fd \
    -drive format=raw,file=fat:rw:./final

# Use GDB with QEMU
qemu-system-x86_64 -s -S \
    -bios /usr/share/ovmf/OVMF.fd \
    -drive format=raw,file=fat:rw:./final

# In another terminal:
gdb final/agi_os_kernel.so
(gdb) target remote :1234
(gdb) continue
```

**Serial Console Debugging**
```bash
# Enable serial output in kernel
# Add to main.c:
void serial_write(const char* str) {
    while (*str) {
        outb(0x3F8, *str++);  // COM1 port
    }
}

# Test with QEMU serial
qemu-system-x86_64 -serial stdio \
    -bios /usr/share/ovmf/OVMF.fd \
    -drive format=raw,file=fat:rw:./final
```

### Log Analysis

**Build Log Analysis**
```bash
# Search for errors
grep -i error logs/*.log
grep -i warning logs/*.log

# Analyze compilation times
grep "real\|user\|sys" logs/compile_*.log

# Check memory usage
grep -i "memory\|malloc\|alloc" logs/*.log
```

**System Log Analysis**
```bash
# Check system logs
journalctl -f
dmesg | tail -20

# Check UEFI logs
ls /sys/firmware/efi/efivars/
cat /sys/firmware/efi/systab
```

## Recovery Procedures

### Build System Recovery

**Clean and Rebuild**
```bash
# Complete clean
make clean-all
rm -rf project/build_system/temp/*
rm -rf project/build_system/logs/*
rm -rf project/final/*

# Rebuild from scratch
./scripts/build.sh --target all
```

**Source Recovery**
```bash
# Re-extract source files
cd project/extracted/
unzip -o ../../machinegod-kernel.zip

# Verify source integrity
md5sum project/final/src/main.c
wc -l project/final/src/main.c
```

### System Recovery

**Boot Recovery**
```bash
# Create recovery USB
sudo ./configs/create_bootable_usb.sh /dev/sdX --recovery

# Boot from recovery media
# Access UEFI shell
# Restore boot configuration
```

**Configuration Recovery**
```bash
# Backup current configuration
cp -r project/build_system/configs/ backup_configs/

# Restore default configuration
git checkout -- project/build_system/configs/

# Apply custom modifications
diff -u backup_configs/ project/build_system/configs/
```

---

**Troubleshooting Version**: 1.0.0  
**Coverage**: Build, Boot, Runtime, CI/CD  
**Debug Tools**: QEMU, GDB, Serial Console  
**Recovery**: Clean rebuild, source restoration, boot recovery