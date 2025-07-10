# UEFI Boot and USB Creation Guide

## Overview

This guide provides comprehensive instructions for creating bootable USB drives and configuring UEFI boot for the AGI OS MachineGod Ternary CPU Kernel. The build system includes automated utilities and configuration files for seamless UEFI deployment.

## UEFI Boot Configuration

### Boot Components

The AGI OS build system includes the following UEFI boot components:

**Configuration Files:**
- `configs/efi_partition_layout.conf`: EFI partition structure definition
- `configs/uefi_boot_entry.conf`: UEFI boot entry configuration
- `configs/startup.nsh`: UEFI shell startup script
- `configs/create_bootable_usb.sh`: Linux USB creation utility
- `configs/create_bootable_usb.bat`: Windows USB creation utility
- `configs/hardware_detection.sh`: Hardware compatibility detection

### EFI Partition Layout

**Partition Structure** (from `configs/efi_partition_layout.conf`):
```
Partition Table: GPT
Partition 1: EFI System Partition (ESP)
  - Type: EFI System (C12A7328-F81F-11D2-BA4B-00A0C93EC93B)
  - Size: 512MB minimum
  - Format: FAT32
  - Mount: /boot/efi

Directory Structure:
/EFI/
├── BOOT/
│   └── BOOTX64.EFI (AGI OS Kernel)
├── AGI_OS/
│   ├── agi_os_kernel.efi
│   └── startup.nsh
└── Microsoft/ (if dual-boot)
```

### UEFI Boot Entry Configuration

**Boot Entry Settings** (from `configs/uefi_boot_entry.conf`):
```
Boot Entry: AGI OS - MachineGod Kernel
Description: Ternary CPU Kernel with 150 AI Innovations
Path: \EFI\BOOT\BOOTX64.EFI
Options: quiet splash ternary_cpu=enabled memory_limit=256M
Priority: 0001
Active: Yes
```

**Boot Parameters:**
- `ternary_cpu=enabled`: Enable ternary CPU processing
- `memory_limit=256M`: Set kernel memory constraint
- `quiet`: Suppress verbose boot messages
- `splash`: Enable boot splash screen

## USB Boot Creation

### Linux USB Creation

**Automated Script** (`configs/create_bootable_usb.sh`):

**Prerequisites:**
```bash
# Install required tools
sudo apt update
sudo apt install parted dosfstools efibootmgr
```

**Usage:**
```bash
# Make script executable
chmod +x configs/create_bootable_usb.sh

# Create bootable USB (replace /dev/sdX with your USB device)
sudo ./configs/create_bootable_usb.sh /dev/sdX

# With custom options
sudo ./configs/create_bootable_usb.sh /dev/sdX --label "AGI_OS" --verify
```

**Script Features:**
- Automatic device detection and validation
- GPT partition table creation
- EFI system partition formatting
- Kernel file copying and verification
- Boot entry configuration
- Safety checks and confirmations

**Manual Linux Process:**
```bash
# 1. Identify USB device
lsblk
sudo fdisk -l

# 2. Unmount existing partitions
sudo umount /dev/sdX*

# 3. Create GPT partition table
sudo parted /dev/sdX mklabel gpt

# 4. Create EFI system partition
sudo parted /dev/sdX mkpart primary fat32 1MiB 513MiB
sudo parted /dev/sdX set 1 esp on

# 5. Format partition
sudo mkfs.fat -F32 /dev/sdX1

# 6. Mount and copy files
sudo mkdir -p /mnt/agi_usb
sudo mount /dev/sdX1 /mnt/agi_usb
sudo mkdir -p /mnt/agi_usb/EFI/BOOT
sudo cp final/agi_os_kernel.efi /mnt/agi_usb/EFI/BOOT/BOOTX64.EFI
sudo cp configs/startup.nsh /mnt/agi_usb/

# 7. Unmount
sudo umount /mnt/agi_usb
```

### Windows USB Creation

**Automated Script** (`configs/create_bootable_usb.bat`):

**Prerequisites:**
- Administrator privileges
- Windows 10/11 with UEFI support
- USB drive (minimum 1GB)

**Usage:**
```cmd
REM Run as Administrator
configs\create_bootable_usb.bat E:

REM With verification
configs\create_bootable_usb.bat E: --verify
```

**Script Features:**
- Automatic USB drive formatting
- EFI partition creation
- File copying and verification
- Boot configuration setup
- Error handling and logging

**Manual Windows Process:**
```cmd
REM 1. Open Command Prompt as Administrator
REM 2. Use diskpart to prepare USB
diskpart
list disk
select disk X (replace X with USB disk number)
clean
convert gpt
create partition efi size=512
format quick fs=fat32 label="AGI_OS"
assign letter=Z
exit

REM 3. Copy kernel files
mkdir Z:\EFI\BOOT
copy final\agi_os_kernel.efi Z:\EFI\BOOT\BOOTX64.EFI
copy configs\startup.nsh Z:\
```

## Hardware Detection and Compatibility

### Hardware Detection Script

**Automated Detection** (`configs/hardware_detection.sh`):

**Features:**
- CPU architecture verification
- UEFI firmware detection
- Memory capacity checking
- Storage device enumeration
- Network interface detection
- Graphics hardware identification

**Usage:**
```bash
# Run hardware detection
chmod +x configs/hardware_detection.sh
./configs/hardware_detection.sh

# Generate compatibility report
./configs/hardware_detection.sh --report > hardware_report.txt

# Check specific components
./configs/hardware_detection.sh --cpu --memory --uefi
```

**Compatibility Requirements:**
- **CPU**: x86_64 architecture required
- **Memory**: Minimum 512MB RAM (256MB for kernel)
- **Firmware**: UEFI 2.0 or later
- **Storage**: USB 2.0/3.0 or SATA/NVMe
- **Graphics**: Basic VGA compatibility

### System Requirements

**Minimum Requirements:**
- 64-bit x86 processor
- 512MB RAM
- UEFI firmware
- 1GB storage space
- USB port or internal storage

**Recommended Requirements:**
- Modern x86_64 processor
- 2GB RAM or more
- UEFI 2.3+ firmware
- 4GB storage space
- USB 3.0 or faster storage

## Boot Process Configuration

### UEFI Shell Startup

**Startup Script** (`configs/startup.nsh`):
```bash
@echo -off
echo AGI OS - MachineGod Kernel Loading...
echo Ternary CPU Architecture with 150 AI Innovations
echo Initializing hardware detection...
echo Starting kernel with memory constraint: 256MB
echo.
\EFI\BOOT\BOOTX64.EFI
```

**Script Features:**
- Boot message display
- Hardware initialization
- Kernel parameter setup
- Error handling and recovery

### Boot Entry Management

**Add Boot Entry (Linux):**
```bash
# Add AGI OS boot entry
sudo efibootmgr -c -d /dev/sdX -p 1 -L "AGI OS" -l "\EFI\BOOT\BOOTX64.EFI"

# Set boot order
sudo efibootmgr -o 0001,0000,0002

# View current entries
sudo efibootmgr -v
```

**Boot Entry Management (Windows):**
```cmd
REM Use bcdedit for Windows boot management
bcdedit /enum firmware
bcdedit /create {fwbootmgr} /d "AGI OS"
bcdedit /set {fwbootmgr} path \EFI\BOOT\BOOTX64.EFI
```

## Advanced Configuration

### Secure Boot Configuration

**Secure Boot Support:**
- Custom key enrollment for AGI OS kernel
- Signature verification setup
- MOK (Machine Owner Key) management
- Shim bootloader integration

**Key Management:**
```bash
# Generate custom keys
openssl req -new -x509 -newkey rsa:2048 -keyout PK.key -out PK.crt -days 3650
openssl req -new -x509 -newkey rsa:2048 -keyout KEK.key -out KEK.crt -days 3650
openssl req -new -x509 -newkey rsa:2048 -keyout db.key -out db.crt -days 3650

# Sign kernel
sbsign --key db.key --cert db.crt --output agi_os_kernel_signed.efi agi_os_kernel.efi
```

### Multi-Boot Configuration

**GRUB Integration:**
```bash
# Add AGI OS entry to GRUB
cat >> /etc/grub.d/40_custom << 'EOF'
menuentry "AGI OS - MachineGod Kernel" {
    insmod part_gpt
    insmod fat
    set root='hd0,gpt1'
    chainloader /EFI/BOOT/BOOTX64.EFI
}
EOF

# Update GRUB configuration
sudo update-grub
```

**systemd-boot Integration:**
```bash
# Create boot entry
cat > /boot/loader/entries/agi-os.conf << 'EOF'
title AGI OS - MachineGod Kernel
efi /EFI/BOOT/BOOTX64.EFI
options ternary_cpu=enabled memory_limit=256M
EOF
```

## Troubleshooting

### Common Boot Issues

**Boot Failure Symptoms:**
- System hangs at UEFI logo
- "No bootable device" error
- Kernel panic during initialization
- Memory allocation failures

**Diagnostic Steps:**
1. **Verify UEFI Settings:**
   - Enable UEFI boot mode
   - Disable Legacy/CSM mode
   - Check Secure Boot status
   - Verify boot order

2. **Check USB Creation:**
   - Verify partition table (GPT)
   - Confirm FAT32 formatting
   - Validate file copying
   - Test on different systems

3. **Hardware Compatibility:**
   - Run hardware detection script
   - Check CPU architecture
   - Verify memory capacity
   - Test UEFI firmware version

### Debug Procedures

**UEFI Shell Debugging:**
```bash
# Access UEFI shell
# Boot to UEFI shell and run:
fs0:
cd EFI\BOOT
BOOTX64.EFI

# Check file integrity
dir
type startup.nsh
```

**Boot Log Analysis:**
```bash
# Enable verbose boot logging
# Modify startup.nsh to include:
echo Verbose boot mode enabled
echo Checking memory: %MEM_SIZE%
echo Loading kernel modules...
```

**Memory Testing:**
```bash
# Test memory constraints
# Add to kernel parameters:
memory_test=enabled
debug_memory=verbose
allocation_limit=256M
```

### Recovery Procedures

**Boot Recovery:**
1. Boot from recovery USB
2. Access UEFI shell
3. Navigate to EFI partition
4. Restore kernel files
5. Reconfigure boot entries

**System Recovery:**
```bash
# Create recovery USB
sudo ./configs/create_bootable_usb.sh /dev/sdX --recovery

# Boot recovery mode
# Select recovery options in startup menu
```

## Performance Optimization

### Boot Performance

**Optimization Techniques:**
- Fast boot UEFI settings
- Minimal hardware initialization
- Optimized kernel loading
- Reduced boot message verbosity

**Boot Time Measurement:**
```bash
# Measure boot performance
systemd-analyze time
systemd-analyze blame
systemd-analyze critical-chain
```

### Memory Optimization

**Memory Configuration:**
- Kernel memory limit: 256MB
- UEFI memory map optimization
- Early memory allocation
- Memory fragmentation prevention

**Memory Monitoring:**
```bash
# Monitor memory usage during boot
cat /proc/meminfo
free -h
vmstat 1
```

## Security Considerations

### Boot Security

**Security Features:**
- UEFI Secure Boot support
- Kernel signature verification
- Boot integrity checking
- Hardware security module integration

**Security Configuration:**
```bash
# Enable security features
# In UEFI settings:
# - Enable Secure Boot
# - Configure TPM 2.0
# - Set boot password
# - Enable hardware encryption
```

### Access Control

**Boot Access Control:**
- UEFI password protection
- Boot device restrictions
- Network boot limitations
- USB boot policies

---

**UEFI Version**: 2.0+ Compatible  
**Supported Architectures**: x86_64  
**Boot Methods**: USB, Internal Storage, Network  
**Security**: Secure Boot, TPM 2.0, Custom Keys