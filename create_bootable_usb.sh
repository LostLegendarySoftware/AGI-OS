#!/bin/bash

# AGI OS Bootable USB Creation Script
# Based on UEFI Specification 2.11 and current best practices
# Supports ternary CPU kernel architecture with 256MB memory constraints

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_SYSTEM_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$BUILD_SYSTEM_DIR")"
OUTPUT_DIR="${PROJECT_ROOT}/final"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

error_exit() {
    log_error "$1"
    exit 1
}

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        error_exit "This script must be run as root for USB device access"
    fi
}

# Check dependencies
check_dependencies() {
    log_info "Checking USB creation dependencies..."
    
    local deps=("parted" "mkfs.fat" "mount" "umount" "lsblk" "fdisk")
    local missing_deps=()
    
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            missing_deps+=("$dep")
        fi
    done
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        log_error "Missing dependencies: ${missing_deps[*]}"
        log_info "Please install: sudo apt-get install parted dosfstools util-linux"
        exit 1
    fi
    
    log_success "All dependencies satisfied"
}

# List available USB devices
list_usb_devices() {
    log_info "Available USB devices:"
    echo
    lsblk -d -o NAME,SIZE,MODEL,TRAN | grep -E "(usb|USB)" || {
        log_warning "No USB devices detected"
        return 1
    }
    echo
}

# Verify kernel files exist
verify_kernel_files() {
    log_info "Verifying AGI OS kernel files..."
    
    if [ ! -f "${OUTPUT_DIR}/agi_os_kernel.efi" ]; then
        error_exit "Kernel EFI file not found: ${OUTPUT_DIR}/agi_os_kernel.efi"
    fi
    
    # Check file size (should be > 1KB for valid EFI)
    local file_size=$(stat -c%s "${OUTPUT_DIR}/agi_os_kernel.efi" 2>/dev/null)
    if [ "$file_size" -lt 1024 ]; then
        error_exit "Kernel EFI file too small (${file_size} bytes), possible build issue"
    fi
    
    log_success "Kernel files verified (${file_size} bytes)"
}

# Create UEFI bootable USB
create_uefi_usb() {
    local device="$1"
    local device_path="/dev/${device}"
    
    log_info "Creating UEFI bootable USB on ${device_path}..."
    
    # Verify device exists and is removable
    if [ ! -b "$device_path" ]; then
        error_exit "Device ${device_path} not found"
    fi
    
    # Check if device is mounted and unmount
    log_info "Unmounting any existing partitions on ${device}..."
    umount "${device_path}"* 2>/dev/null || true
    
    # Create GPT partition table
    log_info "Creating GPT partition table..."
    parted -s "$device_path" mklabel gpt
    
    # Create EFI system partition (256MB as per specifications)
    log_info "Creating EFI system partition (256MB, FAT32)..."
    parted -s "$device_path" mkpart primary fat32 1MiB 257MiB
    parted -s "$device_path" set 1 esp on
    parted -s "$device_path" set 1 boot on
    
    # Wait for partition to be recognized
    sleep 2
    partprobe "$device_path"
    sleep 2
    
    # Format EFI partition as FAT32
    local efi_partition="${device_path}1"
    log_info "Formatting EFI partition as FAT32..."
    mkfs.fat -F32 -n "AGI_OS_EFI" "$efi_partition"
    
    # Create mount point
    local mount_point="/tmp/agi_os_usb_$$"
    mkdir -p "$mount_point"
    
    # Mount EFI partition
    log_info "Mounting EFI partition..."
    mount "$efi_partition" "$mount_point"
    
    # Create EFI directory structure
    log_info "Creating UEFI directory structure..."
    mkdir -p "$mount_point/EFI/BOOT"
    mkdir -p "$mount_point/EFI/AGI_OS"
    mkdir -p "$mount_point/EFI/AGI_OS/config"
    
    # Copy kernel as BOOTX64.EFI (required for x86_64 UEFI boot)
    log_info "Copying AGI OS kernel..."
    cp "${OUTPUT_DIR}/agi_os_kernel.efi" "$mount_point/EFI/BOOT/BOOTX64.EFI"
    cp "${OUTPUT_DIR}/agi_os_kernel.efi" "$mount_point/EFI/AGI_OS/agi_os_kernel.efi"
    
    # Copy startup script
    log_info "Copying UEFI startup script..."
    cp "${SCRIPT_DIR}/startup.nsh" "$mount_point/startup.nsh"
    
    # Copy configuration files
    if [ -f "${SCRIPT_DIR}/efi_partition_layout.conf" ]; then
        cp "${SCRIPT_DIR}/efi_partition_layout.conf" "$mount_point/EFI/AGI_OS/config/"
    fi
    
    # Create boot configuration file
    cat > "$mount_point/EFI/AGI_OS/config/boot.conf" << 'EOF'
# AGI OS Boot Configuration
# Ternary CPU Kernel with UEFI Support

[BOOT_CONFIG]
KERNEL_PATH=\EFI\AGI_OS\agi_os_kernel.efi
FALLBACK_PATH=\EFI\BOOT\BOOTX64.EFI
MEMORY_LIMIT=256MB
CPU_ARCHITECTURE=ternary
AI_INNOVATIONS=150

[DISPLAY]
CONSOLE_MODE=80x25
BOOT_MESSAGES=true
DEBUG_OUTPUT=false

[HARDWARE]
DETECT_TERNARY_CPU=true
REGISTER_COUNT=27
IPC_ENABLED=true
EOF
    
    # Sync and unmount
    log_info "Finalizing USB creation..."
    sync
    umount "$mount_point"
    rmdir "$mount_point"
    
    log_success "UEFI bootable USB created successfully on ${device_path}"
    log_info "USB contains:"
    log_info "  - EFI System Partition (256MB, FAT32)"
    log_info "  - AGI OS Ternary CPU Kernel"
    log_info "  - UEFI startup scripts"
    log_info "  - Boot configuration files"
}

# Verify USB creation
verify_usb() {
    local device="$1"
    local device_path="/dev/${device}"
    
    log_info "Verifying USB creation..."
    
    # Check partition table
    if parted -s "$device_path" print | grep -q "gpt"; then
        log_success "GPT partition table verified"
    else
        log_error "GPT partition table not found"
        return 1
    fi
    
    # Check EFI partition
    if parted -s "$device_path" print | grep -q "esp"; then
        log_success "EFI system partition verified"
    else
        log_error "EFI system partition not found"
        return 1
    fi
    
    # Mount and check files
    local mount_point="/tmp/agi_os_verify_$$"
    mkdir -p "$mount_point"
    
    if mount "${device_path}1" "$mount_point" 2>/dev/null; then
        if [ -f "$mount_point/EFI/BOOT/BOOTX64.EFI" ]; then
            log_success "Kernel file verified on USB"
        else
            log_error "Kernel file not found on USB"
        fi
        
        umount "$mount_point"
        rmdir "$mount_point"
    else
        log_error "Could not mount USB for verification"
        return 1
    fi
    
    log_success "USB verification completed successfully"
}

# Main function
main() {
    echo "AGI OS Bootable USB Creator"
    echo "=========================="
    echo "Ternary CPU Kernel with UEFI Support"
    echo "Based on UEFI Specification 2.11"
    echo
    
    # Parse arguments
    local device=""
    local force=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --device)
                device="$2"
                shift 2
                ;;
            --force)
                force=true
                shift
                ;;
            --help)
                echo "Usage: $0 --device DEVICE [--force]"
                echo "Options:"
                echo "  --device DEVICE    Target USB device (e.g., sdb)"
                echo "  --force           Skip confirmation prompts"
                echo "  --help            Show this help message"
                echo
                echo "Example: $0 --device sdb"
                exit 0
                ;;
            *)
                error_exit "Unknown option: $1"
                ;;
        esac
    done
    
    # Check prerequisites
    check_root
    check_dependencies
    verify_kernel_files
    
    # List USB devices if no device specified
    if [ -z "$device" ]; then
        list_usb_devices
        echo
        read -p "Enter USB device name (e.g., sdb): " device
    fi
    
    # Validate device
    if [ -z "$device" ]; then
        error_exit "No device specified"
    fi
    
    # Remove /dev/ prefix if present
    device="${device#/dev/}"
    
    # Confirmation
    if [ "$force" = false ]; then
        echo
        log_warning "This will DESTROY all data on /dev/${device}"
        log_warning "AGI OS will be installed with UEFI boot support"
        echo
        read -p "Are you sure you want to continue? (yes/no): " confirm
        
        if [ "$confirm" != "yes" ]; then
            log_info "Operation cancelled"
            exit 0
        fi
    fi
    
    # Create bootable USB
    create_uefi_usb "$device"
    verify_usb "$device"
    
    echo
    log_success "AGI OS bootable USB creation completed!"
    log_info "USB device: /dev/${device}"
    log_info "Boot method: UEFI x86_64"
    log_info "Kernel: Ternary CPU with 150 AI innovations"
    log_info "Memory optimization: 256MB constraint"
    echo
    log_info "To boot from USB:"
    log_info "1. Insert USB into target system"
    log_info "2. Enter UEFI/BIOS setup"
    log_info "3. Enable UEFI boot mode"
    log_info "4. Set USB as first boot device"
    log_info "5. Save and restart"
}

# Execute main function with all arguments
main "$@"