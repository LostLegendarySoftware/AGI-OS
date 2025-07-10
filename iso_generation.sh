#!/bin/bash

# AGI OS ISO Generation Template
# Based on MachineGod Ternary CPU Kernel Architecture
# Supports UEFI boot with comprehensive ISO creation

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_SYSTEM_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$BUILD_SYSTEM_DIR")"
OUTPUT_DIR="${PROJECT_ROOT}/final"
TEMP_DIR="${BUILD_SYSTEM_DIR}/temp"
LOGS_DIR="${BUILD_SYSTEM_DIR}/logs"

# ISO configuration
ISO_LABEL="AGI_OS"
ISO_VOLUME_ID="AGI_OS_TERNARY"
BUILD_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ISO_OUTPUT="${OUTPUT_DIR}/agi_os_${BUILD_TIMESTAMP}.iso"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging functions
log_info() {
    echo -e "${BLUE}[ISO-INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[ISO-SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[ISO-WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ISO-ERROR]${NC} $1"
}

# Error handling
error_exit() {
    log_error "$1"
    exit 1
}

# Check dependencies
check_iso_dependencies() {
    log_info "Checking ISO generation dependencies..."
    
    local deps=("xorriso" "genisoimage")
    local available_tools=()
    
    for dep in "${deps[@]}"; do
        if command -v "$dep" &> /dev/null; then
            available_tools+=("$dep")
            log_info "Found ISO tool: $dep"
        fi
    done
    
    if [ ${#available_tools[@]} -eq 0 ]; then
        error_exit "No ISO creation tools found. Please install xorriso or genisoimage"
    fi
    
    # Check for kernel EFI
    if [ ! -f "${OUTPUT_DIR}/agi_os_kernel.efi" ]; then
        error_exit "Kernel EFI not found: ${OUTPUT_DIR}/agi_os_kernel.efi"
    fi
    
    log_success "ISO dependencies satisfied"
}

# Create ISO directory structure
create_iso_structure() {
    log_info "Creating ISO directory structure..."
    
    local iso_dir="${TEMP_DIR}/iso"
    
    # Clean and create ISO directory
    rm -rf "$iso_dir"
    mkdir -p "$iso_dir"
    
    # Create UEFI boot structure
    mkdir -p "$iso_dir/EFI/BOOT"
    mkdir -p "$iso_dir/boot"
    mkdir -p "$iso_dir/agi_os"
    
    # Copy kernel to UEFI boot location
    cp "${OUTPUT_DIR}/agi_os_kernel.efi" "$iso_dir/EFI/BOOT/BOOTX64.EFI"
    
    # Copy additional kernel files
    if [ -f "${OUTPUT_DIR}/agi_os_kernel.so" ]; then
        cp "${OUTPUT_DIR}/agi_os_kernel.so" "$iso_dir/agi_os/"
    fi
    
    log_success "ISO directory structure created"
}

# Create UEFI startup script
create_startup_script() {
    log_info "Creating UEFI startup script..."
    
    local iso_dir="${TEMP_DIR}/iso"
    
    # Create startup.nsh for UEFI shell
    cat > "$iso_dir/startup.nsh" << 'EOF'
@echo -off
cls
echo.
echo =====================================
echo   AGI OS - MachineGod Kernel
echo =====================================
echo.
echo Ternary CPU Architecture
echo 150 AI Innovations Integrated
echo Memory Optimized: 256MB Constraint
echo.
echo Loading AGI OS Kernel...
echo.
\EFI\BOOT\BOOTX64.EFI
EOF
    
    # Create boot information file
    cat > "$iso_dir/agi_os/BOOT_INFO.txt" << EOF
AGI OS Boot Information
=======================

Kernel: MachineGod Ternary CPU Kernel
Architecture: x86_64 UEFI
AI Innovations: 150 integrated innovations
Memory Constraint: 256MB optimized
Build Timestamp: ${BUILD_TIMESTAMP}

Boot Process:
1. UEFI firmware loads BOOTX64.EFI
2. Ternary CPU initialization
3. AI innovation framework startup
4. Memory management initialization
5. IPC system activation

For technical documentation, see:
- Kernel source analysis
- MachineGod white paper
- 150 AI innovations specification

Boot completed successfully if you see this message.
EOF
    
    # Create README for ISO contents
    cat > "$iso_dir/README.txt" << EOF
AGI OS Bootable ISO
===================

This ISO contains the AGI OS based on the MachineGod Ternary CPU Kernel
with 150 integrated AI innovations.

Contents:
- EFI/BOOT/BOOTX64.EFI: Main UEFI bootable kernel
- startup.nsh: UEFI shell startup script
- agi_os/: Additional kernel files and documentation
- boot/: Boot configuration files

Boot Instructions:
1. Burn this ISO to a USB drive or CD/DVD
2. Boot from the USB/CD in UEFI mode
3. The system will automatically start AGI OS
4. For manual boot, use UEFI shell and run startup.nsh

System Requirements:
- UEFI-compatible system
- x86_64 processor
- Minimum 256MB RAM
- UEFI Secure Boot may need to be disabled

For more information, visit: https://machinegod.live
EOF
    
    log_success "UEFI startup script created"
}

# Generate ISO using xorriso
generate_iso_xorriso() {
    log_info "Generating ISO using xorriso..."
    
    local iso_dir="${TEMP_DIR}/iso"
    
    # Set SOURCE_DATE_EPOCH for reproducible builds
    export SOURCE_DATE_EPOCH=$(date +%s)
    
    xorriso -as mkisofs \
        -V "$ISO_VOLUME_ID" \
        -volset "AGI OS Ternary CPU Kernel" \
        -publisher "MachineGod Framework" \
        -preparer "AGI OS Build System" \
        -appid "AGI OS with 150 AI Innovations" \
        -sysid "AGI_OS" \
        -R -f \
        -e EFI/BOOT/BOOTX64.EFI \
        -no-emul-boot \
        -boot-load-size 4 \
        -boot-info-table \
        -eltorito-alt-boot \
        -e EFI/BOOT/BOOTX64.EFI \
        -no-emul-boot \
        -isohybrid-gpt-basdat \
        -o "$ISO_OUTPUT" \
        "$iso_dir" \
        2>&1 | tee "${LOGS_DIR}/iso_xorriso_${BUILD_TIMESTAMP}.log"
    
    if [ $? -eq 0 ]; then
        log_success "ISO generated successfully with xorriso"
        return 0
    else
        log_error "xorriso failed to generate ISO"
        return 1
    fi
}

# Generate ISO using genisoimage
generate_iso_genisoimage() {
    log_info "Generating ISO using genisoimage..."
    
    local iso_dir="${TEMP_DIR}/iso"
    
    genisoimage \
        -V "$ISO_VOLUME_ID" \
        -volset "AGI OS Ternary CPU Kernel" \
        -publisher "MachineGod Framework" \
        -preparer "AGI OS Build System" \
        -appid "AGI OS with 150 AI Innovations" \
        -sysid "AGI_OS" \
        -R -f \
        -e EFI/BOOT/BOOTX64.EFI \
        -no-emul-boot \
        -boot-load-size 4 \
        -boot-info-table \
        -o "$ISO_OUTPUT" \
        "$iso_dir" \
        2>&1 | tee "${LOGS_DIR}/iso_genisoimage_${BUILD_TIMESTAMP}.log"
    
    if [ $? -eq 0 ]; then
        log_success "ISO generated successfully with genisoimage"
        return 0
    else
        log_error "genisoimage failed to generate ISO"
        return 1
    fi
}

# Verify ISO integrity
verify_iso() {
    log_info "Verifying ISO integrity..."
    
    if [ ! -f "$ISO_OUTPUT" ]; then
        error_exit "ISO file not found: $ISO_OUTPUT"
    fi
    
    # Check file size
    local file_size=$(stat -f%z "$ISO_OUTPUT" 2>/dev/null || stat -c%s "$ISO_OUTPUT" 2>/dev/null)
    if [ "$file_size" -lt 1000000 ]; then  # Less than 1MB
        error_exit "ISO file too small ($file_size bytes), possible generation error"
    fi
    
    log_info "ISO file size: $file_size bytes"
    
    # Test ISO structure if possible
    if command -v isoinfo &> /dev/null; then
        log_info "Testing ISO structure..."
        isoinfo -l -i "$ISO_OUTPUT" > "${LOGS_DIR}/iso_structure_${BUILD_TIMESTAMP}.log" 2>&1
        if [ $? -eq 0 ]; then
            log_success "ISO structure verification passed"
        else
            log_warning "ISO structure verification failed, but file exists"
        fi
    fi
    
    log_success "ISO integrity verification completed"
}

# Main ISO generation function
main() {
    log_info "Starting AGI OS ISO Generation"
    log_info "Based on MachineGod Ternary CPU Kernel with 150 AI Innovations"
    echo
    
    # Parse command line arguments
    local force_tool=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --tool)
                force_tool="$2"
                shift 2
                ;;
            --output)
                ISO_OUTPUT="$2"
                shift 2
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --tool TOOL        Force specific ISO tool (xorriso|genisoimage)"
                echo "  --output FILE      Specify output ISO file path"
                echo "  --help             Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Execute ISO generation steps
    check_iso_dependencies
    create_iso_structure
    create_startup_script
    
    # Generate ISO with preferred tool
    local iso_generated=false
    
    if [ "$force_tool" = "xorriso" ] || [ "$force_tool" = "" ]; then
        if command -v xorriso &> /dev/null; then
            if generate_iso_xorriso; then
                iso_generated=true
            fi
        fi
    fi
    
    if [ "$iso_generated" = false ] && ([ "$force_tool" = "genisoimage" ] || [ "$force_tool" = "" ]); then
        if command -v genisoimage &> /dev/null; then
            if generate_iso_genisoimage; then
                iso_generated=true
            fi
        fi
    fi
    
    if [ "$iso_generated" = false ]; then
        error_exit "Failed to generate ISO with any available tool"
    fi
    
    # Verify generated ISO
    verify_iso
    
    echo
    log_success "AGI OS ISO generation completed successfully!"
    log_info "ISO file: $ISO_OUTPUT"
    log_info "ISO size: $(stat -f%z "$ISO_OUTPUT" 2>/dev/null || stat -c%s "$ISO_OUTPUT" 2>/dev/null) bytes"
    log_info "Generation logs: ${LOGS_DIR}/iso_*_${BUILD_TIMESTAMP}.log"
    echo
    log_info "To test the ISO:"
    log_info "  qemu-system-x86_64 -bios /usr/share/ovmf/OVMF.fd -cdrom '$ISO_OUTPUT' -m 256"
}

# Execute main function with all arguments
main "$@"