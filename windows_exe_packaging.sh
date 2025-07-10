#!/bin/bash

# AGI OS Windows EXE Packaging Template
# Based on MachineGod Ternary CPU Kernel Architecture
# Creates Windows deployment package with QEMU launcher

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_SYSTEM_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$BUILD_SYSTEM_DIR")"
OUTPUT_DIR="${PROJECT_ROOT}/final"
TEMP_DIR="${BUILD_SYSTEM_DIR}/temp"
LOGS_DIR="${BUILD_SYSTEM_DIR}/logs"

# Windows package configuration
BUILD_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
WINDOWS_OUTPUT="${OUTPUT_DIR}/agi_os_windows_${BUILD_TIMESTAMP}.zip"
PACKAGE_NAME="AGI_OS_Windows_Package"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging functions
log_info() {
    echo -e "${BLUE}[WIN-INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[WIN-SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WIN-WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[WIN-ERROR]${NC} $1"
}

# Error handling
error_exit() {
    log_error "$1"
    exit 1
}

# Check dependencies
check_windows_dependencies() {
    log_info "Checking Windows packaging dependencies..."
    
    # Check for zip utility
    if ! command -v zip &> /dev/null; then
        error_exit "zip utility not found. Please install zip package"
    fi
    
    # Check for kernel EFI
    if [ ! -f "${OUTPUT_DIR}/agi_os_kernel.efi" ]; then
        error_exit "Kernel EFI not found: ${OUTPUT_DIR}/agi_os_kernel.efi"
    fi
    
    log_success "Windows packaging dependencies satisfied"
}

# Create Windows package structure
create_windows_structure() {
    log_info "Creating Windows package structure..."
    
    local win_dir="${TEMP_DIR}/windows_package"
    
    # Clean and create Windows package directory
    rm -rf "$win_dir"
    mkdir -p "$win_dir"
    mkdir -p "$win_dir/bin"
    mkdir -p "$win_dir/docs"
    mkdir -p "$win_dir/config"
    
    # Copy kernel files
    cp "${OUTPUT_DIR}/agi_os_kernel.efi" "$win_dir/bin/"
    if [ -f "${OUTPUT_DIR}/agi_os_kernel.so" ]; then
        cp "${OUTPUT_DIR}/agi_os_kernel.so" "$win_dir/bin/"
    fi
    
    log_success "Windows package structure created"
}

# Create Windows launcher script
create_launcher_script() {
    log_info "Creating Windows launcher script..."
    
    local win_dir="${TEMP_DIR}/windows_package"
    
    # Create main launcher batch file
    cat > "$win_dir/AGI_OS_Launcher.bat" << 'EOF'
@echo off
title AGI OS - MachineGod Kernel Launcher
color 0A

echo.
echo =====================================
echo   AGI OS - MachineGod Kernel
echo =====================================
echo.
echo Ternary CPU Architecture
echo 150 AI Innovations Integrated
echo Memory Optimized: 256MB Constraint
echo.
echo Starting AGI OS with QEMU emulation...
echo.

REM Check if QEMU is available in PATH
where qemu-system-x86_64 >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: QEMU not found in PATH
    echo.
    echo Please install QEMU for Windows and add it to your PATH:
    echo 1. Download QEMU from: https://www.qemu.org/download/#windows
    echo 2. Install QEMU
    echo 3. Add QEMU installation directory to your PATH
    echo 4. Restart this launcher
    echo.
    pause
    exit /b 1
)

REM Check if OVMF BIOS is available
if not exist "config\OVMF.fd" (
    echo WARNING: OVMF BIOS not found in config directory
    echo AGI OS may not boot properly without UEFI firmware
    echo.
    echo You can download OVMF.fd from:
    echo https://github.com/tianocore/edk2/releases
    echo Place it in the config\ directory
    echo.
    timeout /t 5 >nul
)

REM Create temporary directory for QEMU
if not exist "temp" mkdir temp

REM Launch AGI OS with QEMU
echo Launching AGI OS...
echo.

if exist "config\OVMF.fd" (
    qemu-system-x86_64 ^
        -bios config\OVMF.fd ^
        -drive format=raw,file=fat:rw:bin ^
        -m 256 ^
        -smp 1 ^
        -name "AGI OS - Ternary CPU" ^
        -display gtk ^
        -serial stdio
) else (
    echo Starting without UEFI firmware (legacy mode)...
    qemu-system-x86_64 ^
        -drive format=raw,file=fat:rw:bin ^
        -m 256 ^
        -smp 1 ^
        -name "AGI OS - Ternary CPU" ^
        -display gtk ^
        -serial stdio
)

echo.
echo AGI OS session ended.
pause
EOF
    
    # Create installation script
    cat > "$win_dir/Install_AGI_OS.bat" << 'EOF'
@echo off
title AGI OS Installation Script
color 0B

echo.
echo =====================================
echo   AGI OS Installation Script
echo =====================================
echo.
echo This will install AGI OS on your Windows system
echo.

set INSTALL_DIR=%USERPROFILE%\AGI_OS

echo Installation directory: %INSTALL_DIR%
echo.

REM Create installation directory
if not exist "%INSTALL_DIR%" (
    mkdir "%INSTALL_DIR%"
    echo Created directory: %INSTALL_DIR%
) else (
    echo Directory already exists: %INSTALL_DIR%
)

REM Copy files
echo.
echo Copying AGI OS files...
xcopy /E /I /Y bin "%INSTALL_DIR%\bin\"
xcopy /E /I /Y docs "%INSTALL_DIR%\docs\"
xcopy /E /I /Y config "%INSTALL_DIR%\config\"
copy /Y AGI_OS_Launcher.bat "%INSTALL_DIR%\"
copy /Y Uninstall_AGI_OS.bat "%INSTALL_DIR%\"

REM Create desktop shortcut
echo.
echo Creating desktop shortcut...
set SHORTCUT_PATH=%USERPROFILE%\Desktop\AGI_OS.lnk
powershell -Command "$WshShell = New-Object -comObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%SHORTCUT_PATH%'); $Shortcut.TargetPath = '%INSTALL_DIR%\AGI_OS_Launcher.bat'; $Shortcut.WorkingDirectory = '%INSTALL_DIR%'; $Shortcut.Description = 'AGI OS - MachineGod Ternary CPU Kernel'; $Shortcut.Save()"

REM Create start menu entry
echo Creating start menu entry...
set STARTMENU_DIR=%APPDATA%\Microsoft\Windows\Start Menu\Programs\AGI OS
if not exist "%STARTMENU_DIR%" mkdir "%STARTMENU_DIR%"
set STARTMENU_SHORTCUT=%STARTMENU_DIR%\AGI OS.lnk
powershell -Command "$WshShell = New-Object -comObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%STARTMENU_SHORTCUT%'); $Shortcut.TargetPath = '%INSTALL_DIR%\AGI_OS_Launcher.bat'; $Shortcut.WorkingDirectory = '%INSTALL_DIR%'; $Shortcut.Description = 'AGI OS - MachineGod Ternary CPU Kernel'; $Shortcut.Save()"

echo.
echo =====================================
echo   Installation Completed!
echo =====================================
echo.
echo AGI OS has been installed to: %INSTALL_DIR%
echo.
echo You can now run AGI OS by:
echo 1. Double-clicking the desktop shortcut
echo 2. Using the Start Menu entry
echo 3. Running: %INSTALL_DIR%\AGI_OS_Launcher.bat
echo.
echo Note: QEMU is required to run AGI OS
echo Download from: https://www.qemu.org/download/#windows
echo.
pause
EOF
    
    # Create uninstallation script
    cat > "$win_dir/Uninstall_AGI_OS.bat" << 'EOF'
@echo off
title AGI OS Uninstallation Script
color 0C

echo.
echo =====================================
echo   AGI OS Uninstallation Script
echo =====================================
echo.

set INSTALL_DIR=%USERPROFILE%\AGI_OS

echo This will remove AGI OS from your system.
echo Installation directory: %INSTALL_DIR%
echo.
set /p CONFIRM=Are you sure you want to uninstall AGI OS? (Y/N): 

if /i "%CONFIRM%" NEQ "Y" (
    echo Uninstallation cancelled.
    pause
    exit /b 0
)

echo.
echo Removing AGI OS...

REM Remove desktop shortcut
if exist "%USERPROFILE%\Desktop\AGI_OS.lnk" (
    del "%USERPROFILE%\Desktop\AGI_OS.lnk"
    echo Removed desktop shortcut
)

REM Remove start menu entry
set STARTMENU_DIR=%APPDATA%\Microsoft\Windows\Start Menu\Programs\AGI OS
if exist "%STARTMENU_DIR%" (
    rmdir /s /q "%STARTMENU_DIR%"
    echo Removed start menu entry
)

REM Remove installation directory
if exist "%INSTALL_DIR%" (
    rmdir /s /q "%INSTALL_DIR%"
    echo Removed installation directory
)

echo.
echo =====================================
echo   Uninstallation Completed!
echo =====================================
echo.
echo AGI OS has been removed from your system.
echo.
pause
EOF
    
    log_success "Windows launcher scripts created"
}

# Create documentation
create_windows_documentation() {
    log_info "Creating Windows documentation..."
    
    local win_dir="${TEMP_DIR}/windows_package"
    
    # Create main README
    cat > "$win_dir/README.txt" << EOF
AGI OS for Windows
==================

Welcome to AGI OS - the MachineGod Ternary CPU Kernel with 150 AI innovations!

This package contains everything you need to run AGI OS on your Windows system
using QEMU emulation.

Package Contents:
- AGI_OS_Launcher.bat: Main launcher script
- Install_AGI_OS.bat: Installation script
- Uninstall_AGI_OS.bat: Uninstallation script
- bin/: AGI OS kernel files
- docs/: Documentation and technical information
- config/: Configuration files and UEFI firmware

Quick Start:
1. Run Install_AGI_OS.bat as Administrator
2. Install QEMU for Windows if not already installed
3. Launch AGI OS from desktop shortcut or Start Menu

System Requirements:
- Windows 10/11 (64-bit)
- QEMU for Windows
- Minimum 4GB RAM
- 1GB free disk space
- Hardware virtualization support (recommended)

Installation Instructions:
1. Extract this package to a temporary directory
2. Right-click Install_AGI_OS.bat and select "Run as Administrator"
3. Follow the installation prompts
4. Download and install QEMU from: https://www.qemu.org/download/#windows
5. Launch AGI OS using the desktop shortcut

Technical Information:
- Kernel Architecture: Ternary CPU with UEFI Boot
- AI Innovations: 150 integrated innovations
- Memory Constraint: 256MB optimized
- Build Timestamp: ${BUILD_TIMESTAMP}

For more information:
- Visit: https://machinegod.live
- Technical documentation in docs/ directory
- MachineGod white paper and innovations list

Troubleshooting:
- If QEMU is not found, ensure it's installed and in your PATH
- For UEFI boot issues, place OVMF.fd in the config/ directory
- Check Windows Defender/antivirus if files are blocked
- Run as Administrator if installation fails

Support:
For technical support and updates, visit https://machinegod.live

Build Information:
- Package Version: 1.0.0
- Build Date: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
- Platform: Windows x86_64
- Emulation: QEMU with UEFI support
EOF
    
    # Create technical documentation
    cat > "$win_dir/docs/TECHNICAL_INFO.txt" << EOF
AGI OS Technical Information
============================

Kernel Specifications:
- Architecture: x86_64 UEFI
- CPU Type: Ternary CPU (27 registers, 6 trits each)
- Memory Layout: 256MB constraint optimization
- Boot Method: UEFI with GNU-EFI
- AI Framework: 150 integrated innovations

Ternary CPU Details:
- Register File: 27 registers with 6 trits each (18-bit equivalent)
- Instruction Set: 11 opcodes (NOP, LOAD, STORE, ADD, SUB, AND, OR, NOT, JMP, JZ, HALT)
- Arithmetic: Ternary logic with values {-1, 0, 1}
- Processing: Quantum-inspired fetch-decode-execute cycle

Memory Management:
- Kernel Base: 0x100000 (1MB)
- Kernel Stack: 64KB
- Ternary CPU Memory: 0x300000 (3MB)
- IPC Buffer: 0x200000 (2MB)
- Total Constraint: 256MB maximum

AI Innovations Integration:
- MG-CORE: Core intelligence with stratification engine
- MG-EMO: Emotional systems with resonance engine
- MG-WRP: Warp systems with temporal processing
- MG-CMP: Compression systems with hyper-compression
- MG-MEM: Memory systems with shard isolation
- MG-UIX: Interface systems with truth visualization
- MG-ADV: Advanced intelligence with quantum Bayesian grids
- MG-SYN: Synthesis modules with autopoiesis engines
- MG-AVT: Avatar systems with consciousness rendering
- MG-UOS: Universal OS with ternary logic processing

QEMU Configuration:
- System: qemu-system-x86_64
- Memory: 256MB (-m 256)
- CPU: Single core (-smp 1)
- BIOS: OVMF UEFI firmware
- Storage: FAT filesystem emulation
- Display: GTK interface
- Serial: stdio for debugging

Build System:
- Compiler: GCC with GNU-EFI
- Linker: GNU LD with EFI-specific flags
- Tools: objcopy for EFI conversion
- Testing: QEMU with OVMF
- Packaging: ZIP with batch scripts

For detailed technical specifications, see the MachineGod white paper
and innovations documentation.
EOF
    
    # Create QEMU configuration template
    cat > "$win_dir/config/qemu_config.txt" << EOF
QEMU Configuration for AGI OS
=============================

Basic Configuration:
qemu-system-x86_64 -bios OVMF.fd -drive format=raw,file=fat:rw:bin -m 256

Advanced Configuration:
qemu-system-x86_64 \
  -bios OVMF.fd \
  -drive format=raw,file=fat:rw:bin \
  -m 256 \
  -smp 1 \
  -name "AGI OS - Ternary CPU" \
  -display gtk \
  -serial stdio \
  -netdev user,id=net0 \
  -device rtl8139,netdev=net0

Debug Configuration:
qemu-system-x86_64 \
  -bios OVMF.fd \
  -drive format=raw,file=fat:rw:bin \
  -m 256 \
  -smp 1 \
  -nographic \
  -serial stdio \
  -monitor telnet:127.0.0.1:1234,server,nowait

OVMF Download:
Download OVMF.fd from:
https://github.com/tianocore/edk2/releases

Place OVMF.fd in this config directory for UEFI boot support.
EOF
    
    log_success "Windows documentation created"
}

# Package Windows files
package_windows_files() {
    log_info "Packaging Windows files..."
    
    local win_dir="${TEMP_DIR}/windows_package"
    
    # Create the ZIP package
    cd "$win_dir"
    zip -r "$WINDOWS_OUTPUT" . \
        2>&1 | tee "${LOGS_DIR}/windows_package_${BUILD_TIMESTAMP}.log"
    
    if [ $? -eq 0 ]; then
        log_success "Windows package created successfully"
    else
        error_exit "Failed to create Windows package"
    fi
}

# Verify Windows package
verify_windows_package() {
    log_info "Verifying Windows package..."
    
    if [ ! -f "$WINDOWS_OUTPUT" ]; then
        error_exit "Windows package not found: $WINDOWS_OUTPUT"
    fi
    
    # Check file size
    local file_size=$(stat -f%z "$WINDOWS_OUTPUT" 2>/dev/null || stat -c%s "$WINDOWS_OUTPUT" 2>/dev/null)
    if [ "$file_size" -lt 100000 ]; then  # Less than 100KB
        error_exit "Windows package too small ($file_size bytes), possible packaging error"
    fi
    
    log_info "Windows package size: $file_size bytes"
    
    # Test ZIP integrity
    if command -v unzip &> /dev/null; then
        log_info "Testing ZIP integrity..."
        unzip -t "$WINDOWS_OUTPUT" > "${LOGS_DIR}/windows_zip_test_${BUILD_TIMESTAMP}.log" 2>&1
        if [ $? -eq 0 ]; then
            log_success "ZIP integrity verification passed"
        else
            log_warning "ZIP integrity verification failed, but file exists"
        fi
    fi
    
    log_success "Windows package verification completed"
}

# Main Windows packaging function
main() {
    log_info "Starting AGI OS Windows EXE Packaging"
    log_info "Based on MachineGod Ternary CPU Kernel with 150 AI Innovations"
    echo
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --output)
                WINDOWS_OUTPUT="$2"
                shift 2
                ;;
            --name)
                PACKAGE_NAME="$2"
                shift 2
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --output FILE      Specify output ZIP file path"
                echo "  --name NAME        Specify package name"
                echo "  --help             Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Execute Windows packaging steps
    check_windows_dependencies
    create_windows_structure
    create_launcher_script
    create_windows_documentation
    package_windows_files
    verify_windows_package
    
    echo
    log_success "AGI OS Windows packaging completed successfully!"
    log_info "Windows package: $WINDOWS_OUTPUT"
    log_info "Package size: $(stat -f%z "$WINDOWS_OUTPUT" 2>/dev/null || stat -c%s "$WINDOWS_OUTPUT" 2>/dev/null) bytes"
    log_info "Packaging logs: ${LOGS_DIR}/windows_*_${BUILD_TIMESTAMP}.log"
    echo
    log_info "To test the package:"
    log_info "  1. Extract $WINDOWS_OUTPUT"
    log_info "  2. Run Install_AGI_OS.bat as Administrator"
    log_info "  3. Install QEMU for Windows"
    log_info "  4. Launch AGI OS from desktop shortcut"
}

# Execute main function with all arguments
main "$@"