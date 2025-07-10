@echo off
REM AGI OS Bootable USB Creation Script for Windows
REM Based on UEFI Specification 2.11 and Windows best practices
REM Supports ternary CPU kernel architecture with 256MB memory constraints

setlocal enabledelayedexpansion

REM Script configuration
set SCRIPT_DIR=%~dp0
set BUILD_SYSTEM_DIR=%SCRIPT_DIR%..
set PROJECT_ROOT=%BUILD_SYSTEM_DIR%\..
set OUTPUT_DIR=%PROJECT_ROOT%\final

REM Colors (using Windows color codes)
set RED=[91m
set GREEN=[92m
set YELLOW=[93m
set BLUE=[94m
set NC=[0m

echo %BLUE%AGI OS Bootable USB Creator for Windows%NC%
echo ==========================================
echo Ternary CPU Kernel with UEFI Support
echo Based on UEFI Specification 2.11
echo.

REM Check for administrator privileges
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo %RED%[ERROR]%NC% This script must be run as Administrator
    echo Right-click and select "Run as administrator"
    pause
    exit /b 1
)

echo %GREEN%[SUCCESS]%NC% Running with administrator privileges
echo.

REM Check dependencies
echo %BLUE%[INFO]%NC% Checking Windows USB creation dependencies...

REM Check for diskpart
diskpart /? >nul 2>&1
if %errorLevel% neq 0 (
    echo %RED%[ERROR]%NC% diskpart not available
    pause
    exit /b 1
)

REM Check for format command
format /? >nul 2>&1
if %errorLevel% neq 0 (
    echo %RED%[ERROR]%NC% format command not available
    pause
    exit /b 1
)

echo %GREEN%[SUCCESS]%NC% All dependencies satisfied
echo.

REM Verify kernel files exist
echo %BLUE%[INFO]%NC% Verifying AGI OS kernel files...

if not exist "%OUTPUT_DIR%\agi_os_kernel.efi" (
    echo %RED%[ERROR]%NC% Kernel EFI file not found: %OUTPUT_DIR%\agi_os_kernel.efi
    echo Please build the kernel first using build.sh or Makefile
    pause
    exit /b 1
)

REM Check file size
for %%A in ("%OUTPUT_DIR%\agi_os_kernel.efi") do set KERNEL_SIZE=%%~zA
if %KERNEL_SIZE% lss 1024 (
    echo %RED%[ERROR]%NC% Kernel EFI file too small (%KERNEL_SIZE% bytes)
    pause
    exit /b 1
)

echo %GREEN%[SUCCESS]%NC% Kernel files verified (%KERNEL_SIZE% bytes)
echo.

REM List available USB devices
echo %BLUE%[INFO]%NC% Available disk drives:
echo.
wmic diskdrive get size,model,interfacetype,index
echo.

REM Get USB device selection
set /p DISK_INDEX="Enter disk index number for USB device: "

if "%DISK_INDEX%"=="" (
    echo %RED%[ERROR]%NC% No disk index specified
    pause
    exit /b 1
)

REM Confirmation
echo.
echo %YELLOW%[WARNING]%NC% This will DESTROY all data on disk %DISK_INDEX%
echo %YELLOW%[WARNING]%NC% AGI OS will be installed with UEFI boot support
echo.
set /p CONFIRM="Are you sure you want to continue? (yes/no): "

if not "%CONFIRM%"=="yes" (
    echo %BLUE%[INFO]%NC% Operation cancelled
    pause
    exit /b 0
)

echo.
echo %BLUE%[INFO]%NC% Creating UEFI bootable USB on disk %DISK_INDEX%...

REM Create diskpart script for UEFI USB creation
echo Creating diskpart script...
(
echo select disk %DISK_INDEX%
echo clean
echo convert gpt
echo create partition efi size=256
echo select partition 1
echo format quick fs=fat32 label="AGI_OS_EFI"
echo assign letter=Z
echo active
echo exit
) > "%TEMP%\agi_os_diskpart.txt"

REM Execute diskpart script
echo %BLUE%[INFO]%NC% Partitioning USB drive...
diskpart /s "%TEMP%\agi_os_diskpart.txt"

if %errorLevel% neq 0 (
    echo %RED%[ERROR]%NC% Failed to partition USB drive
    del "%TEMP%\agi_os_diskpart.txt"
    pause
    exit /b 1
)

REM Wait for partition to be ready
echo %BLUE%[INFO]%NC% Waiting for partition to be ready...
timeout /t 3 /nobreak >nul

REM Create UEFI directory structure
echo %BLUE%[INFO]%NC% Creating UEFI directory structure...
if not exist "Z:\EFI" mkdir "Z:\EFI"
if not exist "Z:\EFI\BOOT" mkdir "Z:\EFI\BOOT"
if not exist "Z:\EFI\AGI_OS" mkdir "Z:\EFI\AGI_OS"
if not exist "Z:\EFI\AGI_OS\config" mkdir "Z:\EFI\AGI_OS\config"

REM Copy kernel files
echo %BLUE%[INFO]%NC% Copying AGI OS kernel...
copy "%OUTPUT_DIR%\agi_os_kernel.efi" "Z:\EFI\BOOT\BOOTX64.EFI" >nul
if %errorLevel% neq 0 (
    echo %RED%[ERROR]%NC% Failed to copy kernel as BOOTX64.EFI
    goto cleanup
)

copy "%OUTPUT_DIR%\agi_os_kernel.efi" "Z:\EFI\AGI_OS\agi_os_kernel.efi" >nul
if %errorLevel% neq 0 (
    echo %RED%[ERROR]%NC% Failed to copy kernel to AGI_OS directory
    goto cleanup
)

REM Copy startup script
echo %BLUE%[INFO]%NC% Copying UEFI startup script...
if exist "%SCRIPT_DIR%startup.nsh" (
    copy "%SCRIPT_DIR%startup.nsh" "Z:\startup.nsh" >nul
) else (
    REM Create basic startup script if not found
    (
    echo @echo -off
    echo cls
    echo echo "AGI OS - MachineGod Ternary CPU Kernel"
    echo echo "======================================"
    echo echo "UEFI Boot Loader v1.0"
    echo echo ""
    echo echo "Loading AGI OS kernel..."
    echo \EFI\BOOT\BOOTX64.EFI
    ) > "Z:\startup.nsh"
)

REM Copy configuration files
echo %BLUE%[INFO]%NC% Creating boot configuration...
if exist "%SCRIPT_DIR%efi_partition_layout.conf" (
    copy "%SCRIPT_DIR%efi_partition_layout.conf" "Z:\EFI\AGI_OS\config\" >nul
)

REM Create Windows-specific boot configuration
(
echo # AGI OS Boot Configuration - Windows Created
echo # Ternary CPU Kernel with UEFI Support
echo.
echo [BOOT_CONFIG]
echo KERNEL_PATH=\EFI\AGI_OS\agi_os_kernel.efi
echo FALLBACK_PATH=\EFI\BOOT\BOOTX64.EFI
echo MEMORY_LIMIT=256MB
echo CPU_ARCHITECTURE=ternary
echo AI_INNOVATIONS=150
echo CREATED_ON_WINDOWS=true
echo.
echo [DISPLAY]
echo CONSOLE_MODE=80x25
echo BOOT_MESSAGES=true
echo DEBUG_OUTPUT=false
echo.
echo [HARDWARE]
echo DETECT_TERNARY_CPU=true
echo REGISTER_COUNT=27
echo IPC_ENABLED=true
) > "Z:\EFI\AGI_OS\config\boot.conf"

REM Create README file
echo %BLUE%[INFO]%NC% Creating documentation...
(
echo AGI OS Bootable USB Drive
echo ========================
echo.
echo This USB drive contains the AGI OS with MachineGod Ternary CPU Kernel
echo.
echo System Requirements:
echo - UEFI-compatible system
echo - x86_64 architecture
echo - Minimum 64MB RAM ^(optimized for 256MB^)
echo - UEFI boot mode enabled
echo.
echo Boot Instructions:
echo 1. Insert this USB drive into target system
echo 2. Enter UEFI/BIOS setup ^(usually F2, F12, or DEL during boot^)
echo 3. Enable UEFI boot mode ^(disable Legacy/CSM if present^)
echo 4. Set USB drive as first boot device
echo 5. Save settings and restart
echo.
echo Kernel Features:
echo - Ternary CPU architecture with 27 registers
echo - 150 AI innovations integrated
echo - Inter-process communication ^(IPC^) system
echo - Memory-optimized for 256MB constraint
echo - GNU-EFI compiled for x86_64 UEFI
echo.
echo Created: %DATE% %TIME%
echo Build System: AGI OS Build System v1.0
) > "Z:\README.txt"

REM Verify USB creation
echo %BLUE%[INFO]%NC% Verifying USB creation...

if not exist "Z:\EFI\BOOT\BOOTX64.EFI" (
    echo %RED%[ERROR]%NC% Kernel file not found on USB
    goto cleanup
)

if not exist "Z:\startup.nsh" (
    echo %RED%[ERROR]%NC% Startup script not found on USB
    goto cleanup
)

echo %GREEN%[SUCCESS]%NC% USB verification completed successfully

REM Cleanup and finalize
:cleanup
echo %BLUE%[INFO]%NC% Finalizing USB creation...
del "%TEMP%\agi_os_diskpart.txt" 2>nul

echo.
echo %GREEN%[SUCCESS]%NC% AGI OS bootable USB creation completed!
echo %BLUE%[INFO]%NC% USB device: Disk %DISK_INDEX% ^(Drive Z:^)
echo %BLUE%[INFO]%NC% Boot method: UEFI x86_64
echo %BLUE%[INFO]%NC% Kernel: Ternary CPU with 150 AI innovations
echo %BLUE%[INFO]%NC% Memory optimization: 256MB constraint
echo.
echo %BLUE%[INFO]%NC% To boot from USB:
echo   1. Insert USB into target system
echo   2. Enter UEFI/BIOS setup
echo   3. Enable UEFI boot mode
echo   4. Set USB as first boot device
echo   5. Save and restart
echo.
echo %YELLOW%[NOTE]%NC% USB drive will remain mounted as Z: until you eject it
echo.

pause
exit /b 0