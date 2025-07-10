@echo -off
cls
echo "AGI OS - MachineGod Ternary CPU Kernel"
echo "======================================"
echo "UEFI Boot Loader v1.0"
echo "Kernel Architecture: Ternary CPU with 150 AI Innovations"
echo "Memory Constraint: 256MB optimized"
echo "Target: x86_64 EFI Application"
echo ""

# Set console mode for better compatibility
mode 80 25

# Display system information
echo "Detecting system configuration..."
echo "EFI System Partition: %fs0%"
echo "Boot Path: \EFI\BOOT\"
echo ""

# Check for kernel file existence
if exist fs0:\EFI\BOOT\BOOTX64.EFI then
    echo "Found AGI OS Kernel: BOOTX64.EFI"
    echo "Kernel Size: Optimized for 256MB memory constraint"
    echo "Architecture: GNU-EFI x86_64 compiled"
    echo ""
    echo "Initializing ternary CPU support..."
    echo "Loading MachineGod kernel with AI innovations..."
    echo ""
    
    # Launch the kernel
    fs0:\EFI\BOOT\BOOTX64.EFI
    
    # If kernel returns, show error
    echo ""
    echo "ERROR: Kernel execution completed unexpectedly"
    echo "This may indicate a boot failure or normal shutdown"
    
else
    echo "ERROR: AGI OS Kernel not found!"
    echo "Expected location: fs0:\EFI\BOOT\BOOTX64.EFI"
    echo ""
    echo "Please ensure:"
    echo "1. USB drive is properly formatted (FAT32)"
    echo "2. EFI system partition is correctly structured"
    echo "3. Kernel file (agi_os_kernel.efi) is copied as BOOTX64.EFI"
    echo ""
    echo "Falling back to UEFI shell..."
endif

echo ""
echo "Press any key to continue to UEFI shell..."
pause