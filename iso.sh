# Create AGI-OS bootable ISO
$ISO_DIR = "$PSScriptRoot\AGI-OS-ISO"
$ROOTFS = "$ISO_DIR\rootfs"
$GRUB_CFG = @"
set timeout=0
set default=0

menuentry "AGI-OS (Python AGI Runtime)" {
    insmod linuxefi
    insmod chain
    chainloader /kernel/ternary-kernel-stub.efi
}
"@

# Create folder structure
Remove-Item $ISO_DIR -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path "$ISO_DIR\boot\grub" | Out-Null
New-Item -ItemType Directory -Path "$ISO_DIR\EFI\BOOT" | Out-Null
New-Item -ItemType Directory -Path "$ISO_DIR\kernel" | Out-Null
New-Item -ItemType Directory -Path $ROOTFS | Out-Null

# Copy files
Copy-Item "$PSScriptRoot\ternary-kernel-stub.efi" "$ISO_DIR\kernel\ternary-kernel-stub.efi"
Copy-Item "$PSScriptRoot\trainingless_nlp.py" $ROOTFS
Copy-Item "$PSScriptRoot\warp_system.py" $ROOTFS
Copy-Item "$PSScriptRoot\echofs.py" $ROOTFS
Copy-Item "$PSScriptRoot\mount_fs.py" $ROOTFS
Copy-Item "$PSScriptRoot\go.html" $ROOTFS

# Generate grub.cfg
Set-Content "$ISO_DIR\boot\grub\grub.cfg" $GRUB_CFG

# Add UEFI bootloader
Copy-Item "$PSScriptRoot\ternary-kernel-stub.efi" "$ISO_DIR\EFI\BOOT\BOOTX64.EFI"

# Build ISO
$ISO_OUT = "$PSScriptRoot\AGI-OS.iso"
& wsl mkisofs -o "$ISO_OUT" -R -J -no-emul-boot -boot-load-size 4 -boot-info-table `
    -b boot/grub/i386-pc/eltorito.img `
    -eltorito-alt-boot -e EFI/BOOT/BOOTX64.EFI -no-emul-boot `
    "$ISO_DIR"

Write-Host "✅ ISO created at: $ISO_OUT"