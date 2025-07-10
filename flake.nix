{
  description = "Ternary CPU Kernel Stub";
  
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };
  
  outputs = { self, nixpkgs, ... }: {
    packages.x86_64-linux.default = nixpkgs.legacyPackages.x86_64-linux.stdenv.mkDerivation {
      name = "ternary-kernel-stub";
      src = ../.;
      
      nativeBuildInputs = with nixpkgs.legacyPackages.x86_64-linux; [
        gcc
        gnu-efi
        qemu
        xorriso
        binutils
        make
      ];
      
      buildPhase = ''
        cd build
        make all
      '';
      
      installPhase = ''
        mkdir -p $out/images
        cp build/*.efi $out/images/ 2>/dev/null || true
        cp build/*.iso $out/images/ 2>/dev/null || true
      '';
    };

    devShells.x86_64-linux.default = nixpkgs.legacyPackages.x86_64-linux.mkShell {
      buildInputs = with nixpkgs.legacyPackages.x86_64-linux; [
        gcc
        gnu-efi
        qemu
        xorriso
        binutils
        make
        gdb
      ];
      
      shellHook = ''
        echo "Ternary CPU Kernel Development Environment"
        echo "Available tools: gcc, gnu-efi, qemu, make"
        echo "Run 'make' in the build directory to compile the kernel"
      '';
    };
  };
}