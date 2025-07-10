# Ternary CPU Kernel Stub Makefile
# Based on architecture specification requirements

# Compiler and tools
CC = gcc
LD = ld
OBJCOPY = objcopy

# Architecture target
ARCH = x86_64
EFI_ARCH = x86_64

# Directories
SRC_DIR = ../src
BUILD_DIR = .
IMAGES_DIR = ../images

# GNU-EFI paths (actual system paths)
EFI_INC = /usr/include/efi
EFI_INC_ARCH = /usr/include/efi/$(ARCH)
EFI_LIB = /usr/lib
EFI_CRT_OBJS = $(EFI_LIB)/crt0-efi-$(ARCH).o
EFI_LDS = $(EFI_LIB)/elf_$(ARCH)_efi.lds

# Required compilation flags from architecture specification
CFLAGS = -std=c11 -ffreestanding -fno-stack-protector -fno-stack-check
CFLAGS += -fno-strict-aliasing -fno-merge-all-constants -fno-unwind-tables
CFLAGS += -fno-asynchronous-unwind-tables -mno-sse -mno-mmx -mno-sse2
CFLAGS += -mno-3dnow -msoft-float -mno-red-zone
CFLAGS += -fpic -fshort-wchar -DGNU_EFI_USE_MS_ABI
CFLAGS += -I$(EFI_INC) -I$(EFI_INC_ARCH) -I$(SRC_DIR)

# Linking requirements from architecture specification
LDFLAGS = -nostdlib -znocombreloc -T $(EFI_LDS) -shared -Bsymbolic
LDFLAGS += -L$(EFI_LIB) $(EFI_CRT_OBJS)
LIBS = -lgnuefi -lefi

# Source files
SOURCES = $(SRC_DIR)/main.c
OBJECTS = main.o
TARGET = ternary-kernel-stub
EFI_TARGET = $(TARGET).efi
SO_TARGET = $(TARGET).so

# Default target
all: $(EFI_TARGET)

# Create images directory
$(IMAGES_DIR):
	mkdir -p $(IMAGES_DIR)

# Compile source files
main.o: $(SRC_DIR)/main.c $(SRC_DIR)/ternary.h $(SRC_DIR)/ipc.h
	$(CC) $(CFLAGS) -c $(SRC_DIR)/main.c -o main.o

# Link to create shared object
$(SO_TARGET): $(OBJECTS)
	$(LD) $(LDFLAGS) $(OBJECTS) -o $(SO_TARGET) $(LIBS)

# Convert to EFI executable
$(EFI_TARGET): $(SO_TARGET) | $(IMAGES_DIR)
	$(OBJCOPY) -j .text -j .sdata -j .data -j .dynamic \
		-j .dynsym -j .rel -j .rela -j .reloc \
		--target=efi-app-$(ARCH) $(SO_TARGET) $(EFI_TARGET)
	cp $(EFI_TARGET) $(IMAGES_DIR)/

# Test with QEMU
test: $(EFI_TARGET)
	@echo "Testing kernel with QEMU..."
	qemu-system-x86_64 -bios /usr/share/ovmf/OVMF.fd -drive format=raw,file=fat:rw:$(IMAGES_DIR) -nographic

# Clean build artifacts
clean:
	rm -f *.o *.so *.efi
	rm -rf $(IMAGES_DIR)

# Install target
install: $(EFI_TARGET)
	@echo "Kernel image installed to $(IMAGES_DIR)/$(EFI_TARGET)"

# Help target
help:
	@echo "Available targets:"
	@echo "  all     - Build the kernel EFI image (default)"
	@echo "  test    - Test the kernel with QEMU"
	@echo "  clean   - Remove build artifacts"
	@echo "  install - Install kernel image to images directory"
	@echo "  help    - Show this help message"

.PHONY: all test clean install help