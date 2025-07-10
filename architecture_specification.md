# Kernel Architecture Specification
## Ternary CPU Kernel Stub with UEFI Boot and Message-Passing IPC

### Document Overview
This specification defines the concrete architecture for implementing a bootable kernel stub featuring a ternary virtual CPU, message-passing IPC system, and NixOS-based reproducible builds. All specifications are derived from verified technical research findings.

---

## 1. Kernel Components Architecture

### 1.1 Overall System Structure
```
Kernel Boot Sequence:
UEFI Firmware → EFI Boot Stub → Kernel Initialization → Ternary CPU Init → IPC System Init
```

### 1.2 Core Components
- **EFI Boot Stub**: UEFI-compatible entry point with minimal hardware initialization
- **Ternary Virtual CPU**: Three-state logic processor with 27 registers and 18-trit instruction encoding
- **Message-Passing IPC**: Kernel-mediated communication system with ring buffer implementation
- **Memory Manager**: Basic memory allocation and boundary enforcement

### 1.3 Component Relationships
- Boot stub initializes memory management before CPU initialization
- Ternary CPU operates independently with IPC for inter-component communication
- All IPC communication flows through kernel space for security enforcement
- Memory boundaries strictly enforced between all components

---

## 2. Ternary CPU Instruction Basics

### 2.1 Register Architecture
```c
// Ternary register file specification
typedef struct {
    int8_t trits[6];  // 6 trits per register (18 bits equivalent)
} ternary_register_t;

typedef struct {
    ternary_register_t registers[27];  // 27 registers (3^3 addressing)
    ternary_register_t pc;             // Program counter
    ternary_register_t sp;             // Stack pointer
} ternary_cpu_state_t;
```

### 2.2 Instruction Set Design
- **Instruction Length**: 18 trits (6 trits opcode + 3×3 trits for register addressing + 3 trits immediate)
- **Addressing Mode**: 3-trit register addressing (supports 27 registers)
- **Three-State Logic**: Operations on values {-1, 0, 1}

### 2.3 Core Instructions
```c
// Instruction opcodes (6-trit encoding)
typedef enum {
    TERNARY_NOP    = 0,     // No operation
    TERNARY_LOAD   = 1,     // Load from memory
    TERNARY_STORE  = 2,     // Store to memory
    TERNARY_ADD    = 3,     // Ternary addition
    TERNARY_SUB    = 4,     // Ternary subtraction
    TERNARY_AND    = 5,     // Ternary logical AND
    TERNARY_OR     = 6,     // Ternary logical OR
    TERNARY_NOT    = 7,     // Ternary logical NOT
    TERNARY_JMP    = 8,     // Unconditional jump
    TERNARY_JZ     = 9,     // Jump if zero
    TERNARY_HALT   = 10     // Halt execution
} ternary_opcode_t;
```

### 2.4 Execution Loop Structure
```c
// CPU execution cycle function signature
typedef struct {
    uint32_t (*fetch)(ternary_cpu_state_t *cpu);
    void (*decode)(uint32_t instruction, ternary_instruction_t *decoded);
    void (*execute)(ternary_cpu_state_t *cpu, ternary_instruction_t *instruction);
} ternary_cpu_ops_t;
```

---

## 3. IPC Protocol Specification

### 3.1 Message Queue Structure
```c
// Message structure definition
typedef struct {
    uint32_t sender_id;
    uint32_t receiver_id;
    uint32_t message_type;
    uint32_t data_length;
    uint8_t data[256];      // Fixed-size message payload
    uint64_t timestamp;
} ipc_message_t;

// Message queue implementation
typedef struct {
    ipc_message_t *buffer;
    uint32_t head;
    uint32_t tail;
    uint32_t size;
    uint32_t count;
} ipc_message_queue_t;
```

### 3.2 Ring Buffer Implementation
```c
// Ring buffer with atomic operations
typedef struct {
    volatile uint32_t producer_seq;
    volatile uint32_t consumer_seq;
    uint32_t buffer_size;
    ipc_message_t *messages;
} ipc_ring_buffer_t;

// Ring buffer operations
int ipc_ring_buffer_send(ipc_ring_buffer_t *ring, ipc_message_t *msg);
int ipc_ring_buffer_receive(ipc_ring_buffer_t *ring, ipc_message_t *msg);
```

### 3.3 Communication Primitives
```c
// IPC system interface
typedef struct {
    int (*send_message)(uint32_t dest_id, ipc_message_t *msg);
    int (*receive_message)(uint32_t src_id, ipc_message_t *msg);
    int (*create_queue)(uint32_t queue_id, uint32_t size);
    int (*destroy_queue)(uint32_t queue_id);
} ipc_interface_t;
```

### 3.4 Memory Boundary Enforcement
- All message passing occurs through kernel-allocated buffers
- Direct memory access between processes prohibited
- Kernel validates all message boundaries and sizes
- Memory allocation tracking for each IPC queue

---

## 4. Build Requirements

### 4.1 Compilation Specifications
```makefile
# Required compilation flags
CFLAGS = -std=c11 -ffreestanding -fno-stack-protector -fno-stack-check
CFLAGS += -fno-strict-aliasing -fno-merge-all-constants -fno-unwind-tables
CFLAGS += -fno-asynchronous-unwind-tables -mno-sse -mno-mmx -mno-sse2
CFLAGS += -mno-3dnow -msoft-float -mno-red-zone

# Linking requirements
LDFLAGS = -nostdlib -znocombreloc -T elf_x86_64_efi.lds -shared -Bsymbolic
LIBS = -lgnuefi -lefi
```

### 4.2 NixOS Flake Configuration
```nix
{
  description = "Ternary CPU Kernel Stub";
  
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };
  
  outputs = { self, nixpkgs, ... }: {
    packages.x86_64-linux.default = nixpkgs.legacyPackages.x86_64-linux.stdenv.mkDerivation {
      name = "ternary-kernel-stub";
      src = ./.;
      
      nativeBuildInputs = with nixpkgs.legacyPackages.x86_64-linux; [
        gcc
        gnu-efi
        qemu
        xorriso
      ];
      
      buildPhase = ''
        make all
      '';
      
      installPhase = ''
        mkdir -p $out/images
        cp build/*.efi $out/images/
        cp build/*.iso $out/images/
      '';
    };
  };
}
```

### 4.3 Build Dependencies
- **GNU-EFI Library**: For UEFI development support
- **Cross-compilation GCC**: 64-bit x86_64 target
- **QEMU**: For testing and emulation
- **xorriso**: For ISO image generation

---

## 5. Memory Layout Specification

### 5.1 Kernel Memory Map
```c
// Memory layout constants
#define KERNEL_BASE_ADDR    0x100000    // 1MB
#define KERNEL_STACK_SIZE   0x10000     // 64KB
#define IPC_BUFFER_BASE     0x200000    // 2MB
#define IPC_BUFFER_SIZE     0x100000    // 1MB
#define TERNARY_CPU_MEM     0x300000    // 3MB
#define TERNARY_MEM_SIZE    0x100000    // 1MB
```

### 5.2 Memory Management Interface
```c
// Memory allocation interface
typedef struct {
    void* (*alloc)(size_t size);
    void (*free)(void *ptr);
    void* (*alloc_aligned)(size_t size, size_t alignment);
    int (*map_memory)(uint64_t physical, uint64_t virtual, size_t size);
} memory_manager_t;
```

---

## 6. Boot Sequence Specification

### 6.1 UEFI Entry Point
```c
// Required UEFI entry point signature
EFI_STATUS EFIAPI efi_main(
    EFI_HANDLE ImageHandle,
    EFI_SYSTEM_TABLE *SystemTable
);
```

### 6.2 Initialization Sequence
1. **UEFI Boot Stub Entry**: Initialize EFI system table and basic services
2. **Memory Initialization**: Set up GDT, basic memory mapping, and allocation
3. **Console Setup**: Initialize `SIMPLE_TEXT_OUTPUT_PROTOCOL` for output
4. **Ternary CPU Init**: Initialize virtual CPU state and instruction decoder
5. **IPC System Init**: Create message queues and ring buffers
6. **Kernel Main Loop**: Enter main execution loop with CPU scheduling

### 6.3 Error Handling
- All initialization functions must return status codes
- Failed initialization must halt system with diagnostic output
- Memory allocation failures must be handled gracefully

---

## 7. File Structure Specification

### 7.1 Source Code Organization
```
src/
├── main.c              # UEFI entry point and kernel initialization
├── ternary.h           # Ternary CPU definitions and interfaces
├── ternary.c           # Ternary CPU implementation
├── ipc.h               # IPC system definitions and interfaces
├── ipc.c               # IPC system implementation
├── memory.h            # Memory management interfaces
└── memory.c            # Memory management implementation
```

### 7.2 Build Configuration
```
build/
├── Makefile            # Build automation with UEFI linking
├── flake.nix           # NixOS reproducible build configuration
└── elf_x86_64_efi.lds  # EFI linker script
```

---

## 8. Testing and Validation Requirements

### 8.1 Boot Testing
- Must boot successfully on QEMU with UEFI firmware
- Console output must display initialization progress
- System must halt gracefully on completion

### 8.2 Ternary CPU Testing
- Execute basic arithmetic operations with three-state logic
- Validate register addressing and instruction decoding
- Test jump instructions and program counter management

### 8.3 IPC Testing
- Send and receive messages between kernel components
- Validate ring buffer overflow and underflow handling
- Test atomic operations under concurrent access

---

## 9. Implementation Constraints

### 9.1 Technical Constraints
- **No Standard Library**: Kernel must be freestanding without libc
- **64-bit Architecture**: Target x86_64 UEFI systems only
- **Memory Limitations**: Minimal memory footprint required
- **Real-time Constraints**: IPC operations must be deterministic

### 9.2 Security Constraints
- All inter-component communication through IPC system
- Memory boundary enforcement between all components
- No direct memory access between kernel modules
- Input validation for all IPC messages

---

## 10. Success Criteria

### 10.1 Functional Requirements
- ✅ Bootable .efi image that loads via UEFI
- ✅ Working ternary virtual CPU with instruction execution
- ✅ Functional message-passing IPC system
- ✅ Reproducible builds using NixOS flake

### 10.2 Code Quality Requirements
- Clean C code following kernel development practices
- Comprehensive error handling and validation
- Proper memory management without leaks
- Documented interfaces and data structures

This specification provides concrete, implementable guidance for developing the ternary CPU kernel stub based on verified research findings. All data structures, function signatures, and technical constraints are derived from the technical research documentation.