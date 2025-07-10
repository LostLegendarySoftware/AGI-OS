#ifndef TERNARY_H
#define TERNARY_H

#include <stdint.h>

// Ternary register file specification
typedef struct {
    int8_t trits[6];  // 6 trits per register (18 bits equivalent)
} ternary_register_t;

typedef struct {
    ternary_register_t registers[27];  // 27 registers (3^3 addressing)
    ternary_register_t pc;             // Program counter
    ternary_register_t sp;             // Stack pointer
} ternary_cpu_state_t;

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

// Instruction structure
typedef struct {
    ternary_opcode_t opcode;
    uint8_t reg1;
    uint8_t reg2;
    uint8_t reg3;
    int8_t immediate;
} ternary_instruction_t;

// CPU execution cycle function signature
typedef struct {
    uint32_t (*fetch)(ternary_cpu_state_t *cpu);
    void (*decode)(uint32_t instruction, ternary_instruction_t *decoded);
    void (*execute)(ternary_cpu_state_t *cpu, ternary_instruction_t *instruction);
} ternary_cpu_ops_t;

// Function declarations
void ternary_cpu_init(ternary_cpu_state_t *cpu);
void ternary_cpu_reset(ternary_cpu_state_t *cpu);
int ternary_cpu_step(ternary_cpu_state_t *cpu, ternary_cpu_ops_t *ops);
void ternary_cpu_run(ternary_cpu_state_t *cpu, ternary_cpu_ops_t *ops);

// Ternary arithmetic operations
int8_t ternary_add(int8_t a, int8_t b);
int8_t ternary_sub(int8_t a, int8_t b);
int8_t ternary_and(int8_t a, int8_t b);
int8_t ternary_or(int8_t a, int8_t b);
int8_t ternary_not(int8_t a);

// Register operations
void ternary_reg_set(ternary_register_t *reg, int8_t *trits);
void ternary_reg_get(ternary_register_t *reg, int8_t *trits);
int ternary_reg_is_zero(ternary_register_t *reg);

// Instruction encoding/decoding
uint32_t ternary_encode_instruction(ternary_instruction_t *instr);
void ternary_decode_instruction(uint32_t encoded, ternary_instruction_t *instr);

#endif // TERNARY_H