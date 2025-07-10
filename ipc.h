#ifndef IPC_H
#define IPC_H

#include <stdint.h>
#include <stddef.h>

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

// Ring buffer with atomic operations
typedef struct {
    volatile uint32_t producer_seq;
    volatile uint32_t consumer_seq;
    uint32_t buffer_size;
    ipc_message_t *messages;
} ipc_ring_buffer_t;

// IPC system interface
typedef struct {
    int (*send_message)(uint32_t dest_id, ipc_message_t *msg);
    int (*receive_message)(uint32_t src_id, ipc_message_t *msg);
    int (*create_queue)(uint32_t queue_id, uint32_t size);
    int (*destroy_queue)(uint32_t queue_id);
} ipc_interface_t;

// Memory layout constants for IPC
#define IPC_BUFFER_BASE     0x200000    // 2MB
#define IPC_BUFFER_SIZE     0x100000    // 1MB

// IPC message types
typedef enum {
    IPC_MSG_KERNEL_INIT = 1,
    IPC_MSG_CPU_COMMAND = 2,
    IPC_MSG_MEMORY_REQUEST = 3,
    IPC_MSG_SCHEDULER_UPDATE = 4,
    IPC_MSG_ERROR = 5,
    IPC_MSG_SHUTDOWN = 6
} ipc_message_type_t;

// IPC error codes
typedef enum {
    IPC_SUCCESS = 0,
    IPC_ERROR_INVALID_ID = -1,
    IPC_ERROR_QUEUE_FULL = -2,
    IPC_ERROR_QUEUE_EMPTY = -3,
    IPC_ERROR_INVALID_MESSAGE = -4,
    IPC_ERROR_MEMORY_BOUNDARY = -5,
    IPC_ERROR_PERMISSION_DENIED = -6
} ipc_error_t;

// Function declarations for ring buffer operations
int ipc_ring_buffer_init(ipc_ring_buffer_t *ring, uint32_t size);
int ipc_ring_buffer_send(ipc_ring_buffer_t *ring, ipc_message_t *msg);
int ipc_ring_buffer_receive(ipc_ring_buffer_t *ring, ipc_message_t *msg);
void ipc_ring_buffer_destroy(ipc_ring_buffer_t *ring);

// Function declarations for message queue operations
int ipc_queue_init(ipc_message_queue_t *queue, uint32_t size);
int ipc_queue_enqueue(ipc_message_queue_t *queue, ipc_message_t *msg);
int ipc_queue_dequeue(ipc_message_queue_t *queue, ipc_message_t *msg);
int ipc_queue_is_empty(ipc_message_queue_t *queue);
int ipc_queue_is_full(ipc_message_queue_t *queue);
void ipc_queue_destroy(ipc_message_queue_t *queue);

// Function declarations for IPC system
int ipc_system_init(void);
int ipc_system_shutdown(void);
int ipc_send_message(uint32_t dest_id, ipc_message_t *msg);
int ipc_receive_message(uint32_t src_id, ipc_message_t *msg);
int ipc_create_queue(uint32_t queue_id, uint32_t size);
int ipc_destroy_queue(uint32_t queue_id);

// Memory boundary enforcement functions
int ipc_validate_message_bounds(ipc_message_t *msg);
int ipc_validate_queue_bounds(ipc_message_queue_t *queue);
int ipc_check_memory_access(void *ptr, size_t size);

// Utility functions
uint64_t ipc_get_timestamp(void);
int ipc_validate_message(ipc_message_t *msg);
void ipc_clear_message(ipc_message_t *msg);

#endif // IPC_H