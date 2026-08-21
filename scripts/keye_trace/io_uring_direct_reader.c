// Minimal O_DIRECT + raw io_uring reader for the Figure 2 path benchmark.
//
// This deliberately uses the kernel ABI instead of liburing so the benchmark
// remains reproducible on the Ubuntu 20.04 host where liburing is unavailable.

#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <linux/io_uring.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <time.h>
#include <unistd.h>

struct direct_reader {
    int ring_fd;
    int file_fd;
    unsigned entries;
    unsigned block_size;
    uint64_t file_size;

    void *sq_ring;
    void *cq_ring;
    struct io_uring_sqe *sqes;
    size_t sq_ring_size;
    size_t cq_ring_size;
    size_t sqes_size;
    int single_mmap;

    unsigned *sq_head;
    unsigned *sq_tail;
    unsigned *sq_mask;
    unsigned *sq_entries;
    unsigned *sq_array;
    unsigned *sq_dropped;

    unsigned *cq_head;
    unsigned *cq_tail;
    unsigned *cq_mask;
    unsigned *cq_overflow;
    struct io_uring_cqe *cqes;

    void *buffers;
    struct iovec *iovecs;
    unsigned *free_slots;
    char error[256];
};

static void set_error(struct direct_reader *reader, const char *message) {
    if (reader == NULL) {
        return;
    }
    snprintf(reader->error, sizeof(reader->error), "%s: %s", message,
             strerror(errno));
}

static uint64_t elapsed_ns(const struct timespec *start,
                           const struct timespec *end) {
    return (uint64_t)(end->tv_sec - start->tv_sec) * 1000000000ULL
           + (uint64_t)(end->tv_nsec - start->tv_nsec);
}

static int enter_ring(int fd, unsigned to_submit, unsigned min_complete) {
    int result;
    do {
        result = (int)syscall(__NR_io_uring_enter, fd, to_submit,
                              min_complete, IORING_ENTER_GETEVENTS,
                              NULL, 0);
    } while (result < 0 && errno == EINTR);
    return result;
}

static void destroy_reader(struct direct_reader *reader) {
    if (reader == NULL) {
        return;
    }
    if (reader->sqes != NULL && reader->sqes != MAP_FAILED) {
        munmap(reader->sqes, reader->sqes_size);
    }
    if (reader->single_mmap) {
        if (reader->sq_ring != NULL && reader->sq_ring != MAP_FAILED) {
            size_t size = reader->sq_ring_size > reader->cq_ring_size
                              ? reader->sq_ring_size
                              : reader->cq_ring_size;
            munmap(reader->sq_ring, size);
        }
    } else {
        if (reader->sq_ring != NULL && reader->sq_ring != MAP_FAILED) {
            munmap(reader->sq_ring, reader->sq_ring_size);
        }
        if (reader->cq_ring != NULL && reader->cq_ring != MAP_FAILED) {
            munmap(reader->cq_ring, reader->cq_ring_size);
        }
    }
    if (reader->ring_fd >= 0) {
        close(reader->ring_fd);
    }
    if (reader->file_fd >= 0) {
        close(reader->file_fd);
    }
    free(reader->buffers);
    free(reader->iovecs);
    free(reader->free_slots);
    free(reader);
}

void *direct_reader_create(const char *path, unsigned queue_depth,
                           unsigned block_size) {
    struct direct_reader *reader = calloc(1, sizeof(*reader));
    if (reader == NULL) {
        return NULL;
    }
    reader->ring_fd = -1;
    reader->file_fd = -1;
    reader->entries = queue_depth;
    reader->block_size = block_size;
    reader->error[0] = '\0';

    if (queue_depth == 0 || block_size == 0
        || (block_size & (block_size - 1)) != 0) {
        errno = EINVAL;
        set_error(reader, "invalid queue depth or block size");
        destroy_reader(reader);
        return NULL;
    }

    reader->file_fd = open(path, O_RDONLY | O_DIRECT | O_CLOEXEC);
    if (reader->file_fd < 0) {
        destroy_reader(reader);
        return NULL;
    }
    struct stat file_stat;
    if (fstat(reader->file_fd, &file_stat) != 0) {
        destroy_reader(reader);
        return NULL;
    }
    reader->file_size = (uint64_t)file_stat.st_size;
    if (reader->file_size < block_size
        || reader->file_size % block_size != 0) {
        errno = EINVAL;
        destroy_reader(reader);
        return NULL;
    }

    struct io_uring_params params;
    memset(&params, 0, sizeof(params));
    reader->ring_fd = (int)syscall(__NR_io_uring_setup,
                                   queue_depth, &params);
    if (reader->ring_fd < 0) {
        destroy_reader(reader);
        return NULL;
    }
    reader->entries = params.sq_entries;
    reader->sq_ring_size = params.sq_off.array
                           + params.sq_entries * sizeof(unsigned);
    reader->cq_ring_size = params.cq_off.cqes
                           + params.cq_entries * sizeof(struct io_uring_cqe);
    reader->sqes_size = params.sq_entries * sizeof(struct io_uring_sqe);
    reader->single_mmap = (params.features & IORING_FEAT_SINGLE_MMAP) != 0;

    size_t sq_map_size = reader->sq_ring_size;
    if (reader->single_mmap && reader->cq_ring_size > sq_map_size) {
        sq_map_size = reader->cq_ring_size;
    }
    reader->sq_ring = mmap(NULL, sq_map_size, PROT_READ | PROT_WRITE,
                           MAP_SHARED | MAP_POPULATE, reader->ring_fd,
                           IORING_OFF_SQ_RING);
    if (reader->sq_ring == MAP_FAILED) {
        destroy_reader(reader);
        return NULL;
    }
    if (reader->single_mmap) {
        reader->cq_ring = reader->sq_ring;
    } else {
        reader->cq_ring = mmap(NULL, reader->cq_ring_size,
                               PROT_READ | PROT_WRITE,
                               MAP_SHARED | MAP_POPULATE, reader->ring_fd,
                               IORING_OFF_CQ_RING);
        if (reader->cq_ring == MAP_FAILED) {
            destroy_reader(reader);
            return NULL;
        }
    }
    reader->sqes = mmap(NULL, reader->sqes_size, PROT_READ | PROT_WRITE,
                        MAP_SHARED | MAP_POPULATE, reader->ring_fd,
                        IORING_OFF_SQES);
    if (reader->sqes == MAP_FAILED) {
        destroy_reader(reader);
        return NULL;
    }

    reader->sq_head = (unsigned *)((char *)reader->sq_ring
                                   + params.sq_off.head);
    reader->sq_tail = (unsigned *)((char *)reader->sq_ring
                                   + params.sq_off.tail);
    reader->sq_mask = (unsigned *)((char *)reader->sq_ring
                                   + params.sq_off.ring_mask);
    reader->sq_entries = (unsigned *)((char *)reader->sq_ring
                                      + params.sq_off.ring_entries);
    reader->sq_array = (unsigned *)((char *)reader->sq_ring
                                    + params.sq_off.array);
    reader->sq_dropped = (unsigned *)((char *)reader->sq_ring
                                      + params.sq_off.dropped);
    reader->cq_head = (unsigned *)((char *)reader->cq_ring
                                   + params.cq_off.head);
    reader->cq_tail = (unsigned *)((char *)reader->cq_ring
                                   + params.cq_off.tail);
    reader->cq_mask = (unsigned *)((char *)reader->cq_ring
                                   + params.cq_off.ring_mask);
    reader->cq_overflow = (unsigned *)((char *)reader->cq_ring
                                       + params.cq_off.overflow);
    reader->cqes = (struct io_uring_cqe *)((char *)reader->cq_ring
                                           + params.cq_off.cqes);

    if (posix_memalign(&reader->buffers, block_size,
                       (size_t)reader->entries * block_size) != 0) {
        errno = ENOMEM;
        destroy_reader(reader);
        return NULL;
    }
    reader->iovecs = calloc(reader->entries, sizeof(*reader->iovecs));
    reader->free_slots = calloc(reader->entries, sizeof(*reader->free_slots));
    if (reader->iovecs == NULL || reader->free_slots == NULL) {
        errno = ENOMEM;
        destroy_reader(reader);
        return NULL;
    }
    for (unsigned slot = 0; slot < reader->entries; ++slot) {
        reader->iovecs[slot].iov_base =
            (char *)reader->buffers + (size_t)slot * block_size;
        reader->iovecs[slot].iov_len = block_size;
        reader->free_slots[slot] = slot;
    }
    return reader;
}

void direct_reader_destroy(void *opaque) {
    destroy_reader((struct direct_reader *)opaque);
}

uint64_t direct_reader_file_size(void *opaque) {
    struct direct_reader *reader = (struct direct_reader *)opaque;
    return reader == NULL ? 0 : reader->file_size;
}

const char *direct_reader_last_error(void *opaque) {
    struct direct_reader *reader = (struct direct_reader *)opaque;
    return reader == NULL ? "direct_reader is null" : reader->error;
}

long long direct_reader_read_random(void *opaque, const uint64_t *offsets,
                                    unsigned count) {
    struct direct_reader *reader = (struct direct_reader *)opaque;
    if (reader == NULL || (count > 0 && offsets == NULL)) {
        return -EINVAL;
    }
    if (count == 0) {
        return 0;
    }
    reader->error[0] = '\0';
    unsigned free_count = reader->entries;
    unsigned submitted = 0;
    unsigned completed = 0;
    unsigned inflight = 0;
    unsigned dropped_before = __atomic_load_n(reader->sq_dropped,
                                               __ATOMIC_ACQUIRE);
    unsigned overflow_before = __atomic_load_n(reader->cq_overflow,
                                                __ATOMIC_ACQUIRE);
    struct timespec start;
    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    while (completed < count) {
        unsigned sq_head = __atomic_load_n(reader->sq_head,
                                           __ATOMIC_ACQUIRE);
        unsigned sq_tail = __atomic_load_n(reader->sq_tail,
                                           __ATOMIC_RELAXED);
        unsigned ring_free = *reader->sq_entries - (sq_tail - sq_head);
        unsigned capacity = reader->entries - inflight;
        unsigned remaining = count - submitted;
        unsigned fill = ring_free < capacity ? ring_free : capacity;
        if (fill > remaining) {
            fill = remaining;
        }

        for (unsigned index = 0; index < fill; ++index) {
            unsigned slot = reader->free_slots[--free_count];
            uint64_t offset = offsets[submitted + index];
            if (offset % reader->block_size != 0
                || offset + reader->block_size > reader->file_size) {
                snprintf(reader->error, sizeof(reader->error),
                         "unaligned or out-of-range offset: %llu",
                         (unsigned long long)offset);
                return -EINVAL;
            }
            unsigned sqe_index = sq_tail & *reader->sq_mask;
            struct io_uring_sqe *sqe = &reader->sqes[sqe_index];
            memset(sqe, 0, sizeof(*sqe));
            sqe->opcode = IORING_OP_READV;
            sqe->fd = reader->file_fd;
            sqe->off = offset;
            sqe->addr = (uint64_t)(uintptr_t)&reader->iovecs[slot];
            sqe->len = 1;
            sqe->user_data = slot;
            reader->sq_array[sqe_index] = sqe_index;
            ++sq_tail;
        }
        if (fill > 0) {
            __atomic_store_n(reader->sq_tail, sq_tail, __ATOMIC_RELEASE);
            submitted += fill;
            inflight += fill;
        }

        sq_head = __atomic_load_n(reader->sq_head, __ATOMIC_ACQUIRE);
        sq_tail = __atomic_load_n(reader->sq_tail, __ATOMIC_ACQUIRE);
        unsigned pending = sq_tail - sq_head;
        if (enter_ring(reader->ring_fd, pending, 1) < 0) {
            set_error(reader, "io_uring_enter failed");
            return -errno;
        }

        unsigned cq_head = __atomic_load_n(reader->cq_head,
                                           __ATOMIC_RELAXED);
        unsigned cq_tail = __atomic_load_n(reader->cq_tail,
                                           __ATOMIC_ACQUIRE);
        while (cq_head != cq_tail) {
            struct io_uring_cqe *cqe =
                &reader->cqes[cq_head & *reader->cq_mask];
            if (cqe->res != (int)reader->block_size) {
                int error = cqe->res < 0 ? -cqe->res : EIO;
                snprintf(reader->error, sizeof(reader->error),
                         "short or failed direct read: result=%d", cqe->res);
                return -error;
            }
            unsigned slot = (unsigned)cqe->user_data;
            if (slot >= reader->entries || free_count >= reader->entries) {
                snprintf(reader->error, sizeof(reader->error),
                         "invalid completion slot: %u", slot);
                return -EOVERFLOW;
            }
            reader->free_slots[free_count++] = slot;
            --inflight;
            ++completed;
            ++cq_head;
        }
        __atomic_store_n(reader->cq_head, cq_head, __ATOMIC_RELEASE);
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    if (__atomic_load_n(reader->sq_dropped, __ATOMIC_ACQUIRE)
            != dropped_before
        || __atomic_load_n(reader->cq_overflow, __ATOMIC_ACQUIRE)
            != overflow_before) {
        snprintf(reader->error, sizeof(reader->error),
                 "io_uring queue dropped submissions or completions");
        return -EOVERFLOW;
    }
    return (long long)elapsed_ns(&start, &end);
}
