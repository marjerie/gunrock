#pragma once

#include <linux/perf_event.h>    /* Definition of PERF_* constants */
#include <linux/hw_breakpoint.h> /* Definition of HW_* constants */
#include <sys/syscall.h>         /* Definition of SYS_* constants */
#include <sys/ioctl.h>
#include <unistd.h>
#include <stdio.h>
#include <cassert>

namespace gunrock {
namespace util {

static long
perf_event_open(struct perf_event_attr *hw_event, pid_t pid,
                int cpu, int group_fd, unsigned long flags)
{
    int ret;

    ret = syscall(__NR_perf_event_open, hw_event, pid, cpu,
                    group_fd, flags);
    return ret;
}

struct perf_cpu_t {

    int fd;
    long long count;
    // struct perf_event_attr pe;

    perf_cpu_t() {
        int cpu_num = sched_getcpu();
        printf("Running on CPU %d\n", cpu_num);
        struct perf_event_attr pe;
        memset(&pe, 0, sizeof(pe));
        pe.type = PERF_TYPE_HARDWARE;
        pe.size = sizeof(pe);
        // pe.config = PERF_COUNT_HW_REF_CPU_CYCLES;
        pe.config = PERF_COUNT_HW_CACHE_MISSES;
        // pe.exclude_idle = 0; // count when CPU is idle
        // pe.exclude_user = 0; // count when user is active
        pe.disabled = 1;
        pe.exclude_kernel = 1;
        pe.exclude_hv = 1;

        fd = perf_event_open(&pe, 0, cpu_num, -1, 0);
        if (fd == -1) {
            fprintf(stderr, "Error opening leader %llx\n", pe.config);
            exit(EXIT_FAILURE);
        }

    }

    ~perf_cpu_t() {
        close(fd);
    }

    void start() {
        ioctl(fd, PERF_EVENT_IOC_RESET, 0);
        ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
    }

    void stop() {
        ioctl(fd, PERF_EVENT_IOC_DISABLE, 0);
        read(fd, &count, sizeof(count));
    }

};

}  // namespace util
}  // namespace gunrock