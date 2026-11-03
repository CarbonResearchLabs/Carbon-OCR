// Many cores: the CPU's layout, workers pinned one per core, and
// a pool that spreads tasks smaller than a page over all of them.
//
// Each worker owns its own sessions and buffers, set up on its own thread, so
// a task never takes a lock to run a model and the memory sits in its core's
// NUMA node.  Nothing else in the process starts threads: sessions run with
// one thread each (SessionOptions::threads = 1), so the pool is the only
// parallelism and the OS never juggles more threads than there are cores.
#pragma once

#include <functional>
#include <memory>
#include <vector>

namespace carbon {

// One physical core.
struct CpuCore {
  int group = 0;          // its processor group: Windows splits >64 logical processors into groups
  std::vector<int> cpus;  // its logical processors (two with SMT), numbered within the group
  int numa_node = 0;
  int efficiency = 0;     // on hybrid CPUs, higher is faster (P-cores above E-cores); else 0
  // Within its class, how the OS ranks the core (Windows' scheduling class: its favored cores, the highest turbo,
  // above the rest); 0 where it gives none.  Workers take the higher first.
  int rank = 0;
  bool interrupts = false;  // it holds logical processor 0 of group 0, where Windows sends most interrupts: last
};

struct CpuTopology {
  // The cores this process may use (its affinity, a container's cpuset), in the
  // order workers take them: the fastest class first, alternating NUMA nodes.
  std::vector<CpuCore> cores;
  int logical = 0;     // logical processors among those cores
  int groups = 1;      // processor groups (Windows; 1 elsewhere)
  int numa_nodes = 1;
  int cpu_quota = 0;   // whole cores a cgroup CPU quota allows (Linux containers); 0 for none

  // One per physical core, at most the quota, at least 1.
  int default_workers() const;
};

// Read once: GetLogicalProcessorInformationEx on Windows, /sys and the cgroup
// on Linux.  Elsewhere, or if those cannot be read, a core per hardware thread.
const CpuTopology& cpu_topology();

// Pins the calling thread to one core's logical processors.  False where the
// OS cannot or will not; the thread then runs wherever the OS puts it.
bool pin_this_thread(const CpuCore& core);

struct PoolOptions {
  int workers = 0;   // 0: cpu_topology().default_workers()
  bool pin = true;   // worker i on cpu_topology().cores[i % cores], each on its own core
  // The workers that count as fast: the first `fast` of them.  -1: those on
  // cores of the fastest class (every worker, unless the CPU is hybrid or the
  // pool unpinned).
  int fast = -1;
  // Whether a slow worker also starts tasks from outside the pool (new pages), or
  // only takes the tasks a running task queued (a page's lines): a new task then
  // always starts on a fast core, where it is quickest (the engine's choice).
  bool slow_takes_new = true;
};

// Worker threads that run tasks.  A task gets the index of the worker running
// it (0 to size() - 1), to reach that worker's sessions and buffers.  Tasks
// from outside the pool queue in order; a task that submits tasks (a page its
// line batches) puts them on its own worker's queue, which that worker runs
// newest first.  A worker with nothing of its own takes the oldest task of
// another (work stealing) before it starts a new one from outside, so a few
// pages spread over every core and the pages in flight finish first.  On a
// hybrid CPU a worker on a slower core takes a task only while every fast
// worker is running one, so a lone page runs on the fast cores and a full load
// on all of them; with slow_takes_new false it takes only a running task's
// tasks, never a new one.  Idle workers sleep; they never spin.
class WorkerPool {
 public:
  using Task = std::function<void(int worker)>;

  explicit WorkerPool(PoolOptions options = {});
  // Runs every task still queued, then joins the workers.  Not from a task.
  ~WorkerPool();
  WorkerPool(const WorkerPool&) = delete;
  WorkerPool& operator=(const WorkerPool&) = delete;

  int size() const;
  int pinned() const;  // the workers pinned to a core
  int fast() const;    // the workers on the fastest cores
  bool is_fast(int worker) const;
  // The fast workers asleep now, less the tasks already queued for them: how many
  // more tasks could start at once on the fastest cores.  A hint (it changes as
  // tasks start and end), for splitting a page's work to the cores that are free.
  int idle() const;

  void submit(Task task);

  // Waits until every task submitted so far has run, and every task they
  // submitted.  Rethrows the first exception a task threw; the pool stays
  // usable.  Not from a task.
  void wait();

  // Runs f(worker) once on each worker's own thread and waits: where each
  // worker opens its sessions and allocates its buffers.  Rethrows the first
  // exception.  Not from a task.
  void on_each_worker(const std::function<void(int worker)>& f);

  // The worker the calling thread is, or -1 outside this pool.
  int current_worker() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace carbon
