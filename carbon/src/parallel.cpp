// The CPU's layout, pinning, and the work-stealing pool (carbon/parallel.h).

#include "carbon/parallel.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>
#include <latch>
#include <map>
#include <mutex>
#include <set>
#include <thread>
#include <utility>

#include "carbon/error.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <xmmintrin.h>
#endif

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#elif defined(__linux__)
#include <pthread.h>
#include <sched.h>

#include <filesystem>
#include <fstream>
#include <string>
#endif

namespace carbon {

namespace {

// IEEE arithmetic on the calling thread: denormals kept, not flushed to zero (MXCSR's FTZ and DAZ, bits 15 and 6;
// Arm's FPCR FZ and FZ16, bits 24 and 19, cleared). Every worker the same, so a page's bits do not depend on which
// worker read it: ONNX Runtime's set_denormal_as_zero reached only the thread that made the process's first
// session. Flushing would not be the same on every ISA either (x86 detects tininess after rounding, Arm before).
void ieee_denormals() {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
  _mm_setcsr(_mm_getcsr() & ~0x8040u);
#elif defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
  uint64_t fpcr;
  __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
  fpcr &= ~((uint64_t{1} << 24) | (uint64_t{1} << 19));
  __asm__ volatile("msr fpcr, %0" : : "r"(fpcr));
#endif
}


// Workers take cores fastest class first and, within a class, one NUMA node
// after another, so a pool smaller than the machine spreads its memory traffic;
// within a class and node, the OS's favored cores first and the interrupts'
// core last (a lone pinned worker on logical processor 0 of a laptop ran a page
// 25% slower than unpinned).
void order_cores(std::vector<CpuCore>& cores) {
  std::stable_sort(cores.begin(), cores.end(), [](const CpuCore& a, const CpuCore& b) {
    if (a.efficiency != b.efficiency) return a.efficiency > b.efficiency;
    if (a.rank != b.rank) return a.rank > b.rank;
    return !a.interrupts && b.interrupts;
  });
  std::map<std::pair<int, int>, int> seen;  // (efficiency, node) -> cores so far
  std::vector<std::pair<int, CpuCore>> ranked;
  for (auto& c : cores) {
    int rank = seen[{c.efficiency, c.numa_node}]++;
    ranked.emplace_back(rank, std::move(c));
  }
  std::stable_sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
    if (a.second.efficiency != b.second.efficiency) return a.second.efficiency > b.second.efficiency;
    if (a.first != b.first) return a.first < b.first;
    return a.second.numa_node < b.second.numa_node;
  });
  cores.clear();
  for (auto& rc : ranked) cores.push_back(std::move(rc.second));
}

// A core per hardware thread, when nothing better can be read.
CpuTopology fallback_topology() {
  CpuTopology t;
  int n = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
  for (int i = 0; i < n; ++i) t.cores.push_back({0, {i}, 0, 0});
  t.logical = n;
  return t;
}

#if defined(_WIN32)

CpuTopology read_topology() {
  DWORD bytes = 0;
  GetLogicalProcessorInformationEx(RelationAll, nullptr, &bytes);
  if (GetLastError() != ERROR_INSUFFICIENT_BUFFER || bytes == 0) return fallback_topology();
  std::vector<char> buf(bytes);
  auto* first = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data());
  if (!GetLogicalProcessorInformationEx(RelationAll, first, &bytes)) return fallback_topology();

  struct Node {
    int number;
    WORD group;
    KAFFINITY mask;
  };
  std::vector<Node> nodes;
  CpuTopology t;
  for (DWORD at = 0; at < bytes;) {
    auto* info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data() + at);
    if (info->Relationship == RelationProcessorCore) {
      // A core lies in one group; GroupMask[0] is its logical processors.
      const GROUP_AFFINITY& g = info->Processor.GroupMask[0];
      CpuCore c;
      c.group = g.Group;
      for (int bit = 0; bit < 64; ++bit) {
        if (g.Mask & (KAFFINITY{1} << bit)) c.cpus.push_back(bit);
      }
      c.efficiency = info->Processor.EfficiencyClass;
      t.cores.push_back(std::move(c));
    } else if (info->Relationship == RelationNumaNode) {
      // A node can span groups since Windows 11 and Server 2022; before, GroupCount
      // was reserved (0) and GroupMask the node's one group.
      const auto& node = info->NumaNode;
      for (WORD i = 0; i < std::max<WORD>(1, node.GroupCount); ++i) {
        nodes.push_back({static_cast<int>(node.NodeNumber), node.GroupMasks[i].Group, node.GroupMasks[i].Mask});
      }
    }
    at += info->Size;
  }
  if (t.cores.empty()) return fallback_topology();
  std::set<int> node_numbers;
  for (auto& c : t.cores) {
    for (const Node& n : nodes) {
      if (n.group == c.group && (n.mask & (KAFFINITY{1} << c.cpus.front()))) c.numa_node = n.number;
    }
    node_numbers.insert(c.numa_node);
    t.logical += static_cast<int>(c.cpus.size());
  }
  t.groups = std::max<int>(1, GetActiveProcessorGroupCount());
  t.numa_nodes = static_cast<int>(node_numbers.size());
  // Each core's rank: the highest scheduling class of its logical processors (GetSystemCpuSetInformation).
  ULONG need = 0;
  GetSystemCpuSetInformation(nullptr, 0, &need, GetCurrentProcess(), 0);
  if (need > 0) {
    std::vector<char> sets(need);
    if (GetSystemCpuSetInformation(reinterpret_cast<SYSTEM_CPU_SET_INFORMATION*>(sets.data()), need, &need,
                                   GetCurrentProcess(), 0)) {
      for (ULONG at = 0; at < need;) {
        const auto* s = reinterpret_cast<const SYSTEM_CPU_SET_INFORMATION*>(sets.data() + at);
        if (s->Type == CpuSetInformation) {
          for (CpuCore& c : t.cores) {
            if (c.group != s->CpuSet.Group) continue;
            for (int cpu : c.cpus) {
              if (cpu == s->CpuSet.LogicalProcessorIndex) c.rank = std::max(c.rank, int{s->CpuSet.SchedulingClass});
            }
          }
        }
        if (s->Size == 0) break;
        at += s->Size;
      }
    }
  }
  for (CpuCore& c : t.cores) c.interrupts = c.group == 0 && !c.cpus.empty() && c.cpus.front() == 0;
  return t;
}

#elif defined(__linux__)

long read_long(const std::filesystem::path& p, long fallback) {
  std::ifstream f(p);
  long v = fallback;
  if (f && (f >> v)) return v;
  return fallback;
}

// The whole cores a cgroup CPU quota allows: cgroup v2's cpu.max ("max 100000"
// or "400000 100000"), else v1's cfs_quota_us and cfs_period_us.
int read_cpu_quota() {
  std::ifstream v2("/sys/fs/cgroup/cpu.max");
  std::string quota;
  long period = 0;
  if (v2 && (v2 >> quota >> period)) {
    if (quota == "max" || period <= 0) return 0;
    try {
      return std::max(1, static_cast<int>(std::stol(quota) / period));
    } catch (const std::exception&) {
      return 0;
    }
  }
  long q = read_long("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", -1);
  long p = read_long("/sys/fs/cgroup/cpu/cpu.cfs_period_us", 0);
  if (q <= 0 || p <= 0) return 0;
  return std::max(1, static_cast<int>(q / p));
}

CpuTopology read_topology() {
  cpu_set_t allowed;
  CPU_ZERO(&allowed);
  if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) return fallback_topology();

  namespace fs = std::filesystem;
  std::map<std::pair<long, long>, CpuCore> by_core;  // (package, core id) -> core
  std::map<std::pair<long, long>, long> speed;       // a proxy for the core's class
  std::set<int> node_numbers;
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (!CPU_ISSET(cpu, &allowed)) continue;
    fs::path base = fs::path("/sys/devices/system/cpu") / ("cpu" + std::to_string(cpu));
    long package = read_long(base / "topology" / "physical_package_id", 0);
    long core_id = read_long(base / "topology" / "core_id", cpu);
    // cpu_capacity where the kernel gives it (hybrid and big.LITTLE parts),
    // else the highest clock: P-cores boost above E-cores.
    long s = read_long(base / "cpu_capacity", read_long(base / "cpufreq" / "cpuinfo_max_freq", 0));
    int node = 0;
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(base, ec)) {
      std::string name = e.path().filename().string();
      if (name.size() > 4 && name.starts_with("node") &&
          name.find_first_not_of("0123456789", 4) == std::string::npos) {
        node = std::stoi(name.substr(4));
      }
    }
    CpuCore& c = by_core[{package, core_id}];
    c.cpus.push_back(cpu);
    c.numa_node = node;
    speed[{package, core_id}] = std::max(speed[{package, core_id}], s);
    node_numbers.insert(node);
  }
  if (by_core.empty()) return fallback_topology();

  std::set<long> classes;
  for (const auto& ks : speed) classes.insert(ks.second);
  CpuTopology t;
  for (auto& [k, c] : by_core) {
    c.efficiency = static_cast<int>(std::distance(classes.begin(), classes.find(speed[k])));
    t.logical += static_cast<int>(c.cpus.size());
    t.cores.push_back(std::move(c));
  }
  t.numa_nodes = static_cast<int>(node_numbers.size());
  t.cpu_quota = read_cpu_quota();
  return t;
}

#else

CpuTopology read_topology() { return fallback_topology(); }

#endif

}  // namespace

int CpuTopology::default_workers() const {
  int n = static_cast<int>(cores.size());
  if (cpu_quota > 0) n = std::min(n, cpu_quota);
  return std::max(1, n);
}

const CpuTopology& cpu_topology() {
  static const CpuTopology t = [] {
    CpuTopology r = read_topology();
    order_cores(r.cores);
    return r;
  }();
  return t;
}

bool pin_this_thread(const CpuCore& core) {
  if (core.cpus.empty()) return false;
#if defined(_WIN32)
  GROUP_AFFINITY g{};
  g.Group = static_cast<WORD>(core.group);
  for (int cpu : core.cpus) g.Mask |= KAFFINITY{1} << cpu;
  return SetThreadGroupAffinity(GetCurrentThread(), &g, nullptr) != 0;
#elif defined(__linux__)
  cpu_set_t set;
  CPU_ZERO(&set);
  for (int cpu : core.cpus) CPU_SET(cpu, &set);
  return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
#else
  return false;
#endif
}

struct WorkerPool::Impl {
  struct Worker {
    std::mutex m;
    std::deque<Task> tasks;                     // its own: it pops the front, thieves the back
    std::deque<std::function<void()>> mailbox;  // on_each_worker's, for this worker only
    std::atomic<size_t> queued{0};              // tasks.size(), read without the lock
    std::atomic<size_t> mail{0};
    bool fast = true;                           // on a core of the fastest class
  };

  std::vector<std::unique_ptr<Worker>> workers;
  std::mutex outside_m;
  std::deque<Task> outside;         // from outside the pool, oldest first
  std::atomic<int64_t> queued{0};   // tasks waiting in any queue (not the mailboxes)
  std::atomic<int64_t> inside{0};   // of them, those in the workers' own queues (a running task's)
  bool slow_takes_new = true;
  std::atomic<int64_t> pending{0};  // tasks submitted and not yet finished
  // Sleeping: fast workers wake for any task; a slow worker takes one only while
  // every fast worker is running one (awake is not enough: a burst of tasks
  // wakes them all before they start).  sleeping_fast changes under sleep_m.
  std::mutex sleep_m;
  std::condition_variable wake_fast, wake_slow;
  std::atomic<int> sleeping_fast{0}, busy_fast{0};
  int fast = 0;

  bool fast_all_busy() const { return busy_fast.load() >= fast; }
  std::mutex done_m;
  std::condition_variable done;
  std::mutex error_m;
  std::exception_ptr error;
  std::atomic<bool> stop{false};
  std::atomic<int> pinned{0};
  std::vector<std::thread> threads;

  bool take(int w, Task& task, std::function<void()>& mail) {
    Worker& me = *workers[w];
    if (me.mail.load() > 0) {
      std::lock_guard lk(me.m);
      if (!me.mailbox.empty()) {
        mail = std::move(me.mailbox.front());
        me.mailbox.pop_front();
        me.mail.fetch_sub(1);
        return true;
      }
    }
    if (!me.fast && !fast_all_busy()) return false;  // a fast worker is free for it
    if (me.queued.load() > 0) {
      std::lock_guard lk(me.m);
      if (!me.tasks.empty()) {
        task = std::move(me.tasks.front());
        me.tasks.pop_front();
        me.queued.fetch_sub(1);
        queued.fetch_sub(1);
        inside.fetch_sub(1);
        return true;
      }
    }
    const int n = static_cast<int>(workers.size());
    for (int k = 1; k < n; ++k) {
      Worker& other = *workers[(w + k) % n];
      if (other.queued.load() == 0) continue;
      std::lock_guard lk(other.m);
      if (!other.tasks.empty()) {
        task = std::move(other.tasks.back());
        other.tasks.pop_back();
        other.queued.fetch_sub(1);
        queued.fetch_sub(1);
        inside.fetch_sub(1);
        return true;
      }
    }
    if (!me.fast && !slow_takes_new) return false;  // a new task starts on a fast core
    std::lock_guard lk(outside_m);
    if (!outside.empty()) {
      task = std::move(outside.front());
      outside.pop_front();
      queued.fetch_sub(1);
      return true;
    }
    return false;
  }

  // A task was queued: wake a fast worker if one sleeps, else a slow one if every
  // fast worker is busy (an awake, idle fast worker takes it itself).  Under
  // sleep_m, so a worker between its check and its wait sees the change.
  void notify_task() {
    std::lock_guard lk(sleep_m);
    if (sleeping_fast.load() > 0) {
      wake_fast.notify_one();
    } else if (fast_all_busy() && slow_work()) {
      wake_slow.notify_one();
    }
  }
  // Whether a queued task is one a slow worker may take.
  bool slow_work() const { return (slow_takes_new ? queued.load() : inside.load()) > 0; }

  void notify_all() {
    std::lock_guard lk(sleep_m);
    wake_fast.notify_all();
    wake_slow.notify_all();
  }

  // Sleeps until there is work for this worker; false when the pool stops.  A woken
  // worker that sees more tasks queued passes a wake on, so a burst of tasks
  // wakes as many workers as it needs.
  bool sleep(Worker& me) {
    std::unique_lock lk(sleep_m);
    if (me.fast) {
      sleeping_fast.fetch_add(1);
      wake_fast.wait(lk, [&] { return stop.load() || queued.load() > 0 || me.mail.load() > 0; });
      sleeping_fast.fetch_sub(1);
    } else {
      auto ready = [&] { return stop.load() || me.mail.load() > 0 || (slow_work() && fast_all_busy()); };
      // A slow worker's wake depends on two counts; the timeout is a safety net,
      // not a poll (5 wakes a second).
      while (!wake_slow.wait_for(lk, std::chrono::milliseconds(200), ready)) {
      }
    }
    if (queued.load() > 1) {
      if (sleeping_fast.load() > 0) {
        wake_fast.notify_one();
      } else if (fast_all_busy() && slow_work()) {
        wake_slow.notify_one();
      }
    }
    return !(stop.load() && queued.load() == 0 && me.mail.load() == 0);
  }

  void run(int w);
};

namespace {

struct Current {
  const void* pool = nullptr;  // the WorkerPool::Impl this thread works for
  int worker = -1;
};
thread_local Current current;

}  // namespace

void WorkerPool::Impl::run(int w) {
  current = {this, w};
  Worker& me = *workers[w];
  for (;;) {
    Task task;
    std::function<void()> mail;
    if (!take(w, task, mail)) {
      if (!sleep(me)) return;
      continue;
    }
    if (mail) {
      mail();  // catches its own exceptions
      continue;
    }
    // The last fast worker to start a task lets the slow ones take what is left.
    if (me.fast && busy_fast.fetch_add(1) + 1 >= fast && slow_work()) {
      std::lock_guard lk(sleep_m);
      wake_slow.notify_one();
    }
    try {
      task(w);
    } catch (...) {
      std::lock_guard lk(error_m);
      if (!error) error = std::current_exception();
    }
    if (me.fast) busy_fast.fetch_sub(1);
    if (pending.fetch_sub(1) == 1) {
      std::lock_guard lk(done_m);
      done.notify_all();
    }
  }
}

WorkerPool::WorkerPool(PoolOptions options) : impl_(std::make_unique<Impl>()) {
  const CpuTopology& topo = cpu_topology();
  const int n = options.workers > 0 ? options.workers : topo.default_workers();
  for (int i = 0; i < n; ++i) impl_->workers.push_back(std::make_unique<Impl::Worker>());
  // Fast: the first `fast` workers if given; else those on cores of the fastest
  // class the pool uses (every worker when unpinned, or on a uniform CPU).
  if (options.fast >= 0) {
    for (int i = 0; i < n; ++i) impl_->workers[i]->fast = i < options.fast;
  } else if (options.pin && !topo.cores.empty()) {
    const size_t nc = topo.cores.size();
    int best = 0;
    for (int i = 0; i < n; ++i) best = std::max(best, topo.cores[i % nc].efficiency);
    for (int i = 0; i < n; ++i) impl_->workers[i]->fast = topo.cores[i % nc].efficiency == best;
  }
  for (const auto& w : impl_->workers) impl_->fast += w->fast ? 1 : 0;
  impl_->slow_takes_new = options.slow_takes_new;
  std::latch started(n);  // every worker on its core before the pool is handed out
  for (int i = 0; i < n; ++i) {
    impl_->threads.emplace_back([this, i, pin = options.pin, &topo, &started] {
      if (pin && !topo.cores.empty() && pin_this_thread(topo.cores[i % topo.cores.size()])) {
        impl_->pinned.fetch_add(1);
      }
      ieee_denormals();
      started.count_down();
      impl_->run(i);
    });
  }
  started.wait();
}

WorkerPool::~WorkerPool() {
  try {
    wait();
  } catch (...) {
    // a task's exception nobody waited for: dropped with the pool
  }
  impl_->stop.store(true);
  impl_->notify_all();
  for (auto& t : impl_->threads) t.join();
}

int WorkerPool::size() const { return static_cast<int>(impl_->workers.size()); }

int WorkerPool::pinned() const { return impl_->pinned.load(); }

int WorkerPool::fast() const { return impl_->fast; }

bool WorkerPool::is_fast(int worker) const {
  return worker >= 0 && worker < size() && impl_->workers[worker]->fast;
}

// Sleeping fast workers less the tasks already queued for them: at the start of a call of many pages every
// worker sleeps, but each is about to take a page, and none is free to help another.
int WorkerPool::idle() const {
  return static_cast<int>(std::max<int64_t>(0, impl_->sleeping_fast.load() - impl_->queued.load()));
}

int WorkerPool::current_worker() const { return current.pool == impl_.get() ? current.worker : -1; }

void WorkerPool::submit(Task task) {
  Impl& p = *impl_;
  p.pending.fetch_add(1);
  if (current.pool == &p) {
    Impl::Worker& me = *p.workers[current.worker];
    std::lock_guard lk(me.m);
    me.tasks.push_front(std::move(task));
    me.queued.fetch_add(1);
    p.inside.fetch_add(1);
  } else {
    std::lock_guard lk(p.outside_m);
    p.outside.push_back(std::move(task));
  }
  p.queued.fetch_add(1);
  p.notify_task();
}

void WorkerPool::wait() {
  Impl& p = *impl_;
  if (current.pool == &p) throw Error("WorkerPool::wait called from inside one of its tasks");
  {
    std::unique_lock lk(p.done_m);
    p.done.wait(lk, [&] { return p.pending.load() == 0; });
  }
  std::exception_ptr e;
  {
    std::lock_guard lk(p.error_m);
    std::swap(e, p.error);
  }
  if (e) std::rethrow_exception(e);
}

void WorkerPool::on_each_worker(const std::function<void(int worker)>& f) {
  Impl& p = *impl_;
  if (current.pool == &p) throw Error("WorkerPool::on_each_worker called from inside one of its tasks");
  const int n = size();
  std::latch left(n);
  std::mutex error_m;
  std::exception_ptr error;
  for (int w = 0; w < n; ++w) {
    Impl::Worker& worker = *p.workers[w];
    std::lock_guard lk(worker.m);
    worker.mailbox.push_back([&, w] {
      try {
        f(w);
      } catch (...) {
        std::lock_guard g(error_m);
        if (!error) error = std::current_exception();
      }
      left.count_down();
    });
    worker.mail.fetch_add(1);
  }
  p.notify_all();
  left.wait();
  if (error) std::rethrow_exception(error);
}

}  // namespace carbon
