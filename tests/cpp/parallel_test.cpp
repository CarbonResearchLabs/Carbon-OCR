#include "carbon/parallel.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>

namespace carbon {
namespace {

TEST(CpuTopology, CountsEveryCoreOnce) {
  const CpuTopology& t = cpu_topology();
  ASSERT_FALSE(t.cores.empty());
  int logical = 0;
  std::set<std::pair<int, int>> seen;  // (group, logical processor)
  for (const CpuCore& c : t.cores) {
    ASSERT_FALSE(c.cpus.empty());
    for (int cpu : c.cpus) EXPECT_TRUE(seen.insert({c.group, cpu}).second) << "processor " << cpu << " twice";
    logical += static_cast<int>(c.cpus.size());
  }
  EXPECT_EQ(logical, t.logical);
  EXPECT_GE(t.groups, 1);
  EXPECT_GE(t.numa_nodes, 1);
  EXPECT_GE(t.default_workers(), 1);
  EXPECT_LE(t.default_workers(), static_cast<int>(t.cores.size()));
  for (size_t i = 1; i < t.cores.size(); ++i) {
    EXPECT_GE(t.cores[i - 1].efficiency, t.cores[i].efficiency) << "the fastest cores come first";
  }
}

TEST(WorkerPool, RunsEveryTaskOnce) {
  for (int n : {1, 2, 4}) {
    WorkerPool pool({n});
    EXPECT_EQ(pool.size(), n);
    std::vector<std::atomic<int>> runs(1000);
    for (size_t i = 0; i < runs.size(); ++i) {
      pool.submit([&, i](int worker) {
        EXPECT_GE(worker, 0);
        EXPECT_LT(worker, n);
        runs[i].fetch_add(1);
      });
    }
    pool.wait();
    for (size_t i = 0; i < runs.size(); ++i) EXPECT_EQ(runs[i].load(), 1) << "task " << i;
  }
}

// A page that splits into line batches: the batches are submitted from inside
// a task, and wait() covers them too.
TEST(WorkerPool, TasksSubmitTasks) {
  WorkerPool pool({4});
  std::atomic<int> batches{0};
  for (int page = 0; page < 20; ++page) {
    pool.submit([&](int) {
      for (int b = 0; b < 10; ++b) pool.submit([&](int) { batches.fetch_add(1); });
    });
  }
  pool.wait();
  EXPECT_EQ(batches.load(), 200);
}

// One page's batches spread over the idle workers: the others steal them.
TEST(WorkerPool, IdleWorkersStealFromABusyOne) {
  WorkerPool pool({4});
  std::mutex m;
  std::set<int> workers;
  pool.submit([&](int) {
    for (int b = 0; b < 32; ++b) {
      pool.submit([&](int worker) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        std::lock_guard lk(m);
        workers.insert(worker);
      });
    }
  });
  pool.wait();
  EXPECT_GT(workers.size(), 1u);
}

TEST(WorkerPool, WaitRethrowsAndThePoolGoesOn) {
  WorkerPool pool({2});
  pool.submit([](int) { throw std::runtime_error("a bad page"); });
  EXPECT_THROW(pool.wait(), std::runtime_error);
  std::atomic<int> ran{0};
  pool.submit([&](int) { ran.fetch_add(1); });
  pool.wait();
  EXPECT_EQ(ran.load(), 1);
}

TEST(WorkerPool, OnEachWorkerRunsOncePerWorkerOnItsThread) {
  WorkerPool pool({3});
  std::vector<std::thread::id> ids(3);
  std::vector<int> calls(3, 0);
  pool.on_each_worker([&](int w) {
    ids[w] = std::this_thread::get_id();
    ++calls[w];
    EXPECT_EQ(pool.current_worker(), w);
  });
  EXPECT_EQ(calls, (std::vector<int>{1, 1, 1}));
  EXPECT_EQ(std::set<std::thread::id>(ids.begin(), ids.end()).size(), 3u);
  // A task on worker w runs on the thread on_each_worker saw for w.
  std::atomic<int> mismatches{0};
  for (int i = 0; i < 100; ++i) {
    pool.submit([&](int w) {
      if (ids[w] != std::this_thread::get_id()) mismatches.fetch_add(1);
    });
  }
  pool.wait();
  EXPECT_EQ(mismatches.load(), 0);
  EXPECT_EQ(pool.current_worker(), -1);
  EXPECT_THROW(pool.on_each_worker([](int w) {
    if (w == 1) throw std::runtime_error("no model");
  }), std::runtime_error);
}

// One fast worker and two slow ones: tasks that come one at a time all run on
// the fast worker, and a burst keeps all three busy.
TEST(WorkerPool, FastWorkersFirst) {
  WorkerPool pool({3, false, 1});
  EXPECT_EQ(pool.fast(), 1);
  EXPECT_TRUE(pool.is_fast(0));
  EXPECT_FALSE(pool.is_fast(1));
  EXPECT_FALSE(pool.is_fast(3));
  std::vector<int> by(3, 0);
  for (int i = 0; i < 20; ++i) {
    while (pool.idle() < 1) std::this_thread::yield();  // the fast worker back asleep
    pool.submit([&](int w) { ++by[w]; });
    pool.wait();
  }
  EXPECT_EQ(by, (std::vector<int>{20, 0, 0}));
  std::mutex m;
  std::set<int> workers;
  for (int i = 0; i < 30; ++i) {
    pool.submit([&](int w) {
      std::this_thread::sleep_for(std::chrono::milliseconds(3));
      std::lock_guard lk(m);
      workers.insert(w);
    });
  }
  pool.wait();
  EXPECT_EQ(workers.size(), 3u);
}

TEST(WorkerPool, IdleCountsTheSleepingFastWorkers) {
  WorkerPool pool({4, false});
  EXPECT_EQ(pool.fast(), 4);
  for (int i = 0; i < 1000 && pool.idle() < 4; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  EXPECT_EQ(pool.idle(), 4);
  std::atomic<bool> release{false};
  pool.submit([&](int) {
    while (!release.load()) std::this_thread::yield();
  });
  for (int i = 0; i < 1000 && pool.idle() > 3; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  EXPECT_EQ(pool.idle(), 3);
  release.store(true);
  pool.wait();
}

TEST(WorkerPool, PinsWhereTheOsAllows) {
  WorkerPool pool({2});
#if defined(_WIN32) || defined(__linux__)
  EXPECT_EQ(pool.pinned(), 2);
#endif
  WorkerPool loose({2, false});
  EXPECT_EQ(loose.pinned(), 0);
}

TEST(WorkerPool, DestructorRunsWhatIsQueued) {
  std::atomic<int> ran{0};
  {
    WorkerPool pool({2});
    for (int i = 0; i < 50; ++i) pool.submit([&](int) { ran.fetch_add(1); });
  }
  EXPECT_EQ(ran.load(), 50);
}

}  // namespace
}  // namespace carbon
