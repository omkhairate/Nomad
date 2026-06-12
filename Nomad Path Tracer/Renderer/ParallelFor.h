#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace NomadPathTracer {

class ThreadPool {
public:
  using Task = std::function<void()>;

  static ThreadPool &instance() {
    static ThreadPool pool;
    return pool;
  }

  ~ThreadPool() {
    {
      std::lock_guard<std::mutex> lock(_mutex);
      _shutdown = true;
    }
    _cv.notify_all();
    for (auto &worker : _workers) {
      if (worker.joinable())
        worker.join();
    }
  }

  size_t workerCount() const { return _workers.size(); }

  void enqueue(Task task) {
    {
      std::lock_guard<std::mutex> lock(_mutex);
      _tasks.emplace_back(std::move(task));
    }
    _cv.notify_one();
  }

private:
  ThreadPool() {
    const size_t hardwareThreads =
        std::max<size_t>(1, std::thread::hardware_concurrency());
    const size_t threadCount =
        std::max<size_t>(1, hardwareThreads > 1 ? hardwareThreads - 1 : 1);
    _workers.reserve(threadCount);
    for (size_t i = 0; i < threadCount; ++i) {
      _workers.emplace_back([this]() { workerLoop(); });
    }
  }

  void workerLoop() {
    while (true) {
      Task task;
      {
        std::unique_lock<std::mutex> lock(_mutex);
        _cv.wait(lock, [this]() { return _shutdown || !_tasks.empty(); });
        if (_shutdown && _tasks.empty())
          return;
        task = std::move(_tasks.front());
        _tasks.pop_front();
      }
      task();
    }
  }

  std::vector<std::thread> _workers;
  std::deque<Task> _tasks;
  std::condition_variable _cv;
  std::mutex _mutex;
  bool _shutdown = false;
};

struct ParallelForConfig {
  size_t minChunkSize = 64;
  size_t preferredChunkSize = 0; // 0 means derive from worker count
};

template <typename Func>
void parallelFor(size_t count, Func &&func,
                 const ParallelForConfig &config = ParallelForConfig{}) {
  if (count == 0)
    return;

  ThreadPool &pool = ThreadPool::instance();
  const size_t workerCount = pool.workerCount();

  size_t chunkSize = config.preferredChunkSize;
  if (chunkSize == 0) {
    const size_t lanes = std::max<size_t>(1, workerCount);
    chunkSize = (count + lanes - 1) / lanes;
  }
  chunkSize = std::max(config.minChunkSize, chunkSize);

  size_t taskCount = (count + chunkSize - 1) / chunkSize;
  if (taskCount <= 1 || workerCount == 0) {
    func(0, count);
    return;
  }

  size_t begin = 0;
  size_t end = std::min(chunkSize, count);
  func(begin, end);

  auto remainingTasks = std::make_shared<std::atomic<size_t>>(taskCount - 1);
  auto doneMutex = std::make_shared<std::mutex>();
  auto doneCv = std::make_shared<std::condition_variable>();

  for (size_t i = 1; i < taskCount; ++i) {
    size_t taskBegin = i * chunkSize;
    size_t taskEnd = std::min(taskBegin + chunkSize, count);
    pool.enqueue([&, taskBegin, taskEnd, remainingTasks, doneMutex, doneCv]() {
      func(taskBegin, taskEnd);
      if (--(*remainingTasks) == 0) {
        std::lock_guard<std::mutex> lock(*doneMutex);
        doneCv->notify_one();
      }
    });
  }

  std::unique_lock<std::mutex> lock(*doneMutex);
  doneCv->wait(lock, [&]() { return remainingTasks->load() == 0; });
}

} // namespace NomadPathTracer
