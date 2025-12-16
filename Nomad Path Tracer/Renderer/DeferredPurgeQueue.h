#ifndef DEFERRED_PURGE_QUEUE_H
#define DEFERRED_PURGE_QUEUE_H

#include <algorithm>
#include <cstddef>
#include <functional>
#include <utility>
#include <vector>

// A small helper for deferring resource purges until an associated command
// buffer completes.  The queue tracks which resource should be released once
// the GPU command finishes and takes care of retaining/releasing the command
// buffer while it is referenced by queued entries.
template <typename ResourceT, typename CommandBufferT>
class DeferredPurgeQueue {
public:
  using ResourceType = ResourceT;
  using CommandType = CommandBufferT;
  using RetainFn = std::function<void(CommandType *)>;
  using ReleaseFn = std::function<void(CommandType *)>;
  using PurgeFn = std::function<void(ResourceType &, size_t)>;

  DeferredPurgeQueue(RetainFn retain = RetainFn(),
                     ReleaseFn release = ReleaseFn())
      : _retain(std::move(retain)), _release(std::move(release)) {}

  void enqueue(size_t objectIndex, ResourceType &resource) {
    auto it = std::find_if(_entries.begin(), _entries.end(),
                           [&](const Entry &entry) {
                             return entry.objectIndex == objectIndex &&
                                    entry.resource == &resource;
                           });
    if (it == _entries.end())
      _entries.push_back({objectIndex, &resource, nullptr});
  }

  bool cancel(size_t objectIndex, ResourceType &resource) {
    auto it = std::find_if(_entries.begin(), _entries.end(),
                           [&](const Entry &entry) {
                             return entry.objectIndex == objectIndex &&
                                    entry.resource == &resource;
                           });
    if (it == _entries.end())
      return false;
    if (it->command && _release)
      _release(it->command);
    _entries.erase(it);
    return true;
  }

  void assign(CommandType *command) {
    if (!command)
      return;
    for (auto &entry : _entries) {
      if (entry.command)
        continue;
      if (_retain)
        _retain(command);
      entry.command = command;
    }
  }

  void complete(CommandType *command, bool success, PurgeFn purge) {
    if (!command)
      return;
    auto it = _entries.begin();
    while (it != _entries.end()) {
      Entry &entry = *it;
      if (entry.command != command) {
        ++it;
        continue;
      }

      if (entry.command && _release)
        _release(entry.command);

      if (success) {
        if (purge)
          purge(*entry.resource, entry.objectIndex);
        it = _entries.erase(it);
      } else {
        entry.command = nullptr;
        ++it;
      }
    }
  }

  void clear() {
    if (_release) {
      for (auto &entry : _entries) {
        if (entry.command)
          _release(entry.command);
      }
    }
    _entries.clear();
  }

  bool empty() const { return _entries.empty(); }

private:
  struct Entry {
    size_t objectIndex = 0;
    ResourceType *resource = nullptr;
    CommandType *command = nullptr;
  };

  std::vector<Entry> _entries;
  RetainFn _retain;
  ReleaseFn _release;
};

#endif // DEFERRED_PURGE_QUEUE_H
