#include "MetalCpp Path Tracer/Renderer/DeferredPurgeQueue.h"

#include <cassert>

struct DummyResource {
  int purgeCount = 0;
};

struct DummyCommand {
  int retainCount = 0;
};

int main() {
  DummyResource resourceA;
  DummyResource resourceB;
  DummyCommand command{};

  DeferredPurgeQueue<DummyResource, DummyCommand> queue(
      [](DummyCommand *cmd) {
        if (cmd)
          ++cmd->retainCount;
      },
      [](DummyCommand *cmd) {
        if (cmd)
          --cmd->retainCount;
      });

  queue.enqueue(0, resourceA);
  queue.assign(&command);
  queue.complete(&command, false, [](DummyResource &res, size_t) {
    ++res.purgeCount;
  });
  assert(queue.empty() == false);
  assert(command.retainCount == 0);
  assert(resourceA.purgeCount == 0);

  queue.assign(&command);
  queue.complete(&command, true, [](DummyResource &res, size_t) {
    ++res.purgeCount;
  });
  assert(resourceA.purgeCount == 1);
  assert(queue.empty());
  assert(command.retainCount == 0);

  queue.enqueue(1, resourceB);
  queue.assign(&command);
  bool cancelled = queue.cancel(1, resourceB);
  assert(cancelled);
  assert(queue.empty());
  assert(command.retainCount == 0);

  queue.enqueue(2, resourceA);
  queue.assign(&command);
  queue.clear();
  assert(queue.empty());
  assert(command.retainCount == 0);

  return 0;
}
