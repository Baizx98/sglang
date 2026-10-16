// CUDA 13 Activity collector. External IDs assign GPU work to the innermost
// component, preserving concurrency and avoiding nested stream-span sums.
#include <cupti.h>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
struct Activity {
  uint64_t start, end, bytes;
  uint32_t correlation, runtime_correlation, device, stream, copy_kind;
  std::string kind, name;
};
static std::mutex lock;
static std::vector<Activity> activities;
static std::unordered_map<uint32_t, uint64_t> correlations;
static size_t dropped = 0;
static int callback_error = 0;
static bool initialized = false;
static const CUpti_ActivityKind kinds[] = {
    CUPTI_ACTIVITY_KIND_RUNTIME, CUPTI_ACTIVITY_KIND_DRIVER,
    CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION,
    CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL, CUPTI_ACTIVITY_KIND_MEMCPY,
    CUPTI_ACTIVITY_KIND_MEMSET};
static void CUPTIAPI request_buffer(uint8_t **buffer, size_t *size, size_t *max_records) {
  *size = 8 * 1024 * 1024; *max_records = 0;
  *buffer = static_cast<uint8_t *>(std::malloc(*size));
}
static void CUPTIAPI complete_buffer(CUcontext context, uint32_t stream,
                                     uint8_t *buffer, size_t size, size_t valid_size) {
  std::lock_guard<std::mutex> guard(lock);
  CUpti_Activity *record = nullptr;
  CUptiResult result;
  while ((result = cuptiActivityGetNextRecord(buffer, valid_size, &record)) == CUPTI_SUCCESS) {
    if (record->kind == CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION) {
      auto *v = reinterpret_cast<CUpti_ActivityExternalCorrelation *>(record);
      if (v->externalKind == CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0)
        correlations[v->correlationId] = v->externalId;
    } else if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
      auto *v = reinterpret_cast<CUpti_ActivityKernel10 *>(record);
      activities.push_back({v->start, v->end, 0, v->correlationId, 0,
          v->deviceId, v->streamId, 0, "kernel", v->name ? v->name : ""});
    } else if (record->kind == CUPTI_ACTIVITY_KIND_MEMCPY) {
      auto *v = reinterpret_cast<CUpti_ActivityMemcpy6 *>(record);
      activities.push_back({v->start, v->end, v->bytes, v->correlationId,
          v->runtimeCorrelationId, v->deviceId, v->streamId, v->copyKind, "memcpy", ""});
    } else if (record->kind == CUPTI_ACTIVITY_KIND_MEMSET) {
      auto *v = reinterpret_cast<CUpti_ActivityMemset4 *>(record);
      activities.push_back({v->start, v->end, v->bytes, v->correlationId, 0,
          v->deviceId, v->streamId, 0, "memset", ""});
    }
  }
  if (result != CUPTI_ERROR_MAX_LIMIT_REACHED) callback_error = static_cast<int>(result);
  size_t count = 0;
  result = cuptiActivityGetNumDroppedRecords(context, stream, &count);
  if (result != CUPTI_SUCCESS) callback_error = static_cast<int>(result);
  dropped += count;
  std::free(buffer);
}
static std::string escape(const std::string &text) {
  std::string out;
  for (unsigned char c : text) {
    if (c == '\\' || c == '"') { out += '\\'; out += c; }
    else if (c >= 32) out += c;
  }
  return out;
}
extern "C" int hkv_start() {
  if (!initialized) {
    auto r = cuptiActivityRegisterCallbacks(request_buffer, complete_buffer);
    if (r != CUPTI_SUCCESS) return r;
    initialized = true;
  }
  activities.clear(); correlations.clear(); dropped = 0; callback_error = 0;
  for (auto kind : kinds) { auto r = cuptiActivityEnable(kind); if (r != CUPTI_SUCCESS) return r; }
  return 0;
}
extern "C" uint64_t hkv_clock() { uint64_t value = 0; cuptiGetTimestamp(&value); return value; }
extern "C" int hkv_push(uint64_t id) {
  return cuptiActivityPushExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, id);
}
extern "C" int hkv_pop() {
  uint64_t id; return cuptiActivityPopExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, &id);
}
extern "C" int hkv_stop(const char *path) {
  auto r = cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
  if (r != CUPTI_SUCCESS) return r;
  for (auto kind : kinds) { r = cuptiActivityDisable(kind); if (r != CUPTI_SUCCESS) return r; }
  r = cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
  if (r != CUPTI_SUCCESS) return r;
  std::lock_guard<std::mutex> guard(lock);
  std::ofstream output(path);
  if (!output) return -1;
  size_t unattributed = 0;
  for (const auto &v : activities) {
    auto it = correlations.find(v.correlation);
    if (it == correlations.end()) it = correlations.find(v.runtime_correlation);
    uint64_t external = it == correlations.end() ? 0 : it->second;
    if (!external) ++unattributed;
    output << "{\"kind\":\"" << v.kind << "\",\"name\":\"" << escape(v.name)
           << "\",\"start_ns\":" << v.start << ",\"end_ns\":" << v.end
           << ",\"bytes\":" << v.bytes << ",\"external_id\":" << external
           << ",\"device\":" << v.device << ",\"stream\":" << v.stream
           << ",\"copy_kind\":" << v.copy_kind << "}\n";
  }
  output << "{\"kind\":\"metadata\",\"dropped\":" << dropped
         << ",\"unattributed\":" << unattributed << ",\"activity_count\":" << activities.size() << "}\n";
  return callback_error;
}
