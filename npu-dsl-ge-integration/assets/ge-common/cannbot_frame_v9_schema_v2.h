#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>

// These constants and layouts define this wire protocol, not the target environment.
namespace cannbot_frame_v9_schema_v2 {
static_assert(CANNBOT_HOST_ABI_VERSION == 9 && CANNBOT_SCHEMA_VERSION == 2,
              "Select the decoder matching the detected artifact ABI");
struct FrameHeader {
  uint32_t magic; uint16_t host_abi_version, schema_version;
  uint32_t slot_count, reserved; uint64_t frame_bytes; uint8_t digest[32];
};
struct SlotSchema {
  uint32_t kind, flags, dtype_code, dtype_bits, dtype_lanes, encoding, logical_rank, physical_rank;
  uint64_t offset, size, alignment, pointer_offset, shape_offset, stride_offset,
           physical_shape_offset, physical_stride_offset;
  uint8_t digest[32];
};
struct Schema {
  uint32_t magic; uint16_t host_abi_version, schema_version;
  uint32_t slot_count, reserved; uint64_t frame_bytes, frame_alignment;
  uint8_t digest[32]; const SlotSchema *slots;
};
static_assert(sizeof(FrameHeader)==56 && offsetof(FrameHeader,digest)==24);
static_assert(sizeof(SlotSchema)==128);
inline bool Fits(uint64_t offset, uint64_t bytes, uint64_t size) {
  return offset <= size && bytes <= size-offset;
}

// Generic argument view; slot order follows the schema. shape/stride value
// semantics (units, element vs byte stride) come from the exported host ABI.
struct TensorArg {
  const void *data;
  const int64_t *shape;
  const int64_t *stride;
};

// Schema-driven frame fill for identity-layout tensors of any slot count, rank and dtype.
// Only tensor slots (kind==1) are decoded; scalar or other slot kinds are
// rejected until their layout is read from a real exported artifact and
// implemented here. Do not guess non-tensor layouts.
inline int LaunchTensors(const TensorArg *args, uint32_t count, void *stream) {
  auto *schema=static_cast<const Schema *>(CANNBOT_SCHEMA());
  if (!schema || schema->magic!=0x43425332 ||
      schema->host_abi_version!=CANNBOT_HOST_ABI_VERSION ||
      schema->schema_version!=CANNBOT_SCHEMA_VERSION || schema->slot_count!=count ||
      !schema->slots || schema->frame_bytes<sizeof(FrameHeader) ||
      schema->frame_bytes>std::numeric_limits<size_t>::max() ||
      !schema->frame_alignment || (schema->frame_alignment & (schema->frame_alignment-1)) ||
      schema->frame_alignment>std::numeric_limits<size_t>::max() || !args) { return 1; }
  const size_t alignment=std::max<size_t>(schema->frame_alignment,alignof(void *));
  void *allocation=nullptr;
  if (posix_memalign(&allocation,alignment,static_cast<size_t>(schema->frame_bytes))) { return 1; }
  std::unique_ptr<void,decltype(&std::free)> storage(allocation,&std::free);
  auto *frame=static_cast<uint8_t *>(allocation);
  std::memset(frame,0,static_cast<size_t>(schema->frame_bytes));
  FrameHeader header{0x43424639,schema->host_abi_version,schema->schema_version,
                     schema->slot_count,0,schema->frame_bytes,{}};
  std::memcpy(header.digest,schema->digest,sizeof(header.digest));
  std::memcpy(frame,&header,sizeof(header));
  for (uint32_t i=0;i<schema->slot_count;++i) {
    const auto &s=schema->slots[i];
    if (s.kind!=1) { return 2; }
    // In this schema identity encoding is 0 and physical metadata aliases
    // logical metadata. Encoded storage needs explicit physical shape/stride
    // arguments; reject it rather than leaving those fields zero-filled.
    if (s.encoding!=0 || s.physical_rank!=s.logical_rank ||
        s.physical_shape_offset!=s.shape_offset ||
        s.physical_stride_offset!=s.stride_offset) { return 5; }
    if (s.offset<sizeof(header) || !Fits(s.offset,s.size,schema->frame_bytes) ||
        !Fits(s.pointer_offset,sizeof(args[i].data),s.size) ||
        !Fits(s.shape_offset,uint64_t{s.logical_rank}*sizeof(int64_t),s.size) ||
        !Fits(s.stride_offset,uint64_t{s.logical_rank}*sizeof(int64_t),s.size)) { return 3; }
    if (!args[i].data || (s.logical_rank && (!args[i].shape || !args[i].stride))) { return 4; }
    std::memcpy(frame+s.offset+s.pointer_offset,&args[i].data,sizeof(args[i].data));
    if (s.logical_rank) {
      std::memcpy(frame+s.offset+s.shape_offset,args[i].shape,uint64_t{s.logical_rank}*sizeof(int64_t));
      std::memcpy(frame+s.offset+s.stride_offset,args[i].stride,uint64_t{s.logical_rank}*sizeof(int64_t));
    }
  }
  return CANNBOT_INVOKE(stream,frame,schema->frame_bytes);
}

// Rank-one contiguous convenience wrapper kept for the Add fixtures: element
// stride 1 matches the verified FP32 add contract. Multi-dim, padded or
// non-FP32 operators call LaunchTensors with their real shape/stride.
inline int Launch(const void *x, const void *y, void *z, int64_t n, void *stream) {
  const auto *schema=static_cast<const Schema *>(CANNBOT_SCHEMA());
  if (!schema || schema->magic!=0x43425332 ||
      schema->host_abi_version!=CANNBOT_HOST_ABI_VERSION ||
      schema->schema_version!=CANNBOT_SCHEMA_VERSION || schema->slot_count!=3 || !schema->slots) { return 1; }
  for (uint32_t i=0;i<schema->slot_count;++i) {
    const auto &s=schema->slots[i];
    // The wrapper supplies one shape/stride entry per tensor, so preserve its
    // original FP32/rank-one contract before the generic decoder reads them.
    if (s.kind!=1 || s.logical_rank!=1 || s.dtype_code!=2 ||
        s.dtype_bits!=32 || s.dtype_lanes!=1) { return 2; }
  }
  const int64_t shape[1]{n};
  const int64_t stride[1]{1};
  const TensorArg args[3]{{x,shape,stride},{y,shape,stride},{z,shape,stride}};
  return LaunchTensors(args,3,stream);
}
}
