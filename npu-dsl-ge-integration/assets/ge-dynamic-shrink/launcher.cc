#include "ops.h"
#include "artifact_symbols.h"
#include CANNBOT_ABI_ADAPTER_HEADER
#include "tile_binary.h"
#include "acl/acl_rt.h"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <mutex>

namespace {
int LaunchCannbot(const void *x, const void *y, void *z, int64_t n, aclrtStream stream) {
  return CANNBOT_FRAME_LAUNCH(x,y,z,n,stream);
}
struct TileKernel { aclrtBinHandle binary=nullptr; aclrtFuncHandle function=nullptr; };
std::map<aclrtContext,TileKernel> kernels;
std::mutex kernel_mutex;
int LaunchTilelang(const void *x, const void *y, void *z, int64_t n, aclrtStream stream) {
  aclrtContext context=nullptr;
  auto result=aclrtGetCurrentContext(&context);
  if (result!=ACL_SUCCESS || !context) { return result ? result : 1; }
  aclrtFuncHandle function=nullptr;
  {
    std::lock_guard<std::mutex> lock(kernel_mutex);
    auto &kernel=kernels[context];
    if (!kernel.binary) {
      result=aclrtBinaryLoadFromData(kTileBinary,sizeof(kTileBinary),nullptr,&kernel.binary);
      if (result!=ACL_SUCCESS) { return result; }
    }
    if (!kernel.function) {
      result=aclrtBinaryGetFunction(kernel.binary,"main_kernel",&kernel.function);
      if (result!=ACL_SUCCESS) { return result; }
    }
    function=kernel.function;
  }
  // Dynamic host IR supplies main_kernel(A,B,C,N,grid,dyn_ubuf).
  // N is an int64 device argument; grid/UBUF are launch metadata.
  struct { const void *a; const void *b; void *c; int64_t count; } args{x,y,z,n};
  aclrtLaunchKernelAttr attr{};
  attr.id=ACL_RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE;
  attr.value.dynUBufSize=kTileDynUbuf;
  aclrtLaunchKernelCfg config{&attr,1};
  return aclrtLaunchKernelWithHostArgs(function,kTileGrid,stream,&config,&args,sizeof(args),nullptr,0);
}
}
extern "C" int SmokeLaunch(int backend, const void *x, const void *y, void *z, int64_t n, void *stream) {
  if (!x || !y || !z || !stream) { return 1; }
  int result=backend==0 ? LaunchCannbot(x,y,z,n,stream) : LaunchTilelang(x,y,z,n,stream);
  if (result) { std::cerr<<"[LAUNCH_FAIL] backend="<<backend<<" status="<<result<<std::endl; }
  return result;
}
extern "C" int SmokeRelease() {
  aclrtContext context=nullptr;
  auto result=aclrtGetCurrentContext(&context);
  if (result!=ACL_SUCCESS) { return result; }
  std::lock_guard<std::mutex> lock(kernel_mutex);
  auto found=kernels.find(context);
  if (found==kernels.end()) { return 0; }
  result=aclrtBinaryUnLoad(found->second.binary);
  if (result==ACL_SUCCESS) { kernels.erase(found); }
  return result;
}
