#include "ops.h"
#include "kernel_bundle.h"
#include "graph/custom_op.h"
#include "acl/acl_rt.h"
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <mutex>
#include <map>
#include <string>
#include <sys/mman.h>
#include <unistd.h>

#if DSL_BACKEND == 0
namespace {
class HostKernel {
 public:
  using Schema=const void *(*)() noexcept;
  using Invoke=int32_t (*)(void *,const void *,uint64_t) noexcept;
  HostKernel() {
    fd_=memfd_create("dsl_ge_cannbot_kernel",MFD_CLOEXEC);
    if(fd_<0)return;
    size_t offset=0;
    while(offset<sizeof(kKernelBinary)) {
      const auto written=write(fd_,kKernelBinary+offset,sizeof(kKernelBinary)-offset);
      if(written<0 && errno==EINTR)continue;
      if(written<=0)return;
      offset+=static_cast<size_t>(written);
    }
    const auto path="/proc/self/fd/"+std::to_string(fd_);
    handle_=dlopen(path.c_str(),RTLD_NOW|RTLD_LOCAL);
    if(!handle_) { std::fprintf(stderr,"[KERNEL_LOAD_FAIL] %s\n",dlerror());return; }
    schema=reinterpret_cast<Schema>(dlsym(handle_,kSchemaSymbol));
    invoke=reinterpret_cast<Invoke>(dlsym(handle_,kInvokeSymbol));
  }
  ~HostKernel() { if(handle_)dlclose(handle_);if(fd_>=0)close(fd_); }
  Schema schema=nullptr;Invoke invoke=nullptr;
 private:
  int fd_=-1;void *handle_=nullptr;
};
HostKernel &Kernel() { static HostKernel kernel;return kernel; }
const void *CannbotSchema() { auto &k=Kernel();return k.schema?k.schema():nullptr; }
int32_t CannbotInvoke(void *stream,const void *frame,uint64_t size) { auto &k=Kernel();return k.invoke?k.invoke(stream,frame,size):1; }
}
#define CANNBOT_SCHEMA CannbotSchema
#define CANNBOT_INVOKE CannbotInvoke
#include CANNBOT_ABI_ADAPTER_HEADER
#endif

namespace {
class OmAddBase : public ge::EagerExecuteOp, public ge::ShapeInferOp {
 public:
  ge::graphStatus InferShape(gert::InferShapeContext *ctx) override {
    const auto *x=ctx->GetInputShape(0),*y=ctx->GetInputShape(1);auto *z=ctx->GetOutputShape(0);
    if(!x || !y || !z || x->GetDimNum()!=1 || y->GetDimNum()!=1 || x->GetDim(0)!=kElements || y->GetDim(0)!=kElements)return ge::GRAPH_FAILED;
    *z=*x;return ge::GRAPH_SUCCESS;
  }
  ge::graphStatus InferDataType(gert::InferDataTypeContext *ctx) override {
    if(ctx->GetInputDataType(0)!=ge::DT_FLOAT || ctx->GetInputDataType(1)!=ge::DT_FLOAT)return ge::GRAPH_FAILED;
    return ctx->SetOutputDataType(0,ge::DT_FLOAT);
  }
  ge::graphStatus Execute(gert::EagerOpExecutionContext *ctx) override {
    const auto *x=ctx->GetInputTensor(0),*y=ctx->GetInputTensor(1);
    for(const auto *t:{x,y}) {
      if(t)std::fprintf(stderr,"[OM_INPUT] dtype=%d rank=%zu elements=%lld format=%d addr=%p\n",int(t->GetDataType()),t->GetShape().GetStorageShape().GetDimNum(),static_cast<long long>(t->GetShapeSize()),int(t->GetFormat().GetStorageFormat()),t->GetAddr());
    }
    if(!Valid(x) || !Valid(y))return ge::GRAPH_FAILED;
    auto *z=ctx->MallocOutputTensor(0,x->GetShape(),x->GetFormat(),x->GetDataType());
    if(!z || !z->GetAddr()) { std::fprintf(stderr,"[OM_OUTPUT_FAIL] tensor=%p\n",static_cast<void *>(z));return ge::GRAPH_FAILED; }
    std::fprintf(stderr,"[OM_CUSTOM_EXECUTE] backend=%d elements=%lld kernel_sha256=%s\n",DSL_BACKEND,static_cast<long long>(kElements),kKernelSha256);
#if DSL_BACKEND == 0
    const int result=CANNBOT_FRAME_LAUNCH(x->GetAddr(),y->GetAddr(),z->GetAddr(),kElements,ctx->GetStream());
#else
    aclrtContext context=nullptr;auto result=aclrtGetCurrentContext(&context);
    if(result!=ACL_SUCCESS || !context)return ge::GRAPH_FAILED;
    aclrtFuncHandle function=nullptr;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto &kernel=kernels_[context];
      if(!kernel.binary) { result=aclrtBinaryLoadFromData(kKernelBinary,sizeof(kKernelBinary),nullptr,&kernel.binary);if(result!=ACL_SUCCESS)return ge::GRAPH_FAILED; }
      if(!kernel.function) { result=aclrtBinaryGetFunction(kernel.binary,kKernelName,&kernel.function);if(result!=ACL_SUCCESS)return ge::GRAPH_FAILED; }
      function=kernel.function;
    }
    struct { const void *x,*y;void *z; } args{x->GetAddr(),y->GetAddr(),z->GetAddr()};
    aclrtLaunchKernelAttr attr{};attr.id=ACL_RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE;attr.value.dynUBufSize=kDynUbuf;
    aclrtLaunchKernelCfg config{&attr,1};
    result=aclrtLaunchKernelWithHostArgs(function,kGrid,ctx->GetStream(),&config,&args,sizeof(args),nullptr,0);
#endif
    if(result!=ACL_SUCCESS)std::fprintf(stderr,"[OM_LAUNCH_FAIL] status=%d\n",result);
    return result==ACL_SUCCESS?ge::GRAPH_SUCCESS:ge::GRAPH_FAILED;
  }
#if DSL_BACKEND == 1
  ~OmAddBase() override {
    aclrtContext context=nullptr;
    if(aclrtGetCurrentContext(&context)!=ACL_SUCCESS)return;
    const auto found=kernels_.find(context);
    if(found!=kernels_.end() && found->second.binary)aclrtBinaryUnLoad(found->second.binary);
  }
#endif
 private:
  static bool Valid(const gert::Tensor *t) {
    return t && t->GetAddr() && t->GetDataType()==ge::DT_FLOAT && t->GetShape().GetStorageShape().GetDimNum()==1 && t->GetShapeSize()==kElements && t->GetFormat().GetStorageFormat()==ge::FORMAT_ND;
  }
#if DSL_BACKEND == 1
  struct TileKernel { aclrtBinHandle binary=nullptr;aclrtFuncHandle function=nullptr; };
  std::map<aclrtContext,TileKernel> kernels_;std::mutex mutex_;
#endif
};
}
namespace ge {
#if DSL_BACKEND == 0
class CannbotAddOm final:public OmAddBase {};
REG_AUTO_MAPPING_OP(CannbotAddOm);
#else
class TilelangAddOm final:public OmAddBase {};
REG_AUTO_MAPPING_OP(TilelangAddOm);
#endif
}
