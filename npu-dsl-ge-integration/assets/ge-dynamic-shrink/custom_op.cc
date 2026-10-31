#include "ops.h"
#include "graph/custom_op.h"
#include "acl/acl_rt.h"
#include <atomic>
#include <cstdio>
namespace {
std::atomic<unsigned> execute_count{0};
class DynamicAddBase : public ge::EagerExecuteOp, public ge::ShapeInferOp {
 public:
  explicit DynamicAddBase(int backend,bool alias=false):backend_(backend),alias_(alias) {}
  ge::graphStatus InferShape(gert::InferShapeContext *ctx) override {
    const auto *x=ctx->GetInputShape(0),*y=ctx->GetInputShape(1);
    auto *z=ctx->GetOutputShape(0);
    if (!x || !y || !z || x->GetDimNum()!=1 || y->GetDimNum()!=1) { return ge::GRAPH_FAILED; }
    const int64_t nx=x->GetDim(0),ny=y->GetDim(0);
    if ((nx!=-1 && (nx<1 || nx>4096)) || (ny!=-1 && (ny<1 || ny>4096)) ||
        (nx!=-1 && ny!=-1 && nx!=ny)) { return ge::GRAPH_FAILED; }
    *z=*x;
    return ge::GRAPH_SUCCESS;
  }
  ge::graphStatus InferDataType(gert::InferDataTypeContext *ctx) override {
    if (ctx->GetInputDataType(0)!=ge::DT_FLOAT || ctx->GetInputDataType(1)!=ge::DT_FLOAT) { return ge::GRAPH_FAILED; }
    return ctx->SetOutputDataType(0,ge::DT_FLOAT);
  }
  ge::graphStatus Execute(gert::EagerOpExecutionContext *ctx) override {
    const auto *x=ctx->GetInputTensor(0),*y=ctx->GetInputTensor(1);
    if (!x || !y || x->GetDataType()!=ge::DT_FLOAT || y->GetDataType()!=ge::DT_FLOAT ||
        x->GetShape().GetStorageShape().GetDimNum()!=1 || y->GetShape().GetStorageShape().GetDimNum()!=1 ||
        x->GetShapeSize()!=y->GetShapeSize() || x->GetShapeSize()<1 || x->GetShapeSize()>4096 ||
        x->GetFormat().GetStorageFormat()!=ge::FORMAT_ND || y->GetFormat().GetStorageFormat()!=ge::FORMAT_ND) {
      return ge::GRAPH_FAILED;
    }
    int64_t n=x->GetShapeSize();
    const auto *destination=alias_?ctx->GetInputTensor(2):nullptr;
    if(alias_ && (!destination || destination->GetShapeSize()!=n || destination->GetDataType()!=ge::DT_FLOAT ||
                  destination->GetShape().GetStorageShape().GetDimNum()!=1 ||
                  destination->GetFormat().GetStorageFormat()!=ge::FORMAT_ND)) {return ge::GRAPH_FAILED;}
    auto *z=alias_?ctx->MakeOutputRefInput(0,2):ctx->MallocOutputTensor(0,x->GetShape(),x->GetFormat(),x->GetDataType());
    if (!z || z->GetSize()<static_cast<size_t>(n)*sizeof(float)) { return ge::GRAPH_FAILED; }
    ++execute_count;
    aclrtContext acl_context=nullptr;aclrtGetCurrentContext(&acl_context);
    std::fprintf(stderr,"[GE_DYNAMIC_EXECUTE] backend=%d alias=%d invocation=%u n=%lld ctx=%p op=%p node_ctx=%p x=%p y=%p z=%p capacity=%zu\n",
       backend_,alias_?1:0,execute_count.load(),(long long)n,(void*)acl_context,(void*)this,(void*)ctx,x->GetAddr(),y->GetAddr(),z->GetAddr(),z->GetSize());
    return SmokeLaunch(backend_,x->GetAddr(),y->GetAddr(),z->GetAddr(),n,ctx->GetStream())==0 ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
  }
 private:
  int backend_; bool alias_;
};
}
namespace ge {
class CannbotDynamicAdd final:public DynamicAddBase { public:CannbotDynamicAdd():DynamicAddBase(0) {} };
class TilelangDynamicAdd final:public DynamicAddBase { public:TilelangDynamicAdd():DynamicAddBase(1) {} };
class CannbotDynamicAddOut final:public DynamicAddBase { public:CannbotDynamicAddOut():DynamicAddBase(0,true) {} };
class TilelangDynamicAddOut final:public DynamicAddBase { public:TilelangDynamicAddOut():DynamicAddBase(1,true) {} };
REG_AUTO_MAPPING_OP(CannbotDynamicAddOut);
REG_AUTO_MAPPING_OP(TilelangDynamicAddOut);
REG_AUTO_MAPPING_OP(CannbotDynamicAdd);
REG_AUTO_MAPPING_OP(TilelangDynamicAdd);
}
extern "C" unsigned SmokeExecuteCount() { return execute_count.load(); }
