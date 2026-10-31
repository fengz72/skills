#include "ops.h"
#include "graph/custom_op.h"
#include <atomic>
#include <cstdio>
namespace {
std::atomic<unsigned> execute_count{0};
class AddBase : public ge::EagerExecuteOp, public ge::ShapeInferOp {
 public:
  explicit AddBase(int backend):backend_(backend),elements_(backend==0 ? 256 : 524288) {}
  ge::graphStatus InferShape(gert::InferShapeContext *ctx) override {
    const auto *x=ctx->GetInputShape(0); const auto *y=ctx->GetInputShape(1);
    auto *z=ctx->GetOutputShape(0);
    if (!x || !y || !z || x->GetDimNum()!=1 || y->GetDimNum()!=1 ||
        x->GetDim(0)!=elements_ || y->GetDim(0)!=elements_) { return ge::GRAPH_FAILED; }
    *z=*x;
    return ge::GRAPH_SUCCESS;
  }
  ge::graphStatus InferDataType(gert::InferDataTypeContext *ctx) override {
    if (ctx->GetInputDataType(0)!=ge::DT_FLOAT || ctx->GetInputDataType(1)!=ge::DT_FLOAT) { return ge::GRAPH_FAILED; }
    return ctx->SetOutputDataType(0,ge::DT_FLOAT);
  }
  ge::graphStatus Execute(gert::EagerOpExecutionContext *ctx) override {
    const auto *x=ctx->GetInputTensor(0); const auto *y=ctx->GetInputTensor(1);
    if (!x || !y || x->GetDataType()!=ge::DT_FLOAT || y->GetDataType()!=ge::DT_FLOAT ||
        x->GetShape().GetStorageShape().GetDimNum()!=1 || y->GetShape().GetStorageShape().GetDimNum()!=1 ||
        x->GetShapeSize()!=elements_ || y->GetShapeSize()!=elements_ ||
        x->GetFormat().GetStorageFormat()!=ge::FORMAT_ND || y->GetFormat().GetStorageFormat()!=ge::FORMAT_ND) {
      return ge::GRAPH_FAILED;
    }
    auto *z=ctx->MallocOutputTensor(0,x->GetShape(),x->GetFormat(),x->GetDataType());
    if (!z) { return ge::GRAPH_FAILED; }
    ++execute_count;
    std::fprintf(stderr,"[GE_CUSTOM_EXECUTE] backend=%d invocation=%u elements=%lld op=%p node_ctx=%p x=%p y=%p z=%p capacity=%zu\n",
        backend_,execute_count.load(),static_cast<long long>(elements_),(void*)this,(void*)ctx,
        x->GetAddr(),y->GetAddr(),z->GetAddr(),z->GetSize());
    return SmokeLaunch(backend_,x->GetAddr(),y->GetAddr(),z->GetAddr(),ctx->GetStream())==0 ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
  }
 private:
  int backend_; int64_t elements_;
};
}
namespace ge {
class CannbotAddSmoke final:public AddBase { public: CannbotAddSmoke():AddBase(0) {} };
class TilelangAddSmoke final:public AddBase { public: TilelangAddSmoke():AddBase(1) {} };
REG_AUTO_MAPPING_OP(CannbotAddSmoke);
REG_AUTO_MAPPING_OP(TilelangAddSmoke);
}
extern "C" unsigned SmokeExecuteCount() { return execute_count.load(); }
