#include "register/register.h"
#include <cstdio>
namespace domi {
static Status Parse(const ge::Operator &source, ge::Operator &destination) {
  const auto status=AutoMappingByOpFn(source,destination);
  std::fprintf(stderr,"[ONNX_PARSE] status=%u\n",status);
  return status;
}
REGISTER_CUSTOM_OP("CannbotAddOm").FrameworkType(ONNX).OriginOpType("com.dsl_ge::1::CannbotAdd").ParseParamsByOperatorFn(Parse);
REGISTER_CUSTOM_OP("TilelangAddOm").FrameworkType(ONNX).OriginOpType("com.dsl_ge::1::TilelangAdd").ParseParamsByOperatorFn(Parse);
}
