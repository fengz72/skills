#pragma once
#include <cstdint>
#include "graph/operator_reg.h"
namespace ge {
REG_OP(CannbotDynamicAdd)
    .INPUT(x, TensorType({DT_FLOAT}))
    .INPUT(y, TensorType({DT_FLOAT}))
    .OUTPUT(z, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(CannbotDynamicAdd);
REG_OP(TilelangDynamicAdd)
    .INPUT(x, TensorType({DT_FLOAT}))
    .INPUT(y, TensorType({DT_FLOAT}))
    .OUTPUT(z, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(TilelangDynamicAdd);
REG_OP(CannbotDynamicAddOut)
    .INPUT(x, TensorType({DT_FLOAT}))
    .INPUT(y, TensorType({DT_FLOAT}))
    .INPUT(out_buffer, TensorType({DT_FLOAT}))
    .OUTPUT(out_buffer, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(CannbotDynamicAddOut);
REG_OP(TilelangDynamicAddOut)
    .INPUT(x, TensorType({DT_FLOAT}))
    .INPUT(y, TensorType({DT_FLOAT}))
    .INPUT(out_buffer, TensorType({DT_FLOAT}))
    .OUTPUT(out_buffer, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(TilelangDynamicAddOut);
}
extern "C" int SmokeLaunch(int backend, const void *x, const void *y, void *z, int64_t n, void *stream);
extern "C" int SmokeRelease();
extern "C" unsigned SmokeExecuteCount();
