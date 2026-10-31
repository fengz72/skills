#pragma once
#include "graph/operator_reg.h"
namespace ge {
REG_OP(CannbotAddSmoke)
    .INPUT(x, TensorType({DT_FLOAT}))
    .INPUT(y, TensorType({DT_FLOAT}))
    .OUTPUT(z, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(CannbotAddSmoke);
REG_OP(TilelangAddSmoke)
    .INPUT(x, TensorType({DT_FLOAT}))
    .INPUT(y, TensorType({DT_FLOAT}))
    .OUTPUT(z, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(TilelangAddSmoke);
}
extern "C" int SmokeLaunch(int backend, const void *x, const void *y, void *z, void *stream);
extern "C" int SmokeRelease();
extern "C" unsigned SmokeExecuteCount();
