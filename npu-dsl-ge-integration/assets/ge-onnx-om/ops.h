#pragma once
#include "graph/operator_reg.h"
namespace ge {
REG_OP(CannbotAddOm).INPUT(x,TensorType({DT_FLOAT})).INPUT(y,TensorType({DT_FLOAT})).OUTPUT(z,TensorType({DT_FLOAT})).OP_END_FACTORY_REG(CannbotAddOm);
REG_OP(TilelangAddOm).INPUT(x,TensorType({DT_FLOAT})).INPUT(y,TensorType({DT_FLOAT})).OUTPUT(z,TensorType({DT_FLOAT})).OP_END_FACTORY_REG(TilelangAddOm);
}
