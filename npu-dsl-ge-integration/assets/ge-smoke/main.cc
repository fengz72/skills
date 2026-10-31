#include "ops.h"
#include "acl/acl.h"
#include "ge/ge_api.h"
#include "ops_proto_legacy.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void Acl(aclError status,const char *label) {
  if (status!=ACL_SUCCESS) {
    const char *detail=aclGetRecentErrMsg();
    throw std::runtime_error(std::string(label)+" acl="+std::to_string(status)+" "+(detail?detail:""));
  }
}
void Ge(ge::Status status,const char *label) {
  if (status!=ge::SUCCESS) {
    const char *detail=aclGetRecentErrMsg();
    throw std::runtime_error(std::string(label)+" ge="+std::to_string(status)+" "+(detail?detail:""));
  }
}
constexpr size_t kGuard=256;
struct Buffers {
  aclrtStream stream=nullptr;
  void *x=nullptr,*y=nullptr,*raw_z=nullptr;
  size_t bytes=0;
  void Init(size_t count) {
    bytes=count*sizeof(float);
    Acl(aclrtCreateStream(&stream),"create stream");
    Acl(aclrtMalloc(&x,bytes,ACL_MEM_MALLOC_HUGE_FIRST),"malloc x");
    Acl(aclrtMalloc(&y,bytes,ACL_MEM_MALLOC_HUGE_FIRST),"malloc y");
    Acl(aclrtMalloc(&raw_z,bytes+2*kGuard,ACL_MEM_MALLOC_HUGE_FIRST),"malloc guarded output");
  }
  void *Z() const { return static_cast<uint8_t *>(raw_z)+kGuard; }
  ~Buffers() {
    if (stream) { aclrtSynchronizeStream(stream); }
    for (void *p:{x,y,raw_z}) { if (p) { aclrtFree(p); } }
    if (stream) { aclrtDestroyStream(stream); }
  }
};
void Fixture(size_t n,int variant,std::vector<float> &a,std::vector<float> &b) {
  std::mt19937 gen(20261027+variant);
  std::uniform_real_distribution<float> random(-8.0F,8.0F);
  for (size_t i=0;i<n;++i) {
    switch(variant) {
      case 0:a[i]=0;b[i]=0;break;
      case 1:a[i]=1;b[i]=2;break;
      case 2:a[i]=(static_cast<int>(i%257)-128)*0.125F;b[i]=(static_cast<int>(i%113)-56)*0.25F;break;
      case 3:a[i]=random(gen);b[i]=random(gen);break;
      case 4:a[i]=random(gen);b[i]=-a[i];break;
      default:a[i]=(i%2 ? 1 : -1)*1024.0F;b[i]=static_cast<float>(i%31)*0.03125F;break;
    }
  }
}
ge::Graph Build(int backend,size_t n) {
  ge::TensorDesc d(ge::Shape({static_cast<int64_t>(n)}),ge::FORMAT_ND,ge::DT_FLOAT);
  auto x=ge::op::Data("x").set_attr_index(0),y=ge::op::Data("y").set_attr_index(1);
  x.update_input_desc_x(d);x.update_output_desc_y(d);
  y.update_input_desc_x(d);y.update_output_desc_y(d);
  ge::Operator op;
  if (backend==0) { op=ge::op::CannbotAddSmoke("smoke_add").set_input_x(x).set_input_y(y); }
  else { op=ge::op::TilelangAddSmoke("smoke_add").set_input_x(x).set_input_y(y); }
  ge::Graph graph("dsl_add_smoke");
  std::vector<ge::Operator> inputs{x,y},outputs{op};
  Ge(graph.AddOp(x),"add x node"); Ge(graph.AddOp(y),"add y node"); Ge(graph.AddOp(op),"add custom node");
  graph.SetInputs(inputs).SetOutputs(outputs);
  Ge(graph.SaveToFile(backend==0?"cannbot_graph.air":"tilelang_graph.air"),"save graph AIR");
  std::vector<ge::AscendString> names;Ge(graph.GetAllOpName(names),"list graph operators");
  std::printf("[GRAPH_STRUCTURE] custom_type=%s nodes=%zu\n",backend==0?"CannbotAddSmoke":"TilelangAddSmoke",names.size());
  return graph;
}
gert::Tensor Tensor(size_t n,void *address) {
  int64_t size=static_cast<int64_t>(n);
  return {{{size},{size}},{ge::FORMAT_ND,ge::FORMAT_ND,{}},gert::kOnDeviceHbm,ge::DT_FLOAT,address};
}
bool Run(int backend,bool graph_mode,bool external,int device) {
  size_t n=backend==0?256:524288;
  Buffers memory;memory.Init(n);
  std::vector<float> a(n),b(n),actual(n),readback(n);
  auto execute=[&](ge::Session *session) {
    std::vector<gert::Tensor> inputs(2),outputs(1);
    inputs[0]=Tensor(n,memory.x); inputs[1]=Tensor(n,memory.y);
    outputs[0]=Tensor(n,external?memory.Z():nullptr);
    size_t checked=0;float max_error=0;double squared=0;
    for (int variant=0;variant<6;++variant) {
      Fixture(n,variant,a,b);
      Acl(aclrtMemcpy(memory.x,memory.bytes,a.data(),memory.bytes,ACL_MEMCPY_HOST_TO_DEVICE),"copy a");
      Acl(aclrtMemcpy(memory.y,memory.bytes,b.data(),memory.bytes,ACL_MEMCPY_HOST_TO_DEVICE),"copy b");
      Acl(aclrtMemsetAsync(memory.raw_z,memory.bytes+2*kGuard,0xa5,memory.bytes+2*kGuard,memory.stream),"guard output");
      void *output=memory.Z();
      if (session) {
        Ge(session->ExecuteGraphWithStreamAsync(0,memory.stream,inputs,outputs),"execute GE graph");
      } else { Acl(SmokeLaunch(backend,memory.x,memory.y,output,memory.stream),"direct kernel launch"); }
      Acl(aclrtSynchronizeStream(memory.stream),"sync execution stream");
      if (session) {
        output=outputs[0].GetAddr();
        if (!output || outputs[0].GetDataType()!=ge::DT_FLOAT || outputs[0].GetShapeSize()!=static_cast<int64_t>(n)) {
          throw std::runtime_error("GE output metadata invalid: address="+std::to_string(reinterpret_cast<uintptr_t>(output))+
            " dtype="+std::to_string(outputs[0].GetDataType())+" elements="+std::to_string(outputs[0].GetShapeSize()));
        }
        if (external && output!=memory.Z()) { throw std::runtime_error("GE changed external output pointer"); }
      }
      Acl(aclrtMemcpy(actual.data(),memory.bytes,output,memory.bytes,ACL_MEMCPY_DEVICE_TO_HOST),"copy output");
      float step_error=0;
      for (size_t i=0;i<n;++i) {
        const float expected=static_cast<float>(static_cast<double>(a[i])+static_cast<double>(b[i]));
        const float error=std::fabs(actual[i]-expected);
        if (!std::isfinite(actual[i]) || error>1e-6F+1e-6F*std::fabs(expected)) {
          throw std::runtime_error("mismatch case="+std::to_string(variant)+" index="+std::to_string(i)+
            " got="+std::to_string(actual[i])+" expected="+std::to_string(expected));
        }
        step_error=std::max(step_error,error);max_error=std::max(max_error,error);
        squared+=static_cast<double>(error)*error;++checked;
      }
      for (const auto &source:std::vector<std::pair<void *,std::vector<float> *>>{{memory.x,&a},{memory.y,&b}}) {
        Acl(aclrtMemcpy(readback.data(),memory.bytes,source.first,memory.bytes,ACL_MEMCPY_DEVICE_TO_HOST),"read input");
        if (std::memcmp(readback.data(),source.second->data(),memory.bytes)) { throw std::runtime_error("input modified"); }
      }
      std::vector<uint8_t> guard(kGuard);
      for (size_t offset:{size_t(0),kGuard+memory.bytes}) {
        if (session && !external) { break; }
        Acl(aclrtMemcpy(guard.data(),kGuard,static_cast<uint8_t *>(memory.raw_z)+offset,kGuard,ACL_MEMCPY_DEVICE_TO_HOST),"read guard");
        if (!std::all_of(guard.begin(),guard.end(),[](uint8_t b){return b==0xa5;})) { throw std::runtime_error("output guard damaged"); }
      }
      std::printf("[CASE] variant=%d checked=%zu max_abs=%.9g inputs_unchanged=1 guards=%s status=PASS\n",variant,n,step_error,(!session || external)?"PASS":"not_applicable");
    }
    std::printf("[RESULT] backend=%s mode=%s output=%s device=%d calls=6 checked=%zu max_abs=%.9g rmse=%.9g ge_execute_callbacks=%u status=PASS\n",
      backend==0?"cannbot":"tilelang",graph_mode?"ge":"direct",external?"external":"ge_allocated",device,checked,max_error,std::sqrt(squared/checked),SmokeExecuteCount());
  };
  if (graph_mode) {
    std::map<ge::AscendString,ge::AscendString> opts{{"ge.exec.deviceId",std::to_string(device).c_str()},{"ge.graphRunMode","1"}};
    ge::Session session(opts);
    auto graph=Build(backend,n);
    Ge(session.AddGraph(0,graph),"AddGraph");std::printf("[GRAPH] AddGraph=PASS\n");
    Ge(session.CompileGraph(0),"CompileGraph");std::printf("[GRAPH] CompileGraph=PASS\n");
    std::map<ge::AscendString,ge::AscendString> load;
    Ge(session.LoadGraph(0,load,memory.stream),"LoadGraph");std::printf("[GRAPH] LoadGraph=PASS\n");
    execute(&session);
    Ge(session.RemoveGraph(0),"RemoveGraph");
    if (!SmokeExecuteCount()) { throw std::runtime_error("GE did not invoke custom implementation"); }
  } else { execute(nullptr); }
  Acl(SmokeRelease(),"release binary after model unload");
  std::printf("[CLEANUP] status=PASS\n");
  return true;
}
}
int main(int argc,char **argv) {
  std::setvbuf(stdout,nullptr,_IONBF,0);
  if (argc<4) { std::fprintf(stderr,"usage: smoke_runner cannbot|tilelang direct|ge device [external|auto]\n");return 2; }
  int backend=std::string(argv[1])=="cannbot"?0:1;
  bool graph=std::string(argv[2])=="ge";
  int device=std::stoi(argv[3]);
  if (device<0) { return 2; }
  bool external=argc<5 || std::string(argv[4])!="auto";
  bool initialized=false;
  try {
    if (graph) {
      std::map<ge::AscendString,ge::AscendString> opts{{"ge.exec.deviceId",std::to_string(device).c_str()},{"ge.graphRunMode","1"}};
      Ge(ge::GEInitialize(opts),"GEInitialize");
    } else { Acl(aclInit(nullptr),"aclInit"); }
    initialized=true;
    Acl(aclrtSetDevice(device),"set device");
    bool ok=Run(backend,graph,external,device);
    if (graph) { Ge(ge::GEFinalize(),"GEFinalize"); }
    else { Acl(aclrtResetDevice(device),"reset device");Acl(aclFinalize(),"aclFinalize"); }
    return ok?0:1;
  } catch(const std::exception &error) {
    std::fprintf(stderr,"[FAIL] %s\n",error.what());
    if (initialized) { SmokeRelease();if(graph) { ge::GEFinalize(); } else { aclrtResetDevice(device);aclFinalize(); } }
    return 1;
  }
}
