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
#include <memory>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
constexpr size_t kCapacity=4096,kGuard=4096,kPayload=kCapacity*sizeof(float),kTotal=kPayload+2*kGuard;
const std::vector<int64_t> kShapes{4096,4096,4096,2049,2049,1025,257,1,1};
void Acl(aclError r,const char *name) {
  if(r!=ACL_SUCCESS) {
    const char *detail=aclGetRecentErrMsg();
    throw std::runtime_error(std::string(name)+" acl="+std::to_string(r)+" "+(detail?detail:""));
  }
}
void Ge(ge::Status r,const char *name) {
  if(r!=ge::SUCCESS) {const char *d=aclGetRecentErrMsg();throw std::runtime_error(std::string(name)+" ge="+std::to_string(r)+" "+(d?d:""));}
}
struct Stream {
  aclrtStream value=nullptr;
  Stream() {Acl(aclrtCreateStream(&value),"create stream");}
  ~Stream() {if(value) {aclrtSynchronizeStream(value);aclrtDestroyStream(value);}}
};
struct Block {
  void *raw=nullptr;
  Block() {Acl(aclrtMalloc(&raw,kTotal,ACL_MEM_MALLOC_HUGE_FIRST),"malloc padded block");}
  ~Block() {if(raw) {aclrtFree(raw);}}
  void *Data() const {return static_cast<uint8_t *>(raw)+kGuard;}
  void Reset() {Acl(aclrtMemset(raw,kTotal,0xa5,kTotal),"reset guard/payload");}
  std::vector<uint8_t> Read() const {
    std::vector<uint8_t> bytes(kTotal);
    Acl(aclrtMemcpy(bytes.data(),bytes.size(),raw,bytes.size(),ACL_MEMCPY_DEVICE_TO_HOST),"read padded block");return bytes;
  }
  void Check(const std::vector<float> &expected,const char *name) const {
    auto bytes=Read();size_t logical=expected.size()*sizeof(float);
    if(std::memcmp(bytes.data()+kGuard,expected.data(),logical)) {throw std::runtime_error(std::string(name)+" input modified");}
    if(!std::all_of(bytes.begin(),bytes.begin()+kGuard,[](uint8_t b){return b==0xa5;}) ||
       !std::all_of(bytes.begin()+kGuard+logical,bytes.end(),[](uint8_t b){return b==0xa5;})) {
      throw std::runtime_error(std::string(name)+" guard/logical-tail corrupted");
    }
  }
};
struct Bundle {
  Block x,y;
  std::unique_ptr<Block> z;
  std::vector<uint8_t> sx,sy,sz;
  explicit Bundle(bool external) {if(external) {z=std::make_unique<Block>();}}
  void Snapshot() {sx=x.Read();sy=y.Read();if(z) {sz=z->Read();}}
  void CheckRetired() const {
    if(x.Read()!=sx || y.Read()!=sy || (z && z->Read()!=sz)) {throw std::runtime_error("retired buffer was modified: stale pointer write");}
  }
};
gert::Tensor Tensor(int64_t n,void *p) {return {{{n},{n}},{ge::FORMAT_ND,ge::FORMAT_ND,{}},gert::kOnDeviceHbm,ge::DT_FLOAT,p};}
ge::Graph Build(int backend,bool alias) {
  ge::TensorDesc d(ge::Shape({-1}),ge::FORMAT_ND,ge::DT_FLOAT);
  auto x=ge::op::Data("x").set_attr_index(0),y=ge::op::Data("y").set_attr_index(1);
  x.update_input_desc_x(d);x.update_output_desc_y(d);y.update_input_desc_x(d);y.update_output_desc_y(d);
  ge::Operator add;
  auto dest=ge::op::Data("destination").set_attr_index(2);
  dest.update_input_desc_x(d);dest.update_output_desc_y(d);
  if(alias && backend==0) {add=ge::op::CannbotDynamicAddOut("dynamic_add").set_input_x(x).set_input_y(y).set_input_out_buffer(dest);}
  else if(alias) {add=ge::op::TilelangDynamicAddOut("dynamic_add").set_input_x(x).set_input_y(y).set_input_out_buffer(dest);}
  else if(backend==0) {add=ge::op::CannbotDynamicAdd("dynamic_add").set_input_x(x).set_input_y(y);}
  else {add=ge::op::TilelangDynamicAdd("dynamic_add").set_input_x(x).set_input_y(y);}
  ge::Graph graph("dsl_dynamic_shrink");
  Ge(graph.AddOp(x),"add x");Ge(graph.AddOp(y),"add y");
  if(alias) {Ge(graph.AddOp(dest),"add destination");}
  Ge(graph.AddOp(add),"add custom");
  std::vector<ge::Operator> in{x,y},out{add};if(alias) {in.push_back(dest);}
  graph.SetInputs(in).SetOutputs(out);
  Ge(graph.SaveToFile(backend==0?"cannbot_dynamic.air":"tilelang_dynamic.air"),"save dynamic graph");
  std::printf("[GRAPH_STRUCTURE] input_shape=-1 alias=%d nodes=%d\n",alias?1:0,alias?4:3);
  return graph;
}
void Run(int backend,const std::string &mode,int device) {
  const bool direct=mode=="direct",reuse=mode=="reuse",automatic=mode=="auto",alias=mode=="alias";
  // Validate the entire schedule before constructing/launching a model.
  for(size_t i=0;i<kShapes.size();++i) {
    if(kShapes[i]<1 || kShapes[i]>static_cast<int64_t>(kCapacity) || (i && kShapes[i]>kShapes[i-1])) {
      throw std::runtime_error("SHRINK_ONLY: invalid or increasing shape schedule");
    }
  }
  Stream stream;
  aclrtContext context=nullptr;Acl(aclrtGetCurrentContext(&context),"get context");
  std::vector<std::unique_ptr<Bundle>> pool;
  std::set<void*> x_addresses,y_addresses,z_addresses;
  size_t checked=0;float maximum=0;double square=0;
  auto steps=[&](ge::Session *session) {
    for(size_t step=0;step<kShapes.size();++step) {
      const int64_t n=kShapes[step];size_t bytes=n*sizeof(float);
      Acl(aclrtSetCurrentContext(context),"restore caller context");
      if(!reuse || pool.empty()) {pool.push_back(std::make_unique<Bundle>(!automatic));}
      auto &active=*pool.back();
      active.x.Reset();active.y.Reset();if(active.z) {active.z->Reset();}
      std::vector<float> a(n),b(n),actual(n);
      std::mt19937 random(20261028+step);std::uniform_real_distribution<float> uniform(-8,8);
      for(int64_t i=0;i<n;++i) {a[i]=uniform(random)+static_cast<float>(step)*0.25F;b[i]=uniform(random)-static_cast<float>(i%7)*0.125F;}
      Acl(aclrtMemcpy(active.x.Data(),bytes,a.data(),bytes,ACL_MEMCPY_HOST_TO_DEVICE),"copy input x");
      Acl(aclrtMemcpy(active.y.Data(),bytes,b.data(),bytes,ACL_MEMCPY_HOST_TO_DEVICE),"copy input y");
      void *output=active.z?active.z->Data():nullptr;
      if(output) {Acl(aclrtMemsetAsync(output,bytes,0xff,bytes,stream.value),"poison valid output");}
      x_addresses.insert(active.x.Data());y_addresses.insert(active.y.Data());
      unsigned before=SmokeExecuteCount();
      std::vector<gert::Tensor> inputs(alias?3:2),outputs(1);
      inputs[0]=Tensor(n,active.x.Data());inputs[1]=Tensor(n,active.y.Data());
      if(alias) {inputs[2]=Tensor(n,active.z->Data());}
      outputs[0]=Tensor(n,output);
      if(session) {Ge(session->ExecuteGraphWithStreamAsync(0,stream.value,inputs,outputs),"execute same dynamic graph");}
      else {Acl(SmokeLaunch(backend,active.x.Data(),active.y.Data(),output,n,stream.value),"launch dynamic kernel");}
      Acl(aclrtSynchronizeStream(stream.value),"sync execution stream");
      size_t capacity=bytes;
      if(session) {
        if(outputs.size()!=1 || !outputs[0].GetAddr() || outputs[0].GetSize()<bytes ||
           outputs[0].GetShape().GetStorageShape()!=gert::Shape({n}) ||
           outputs[0].GetShape().GetOriginShape()!=gert::Shape({n}) || outputs[0].GetDataType()!=ge::DT_FLOAT) {
          throw std::runtime_error("invalid dynamic output shape/capacity");
        }
        if(!automatic && outputs[0].GetAddr()!=output) {throw std::runtime_error("external output address was replaced");}
        output=outputs[0].GetAddr();capacity=outputs[0].GetSize();
        if(SmokeExecuteCount()!=before+1) {throw std::runtime_error("expected one dynamic Execute callback for this invocation");}
      }
      z_addresses.insert(output);
      Acl(aclrtMemcpy(actual.data(),bytes,output,bytes,ACL_MEMCPY_DEVICE_TO_HOST),"read output");
      float step_max=0;
      for(int64_t i=0;i<n;++i) {
        float expected=static_cast<float>(static_cast<double>(a[i])+static_cast<double>(b[i]));
        float error=std::fabs(actual[i]-expected);
        if(!std::isfinite(actual[i]) || error>1e-6F+1e-6F*std::fabs(expected)) {
          throw std::runtime_error("precision mismatch step="+std::to_string(step)+" n="+std::to_string(n)+" index="+std::to_string(i));
        }
        step_max=std::max(step_max,error);maximum=std::max(maximum,error);square+=static_cast<double>(error)*error;++checked;
      }
      active.x.Check(a,"x");active.y.Check(b,"y");
      if(active.z) {
        auto raw=active.z->Read();
        if(!std::all_of(raw.begin(),raw.begin()+kGuard,[](uint8_t v){return v==0xa5;}) ||
           !std::all_of(raw.begin()+kGuard+bytes,raw.end(),[](uint8_t v){return v==0xa5;})) {
          throw std::runtime_error("output guard/logical tail was overwritten");
        }
      }
      for(size_t i=0;i+1<pool.size();++i) {pool[i]->CheckRetired();}
      active.Snapshot();
      std::printf("[STEP] step=%zu n=%lld x=%p y=%p z=%p capacity=%zu callback_delta=%u checked=%lld max_abs=%.9g input_guards=PASS output_guards=%s retired_buffers=PASS status=PASS\n",
          step,(long long)n,active.x.Data(),active.y.Data(),output,capacity,SmokeExecuteCount()-before,(long long)n,step_max,automatic?"not_applicable":"PASS");
    }
  };
  if(direct) {steps(nullptr);}
  else {
    std::map<ge::AscendString,ge::AscendString> opts{{"ge.exec.deviceId",std::to_string(device).c_str()},{"ge.graphRunMode","1"}};
    ge::Session session(opts);auto graph=Build(backend,alias);
    Ge(session.AddGraph(0,graph),"AddGraph");std::printf("[GRAPH] AddGraph=PASS\n");
    Ge(session.CompileGraph(0),"CompileGraph");std::printf("[GRAPH] CompileGraph=PASS\n");
    std::map<ge::AscendString,ge::AscendString> load;
    Ge(session.LoadGraph(0,load,stream.value),"LoadGraph");std::printf("[GRAPH] LoadGraph=PASS\n");
    steps(&session);Ge(session.RemoveGraph(0),"RemoveGraph");
  }
  const size_t expected_unique=reuse?1:kShapes.size();
  if(x_addresses.size()!=expected_unique || y_addresses.size()!=expected_unique ||
     (!automatic && z_addresses.size()!=expected_unique)) {throw std::runtime_error("requested caller address variation was not achieved");}
  Acl(aclrtSetCurrentContext(context),"restore cleanup context");Acl(SmokeRelease(),"unload dynamic kernel binary");
  std::printf("[RESULT] backend=%s mode=%s device=%d same_model=%d graphs_compiled=%d calls=%zu checked=%zu max_abs=%.9g rmse=%.9g ge_callbacks=%u unique_x=%zu unique_y=%zu unique_z=%zu growth_executed=0 status=PASS\n",
      backend==0?"cannbot":"tilelang",mode.c_str(),device,direct?0:1,direct?0:1,kShapes.size(),checked,maximum,std::sqrt(square/checked),SmokeExecuteCount(),x_addresses.size(),y_addresses.size(),z_addresses.size());
  std::printf("[CLEANUP] status=PASS\n");
}
}
int main(int argc,char **argv) {
  std::setvbuf(stdout,nullptr,_IONBF,0);
  if(argc!=4) {std::fprintf(stderr,"usage: dynamic_runner cannbot|tilelang direct|reuse|rotate|auto|alias device\n");return 2;}
  const std::string name=argv[1],mode=argv[2];
  if((name!="cannbot" && name!="tilelang") || (mode!="direct" && mode!="reuse" && mode!="rotate" && mode!="auto" && mode!="alias")) {return 2;}
  int backend=name=="cannbot"?0:1,device=std::stoi(argv[3]);if(device<0) {return 2;}
  bool initialized=false;
  try {
    if(mode=="direct") {Acl(aclInit(nullptr),"aclInit");}
    else {std::map<ge::AscendString,ge::AscendString> opts{{"ge.exec.deviceId",std::to_string(device).c_str()},{"ge.graphRunMode","1"}};Ge(ge::GEInitialize(opts),"GEInitialize");}
    initialized=true;Acl(aclrtSetDevice(device),"set device");Run(backend,mode,device);
    if(mode=="direct") {Acl(aclrtResetDevice(device),"reset device");Acl(aclFinalize(),"aclFinalize");}
    else {Ge(ge::GEFinalize(),"GEFinalize");}
    return 0;
  } catch(const std::exception &e) {
    std::fprintf(stderr,"[FAIL] %s\n",e.what());
    if(initialized) {SmokeRelease();if(mode=="direct") {aclrtResetDevice(device);aclFinalize();} else {ge::GEFinalize();}}
    return 1;
  }
}
