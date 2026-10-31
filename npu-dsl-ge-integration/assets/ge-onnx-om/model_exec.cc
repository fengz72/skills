#include "acl/acl.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void Check(aclError status,const char *where) {
  if (status!=ACL_SUCCESS) {
    const char *detail=aclGetRecentErrMsg();
    throw std::runtime_error(std::string(where)+" status="+std::to_string(status)+" "+(detail?detail:""));
  }
}
constexpr size_t kGuard=512;
struct Dataset {
  aclmdlDataset *dataset=aclmdlCreateDataset();
  std::vector<void *> allocations;
  std::vector<aclDataBuffer *> buffers;
  void *Add(size_t bytes) {
    if (!dataset) { throw std::runtime_error("create dataset failed"); }
    void *raw=nullptr;
    Check(aclrtMalloc(&raw,bytes+2*kGuard,ACL_MEM_MALLOC_NORMAL_ONLY),"allocate guarded buffer");
    allocations.push_back(raw);
    auto *data=aclCreateDataBuffer(static_cast<uint8_t *>(raw)+kGuard,bytes);
    if (!data) { throw std::runtime_error("create data buffer failed"); }
    buffers.push_back(data);
    Check(aclmdlAddDatasetBuffer(dataset,data),"add dataset buffer");
    Check(aclrtMemset(raw,bytes+2*kGuard,0xa5,bytes+2*kGuard),"initialize guards");
    return static_cast<uint8_t *>(raw)+kGuard;
  }
  void Guards(size_t bytes) const {
    std::vector<uint8_t> guard(kGuard);
    for (void *raw:allocations) {
      for (size_t offset:{size_t{0},kGuard+bytes}) {
        Check(aclrtMemcpy(guard.data(),kGuard,static_cast<uint8_t *>(raw)+offset,kGuard,ACL_MEMCPY_DEVICE_TO_HOST),"read guard");
        if (!std::all_of(guard.begin(),guard.end(),[](uint8_t v){return v==0xa5;})) { throw std::runtime_error("guard damaged"); }
      }
    }
  }
  ~Dataset() {
    for (auto *b:buffers) { aclDestroyDataBuffer(b); }
    if (dataset) { aclmdlDestroyDataset(dataset); }
    for (auto *p:allocations) { aclrtFree(p); }
  }
};
void Fixture(size_t n,int variant,std::vector<float>&a,std::vector<float>&b) {
  std::mt19937 generator(7101+variant);
  std::uniform_real_distribution<float> random(-8,8);
  for(size_t i=0;i<n;++i) {
    switch(variant) {
      case 0:a[i]=0;b[i]=0;break;
      case 1:a[i]=1;b[i]=2;break;
      case 2:a[i]=(int(i%257)-128)*.125F;b[i]=(int(i%113)-56)*.25F;break;
      case 3:a[i]=random(generator);b[i]=random(generator);break;
      case 4:a[i]=random(generator);b[i]=-a[i];break;
      default:a[i]=(i%2?1:-1)*1024.F;b[i]=float(i%31)*.03125F;break;
    }
  }
}
void Run(uint32_t model,aclmdlDesc *desc,size_t n,int device) {
  const size_t bytes=n*sizeof(float);
  if (aclmdlGetNumInputs(desc)!=2 || aclmdlGetNumOutputs(desc)!=1) { throw std::runtime_error("unexpected model I/O count"); }
  for (size_t i=0;i<3;++i) {
    const bool input=i<2;const size_t index=input?i:0;
    aclmdlIODims dims{};
    Check(input?aclmdlGetInputDims(desc,index,&dims):aclmdlGetOutputDims(desc,index,&dims),"get model dims");
    auto dtype=input?aclmdlGetInputDataType(desc,index):aclmdlGetOutputDataType(desc,index);
    auto size=input?aclmdlGetInputSizeByIndex(desc,index):aclmdlGetOutputSizeByIndex(desc,index);
    if (dims.dimCount!=1 || dims.dims[0]!=static_cast<int64_t>(n) || dtype!=ACL_FLOAT || size!=bytes) { throw std::runtime_error("model metadata does not match fixture"); }
  }
  Dataset inputs,outputs;
  void *x=inputs.Add(bytes),*y=inputs.Add(bytes),*z=outputs.Add(bytes);
  std::vector<float>a(n),b(n),actual(n),readback(n);
  double squared=0,max_abs=0;size_t checked=0;
  for(int step=0;step<6;++step) {
    Fixture(n,step,a,b);
    Check(aclrtMemcpy(x,bytes,a.data(),bytes,ACL_MEMCPY_HOST_TO_DEVICE),"copy x");
    Check(aclrtMemcpy(y,bytes,b.data(),bytes,ACL_MEMCPY_HOST_TO_DEVICE),"copy y");
    Check(aclrtMemset(z,bytes,0xff,bytes),"poison output");
    Check(aclmdlExecute(model,inputs.dataset,outputs.dataset),"execute OM");
    Check(aclrtMemcpy(actual.data(),bytes,z,bytes,ACL_MEMCPY_DEVICE_TO_HOST),"read output");
    double step_max=0;
    for(size_t i=0;i<n;++i) {
      const float reference=float(double(a[i])+double(b[i]));
      const double error=std::fabs(double(actual[i])-reference);
      if (!std::isfinite(actual[i]) || error>1e-6+1e-6*std::fabs(reference)) { throw std::runtime_error("precision mismatch step="+std::to_string(step)+" index="+std::to_string(i)); }
      step_max=std::max(step_max,error);max_abs=std::max(max_abs,error);squared+=error*error;++checked;
    }
    for (size_t i=0;i<2;++i) {
      Check(aclrtMemcpy(readback.data(),bytes,i==0?x:y,bytes,ACL_MEMCPY_DEVICE_TO_HOST),"read input");
      if (std::memcmp(readback.data(),(i==0?a:b).data(),bytes)!=0) { throw std::runtime_error("input modified"); }
    }
    inputs.Guards(bytes);outputs.Guards(bytes);
    std::printf("[OM_CASE] step=%d n=%zu max_abs=%.9g guards=PASS inputs_unchanged=1 status=PASS\n",step,n,step_max);
  }
  std::ifstream maps("/proc/self/maps");std::string line;
  while(std::getline(maps,line)) {
    if (line.find("cust_opapi")!=std::string::npos || line.find("cannbot")!=std::string::npos || line.find("memfd:")!=std::string::npos) { std::printf("[LOADED_MAP] %s\n",line.c_str()); }
  }
  std::printf("[OM_RESULT] device=%d calls=6 checked=%zu max_abs=%.9g rmse=%.9g guards=PASS status=PASS\n",device,checked,max_abs,std::sqrt(squared/checked));
}
}
int main(int argc,char **argv) {
  std::setvbuf(stdout,nullptr,_IONBF,0);
  if(argc!=4) { std::fprintf(stderr,"usage: model_exec MODEL.om DEVICE ELEMENTS\n");return 2; }
  int device=-1;size_t n=0;bool init=false,selected=false,loaded=false;uint32_t model=0;aclmdlDesc *desc=nullptr;
  int result=1;
  try {
    device=std::stoi(argv[2]);n=std::stoull(argv[3]);
    if(device<0 || !n || n>524288) { throw std::runtime_error("invalid device or fixture size"); }
    Check(aclInit(nullptr),"ACL init");init=true;
    Check(aclrtSetDevice(device),"select device");selected=true;
    Check(aclmdlLoadFromFile(argv[1],&model),"load OM");loaded=true;std::printf("[OM_LOAD] status=PASS\n");
    desc=aclmdlCreateDesc();if (!desc) { throw std::runtime_error("create model descriptor failed"); }
    Check(aclmdlGetDesc(desc,model),"get model descriptor");
    Run(model,desc,n,device);result=0;
  } catch(const std::exception &error) { std::fprintf(stderr,"[OM_FAIL] %s\n",error.what()); }
  if(desc) { if(aclmdlDestroyDesc(desc)!=ACL_SUCCESS)result=1; }
  if(loaded) { if(aclmdlUnload(model)!=ACL_SUCCESS)result=1; }
  if(selected) { if(aclrtResetDevice(device)!=ACL_SUCCESS)result=1; }
  if(init) { if(aclFinalize()!=ACL_SUCCESS)result=1; }
  std::printf("[OM_CLEANUP] status=%s\n",result==0?"PASS":"FAIL");return result;
}
