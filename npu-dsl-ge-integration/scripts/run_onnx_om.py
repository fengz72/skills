#!/usr/bin/env python3
"""Verify ONNX parser, GE OM packaging and isolated NPU execution of both fixed Add fixtures."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def custom_nodes(value, op_type):
    found=[]
    if isinstance(value,dict):
        if value.get('type')==op_type:found.append(value)
        for child in value.values():found.extend(custom_nodes(child,op_type))
    elif isinstance(value,list):
        for child in value:found.extend(custom_nodes(child,op_type))
    return found


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-fixture',type=Path,required=True,help='Previously validated fixed Add export/build directory; exports are reused')
    parser.add_argument('--device',type=int,required=True)
    parser.add_argument('--soc-version',required=True,help='Confirmed ATC SoC name, e.g. obtained via ACL on the target device')
    parser.add_argument('--cxx11-abi',type=int,choices=[0,1],required=True)
    parser.add_argument('--build-jobs',type=int,default=os.cpu_count() or 1)
    parser.add_argument('--stage-timeout',type=int,default=600)
    parser.add_argument('--work-dir',type=Path,required=True)
    args=parser.parse_args()
    if args.device<0 or min(args.build_jobs,args.stage_timeout)<1 or not os.environ.get('ASCEND_HOME_PATH'):
        parser.error('Confirm device, positive build jobs/timeout and source target CANN environment')
    import onnx
    from onnx import helper,TensorProto
    fixture=args.native_fixture.resolve(strict=True)
    work=args.work_dir.resolve();work.mkdir(parents=True,exist_ok=False)
    skill=Path(__file__).resolve().parents[1]
    report={'status':'running','parameters':{key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()},
            'scope':'fixed, contiguous rank-one FP32 Add; ONNX/ATC/ACL; kernel embedded in SO_BINS, no PortableOp payload',
            'source_fixture':str(fixture),'source_reexported':False,'stages':[],'backends':[],
            'onnx_version':onnx.__version__,'runtime_requirements':['Matching CANN/GE/ACL runtime and driver','Compatible NPU and host architecture','System C/C++ libraries'],
            'limits':['No dynamic-shape or cross-machine compatibility validation','Sequential execution; no concurrency claim']}
    def save():
        (work/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    def run(stage,command,cwd,env=None,expect_failure=False):
        log=cwd/(stage+'.log')
        item={'stage':stage,'command':list(map(str,command)),'log':str(log.relative_to(work)),'status':'running'}
        report['stages'].append(item);save();print('[ONNX_OM_STAGE]',stage,flush=True)
        try:
            with log.open('w') as stream:
                result=subprocess.run(item['command'],cwd=cwd,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=args.stage_timeout)
        except subprocess.TimeoutExpired:
            item['status']='timeout';save();raise
        item['returncode']=result.returncode
        item['status']='expected_failure' if expect_failure and result.returncode else ('passed' if result.returncode==0 and not expect_failure else 'failed')
        save()
        if item['status']=='failed':raise RuntimeError(f'{stage} failed; inspect {log}')
        return log.read_text()
    save()
    try:
        deployment=work/'deployment';deployment.mkdir()
        for backend,index,elements,title in [('cannbot',0,256,'Cannbot'),('tilelang',1,524288,'Tilelang')]:
            target=work/backend
            shutil.copytree(skill/'assets/ge-onnx-om',target,ignore=shutil.ignore_patterns('__pycache__'))
            for path in (skill/'assets/ge-common').iterdir():
                if path.is_file():shutil.copy2(path,target/path.name)
            model=helper.make_model(helper.make_graph([helper.make_node(title+'Add',['x','y'],['z'],name='dsl_add',domain='com.dsl_ge')],backend,
                [helper.make_tensor_value_info(name,TensorProto.FLOAT,[elements]) for name in ['x','y']],
                [helper.make_tensor_value_info('z',TensorProto.FLOAT,[elements])]),
                opset_imports=[helper.make_opsetid('',13),helper.make_opsetid('com.dsl_ge',1)])
            # Minimal example wire-format versions, independent of installed package version.
            model.ir_version=8
            onnx.checker.check_model(model,check_custom_domain=False)
            onnx.save(model,target/'model.onnx')
            run(backend+'-prepare',[sys.executable,target/'prepare.py','--native-fixture',fixture,'--backend',backend],target)
            run(backend+'-configure',['cmake','-S',target,'-B',target/'build',f'-DGE_CXX11_ABI={args.cxx11_abi}',f'-DDSL_BACKEND={index}','-DCMAKE_BUILD_TYPE=Release'],target)
            run(backend+'-build',['cmake','--build',target/'build','--parallel',args.build_jobs],target)
            opp=target/'build/opp'
            run(backend+'-check-opp',[sys.executable,skill/'scripts/check_opp.py',opp],target)
            op_library=next((opp/'op_graph/lib').rglob('libcust_opapi.so'))
            exports=subprocess.check_output(['nm','-D','--defined-only',str(op_library)],text=True)
            for symbol in ['GetRegisteredCustomOpCreatorAbiVersion','GetRegisteredCustomOpCreatorNum','GetRegisteredCustomOpCreators']:
                assert symbol in {s.split()[-1] for s in exports.splitlines() if s.split()},symbol
            linkage=subprocess.check_output(['ldd',str(op_library)],text=True)
            assert 'not found' not in linkage and 'libcannbot_add.so' not in linkage,linkage
            (target/'implementation-ldd.txt').write_text(linkage)
            env=os.environ.copy();env['ASCEND_CUSTOM_OPP_PATH']=str(opp)
            parser_dir=opp/'framework/onnx';hidden=target/'parser-disabled'
            parser_dir.rename(hidden)
            try:
                text=run(backend+'-missing-parser',['atc','--model='+str(target/'model.onnx'),'--framework=5','--input_format=ND','--output='+str(target/'negative'),'--soc_version='+args.soc_version,'--log=info'],target,env,expect_failure=True)
                assert 'com.dsl_ge::1::'+title+'Add' in text and re.search(r'no.*plugin.*registered|not.*support|parse.*fail',text,re.I),text[-3000:]
                assert not (target/'negative.om').exists()
            finally:hidden.rename(parser_dir)
            dumps=target/'dumps';dumps.mkdir()
            env.update(DUMP_GE_GRAPH='2',DUMP_GRAPH_LEVEL='2',DUMP_GRAPH_PATH=str(dumps))
            text=run(backend+'-atc',['atc','--model='+str(target/'model.onnx'),'--framework=5','--input_format=ND','--output='+str(target/'model'),'--soc_version='+args.soc_version,'--log=info'],target,env)
            assert '[ONNX_PARSE] status=0' in text
            run(backend+'-om-json',['atc','--mode=1','--om='+str(target/'model.om'),'--json='+str(target/'model.json')],target,env)
            nodes=custom_nodes(json.loads((target/'model.json').read_text()),title+'AddOm')
            assert len(nodes)==1,(backend,len(nodes))
            bundle=json.loads((target/'bundle.json').read_text())
            binary=(fixture/bundle['binary_source']).read_bytes()
            library=op_library.read_bytes();om=(target/'model.om').read_bytes()
            assert hashlib.sha256(binary).hexdigest()==bundle['binary_sha256']
            assert binary in library and library in om and binary in om,'OM does not contain exact implementation/kernel bytes'
            shutil.copy2(target/'model.om',deployment/(backend+'.om'))
            if not (deployment/'model_exec').exists():shutil.copy2(target/'build/model_exec',deployment/'model_exec')
            result={'backend':backend,'ge_type':title+'AddOm','onnx_sha256':digest(target/'model.onnx'),'onnx_checker':'structure only; no custom schema',
                    'missing_parser_control':'failed as expected','parsed_node':nodes[0],
                    'om_sha256':digest(target/'model.om'),'om_bytes':len(om),'implementation_sha256':digest(op_library),
                    'implementation_bytes_exact_in_om':True,'kernel_bytes_exact_in_om':True,'kernel_bundle':bundle,
                    'source_hashes':{p.name:digest(p) for p in target.iterdir() if p.is_file() and p.suffix in ['.cc','.h','.py','.txt']},
                    'status':'compiled'}
            report['backends'].append(result);save()
        assert sorted(p.name for p in deployment.iterdir())==['cannbot.om','model_exec','tilelang.om']
        isolate=work/'cannbot/isolate.sh'
        for result in report['backends']:
            backend=result['backend'];target=work/backend
            # Namespace mounts mask development files and all user/temp caches,
            # while the read-only deployment bind mount retains only OM + ACL runner.
            command=['unshare','--mount','bash',isolate,deployment,backend+'.om',args.device,result['kernel_bundle']['elements']]
            text=run(backend+'-isolated-run',command,target)
            assert '[ISOLATION] private_mounts=1' in text
            assert '[OM_LOAD] status=PASS' in text and '[OM_CLEANUP] status=PASS' in text
            assert f'[OM_CUSTOM_EXECUTE] backend={int(backend=="tilelang")}' in text
            assert 'kernel_sha256='+result['kernel_bundle']['binary_sha256'] in text
            assert len(re.findall(r'^\[OM_CASE\].*status=PASS$',text,re.M))==6
            match=re.search(r'^\[OM_RESULT\] (.+)$',text,re.M)
            assert match
            metrics=dict(re.findall(r'(\w+)=([^\s]+)',match.group(1)))
            assert metrics['status']=='PASS' and metrics['guards']=='PASS'
            result.update(status='passed',self_contained=True,isolated_metrics=metrics)
            save()
        report.update(status='passed',self_contained=True,calls=sum(int(b['isolated_metrics']['calls']) for b in report['backends']),
            checked=sum(int(b['isolated_metrics']['checked']) for b in report['backends']),
            max_abs=max(float(b['isolated_metrics']['max_abs']) for b in report['backends']),
            max_rmse=max(float(b['isolated_metrics']['rmse']) for b in report['backends']),
            deployment_files={p.name:digest(p) for p in deployment.iterdir()})
        save();print('[ONNX_OM_RESULT]',json.dumps({k:report[k] for k in ['status','self_contained','calls','checked','max_abs','max_rmse']}),flush=True)
    except Exception as error:
        report.update(status='failed',error=str(error));save();raise


if __name__=='__main__':main()
