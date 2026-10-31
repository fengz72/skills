"""One AOT artifact with a runtime tensor dimension; no source specialization per N."""
import cannbotdsl as cb

@cb.kernel
def dynamic_add_kernel(x,y,z):
    n=x.shape[0]
    # Thread count and loop stride are a paired specialization of this example.
    with cb.vf(mode="simt",thread=256):
        tid=cb.simt.thread_idx()[0]
        for i in cb.range(tid,n,256):
            z[i]=x[i]+y[i]

@cb.host
def dynamic_add_host(x,y,z):
    dynamic_add_kernel[1](x,y,z)

if __name__=='__main__':
    n=cb.Dim('N',min=1,max=4096)
    spec=cb.TensorSpec((n,),cb.dtypes.float32)
    cb.compile(dynamic_add_host,spec,spec,spec)
