"""Fixed 256-element FP32 vector add, using the installed cannbot-dsl API."""
import cannbotdsl as cb

@cb.kernel
def add_kernel(x, y, z):
    a = cb.Buffer(cb.MemLoc.UB, (256,), cb.dtypes.float32)
    b = cb.Buffer(cb.MemLoc.UB, (256,), cb.dtypes.float32)
    c = cb.Buffer(cb.MemLoc.UB, (256,), cb.dtypes.float32)
    cb.mem_copy(a, x)
    cb.mem_copy(b, y)
    # One thread per element in this fixed-size example; not a device capability limit.
    with cb.vf(mode="simt", thread=256):
        i = cb.simt.thread_idx()[0]
        c[i] = a[i] + b[i]
    cb.mem_copy(z, c)

@cb.host
def add_host(x, y, z):
    add_kernel[1](x, y, z)

if __name__ == '__main__':
    spec = cb.TensorSpec((256,), cb.dtypes.float32)
    cb.compile(add_host, spec, spec, spec)
