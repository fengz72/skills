"""AOT-only source-export smoke driver; fixed 256-element float32 copy."""
import cannbotdsl as cb


@cb.kernel
def copy_kernel(x, y):
    tmp = cb.Buffer(cb.MemLoc.UB, (256,), cb.dtypes.float32)
    cb.mem_copy(tmp, x)
    cb.mem_copy(y, tmp)


@cb.host
def copy_host(x, y):
    copy_kernel[1](x, y)


if __name__ == '__main__':
    spec = cb.TensorSpec((256,), cb.dtypes.float32)
    cb.compile(copy_host, spec, spec)
