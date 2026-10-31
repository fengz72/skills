"""Dynamic FP32 Add using a runtime N and guarded final tile."""
import tilelang.ascend.language as T

def dynamic_add():
    n=T.dynamic('N',dtype='int64')
    @T.prim_func
    def main(A:T.Tensor((n,),'float32'),B:T.Tensor((n,),'float32'),C:T.Tensor((n,),'float32')):
        # Single-block, 256-element tiles are this example's algorithm configuration.
        with T.Kernel(1):
            a=T.alloc_shared((256,),'float32')
            b=T.alloc_shared((256,),'float32')
            c=T.alloc_shared((256,),'float32')
            for tile in T.serial(T.ceildiv(n,256)):
                T.copy(A[tile*256:tile*256+256],a)
                T.copy(B[tile*256:tile*256+256],b)
                with T.SimtVF(threads=256):
                    for i in T.Parallel(256):
                        c[i]=a[i]+b[i]
                T.copy(c,C[tile*256:tile*256+256])
    return main
