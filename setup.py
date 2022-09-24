from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
setup(
name="caculate",# python中导入的名称
ext_modules=[
CppExtension("caculate", ["./caculate.cpp"])# 若没有使用到cuda，也可以使用CppExtension， 此处的test名和上面name的需要一样
],
cmdclass={
'build_ext': BuildExtension
}
)

