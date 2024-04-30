# The LLVM Compiler Infrastructure

[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/llvm/llvm-project/badge)](https://securityscorecards.dev/viewer/?uri=github.com/llvm/llvm-project)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/8273/badge)](https://www.bestpractices.dev/projects/8273)
[![libc++](https://github.com/llvm/llvm-project/actions/workflows/libcxx-build-and-test.yaml/badge.svg?branch=main&event=schedule)](https://github.com/llvm/llvm-project/actions/workflows/libcxx-build-and-test.yaml?query=event%3Aschedule)

Welcome to the LLVM project!

This repository contains the source code for LLVM, a toolkit for the
construction of highly optimized compilers, optimizers, and run-time
environments.

The LLVM project has multiple components. The core of the project is
itself called "LLVM". This contains all of the tools, libraries, and header
files needed to process intermediate representations and convert them into
object files. Tools include an assembler, disassembler, bitcode analyzer, and
bitcode optimizer.

C-like languages use the [Clang](https://clang.llvm.org/) frontend. This
component compiles C, C++, Objective-C, and Objective-C++ code into LLVM bitcode
-- and from there into object files, using LLVM.

Other components include:
the [libc++ C++ standard library](https://libcxx.llvm.org),
the [LLD linker](https://lld.llvm.org), and more.

## Getting the Source Code and Building LLVM

Consult the
[Getting Started with LLVM](https://llvm.org/docs/GettingStarted.html#getting-the-source-code-and-building-llvm)
page for information on building and running LLVM.

For information on how to contribute to the LLVM project, please take a look at
the [Contributing to LLVM](https://llvm.org/docs/Contributing.html) guide.

## Getting in touch

Join the [LLVM Discourse forums](https://discourse.llvm.org/), [Discord
chat](https://discord.gg/xS7Z362),
[LLVM Office Hours](https://llvm.org/docs/GettingInvolved.html#office-hours) or
[Regular sync-ups](https://llvm.org/docs/GettingInvolved.html#online-sync-ups).

The LLVM project has adopted a [code of conduct](https://llvm.org/docs/CodeOfConduct.html) for
participants to all modes of communication within the project.

## 6156 Additions

Please find all project milestone assignments in the 6156 directory.

Test programs can be found in the `sparse-test` directory of the project root. 

To run the tests, the fork of our LLVM project must first be built. The following commands have worked for us:

```
$ cd ./build
$ cmake -DCMAKE_C_COMPILER=clang \
        -DCMAKE_CXX_COMPILER=clang++ \
        -DCMAKE_BUILD_TYPE=Debug \
        -DLLVM_ENABLE_PROJECTS=clang \
        -GNinja ../llvm
$ ninja -j<N_THREADS>
\end{lstlisting}
The commands above will build LLVM and Clang. Using the built binaries, we emit the LLVM IR of test \lstinline{C} programs via:
\begin{lstlisting}
$ <LLVM_ROOT>/build/bin/clang -S -fenable-matrix -emit-llvm <FILE>.c -o <FILE>.ll
```

We also designed a test for our `store` intrinsic, which may be run via 

```
$ <LLVM_ROOT>/build/bin/llvm-lit llvm/test/Transforms/LowerSparseIntrinsics/sparse-store-double.ll
```

For WIP code that depicts some of the details of our attempt in implementing more complex sparse matrix operations, please refer to the branches on our GitHub repository that is not `main`.
