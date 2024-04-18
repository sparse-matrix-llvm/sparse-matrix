
//===- llvm/MatrixBuilder.h - Builder to lower matrix ops -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the MatrixBuilder class, which is used as a convenient way
// to lower matrix operations to LLVM IR.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_SPARSEMATRIXBUILDER_H
#define LLVM_IR_SPARSEMATRIXBUILDER_H

#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Alignment.h"

namespace llvm {

class Function;
class Twine;
class Module;

class SparseMatrixBuilder {
  IRBuilderBase &B;
  Module *getModule() { return B.GetInsertBlock()->getParent()->getParent(); }


public:
  SparseMatrixBuilder(IRBuilderBase &Builder) : B(Builder) {}

  /// Create a column major, strided matrix load.
  /// \p EltTy   - Matrix element type
  /// \p DataPtr - Start address of the matrix read
  /// \p Rows    - Number of rows in matrix (must be a constant)
  /// \p Columns - Number of columns in matrix (must be a constant)
  /// \p Stride  - Space between columns
  CallInst *CreateSparseMatrixLoad(Type *EltTy, Value *DataPtr, unsigned Rows, unsigned Columns) {
    DEBUG_WITH_TYPE("sparse", llvm::dbgs() << "hi hello CreateSparseMatrixLoad\n");
    auto *RetType = FixedVectorType::get(EltTy, Rows * Columns);
    DEBUG_WITH_TYPE("sparse", llvm::dbgs() << "hi hello CreateSparseMatrixLoad 1\n");
    Value *Ops[] = {DataPtr, B.getInt32(Rows), B.getInt32(Columns)};
    DEBUG_WITH_TYPE("sparse", llvm::dbgs() << "hi hello CreateSparseMatrixLoad 2\n");
    Type *OverloadedTypes[] = {RetType};
    DEBUG_WITH_TYPE("sparse", llvm::dbgs() << "hi hello CreateSparseMatrixLoad 3\n");

    Function *TheFn = Intrinsic::getDeclaration(
        getModule(), Intrinsic::csc_matrix_load, OverloadedTypes);
    
    DEBUG_WITH_TYPE("sparse", llvm::dbgs() << "hi hello CreateSparseMatrixLoad 4\n");

    CallInst *Call = B.CreateCall(TheFn->getFunctionType(), TheFn, Ops, "");

    DEBUG_WITH_TYPE("sparse", llvm::dbgs() << "hi hello CreateSparseMatrixLoad 5\n");

    Attribute AlignAttr =
        Attribute::getWithAlignment(Call->getContext(), Align());

    Call->addParamAttr(0, AlignAttr);

    return Call;
  }

  /// Create a column major, strided matrix store.
  /// \p Matrix  - Matrix to store
  /// \p Ptr     - Pointer to write back to
  /// \p Stride  - Space between columns
//  CallInst *CreateSparseMatrixStore(Value *Matrix, Value *Ptr, Align Alignment,
//                                   Value *Stride, bool IsVolatile,
//                                   unsigned Rows, unsigned Columns,
//                                   const Twine &Name = "") {
//    Value *Ops[] = {Matrix,           Ptr,
//                    Stride,           B.getInt1(IsVolatile),
//                    B.getInt32(Rows), B.getInt32(Columns)};
//    Type *OverloadedTypes[] = {Matrix->getType(), Stride->getType()};
//
//    Function *TheFn = Intrinsic::getDeclaration(
//        getModule(), Intrinsic::matrix_column_major_store, OverloadedTypes);
//
//    CallInst *Call = B.CreateCall(TheFn->getFunctionType(), TheFn, Ops, Name);
//    Attribute AlignAttr =
//        Attribute::getWithAlignment(Call->getContext(), Alignment);
//    Call->addParamAttr(1, AlignAttr);
//    return Call;
//  }
};

} // end namespace llvm

#endif // LLVM_IR_MATRIXBUILDER_H
