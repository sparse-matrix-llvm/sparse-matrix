//===- LowerSparseIntrinsics.cpp -  Lower sparse matrix intrinsics -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower sparse intrinsics to vector operations.
//
// TODO:
//  * A lot
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/LowerSparseIntrinsics.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Scalar.h"

using namespace llvm;

#define DEBUG_TYPE "lower-sparse-intrinsics"

namespace {

class LowerSparseIntrinsics {
  Function &Func;
  const DataLayout &DL;
  const TargetTransformInfo &TTI;

//   /// Wrapper class representing a matrix as a set of column vectors.
//   /// All column vectors must have the same vector type.
  struct CSCMatrixTy {
    SmallVector<Value *, 16> Values;
    SmallVector<Value *, 16> RowIndices;
    SmallVector<Value *, 16> ColPointers;
    std::uint32_t nnz;

  // public:
    CSCMatrixTy() : Values(), RowIndices(), ColPointers(), nnz(0) {}
    CSCMatrixTy(ArrayRef<Value *> Values, ArrayRef<Value *> RowIndices,
                   ArrayRef<Value *> ColPointers, uint32_t nnz)
        : Values(Values.begin(), Values.end()),
          RowIndices(RowIndices.begin(), RowIndices.end()),
          ColPointers(ColPointers.begin(), ColPointers.end()), nnz(nnz) {}

    // // Value *getColumn(unsigned i) const { return Columns[i]; }

    // void setColumn(unsigned i, Value *V) { Columns[i] = V; }

    // size_t getNumColumns() const { return Columns.size(); }

    // const SmallVectorImpl<Value *> &getColumnVectors() const { return Columns; }

    // SmallVectorImpl<Value *> &getColumnVectors() { return Columns; }

    // void addColumn(Value *V) { Columns.push_back(V); }

    // iterator_range<SmallVector<Value *, 8>::iterator> columns() {
    //   return make_range(Columns.begin(), Columns.end());
    // }

    /// Embed the columns of the matrix into a flat vector by concatenating
    /// them.
    // Value *embedInVector(IRBuilder<> &Builder) const {
    //   return Columns.size() == 1 ? Columns[0]
    //                              : concatenateVectors(Builder, Columns);
    // }
  };

  struct ShapeInfo {
    unsigned NumRows;
    unsigned NumColumns;

    ShapeInfo(unsigned NumRows = 0, unsigned NumColumns = 0)
        : NumRows(NumRows), NumColumns(NumColumns) {}

    ShapeInfo(ConstantInt *NumRows, ConstantInt *NumColumns)
        : NumRows(NumRows->getZExtValue()),
          NumColumns(NumColumns->getZExtValue()) {}
  };

  // namespace {
  // Value *computeVectorAddr(Value *BasePtr, Value *VecIdx, Value *Stride,
  //                          unsigned NumElements, Type *EltType,
  //                          IRBuilder<> &Builder) {

  //   // assert((!isa<ConstantInt>(Stride) ||
  //           // cast<ConstantInt>(Stride)->getZExtValue() >= NumElements) &&
  //         //  "Stride must be >= the number of elements in the result vector.");

  //   // Compute the start of the vector with index VecIdx as VecIdx * Stride.
  //   // Value *VecStart = Builder.CreateMul(VecIdx, Stride, "vec.start");

  //   // Get pointer to the start of the selected vector. Skip GEP creation,
  //   // if we select vector 0.
  //   if (isa<ConstantInt>(VecStart) && cast<ConstantInt>(VecStart)->isZero())
  //     VecStart = BasePtr;
  //   else
  //     VecStart = Builder.CreateGEP(EltType, BasePtr, VecStart, "vec.gep");

  //   return VecStart;
  // }
  // }

public:
  LowerSparseIntrinsics(Function &F, TargetTransformInfo &TTI)
      : Func(F), DL(F.getParent()->getDataLayout()), TTI(TTI) {}

  // / Return the set of column vectors that a matrix value is lowered to.
  // /
  // / We split the flat vector \p MatrixVal containing a matrix with shape \p
  // SI / into column vectors.
  CSCMatrixTy getMatrix(Value *MatrixVal, const ShapeInfo &SI,
                        IRBuilder<> &Builder) {
    VectorType *VType = dyn_cast<VectorType>(MatrixVal->getType());
    assert(VType && "CSCMatrixVal must be a vector type");
    // assert(VType->getNumElements() == SI.NumRows * SI.NumColumns &&
    //        "The vector size must match the number of matrix elements");

    Value *Index = Builder.getInt32(0);
    Value *FirstElement =
        Builder.CreateExtractElement(MatrixVal, Index, "firstElement");
    ConstantInt *CInt = dyn_cast<ConstantInt>(FirstElement);
    assert(CInt && "nnz must be an integer");
    assert(!CInt->isNegative() && "nnz must be nonnegative");
    unsigned nnz = CInt->getZExtValue();

    unsigned OffsetColPointers = 1;
    unsigned OffsetRowIndices = OffsetColPointers + SI.NumColumns + 1;
    unsigned OffsetValues = OffsetRowIndices + nnz;

    auto MaskColPtrs =
        std::move(createSequentialMask(OffsetColPointers, OffsetRowIndices, 0));
    Value *ColPointers =
        Builder.CreateShuffleVector(MatrixVal, MaskColPtrs, "colPointers");

    auto MaskRowIndices =
        std::move(createSequentialMask(OffsetRowIndices, OffsetValues, 0));
    Value *RowIndices =
        Builder.CreateShuffleVector(MatrixVal, MaskRowIndices, "rowIndices");

    auto MaskValues =
        std::move(createSequentialMask(OffsetValues, OffsetValues + nnz, 0));
    Value *Values =
        Builder.CreateShuffleVector(MatrixVal, MaskValues, "values");

    return {ColPointers, RowIndices, Values, nnz};
  }

  // Replace intrinsic calls
  bool VisitCallInst(CallInst *Inst) {
    if (!Inst->getCalledFunction() || !Inst->getCalledFunction()->isIntrinsic())
      return false;

    switch (Inst->getCalledFunction()->getIntrinsicID()) {
    // case Intrinsic::matrix_multiply:
    //   LowerMultiply(Inst);
    //   break;
    // case Intrinsic::matrix_transpose:
    //   LowerTranspose(Inst);
    //   break;
    // case Intrinsic::matrix_columnwise_load:
    //   LowerColumnwiseLoad(Inst);
    //   break;
    case Intrinsic::csc_matrix_load:
      LowerCSCLoad(Inst);
      break;
    case Intrinsic::csc_matrix_store:
      LowerCSCStore(Inst);
      break;
    default:
      return false;
    }
    Inst->eraseFromParent();
    return true;
  }

  bool Visit() {
    ReversePostOrderTraversal<Function *> RPOT(&Func);
    bool Changed = false;
    for (auto *BB : RPOT) {
      for (Instruction &Inst : make_early_inc_range(*BB)) {
        if (CallInst *CInst = dyn_cast<CallInst>(&Inst))
          Changed |= VisitCallInst(CInst);
      }
    }

    return Changed;
  }

//   LoadInst *createColumnLoad(Value *ColumnPtr, Type *EltType,
//                              IRBuilder<> Builder) {
//     unsigned Align = DL.getABITypeAlignment(EltType);
//     return Builder.CreateAlignedLoad(ColumnPtr, Align);
//   }

  // StoreInst *createColumnStore(Value *ColumnValue, Value *ColumnPtr,
  //                              Type *EltType, IRBuilder<> Builder) {
  //   unsigned Align = DL.getABITypeAlignment(EltType);
  //   return Builder.dStore(ColumnValue, ColumnPtr, Align);
  // }

  /// Turns \p BasePtr into an elementwise pointer to \p EltType.
  Value *createElementPtr(Value *BasePtr, Type *EltType, IRBuilder<> &Builder) {
    unsigned AS = cast<PointerType>(BasePtr->getType())->getAddressSpace();
    Type *EltPtrType = PointerType::get(EltType, AS);
    return Builder.CreatePointerCast(BasePtr, EltPtrType);
  }

  /// Lowers llvm.matrix.columnwise.load.
  ///
  /// The intrinsic loads a matrix from memory using a stride between columns.
//   void LowerColumnwiseLoad(CallInst *Inst) {
//     IRBuilder<> Builder(Inst);
//     Value *Ptr = Inst->getArgOperand(0);
//     Value *Stride = Inst->getArgOperand(1);
//     auto VType = cast<VectorType>(Inst->getType());
//     ShapeInfo Shape(cast<ConstantInt>(Inst->getArgOperand(2)),
//                     cast<ConstantInt>(Inst->getArgOperand(3)));
//     Value *EltPtr = createElementPtr(Ptr, VType->getElementType(), Builder);

//     ColumnMatrixTy Result;
//     // Distance between start of one column and the start of the next
//     for (unsigned C = 0, E = Shape.NumColumns; C < E; ++C) {
//       Value *GEP =
//           computeColumnAddr(EltPtr, Builder.getInt32(C), Stride, Shape.NumRows,
//                             VType->getElementType(), Builder);
//       Value *Column = createColumnLoad(GEP, VType->getElementType(), Builder);
//       Result.addColumn(Column);
//     }

//     Inst->replaceAllUsesWith(Result.embedInVector(Builder));
//   }

  void LowerCSCLoad(CallInst *Inst) {
    IRBuilder<> Builder(Inst);
    DEBUG_WITH_TYPE("sparse", llvm::dbgs() << "hi hello LowerCSCLoad 1 \n");

    Value *Ptr = Inst->getArgOperand(0);
    DEBUG_WITH_TYPE("sparse", llvm::dbgs() << "hi hello LowerCSCLoad 2\n");
    
    ShapeInfo Shape(cast<ConstantInt>(Inst->getArgOperand(1)),
                    cast<ConstantInt>(Inst->getArgOperand(2)));

    DEBUG_WITH_TYPE("sparse", llvm::dbgs() << "hi hello LowerCSCLoad 3\n");
    CSCMatrixTy Result;

    Value *Index = Builder.getInt32(0);
    Value *FirstElement = Builder.CreateExtractElement(Ptr, Index, "firstElement");
    ConstantInt *CInt = dyn_cast<ConstantInt>(FirstElement);
    
    unsigned nnz = CInt->getZExtValue();
    unsigned OffsetColPointers = 1;
    unsigned OffsetRowIndices = OffsetColPointers + Shape.NumColumns + 1;

    // Builder.CreateAlignedLoad(row_indices, ...)
    // Builder.CreateAlignedLoad(col_pointers, ...)
    // Builder.CreateAlignedLoad(col_pointers, ...)

    DEBUG_WITH_TYPE("sparse", llvm::dbgs() << "load nnz: " << nnz << "\n");
    DEBUG_WITH_TYPE("sparse", llvm::dbgs() << "load col: " << OffsetColPointers << "\n");
    DEBUG_WITH_TYPE("sparse", llvm::dbgs() << "load row: " << OffsetRowIndices << "\n");
  }

  /// Lowers llvm.matrix.columnwise.store.
  ///
  /// The intrinsic store a matrix back memory using a stride between columns.
  void LowerCSCStore(CallInst *Inst) {
    IRBuilder<> Builder(Inst);
    Value *Matrix = Inst->getArgOperand(0);
    Value *Ptr = Inst->getArgOperand(1);
    // later check if this makes sense
    // ShapeInfo Shape(cast<ConstantInt>(Inst->getArgOperand(2)),
    //                 cast<ConstantInt>(Inst->getArgOperand(3)));
    auto VType = cast<VectorType>(Matrix->getType());
    // Value *EltPtr = createElementPtr(Ptr, VType->getElementType(), Builder);
    Align InitialAlign =
        DL.getValueOrABITypeAlignment(Inst->getParamAlign(1), VType->getElementType());
    Builder.CreateAlignedStore(Matrix, Ptr, InitialAlign);
  }

//   /// Extract a column vector of \p NumElts starting at index (\p I, \p J) from
//   /// the matrix \p LM represented as a vector of column vectors.
//   Value *extractVector(const ColumnMatrixTy &LM, unsigned I, unsigned J,
//                        unsigned NumElts, IRBuilder<> Builder) {
//     Value *Col = LM.getColumn(J);
//     Value *Undef = UndefValue::get(Col->getType());
//     Constant *Mask = createSequentialMask(Builder, I, NumElts, 0);
//     return Builder.CreateShuffleVector(Col, Undef, Mask, "block");
//   }

//   // Set elements I..I+NumElts-1 to Block
//   Value *insertVector(Value *Col, unsigned I, Value *Block,
//                       IRBuilder<> Builder) {

//     // First, bring Block to the same size as Col
//     unsigned BlockNumElts =
//         cast<VectorType>(Block->getType())->getNumElements();
//     unsigned NumElts = cast<VectorType>(Col->getType())->getNumElements();
//     assert(NumElts >= BlockNumElts && "Too few elements for current block");

//     Value *ExtendMask =
//         createSequentialMask(Builder, 0, BlockNumElts, NumElts - BlockNumElts);
//     Value *Undef = UndefValue::get(Block->getType());
//     Block = Builder.CreateShuffleVector(Block, Undef, ExtendMask);

//     // If Col is 7 long and I is 2 and BlockNumElts is 2 the mask is: 0, 1, 7,
//     // 8, 4, 5, 6
//     SmallVector<Constant *, 16> Mask;
//     unsigned i;
//     for (i = 0; i < I; i++)
//       Mask.push_back(Builder.getInt32(i));

//     unsigned VecNumElts = cast<VectorType>(Col->getType())->getNumElements();
//     for (; i < I + BlockNumElts; i++)
//       Mask.push_back(Builder.getInt32(i - I + VecNumElts));

//     for (; i < VecNumElts; i++)
//       Mask.push_back(Builder.getInt32(i));

//     Value *MaskVal = ConstantVector::get(Mask);

//     return Builder.CreateShuffleVector(Col, Block, MaskVal);
//   }

//   Value *createMulAdd(Value *Sum, Value *A, Value *B, bool UseFPOp,
//                       IRBuilder<> &Builder) {
//     Value *Mul = UseFPOp ? Builder.CreateFMul(A, B) : Builder.CreateMul(A, B);
//     if (!Sum)
//       return Mul;

//     return UseFPOp ? Builder.CreateFAdd(Sum, Mul) : Builder.CreateAdd(Sum, Mul);
//   }

//   /// Lowers llvm.matrix.multiply.
//   void LowerMultiply(CallInst *MatMul) {
//     IRBuilder<> Builder(MatMul);
//     auto *EltType = cast<VectorType>(MatMul->getType())->getElementType();
//     ShapeInfo LShape(cast<ConstantInt>(MatMul->getArgOperand(2)),
//                      cast<ConstantInt>(MatMul->getArgOperand(3)));
//     ShapeInfo RShape(cast<ConstantInt>(MatMul->getArgOperand(3)),
//                      cast<ConstantInt>(MatMul->getArgOperand(4)));

//     const ColumnMatrixTy &Lhs =
//         getMatrix(MatMul->getArgOperand(0), LShape, Builder);
//     const ColumnMatrixTy &Rhs =
//         getMatrix(MatMul->getArgOperand(1), RShape, Builder);

//     const unsigned R = LShape.NumRows;
//     const unsigned M = LShape.NumColumns;
//     const unsigned C = RShape.NumColumns;
//     assert(M == RShape.NumRows);

//     // Initialize the output
//     ColumnMatrixTy Result;
//     for (unsigned J = 0; J < C; ++J)
//       Result.addColumn(UndefValue::get(VectorType::get(EltType, R)));

//     const unsigned VF = std::max(TTI.getRegisterBitWidth(true) /
//                                      EltType->getPrimitiveSizeInBits(),
//                                  uint64_t(1));

//     // Multiply columns from the first operand with scalars from the second
//     // operand.  Then move along the K axes and accumulate the columns.  With
//     // this the adds can be vectorized without reassociation.
//     for (unsigned J = 0; J < C; ++J) {
//       unsigned BlockSize = VF;
//       for (unsigned I = 0; I < R; I += BlockSize) {
//         // Gradually lower the vectorization factor to cover the remainder.
//         while (I + BlockSize > R)
//           BlockSize /= 2;

//         Value *Sum = nullptr;
//         for (unsigned K = 0; K < M; ++K) {
//           Value *L = extractVector(Lhs, I, K, BlockSize, Builder);
//           Value *RH = Builder.CreateExtractElement(Rhs.getColumn(J), K);
//           Value *Splat = Builder.CreateVectorSplat(BlockSize, RH, "splat");
//           Sum = createMulAdd(Sum, L, Splat, EltType->isFloatingPointTy(),
//                              Builder);
//         }
//         Result.setColumn(J, insertVector(Result.getColumn(J), I, Sum, Builder));
//       }
//     }

//     MatMul->replaceAllUsesWith(Result.embedInVector(Builder));
//   }

//   /// Lowers llvm.matrix.transpose.
//   void LowerTranspose(CallInst *Inst) {
//     ColumnMatrixTy Result;
//     IRBuilder<> Builder(Inst);
//     Value *InputVal = Inst->getArgOperand(0);
//     VectorType *VectorTy = cast<VectorType>(InputVal->getType());
//     ShapeInfo ArgShape(cast<ConstantInt>(Inst->getArgOperand(1)),
//                        cast<ConstantInt>(Inst->getArgOperand(2)));
//     ColumnMatrixTy InputMatrix = getMatrix(InputVal, ArgShape, Builder);

//     for (unsigned Row = 0; Row < ArgShape.NumRows; ++Row) {
//       // Build a single column vector for this row. First initialize it.
//       Value *ResultColumn = UndefValue::get(
//           VectorType::get(VectorTy->getElementType(), ArgShape.NumColumns));

//       // Go through the elements of this row and insert it into the resulting
//       // column vector.
//       for (auto C : enumerate(InputMatrix.columns())) {
//         Value *Elt = Builder.CreateExtractElement(C.value(), Row);
//         // We insert at index Column since that is the row index after the
//         // transpose.
//         ResultColumn =
//             Builder.CreateInsertElement(ResultColumn, Elt, C.index());
//       }
//       Result.addColumn(ResultColumn);
//     }

//     Inst->replaceAllUsesWith(Result.embedInVector(Builder));
//   }
};
} // namespace

PreservedAnalyses LowerSparseIntrinsicsPass::run(Function &F,
                                                 FunctionAnalysisManager &AM) {
  auto &TTI = AM.getResult<TargetIRAnalysis>(F);
  LowerSparseIntrinsics LMT(F, TTI);
  if (LMT.Visit()) {
    PreservedAnalyses PA;
    PA.preserveSet<CFGAnalyses>();
    return PA;
  }
  return PreservedAnalyses::all();
}
