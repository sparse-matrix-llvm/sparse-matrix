; ModuleID = 'test.c'
source_filename = "test.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@__const.main.pa = private unnamed_addr constant [4 x float] [float 1.000000e+00, float 2.000000e+00, float 3.000000e+00, float 4.000000e+00], align 16
@__const.main.pb = private unnamed_addr constant [4 x float] [float 1.000000e+00, float 0.000000e+00, float 0.000000e+00, float 1.000000e+00], align 16

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @f(ptr noundef %pa, ptr noundef %pb) #0 {
entry:
  %pa.addr = alloca ptr, align 8
  %pb.addr = alloca ptr, align 8
  %a = alloca [4 x float], align 4
  %b = alloca [4 x float], align 4
  %r = alloca [4 x float], align 4
  store ptr %pa, ptr %pa.addr, align 8
  store ptr %pb, ptr %pb.addr, align 8
  %0 = load ptr, ptr %pa.addr, align 8
  %col.load = load <2 x float>, ptr %0, align 4
  %vec.gep = getelementptr float, ptr %0, i64 4
  %col.load2 = load <2 x float>, ptr %vec.gep, align 4
  store <2 x float> %col.load, ptr %a, align 4
  %vec.gep3 = getelementptr float, ptr %a, i64 2
  store <2 x float> %col.load2, ptr %vec.gep3, align 4
  %1 = load ptr, ptr %pb.addr, align 8
  %col.load4 = load <2 x float>, ptr %1, align 4
  %vec.gep5 = getelementptr float, ptr %1, i64 4
  %col.load6 = load <2 x float>, ptr %vec.gep5, align 4
  store <2 x float> %col.load4, ptr %b, align 4
  %vec.gep7 = getelementptr float, ptr %b, i64 2
  store <2 x float> %col.load6, ptr %vec.gep7, align 4
  %col.load8 = load <2 x float>, ptr %a, align 4
  %vec.gep9 = getelementptr float, ptr %a, i64 2
  %col.load10 = load <2 x float>, ptr %vec.gep9, align 4
  %col.load11 = load <2 x float>, ptr %b, align 4
  %vec.gep12 = getelementptr float, ptr %b, i64 2
  %col.load13 = load <2 x float>, ptr %vec.gep12, align 4
  %block = shufflevector <2 x float> %col.load8, <2 x float> poison, <2 x i32> <i32 0, i32 1>
  %2 = extractelement <2 x float> %col.load11, i64 0
  %splat.splatinsert = insertelement <2 x float> poison, float %2, i64 0
  %splat.splat = shufflevector <2 x float> %splat.splatinsert, <2 x float> poison, <2 x i32> zeroinitializer
  %3 = fmul <2 x float> %block, %splat.splat
  %block14 = shufflevector <2 x float> %col.load10, <2 x float> poison, <2 x i32> <i32 0, i32 1>
  %4 = extractelement <2 x float> %col.load11, i64 1
  %splat.splatinsert15 = insertelement <2 x float> poison, float %4, i64 0
  %splat.splat16 = shufflevector <2 x float> %splat.splatinsert15, <2 x float> poison, <2 x i32> zeroinitializer
  %5 = fmul <2 x float> %block14, %splat.splat16
  %6 = fadd <2 x float> %3, %5
  %7 = shufflevector <2 x float> %6, <2 x float> poison, <2 x i32> <i32 0, i32 1>
  %8 = shufflevector <2 x float> poison, <2 x float> %7, <2 x i32> <i32 2, i32 3>
  %block17 = shufflevector <2 x float> %col.load8, <2 x float> poison, <2 x i32> <i32 0, i32 1>
  %9 = extractelement <2 x float> %col.load13, i64 0
  %splat.splatinsert18 = insertelement <2 x float> poison, float %9, i64 0
  %splat.splat19 = shufflevector <2 x float> %splat.splatinsert18, <2 x float> poison, <2 x i32> zeroinitializer
  %10 = fmul <2 x float> %block17, %splat.splat19
  %block20 = shufflevector <2 x float> %col.load10, <2 x float> poison, <2 x i32> <i32 0, i32 1>
  %11 = extractelement <2 x float> %col.load13, i64 1
  %splat.splatinsert21 = insertelement <2 x float> poison, float %11, i64 0
  %splat.splat22 = shufflevector <2 x float> %splat.splatinsert21, <2 x float> poison, <2 x i32> zeroinitializer
  %12 = fmul <2 x float> %block20, %splat.splat22
  %13 = fadd <2 x float> %10, %12
  %14 = shufflevector <2 x float> %13, <2 x float> poison, <2 x i32> <i32 0, i32 1>
  %15 = shufflevector <2 x float> poison, <2 x float> %14, <2 x i32> <i32 2, i32 3>
  %16 = fadd <2 x float> %8, <float 1.000000e+01, float 1.000000e+01>
  %17 = fadd <2 x float> %15, <float 1.000000e+01, float 1.000000e+01>
  store <2 x float> %16, ptr %r, align 4
  %vec.gep23 = getelementptr float, ptr %r, i64 2
  store <2 x float> %17, ptr %vec.gep23, align 4
  %col.load24 = load <2 x float>, ptr %r, align 4
  %vec.gep25 = getelementptr float, ptr %r, i64 2
  %col.load26 = load <2 x float>, ptr %vec.gep25, align 4
  %18 = load ptr, ptr %pa.addr, align 8
  store <2 x float> %col.load24, ptr %18, align 4
  %vec.gep27 = getelementptr float, ptr %18, i64 4
  store <2 x float> %col.load26, ptr %vec.gep27, align 4
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare <4 x float> @llvm.matrix.column.major.load.v4f32.i64(ptr nocapture, i64, i1 immarg, i32 immarg, i32 immarg) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <4 x float> @llvm.matrix.multiply.v4f32.v4f32.v4f32(<4 x float>, <4 x float>, i32 immarg, i32 immarg, i32 immarg) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: write)
declare void @llvm.matrix.column.major.store.v4f32.i64(<4 x float>, ptr nocapture writeonly, i64, i1 immarg, i32 immarg, i32 immarg) #3

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 {
entry:
  %retval = alloca i32, align 4
  %pa = alloca [4 x float], align 16
  %pb = alloca [4 x float], align 16
  store i32 0, ptr %retval, align 4
  call void @llvm.memcpy.p0.p0.i64(ptr align 16 %pa, ptr align 16 @__const.main.pa, i64 16, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr align 16 %pb, ptr align 16 @__const.main.pb, i64 16, i1 false)
  ret i32 0
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #4

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: write) }
attributes #4 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 2}
!5 = !{!"clang version 19.0.0git (git@github.com:sparse-matrix-llvm/sparse-matrix.git cebf77fb936a7270c7e3fa5c4a7e76216321d385)"}
