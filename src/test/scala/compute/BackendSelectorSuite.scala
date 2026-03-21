package compute

import linalg.Matrix
import munit.FunSuite

class BackendSelectorSuite extends FunSuite:
  test("normalize backend aliases") {
    assertEquals(BackendSelector.normalizeBackend("cpu"), "cpu")
    assertEquals(BackendSelector.normalizeBackend("metal"), "gpu")
    assertEquals(BackendSelector.normalizeBackend("GPU"), "gpu")
    assertEquals(BackendSelector.normalizeBackend("anything"), "cpu")
  }

  test("normalize precision") {
    assertEquals(BackendSelector.normalizePrecision("fp32"), "fp32")
    assertEquals(BackendSelector.normalizePrecision("FP64"), "fp64")
    assertEquals(BackendSelector.normalizePrecision("invalid"), "fp64")
  }

  test("cpu backend selection") {
    val b = BackendSelector.fromConfig("cpu", "fp64")
    assertEquals(b.name, "cpu")
    assert(!b.isGpu)
  }

  test("gpu selection returns metal wrapper and is safe") {
    val b = BackendSelector.fromConfig("gpu", "fp64")
    assertEquals(b.name, "gpu")
    // In CI/dev without JNI, this can be false and should still be safe.
    assert(b.diagnostics.nonEmpty)
  }

  test("cpu backend exposes no gpu ops and supports fused linearActivation") {
    val b = CpuBackend("fp64")
    assertEquals(b.gpuOpsEnabled, Set.empty)
    val m = Matrix.fromFunction(2, 2)((r, c) => if r == c then 1.0 else 0.0)
    val x = Vector(1.0, -2.0)
    val bias = Vector(0.5, 0.5)
    val (z, a) = b.linearActivation(m, x, bias, "relu")
    assertEquals(z, Vector(1.5, -1.5))
    assertEquals(a, Vector(1.5, 0.0))
  }

  test("metal outer parity when gpu is available") {
    val b = MetalBackend.create("fp32")
    if !b.isGpu then assert(true)
    else
      val a = Vector(1.2, -0.3, 2.0)
      val c = Vector(0.5, -1.0)
      val cpu = CpuBackend("fp64").outer(a, c)
      val gpu = b.outer(a, c)
      val maxErr = cpu.data.zip(gpu.data).map((x, y) => math.abs(x - y)).max
      assert(maxErr < 1e-4)
  }

  test("metal fused linearActivation parity when gpu is available") {
    val b = MetalBackend.create("fp32")
    if !b.isGpu then assert(true)
    else
      val m = Matrix.fromFunction(4, 3)((r, c) => ((r + c) % 5) * 0.1)
      val x = Vector(0.2, -0.1, 0.7)
      val bias = Vector(0.01, -0.02, 0.03, 0.04)
      val cpu = CpuBackend("fp64").linearActivation(m, x, bias, "tanh")
      val gpu = b.linearActivation(m, x, bias, "tanh")
      val maxErrZ = cpu._1.zip(gpu._1).map((x, y) => math.abs(x - y)).max
      val maxErrA = cpu._2.zip(gpu._2).map((x, y) => math.abs(x - y)).max
      assert(maxErrZ < 1e-4)
      assert(maxErrA < 1e-4)
  }

  test("metal batch matMul parity when gpu is available") {
    val b = MetalBackend.create("fp32")
    if !b.isGpu then assert(true)
    else
      val a = Matrix.fromFunction(3, 4)((r, c) => ((r * 7 + c * 3) % 11) * 0.1 - 0.3)
      val c = Matrix.fromFunction(4, 2)((r, col) => ((r + col * 5) % 9) * 0.07 - 0.2)
      val cpu = CpuBackend("fp64").matMul(a, c)
      val gpu = b.matMul(a, c)
      val maxErr = cpu.data.zip(gpu.data).map((x, y) => math.abs(x - y)).max
      assert(maxErr < 1e-4)
  }

  test("metal batch linearActivation parity when gpu is available") {
    val b = MetalBackend.create("fp32")
    if !b.isGpu then assert(true)
    else
      val x = Matrix.fromFunction(5, 3)((r, c) => ((r + c * 2) % 7) * 0.1 - 0.2)
      val wT = Matrix.fromFunction(3, 4)((r, c) => ((r * 2 + c * 3) % 13) * 0.05 - 0.1)
      val bias = Vector(0.01, -0.02, 0.03, 0.04)
      val cpu = CpuBackend("fp64").linearActivationBatch(x, wT, bias, "tanh")
      val gpu = b.linearActivationBatch(x, wT, bias, "tanh")
      val maxErrZ = cpu._1.data.zip(gpu._1.data).map((a, z) => math.abs(a - z)).max
      val maxErrA = cpu._2.data.zip(gpu._2.data).map((a, z) => math.abs(a - z)).max
      assert(maxErrZ < 1e-4)
      assert(maxErrA < 1e-4)
  }

  test("metal batch softmax and crossEntropy parity when gpu is available") {
    val b = MetalBackend.create("fp32")
    if !b.isGpu then assert(true)
    else
      val logits = Matrix.fromFunction(4, 5)((r, c) => (r * 5 + c).toDouble * 0.2 - 1.0)
      val targets = Vector(0, 3, 1, 4)
      val cpuBackend = CpuBackend("fp64")
      val cpuSoftmax = cpuBackend.softmaxStableBatch(logits)
      val gpuSoftmax = b.softmaxStableBatch(logits)
      val maxErrSoftmax = cpuSoftmax.data.zip(gpuSoftmax.data).map((x, y) => math.abs(x - y)).max
      assert(maxErrSoftmax < 1e-4)

      val cpuLoss = cpuBackend.crossEntropyBatch(cpuSoftmax, targets)
      val gpuLoss = b.crossEntropyBatch(gpuSoftmax, targets)
      val maxErrLoss = cpuLoss.zip(gpuLoss).map((x, y) => math.abs(x - y)).max
      assert(maxErrLoss < 1e-4)
  }

  test("metal fallback works when native ops fail") {
    val b = MetalBackend.create("fp32")
    val m = Matrix.fromFunction(2, 2)((r, c) => if r == c then 2.0 else 1.0)
    val x = Vector(1.0, 2.0)
    val bias = Vector(0.0, 0.0)
    val cpu = CpuBackend("fp64")

    System.setProperty("metal.simulate.fail.matvec", "1")
    System.setProperty("metal.simulate.fail.outer", "1")
    System.setProperty("metal.simulate.fail.linear", "1")
    try
      assertEquals(b.matVecMul(m, x), cpu.matVecMul(m, x))
      assertEquals(b.outer(Vector(1.0, 2.0), Vector(3.0)), cpu.outer(Vector(1.0, 2.0), Vector(3.0)))
      assertEquals(b.linearActivation(m, x, bias, "relu"), cpu.linearActivation(m, x, bias, "relu"))
    finally
      System.clearProperty("metal.simulate.fail.matvec")
      System.clearProperty("metal.simulate.fail.outer")
      System.clearProperty("metal.simulate.fail.linear")
  }

  test("metal batch fallback works and gpuOpsEnabled reflects disabled batch ops") {
    val b = MetalBackend.create("fp32")
    val cpu = CpuBackend("fp64")
    val a = Matrix.fromFunction(2, 3)((r, c) => (r + c).toDouble)
    val c = Matrix.fromFunction(3, 2)((r, col) => (r - col).toDouble)
    val x = Matrix.fromFunction(2, 3)((r, col) => (r * 3 + col).toDouble * 0.1)
    val wT = Matrix.fromFunction(3, 2)((r, col) => (r + col).toDouble * 0.2)
    val bias = Vector(0.1, -0.2)
    val logits = Matrix.fromFunction(2, 3)((r, col) => (r - col).toDouble)
    val targets = Vector(0, 2)

    System.setProperty("metal.simulate.fail.matmul", "1")
    System.setProperty("metal.simulate.fail.linear_batch", "1")
    System.setProperty("metal.simulate.fail.softmax_batch", "1")
    System.setProperty("metal.simulate.fail.ce_batch", "1")
    try
      assertEquals(b.matMul(a, c), cpu.matMul(a, c))
      assertEquals(b.linearActivationBatch(x, wT, bias, "relu"), cpu.linearActivationBatch(x, wT, bias, "relu"))
      assertEquals(b.softmaxStableBatch(logits), cpu.softmaxStableBatch(logits))
      assertEquals(b.crossEntropyBatch(cpu.softmaxStableBatch(logits), targets), cpu.crossEntropyBatch(cpu.softmaxStableBatch(logits), targets))
      if b.isGpu then
        assert(!b.gpuOpsEnabled.contains("batchMatMul"))
        assert(!b.gpuOpsEnabled.contains("batchLinearActivation"))
        assert(!b.gpuOpsEnabled.contains("batchSoftmax"))
        assert(!b.gpuOpsEnabled.contains("batchCrossEntropy"))
    finally
      System.clearProperty("metal.simulate.fail.matmul")
      System.clearProperty("metal.simulate.fail.linear_batch")
      System.clearProperty("metal.simulate.fail.softmax_batch")
      System.clearProperty("metal.simulate.fail.ce_batch")
  }
