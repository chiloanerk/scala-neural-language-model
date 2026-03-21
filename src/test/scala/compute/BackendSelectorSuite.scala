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
