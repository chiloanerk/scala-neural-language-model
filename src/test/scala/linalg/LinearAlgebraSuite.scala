package linalg

import munit.FunSuite

class LinearAlgebraSuite extends FunSuite:
  test("vector ops and dot") {
    val a = Vector(1.0, 2.0, 3.0)
    val b = Vector(4.0, 5.0, 6.0)
    assertEquals(LinearAlgebra.vecAdd(a, b), Vector(5.0, 7.0, 9.0))
    assertEquals(LinearAlgebra.vecSub(b, a), Vector(3.0, 3.0, 3.0))
    assertEquals(LinearAlgebra.scalarMul(a, 2.0), Vector(2.0, 4.0, 6.0))
    assertEquals(LinearAlgebra.hadamard(a, b), Vector(4.0, 10.0, 18.0))
    assertEquals(LinearAlgebra.dot(a, b), 32.0)
  }

  test("matVecMul and outer") {
    val m = Matrix.fromFunction(2, 3)((r, c) => (r + c + 1).toDouble)
    val v = Vector(1.0, 2.0, 3.0)
    assertEquals(LinearAlgebra.matVecMul(m, v), Vector(14.0, 20.0))

    val o = LinearAlgebra.outer(Vector(2.0, 3.0), Vector(4.0, 5.0))
    assertEquals(o.get(1, 1), 15.0)
  }

  test("batched matrix helpers") {
    val a = Matrix.fromFunction(2, 3)((r, c) => (r + c + 1).toDouble)
    val b = Matrix.fromFunction(3, 2)((r, c) => (r * 2 + c + 1).toDouble)
    val mm = LinearAlgebra.matMul(a, b)
    assertEquals(mm.rows, 2)
    assertEquals(mm.cols, 2)
    assert(math.abs(mm.get(0, 0) - 22.0) < 1e-9)

    val biased = LinearAlgebra.addRowBias(mm, Vector(1.0, -1.0))
    assert(math.abs(biased.get(0, 0) - 23.0) < 1e-9)
    assert(math.abs(biased.get(0, 1) - (28.0 - 1.0)) < 1e-9)

    val rowSum = LinearAlgebra.reduceSumRows(mm)
    assertEquals(rowSum.length, 2)
  }

  test("softmax/cross-entropy batch") {
    val logits = Matrix(Vector(1.0, 2.0, 3.0, 1000.0, 1001.0, 999.0), rows = 2, cols = 3)
    val probs = LinearAlgebra.softmaxStableBatch(logits)
    assertEquals(probs.rows, 2)
    assertEquals(probs.cols, 3)
    assert(math.abs(probs.rowSlice(0).sum - 1.0) < 1e-9)
    assert(math.abs(probs.rowSlice(1).sum - 1.0) < 1e-9)
    val losses = LinearAlgebra.crossEntropyBatch(probs, Vector(2, 1))
    assertEquals(losses.length, 2)
    assert(losses.forall(_ >= 0.0))
  }

  test("activations and grads") {
    val v = Vector(-1.0, 0.0, 1.0)
    assertEquals(LinearAlgebra.relu(v), Vector(0.0, 0.0, 1.0))
    assertEquals(LinearAlgebra.reluGrad(v), Vector(0.0, 0.0, 1.0))
    val tg = LinearAlgebra.tanhGrad(Vector(0.0))
    assert(math.abs(tg.head - 1.0) < 1e-12)
  }

  test("stable softmax and cross entropy") {
    val probs = LinearAlgebra.softmaxStable(Vector(1000.0, 1001.0, 999.0))
    assert(probs.forall(_.isFinite))
    assert(math.abs(probs.sum - 1.0) < 1e-9)
    val ce = LinearAlgebra.crossEntropy(probs, 1)
    assert(ce >= 0.0)
  }

  test("argTopK and norms") {
    val top = LinearAlgebra.argTopK(Vector(0.1, 0.7, 0.4), 2)
    assertEquals(top.map(_._1), Vector(1, 2))
    assert(math.abs(LinearAlgebra.l2Norm(Vector(3.0, 4.0)) - 5.0) < 1e-12)
    assert(math.abs(LinearAlgebra.l2Norm(Matrix.fromFunction(1, 2)((_, c) => if c == 0 then 3.0 else 4.0)) - 5.0) < 1e-12)
  }
