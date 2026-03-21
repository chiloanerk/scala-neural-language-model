package linalg

import scala.math.{exp, log, tanh}

object LinearAlgebra:
  def requireSameLength(a: Vec, b: Vec, label: String): Unit =
    require(a.length == b.length, s"$label length mismatch: ${a.length} vs ${b.length}")

  def zeros(n: Int): Vec = Vector.fill(n)(0.0)

  def vecAdd(a: Vec, b: Vec): Vec =
    requireSameLength(a, b, "vecAdd")
    a.indices.map(i => a(i) + b(i)).toVector

  def vecSub(a: Vec, b: Vec): Vec =
    requireSameLength(a, b, "vecSub")
    a.indices.map(i => a(i) - b(i)).toVector

  def scalarMul(a: Vec, s: Double): Vec = a.map(_ * s)

  def hadamard(a: Vec, b: Vec): Vec =
    requireSameLength(a, b, "hadamard")
    a.indices.map(i => a(i) * b(i)).toVector

  def dot(a: Vec, b: Vec): Double =
    requireSameLength(a, b, "dot")
    var sum = 0.0
    var i = 0
    while i < a.length do
      sum += a(i) * b(i)
      i += 1
    sum

  def matVecMul(m: Matrix, v: Vec): Vec =
    require(m.cols == v.length, s"matVecMul shape mismatch: m=(${m.rows},${m.cols}) v=${v.length}")
    Vector.tabulate(m.rows) { r =>
      var sum = 0.0
      var c = 0
      while c < m.cols do
        sum += m.get(r, c) * v(c)
        c += 1
      sum
    }

  def matMul(a: Matrix, b: Matrix): Matrix =
    require(a.cols == b.rows, s"matMul shape mismatch: a=(${a.rows},${a.cols}) b=(${b.rows},${b.cols})")
    Matrix.fromFunction(a.rows, b.cols) { (r, c) =>
      var sum = 0.0
      var k = 0
      while k < a.cols do
        sum += a.get(r, k) * b.get(k, c)
        k += 1
      sum
    }

  def addRowBias(m: Matrix, bias: Vec): Matrix =
    require(m.cols == bias.length, s"addRowBias shape mismatch: m.cols=${m.cols} bias=${bias.length}")
    Matrix.fromFunction(m.rows, m.cols)((r, c) => m.get(r, c) + bias(c))

  def reduceSumRows(m: Matrix): Vec =
    Vector.tabulate(m.cols) { c =>
      var sum = 0.0
      var r = 0
      while r < m.rows do
        sum += m.get(r, c)
        r += 1
      sum
    }

  def softmaxStableBatch(logits: Matrix): Matrix =
    if logits.rows == 0 then Matrix.zeros(0, logits.cols)
    else
      val out = Vector.newBuilder[Double]
      out.sizeHint(logits.rows * logits.cols)
      var r = 0
      while r < logits.rows do
        val probs = softmaxStable(logits.rowSlice(r))
        var c = 0
        while c < logits.cols do
          out += probs(c)
          c += 1
        r += 1
      Matrix(out.result(), logits.rows, logits.cols)

  def crossEntropyBatch(probs: Matrix, targets: Vector[Int], epsilon: Double = 1e-12): Vector[Double] =
    require(probs.rows == targets.length, s"crossEntropyBatch rows ${probs.rows} != targets ${targets.length}")
    Vector.tabulate(probs.rows) { r =>
      val t = targets(r)
      require(t >= 0 && t < probs.cols, s"target index out of range at row $r: $t")
      -log(math.max(probs.get(r, t), epsilon))
    }

  def outer(a: Vec, b: Vec): Matrix =
    Matrix.fromFunction(a.length, b.length)((r, c) => a(r) * b(c))

  def tanhVec(v: Vec): Vec = v.map(tanh)

  def relu(v: Vec): Vec = v.map(x => math.max(0.0, x))

  def reluGrad(v: Vec): Vec = v.map(x => if x > 0 then 1.0 else 0.0)

  def tanhGrad(v: Vec): Vec =
    val tanhV = v.map(tanh)
    tanhV.map(x => 1.0 - x * x)

  def softmaxStable(logits: Vec): Vec =
    require(logits.nonEmpty, "softmax requires non-empty vector")
    val maxLogit = logits.max
    val shiftedExp = logits.map(x => exp(x - maxLogit))
    val denom = shiftedExp.sum
    shiftedExp.map(_ / denom)

  def crossEntropy(probs: Vec, target: Int, epsilon: Double = 1e-12): Double =
    require(target >= 0 && target < probs.length, s"target index out of range: $target")
    -log(math.max(probs(target), epsilon))

  def argTopK(v: Vec, k: Int): Vector[(Int, Double)] =
    v.zipWithIndex
      .map { case (value, idx) => (idx, value) }
      .sortBy { case (_, value) => -value }
      .take(math.max(k, 1))
      .toVector

  def l2Norm(vec: Vec): Double = math.sqrt(vec.map(x => x * x).sum)

  def l2Norm(mat: Matrix): Double = math.sqrt(mat.data.map(x => x * x).sum)
