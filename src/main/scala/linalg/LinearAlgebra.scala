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

  def outer(a: Vec, b: Vec): Matrix =
    Matrix.fromFunction(a.length, b.length)((r, c) => a(r) * b(c))

  def tanhVec(v: Vec): Vec = v.map(tanh)

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
