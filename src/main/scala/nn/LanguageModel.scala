package nn

import data.Example
import linalg.{LinearAlgebra, Matrix, Vec}
import scala.util.Random

final case class Params(E: Matrix, W1: Matrix, b1: Vec, W2: Matrix, b2: Vec)
final case class Grads(dE: Matrix, dW1: Matrix, db1: Vec, dW2: Matrix, db2: Vec)
final case class ForwardCache(x: Vec, z1: Vec, a1: Vec, logits: Vec, probs: Vec, context: Vector[Int])
final case class ActivationCache(z1: Vec, a1: Vec)

final case class ModelConfig(contextSize: Int, embedDim: Int, hiddenDim: Int, vocabSize: Int, activation: String = "tanh")

object LanguageModel:

  def initParams(cfg: ModelConfig, seed: Int): Params =
    val rnd = Random(seed)

    val E = xavierUniform(cfg.vocabSize, cfg.embedDim, rnd)
    val W1 = xavierUniform(cfg.hiddenDim, cfg.contextSize * cfg.embedDim, rnd)
    val b1 = Vector.fill(cfg.hiddenDim)(0.0)
    val W2 = xavierUniform(cfg.vocabSize, cfg.hiddenDim, rnd)
    val b2 = Vector.fill(cfg.vocabSize)(0.0)

    Params(E = E, W1 = W1, b1 = b1, W2 = W2, b2 = b2)

  private def xavierUniform(fanOut: Int, fanIn: Int, rnd: Random): Matrix =
    val bound = math.sqrt(6.0) / math.sqrt(fanIn.toDouble + fanOut.toDouble)
    Matrix.fromFunction(fanOut, fanIn)((_, _) => rnd.between(-bound, bound))

  def forward(p: Params, context: Vector[Int], activation: String = "tanh"): ForwardCache =
    require(context.nonEmpty, "context cannot be empty")

    val x = context.flatMap(id => p.E.rowSlice(id))
    val z1 = LinearAlgebra.vecAdd(LinearAlgebra.matVecMul(p.W1, x), p.b1)
    val a1 = applyActivation(z1, activation)
    val logits = LinearAlgebra.vecAdd(LinearAlgebra.matVecMul(p.W2, a1), p.b2)
    val probs = LinearAlgebra.softmaxStable(logits)

    ForwardCache(x = x, z1 = z1, a1 = a1, logits = logits, probs = probs, context = context)

  private def applyActivation(v: Vec, activation: String): Vec =
    activation.toLowerCase match
      case "relu" => LinearAlgebra.relu(v)
      case "tanh" => LinearAlgebra.tanhVec(v)
      case _      => throw new IllegalArgumentException(s"Unknown activation: $activation. Use 'relu' or 'tanh'.")

  def lossFromCache(cache: ForwardCache, target: Int): Double =
    LinearAlgebra.crossEntropy(cache.probs, target)

  def backward(p: Params, cache: ForwardCache, target: Int, activation: String = "tanh"): Grads =
    val vocabSize = cache.probs.length
    require(target >= 0 && target < vocabSize, s"target out of range: $target")

    val dLogits = cache.probs.indices.map { i =>
      val t = if i == target then 1.0 else 0.0
      cache.probs(i) - t
    }.toVector

    val dW2 = LinearAlgebra.outer(dLogits, cache.a1)
    val db2 = dLogits

    val da1 = LinearAlgebra.matVecMul(p.W2.transposeView, dLogits)

    val dz1 = applyActivationGrad(da1, cache.z1, activation)

    val dW1 = LinearAlgebra.outer(dz1, cache.x)
    val db1 = dz1

    val dx = LinearAlgebra.matVecMul(p.W1.transposeView, dz1)

    val dE = scatterEmbeddingGrad(dx, cache.context, p.E.rows, p.E.cols)

    Grads(dE = dE, dW1 = dW1, db1 = db1, dW2 = dW2, db2 = db2)

  private def applyActivationGrad(grad: Vec, z: Vec, activation: String): Vec =
    activation.toLowerCase match
      case "relu" => LinearAlgebra.hadamard(grad, LinearAlgebra.reluGrad(z))
      case "tanh" => LinearAlgebra.hadamard(grad, LinearAlgebra.tanhGrad(z))
      case _      => throw new IllegalArgumentException(s"Unknown activation: $activation. Use 'relu' or 'tanh'.")

  private def scatterEmbeddingGrad(dx: Vec, context: Vector[Int], vocabSize: Int, embedDim: Int): Matrix =
    require(dx.length == context.length * embedDim, s"dx length ${dx.length} must equal contextSize*embedDim ${context.length * embedDim}")

    val accum = Array.fill(vocabSize * embedDim)(0.0)

    var pos = 0
    while pos < context.length do
      val tokenId = context(pos)
      require(tokenId >= 0 && tokenId < vocabSize, s"context token id out of range: $tokenId")
      val baseDx = pos * embedDim
      val baseRow = tokenId * embedDim

      var j = 0
      while j < embedDim do
        accum(baseRow + j) += dx(baseDx + j)
        j += 1

      pos += 1

    Matrix(accum.toVector, vocabSize, embedDim)

  def update(
      p: Params,
      g: Grads,
      lr: Double,
      l2: Double = 0.0,
      clipNorm: Option[Double] = None
  ): Params =
    require(lr > 0.0, s"learning rate must be > 0, got $lr")

    val gradients = clipNorm match
      case Some(maxNorm) if maxNorm > 0.0 => clipGradients(g, maxNorm)
      case _                               => g

    def applyWeightDecay(w: Matrix, dw: Matrix): Matrix =
      if l2 <= 0.0 then dw
      else dw.zipMap(w)((grad, weight) => grad + l2 * weight)

    def updateMat(w: Matrix, dw: Matrix): Matrix =
      w.zipMap(dw)((wv, gv) => wv - lr * gv)

    def updateVec(v: Vec, dv: Vec): Vec =
      LinearAlgebra.vecSub(v, LinearAlgebra.scalarMul(dv, lr))

    val dEAdj = applyWeightDecay(p.E, gradients.dE)
    val dW1Adj = applyWeightDecay(p.W1, gradients.dW1)
    val dW2Adj = applyWeightDecay(p.W2, gradients.dW2)

    Params(
      E = updateMat(p.E, dEAdj),
      W1 = updateMat(p.W1, dW1Adj),
      b1 = updateVec(p.b1, gradients.db1),
      W2 = updateMat(p.W2, dW2Adj),
      b2 = updateVec(p.b2, gradients.db2)
    )

  private def clipGradients(g: Grads, maxNorm: Double): Grads =
    val sqNorm =
      g.dE.data.map(x => x * x).sum +
        g.dW1.data.map(x => x * x).sum +
        g.db1.map(x => x * x).sum +
        g.dW2.data.map(x => x * x).sum +
        g.db2.map(x => x * x).sum

    val norm = math.sqrt(sqNorm)
    if norm <= maxNorm || norm == 0.0 then g
    else
      val scale = maxNorm / norm

      def scaleMat(m: Matrix): Matrix = m.map(_ * scale)
      def scaleVec(v: Vec): Vec = v.map(_ * scale)

      Grads(
        dE = scaleMat(g.dE),
        dW1 = scaleMat(g.dW1),
        db1 = scaleVec(g.db1),
        dW2 = scaleMat(g.dW2),
        db2 = scaleVec(g.db2)
      )

  def trainStep(p: Params, ex: Example, lr: Double, l2: Double = 0.0, clipNorm: Option[Double] = None, activation: String = "tanh"): (Params, Double) =
    val cache = forward(p, ex.context, activation)
    val loss = lossFromCache(cache, ex.target)
    val grads = backward(p, cache, ex.target, activation)
    val updated = update(p, grads, lr, l2 = l2, clipNorm = clipNorm)
    (updated, loss)
