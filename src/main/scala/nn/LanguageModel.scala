package nn

import compute.{ComputeBackend, CpuBackend}
import data.Example
import linalg.{LinearAlgebra, Matrix, Vec}
import scala.util.Random

final case class Params(E: Matrix, W1: Matrix, b1: Vec, W2: Matrix, b2: Vec)
final case class Grads(dE: Matrix, dW1: Matrix, db1: Vec, dW2: Matrix, db2: Vec)
final case class ForwardCache(x: Vec, z1: Vec, a1: Vec, logits: Vec, probs: Vec, context: Vector[Int])
final case class ForwardBatchCache(
    x: Matrix,
    z1: Matrix,
    a1: Matrix,
    logits: Matrix,
    probs: Matrix,
    contexts: Vector[Vector[Int]]
)
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

  def forward(
      p: Params,
      context: Vector[Int],
      activation: String = "tanh",
      backend: ComputeBackend = CpuBackend.Default
  ): ForwardCache =
    require(context.nonEmpty, "context cannot be empty")

    val x = context.flatMap(id => p.E.rowSlice(id))
    val (z1, a1) = backend.linearActivation(p.W1, x, p.b1, activation)
    val logits = backend.vecAdd(backend.matVecMul(p.W2, a1), p.b2)
    val probs = backend.softmaxStable(logits)

    ForwardCache(x = x, z1 = z1, a1 = a1, logits = logits, probs = probs, context = context)

  def lossFromCache(cache: ForwardCache, target: Int, backend: ComputeBackend = CpuBackend.Default): Double =
    backend.crossEntropy(cache.probs, target)

  def forwardBatch(
      p: Params,
      contexts: Vector[Vector[Int]],
      activation: String = "tanh",
      backend: ComputeBackend = CpuBackend.Default
  ): ForwardBatchCache =
    require(contexts.nonEmpty, "contexts cannot be empty")
    val inputDim = contexts.head.length * p.E.cols
    require(contexts.forall(_.length == contexts.head.length), "all contexts in batch must have same length")

    val x = Matrix(
      contexts.flatMap(ctx => ctx.flatMap(id => p.E.rowSlice(id))),
      rows = contexts.length,
      cols = inputDim
    )
    val (z1, a1) = backend.linearActivationBatch(x, p.W1.transposeView, p.b1, activation)
    val logits = backend.addRowBias(backend.matMul(a1, p.W2.transposeView), p.b2)
    val probs = backend.softmaxStableBatch(logits)
    ForwardBatchCache(x = x, z1 = z1, a1 = a1, logits = logits, probs = probs, contexts = contexts)

  def lossFromBatchCache(cache: ForwardBatchCache, targets: Vector[Int], backend: ComputeBackend = CpuBackend.Default): Vec =
    backend.crossEntropyBatch(cache.probs, targets)

  def backward(
      p: Params,
      cache: ForwardCache,
      target: Int,
      activation: String = "tanh",
      backend: ComputeBackend = CpuBackend.Default
  ): Grads =
    val vocabSize = cache.probs.length
    require(target >= 0 && target < vocabSize, s"target out of range: $target")

    val dLogits = cache.probs.indices.map { i =>
      val t = if i == target then 1.0 else 0.0
      cache.probs(i) - t
    }.toVector

    val dW2 = backend.outer(dLogits, cache.a1)
    val db2 = dLogits

    val da1 = backend.matVecMul(p.W2.transposeView, dLogits)

    val dz1 = applyActivationGrad(da1, cache.z1, activation, backend)

    val dW1 = backend.outer(dz1, cache.x)
    val db1 = dz1

    val dx = backend.matVecMul(p.W1.transposeView, dz1)

    val dE = scatterEmbeddingGrad(dx, cache.context, p.E.rows, p.E.cols)

    Grads(dE = dE, dW1 = dW1, db1 = db1, dW2 = dW2, db2 = db2)

  def backwardBatch(
      p: Params,
      cache: ForwardBatchCache,
      targets: Vector[Int],
      activation: String = "tanh",
      backend: ComputeBackend = CpuBackend.Default
  ): Grads =
    require(targets.length == cache.probs.rows, s"targets length ${targets.length} must equal batch size ${cache.probs.rows}")

    val dLogits = Matrix.fromFunction(cache.probs.rows, cache.probs.cols) { (r, c) =>
      val t = if c == targets(r) then 1.0 else 0.0
      cache.probs.get(r, c) - t
    }

    val dW2 = backend.matMul(dLogits.transposeView, cache.a1)
    val db2 = backend.reduceSumRows(dLogits)

    val da1 = backend.matMul(dLogits, p.W2)
    val dz1 = applyActivationGradBatch(da1, cache.z1, activation)

    val dW1 = backend.matMul(dz1.transposeView, cache.x)
    val db1 = backend.reduceSumRows(dz1)

    val dx = backend.matMul(dz1, p.W1)
    val dE = scatterEmbeddingGradBatch(dx, cache.contexts, p.E.rows, p.E.cols)

    Grads(dE = dE, dW1 = dW1, db1 = db1, dW2 = dW2, db2 = db2)

  private def applyActivationGrad(grad: Vec, z: Vec, activation: String, backend: ComputeBackend): Vec =
    activation.toLowerCase match
      case "relu" => backend.hadamard(grad, backend.reluGrad(z))
      case "tanh" => backend.hadamard(grad, backend.tanhGrad(z))
      case _      => throw new IllegalArgumentException(s"Unknown activation: $activation. Use 'relu' or 'tanh'.")

  private def applyActivationGradBatch(grad: Matrix, z: Matrix, activation: String): Matrix =
    require(grad.rows == z.rows && grad.cols == z.cols, "activation grad batch shape mismatch")
    Matrix.fromFunction(grad.rows, grad.cols) { (r, c) =>
      val g = grad.get(r, c)
      val zv = z.get(r, c)
      activation.toLowerCase match
        case "relu" => if zv > 0 then g else 0.0
        case "tanh" =>
          val t = math.tanh(zv)
          g * (1.0 - t * t)
        case _ => throw new IllegalArgumentException(s"Unknown activation: $activation. Use 'relu' or 'tanh'.")
    }

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

  private def scatterEmbeddingGradBatch(dx: Matrix, contexts: Vector[Vector[Int]], vocabSize: Int, embedDim: Int): Matrix =
    require(dx.rows == contexts.length, s"dx rows ${dx.rows} must equal batch size ${contexts.length}")
    require(contexts.nonEmpty, "contexts cannot be empty")
    val contextSize = contexts.head.length
    require(dx.cols == contextSize * embedDim, s"dx cols ${dx.cols} must equal contextSize*embedDim ${contextSize * embedDim}")
    require(contexts.forall(_.length == contextSize), "all contexts must have same length")

    val accum = Array.fill(vocabSize * embedDim)(0.0)
    var r = 0
    while r < contexts.length do
      val context = contexts(r)
      var pos = 0
      while pos < context.length do
        val tokenId = context(pos)
        require(tokenId >= 0 && tokenId < vocabSize, s"context token id out of range: $tokenId")
        val baseDx = pos * embedDim
        val baseRow = tokenId * embedDim
        var j = 0
        while j < embedDim do
          accum(baseRow + j) += dx.get(r, baseDx + j)
          j += 1
        pos += 1
      r += 1
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

  def trainStep(
      p: Params,
      ex: Example,
      lr: Double,
      l2: Double = 0.0,
      clipNorm: Option[Double] = None,
      activation: String = "tanh",
      backend: ComputeBackend = CpuBackend.Default
  ): (Params, Double) =
    val cache = forward(p, ex.context, activation, backend)
    val loss = lossFromCache(cache, ex.target, backend)
    val grads = backward(p, cache, ex.target, activation, backend)
    val updated = update(p, grads, lr, l2 = l2, clipNorm = clipNorm)
    (updated, loss)

  def trainBatchStep(
      p: Params,
      batch: Vector[Example],
      lr: Double,
      l2: Double = 0.0,
      clipNorm: Option[Double] = None,
      activation: String = "tanh",
      backend: ComputeBackend = CpuBackend.Default
  ): (Params, Double) =
    require(batch.nonEmpty, "batch cannot be empty")
    val contexts = batch.map(_.context)
    val targets = batch.map(_.target)
    val cache = forwardBatch(p, contexts, activation, backend)
    val losses = lossFromBatchCache(cache, targets, backend)
    val grads = backwardBatch(p, cache, targets, activation, backend)
    val scale = 1.0 / batch.length.toDouble
    val scaled = Grads(
      dE = grads.dE.map(_ * scale),
      dW1 = grads.dW1.map(_ * scale),
      db1 = grads.db1.map(_ * scale),
      dW2 = grads.dW2.map(_ * scale),
      db2 = grads.db2.map(_ * scale)
    )
    val updated = update(p, scaled, lr, l2 = l2, clipNorm = clipNorm)
    (updated, losses.sum / losses.length.toDouble)
