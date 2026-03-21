package compute

import linalg.{LinearAlgebra, Matrix, Vec}

final case class CpuBackend(precision: String = "fp64") extends ComputeBackend:
  override val name: String = "cpu"
  override val isGpu: Boolean = false
  override val diagnostics: String = s"CPU backend (precision=$precision)"
  override val gpuOpsEnabled: Set[String] = Set.empty

  override def vecAdd(a: Vec, b: Vec): Vec = LinearAlgebra.vecAdd(a, b)
  override def vecSub(a: Vec, b: Vec): Vec = LinearAlgebra.vecSub(a, b)
  override def scalarMul(a: Vec, s: Double): Vec = LinearAlgebra.scalarMul(a, s)
  override def hadamard(a: Vec, b: Vec): Vec = LinearAlgebra.hadamard(a, b)

  override def matVecMul(m: Matrix, v: Vec): Vec = LinearAlgebra.matVecMul(m, v)
  override def outer(a: Vec, b: Vec): Matrix = LinearAlgebra.outer(a, b)
  override def matMul(a: Matrix, b: Matrix): Matrix = LinearAlgebra.matMul(a, b)
  override def addRowBias(m: Matrix, bias: Vec): Matrix = LinearAlgebra.addRowBias(m, bias)
  override def reduceSumRows(m: Matrix): Vec = LinearAlgebra.reduceSumRows(m)

  override def tanh(v: Vec): Vec = LinearAlgebra.tanhVec(v)
  override def tanhGrad(v: Vec): Vec = LinearAlgebra.tanhGrad(v)
  override def relu(v: Vec): Vec = LinearAlgebra.relu(v)
  override def reluGrad(v: Vec): Vec = LinearAlgebra.reluGrad(v)
  override def linearActivation(m: Matrix, x: Vec, b: Vec, activation: String): (Vec, Vec) =
    val z = vecAdd(matVecMul(m, x), b)
    val a = activation.toLowerCase match
      case "relu" => relu(z)
      case "tanh" => tanh(z)
      case _      => throw new IllegalArgumentException(s"Unknown activation: $activation")
    (z, a)
  override def linearActivationBatch(x: Matrix, wT: Matrix, b: Vec, activation: String): (Matrix, Matrix) =
    val z = addRowBias(matMul(x, wT), b)
    val a = activation.toLowerCase match
      case "relu" => z.map(v => math.max(0.0, v))
      case "tanh" => z.map(math.tanh)
      case _      => throw new IllegalArgumentException(s"Unknown activation: $activation")
    (z, a)

  override def softmaxStable(logits: Vec): Vec = LinearAlgebra.softmaxStable(logits)
  override def softmaxStableBatch(logits: Matrix): Matrix = LinearAlgebra.softmaxStableBatch(logits)
  override def crossEntropy(probs: Vec, target: Int): Double = LinearAlgebra.crossEntropy(probs, target)
  override def crossEntropyBatch(probs: Matrix, targets: Vector[Int]): Vec = LinearAlgebra.crossEntropyBatch(probs, targets)

object CpuBackend:
  val Default: CpuBackend = CpuBackend("fp64")
