package compute

import linalg.{Matrix, Vec}

trait ComputeBackend:
  def name: String
  def precision: String
  def isGpu: Boolean
  def diagnostics: String
  def gpuOpsEnabled: Set[String] = Set.empty

  def vecAdd(a: Vec, b: Vec): Vec
  def vecSub(a: Vec, b: Vec): Vec
  def scalarMul(a: Vec, s: Double): Vec
  def hadamard(a: Vec, b: Vec): Vec

  def matVecMul(m: Matrix, v: Vec): Vec
  def outer(a: Vec, b: Vec): Matrix
  def matMul(a: Matrix, b: Matrix): Matrix
  def addRowBias(m: Matrix, bias: Vec): Matrix
  def reduceSumRows(m: Matrix): Vec

  def tanh(v: Vec): Vec
  def tanhGrad(v: Vec): Vec
  def relu(v: Vec): Vec
  def reluGrad(v: Vec): Vec
  def linearActivation(m: Matrix, x: Vec, b: Vec, activation: String): (Vec, Vec)
  def linearActivationBatch(x: Matrix, wT: Matrix, b: Vec, activation: String): (Matrix, Matrix)

  def softmaxStable(logits: Vec): Vec
  def softmaxStableBatch(logits: Matrix): Matrix
  def crossEntropy(probs: Vec, target: Int): Double
  def crossEntropyBatch(probs: Matrix, targets: Vector[Int]): Vec

  def resetProfile(): Unit = ()
  def profileSummary: String = ""
