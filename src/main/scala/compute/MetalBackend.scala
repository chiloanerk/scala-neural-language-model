package compute

import linalg.{Matrix, Vec}

/**
  * Phase-1 backend: routes hot-path ops through CPU fallback unless native Metal kernels are available.
  * Native probe is real (JNI + Metal device check), so CLI can report accurate GPU status.
  */
final case class MetalBackend(precision: String, probe: MetalProbe, cpuFallback: CpuBackend = CpuBackend.Default) extends ComputeBackend:
  override val name: String = "gpu"
  override val isGpu: Boolean = probe.available
  @volatile private var matVecEnabled: Boolean = probe.available
  @volatile private var outerEnabled: Boolean = probe.available
  @volatile private var linearEnabled: Boolean = probe.available
  @volatile private var matVecError: Option[String] = None
  @volatile private var outerError: Option[String] = None
  @volatile private var linearError: Option[String] = None
  @volatile private var matVecCalls: Long = 0L
  @volatile private var outerCalls: Long = 0L
  @volatile private var linearCalls: Long = 0L
  @volatile private var matVecNanos: Long = 0L
  @volatile private var outerNanos: Long = 0L
  @volatile private var linearNanos: Long = 0L
  override val gpuOpsEnabled: Set[String] =
    Set.newBuilder[String]
      .addAll(if matVecEnabled then List("matVecMul") else Nil)
      .addAll(if outerEnabled then List("outer") else Nil)
      .addAll(if linearEnabled then List("linearActivation") else Nil)
      .result()
  override def diagnostics: String =
    if !probe.available then s"Metal unavailable, using CPU fallback (reason=${probe.error.getOrElse("unknown")})"
    else
      val disabled =
        List(
          matVecError.map(r => s"matVecMul:$r"),
          outerError.map(r => s"outer:$r"),
          linearError.map(r => s"linearActivation:$r")
        ).flatten
      if disabled.isEmpty then s"Metal backend (device=${probe.deviceName}, precision=$precision)"
      else s"Metal backend (device=${probe.deviceName}, precision=$precision, disabled=${disabled.mkString(";")})"

  private def disableOp(op: String, reason: String): Unit =
    op match
      case "matVecMul" =>
        matVecEnabled = false
        matVecError = Some(reason)
      case "outer" =>
        outerEnabled = false
        outerError = Some(reason)
      case "linearActivation" =>
        linearEnabled = false
        linearError = Some(reason)
      case _ => ()
    Console.err.println(s"[warn] GPU op '$op' disabled, falling back to CPU (reason=$reason)")

  override def vecAdd(a: Vec, b: Vec): Vec = cpuFallback.vecAdd(a, b)
  override def vecSub(a: Vec, b: Vec): Vec = cpuFallback.vecSub(a, b)
  override def scalarMul(a: Vec, s: Double): Vec = cpuFallback.scalarMul(a, s)
  override def hadamard(a: Vec, b: Vec): Vec = cpuFallback.hadamard(a, b)

  override def matVecMul(m: Matrix, v: Vec): Vec =
    val t0 = System.nanoTime()
    matVecCalls += 1
    if matVecEnabled then
      MetalNativeBridge.tryMatVecMul(m, v) match
        case Right(out) =>
          matVecNanos += (System.nanoTime() - t0)
          out
        case Left(reason) =>
          disableOp("matVecMul", reason)
          val out = cpuFallback.matVecMul(m, v)
          matVecNanos += (System.nanoTime() - t0)
          out
    else
      val out = cpuFallback.matVecMul(m, v)
      matVecNanos += (System.nanoTime() - t0)
      out
  override def outer(a: Vec, b: Vec): Matrix =
    val t0 = System.nanoTime()
    outerCalls += 1
    if outerEnabled then
      MetalNativeBridge.tryOuter(a, b) match
        case Right(out) =>
          outerNanos += (System.nanoTime() - t0)
          out
        case Left(reason) =>
          disableOp("outer", reason)
          val out = cpuFallback.outer(a, b)
          outerNanos += (System.nanoTime() - t0)
          out
    else
      val out = cpuFallback.outer(a, b)
      outerNanos += (System.nanoTime() - t0)
      out

  override def tanh(v: Vec): Vec = cpuFallback.tanh(v)
  override def tanhGrad(v: Vec): Vec = cpuFallback.tanhGrad(v)
  override def relu(v: Vec): Vec = cpuFallback.relu(v)
  override def reluGrad(v: Vec): Vec = cpuFallback.reluGrad(v)
  override def linearActivation(m: Matrix, x: Vec, b: Vec, activation: String): (Vec, Vec) =
    val t0 = System.nanoTime()
    linearCalls += 1
    if linearEnabled then
      MetalNativeBridge.tryLinearActivation(m, x, b, activation) match
        case Right(out) =>
          linearNanos += (System.nanoTime() - t0)
          out
        case Left(reason) =>
          disableOp("linearActivation", reason)
          val out = cpuFallback.linearActivation(m, x, b, activation)
          linearNanos += (System.nanoTime() - t0)
          out
    else
      val out = cpuFallback.linearActivation(m, x, b, activation)
      linearNanos += (System.nanoTime() - t0)
      out

  override def softmaxStable(logits: Vec): Vec = cpuFallback.softmaxStable(logits)
  override def softmaxStableBatch(logits: Matrix): Matrix = cpuFallback.softmaxStableBatch(logits)
  override def crossEntropy(probs: Vec, target: Int): Double = cpuFallback.crossEntropy(probs, target)
  override def crossEntropyBatch(probs: Matrix, targets: Vector[Int]): Vec = cpuFallback.crossEntropyBatch(probs, targets)
  override def matMul(a: Matrix, b: Matrix): Matrix = cpuFallback.matMul(a, b)
  override def addRowBias(m: Matrix, bias: Vec): Matrix = cpuFallback.addRowBias(m, bias)
  override def reduceSumRows(m: Matrix): Vec = cpuFallback.reduceSumRows(m)
  override def linearActivationBatch(x: Matrix, wT: Matrix, b: Vec, activation: String): (Matrix, Matrix) =
    cpuFallback.linearActivationBatch(x, wT, b, activation)

  override def resetProfile(): Unit =
    matVecCalls = 0L
    outerCalls = 0L
    linearCalls = 0L
    matVecNanos = 0L
    outerNanos = 0L
    linearNanos = 0L

  override def profileSummary: String =
    def ms(nanos: Long): Double = nanos.toDouble / 1e6
    s"matVec: calls=$matVecCalls timeMs=${"%.2f".format(ms(matVecNanos))}, " +
      s"outer: calls=$outerCalls timeMs=${"%.2f".format(ms(outerNanos))}, " +
      s"linearActivation: calls=$linearCalls timeMs=${"%.2f".format(ms(linearNanos))}"

object MetalBackend:
  def create(precision: String): MetalBackend =
    val probe = MetalNativeBridge.probe()
    MetalBackend(precision = precision, probe = probe)
