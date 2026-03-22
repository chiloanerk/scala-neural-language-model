package compute

import linalg.{Matrix, Vec}
import java.util.Locale

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
  @volatile private var matMulEnabled: Boolean = probe.available
  @volatile private var linearBatchEnabled: Boolean = probe.available
  @volatile private var softmaxBatchEnabled: Boolean = probe.available
  @volatile private var ceBatchEnabled: Boolean = probe.available
  @volatile private var matVecError: Option[String] = None
  @volatile private var outerError: Option[String] = None
  @volatile private var linearError: Option[String] = None
  @volatile private var matMulError: Option[String] = None
  @volatile private var linearBatchError: Option[String] = None
  @volatile private var softmaxBatchError: Option[String] = None
  @volatile private var ceBatchError: Option[String] = None
  @volatile private var matVecCalls: Long = 0L
  @volatile private var outerCalls: Long = 0L
  @volatile private var linearCalls: Long = 0L
  @volatile private var matVecNanos: Long = 0L
  @volatile private var outerNanos: Long = 0L
  @volatile private var linearNanos: Long = 0L
  @volatile private var matMulCalls: Long = 0L
  @volatile private var linearBatchCalls: Long = 0L
  @volatile private var softmaxBatchCalls: Long = 0L
  @volatile private var ceBatchCalls: Long = 0L
  @volatile private var matMulNanos: Long = 0L
  @volatile private var linearBatchNanos: Long = 0L
  @volatile private var softmaxBatchNanos: Long = 0L
  @volatile private var ceBatchNanos: Long = 0L
  override def gpuOpsEnabled: Set[String] =
    Set.newBuilder[String]
      .addAll(if matVecEnabled then List("matVecMul") else Nil)
      .addAll(if outerEnabled then List("outer") else Nil)
      .addAll(if linearEnabled then List("linearActivation") else Nil)
      .addAll(if matMulEnabled then List("batchMatMul") else Nil)
      .addAll(if linearBatchEnabled then List("batchLinearActivation") else Nil)
      .addAll(if softmaxBatchEnabled then List("batchSoftmax") else Nil)
      .addAll(if ceBatchEnabled then List("batchCrossEntropy") else Nil)
      .result()
  override def diagnostics: String =
    if !probe.available then s"Metal unavailable, using CPU fallback (reason=${probe.error.getOrElse("unknown")})"
    else
      val disabled =
        List(
          matVecError.map(r => s"matVecMul:$r"),
          outerError.map(r => s"outer:$r"),
          linearError.map(r => s"linearActivation:$r"),
          matMulError.map(r => s"batchMatMul:$r"),
          linearBatchError.map(r => s"batchLinearActivation:$r"),
          softmaxBatchError.map(r => s"batchSoftmax:$r"),
          ceBatchError.map(r => s"batchCrossEntropy:$r")
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
      case "batchMatMul" =>
        matMulEnabled = false
        matMulError = Some(reason)
      case "batchLinearActivation" =>
        linearBatchEnabled = false
        linearBatchError = Some(reason)
      case "batchSoftmax" =>
        softmaxBatchEnabled = false
        softmaxBatchError = Some(reason)
      case "batchCrossEntropy" =>
        ceBatchEnabled = false
        ceBatchError = Some(reason)
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
  override def softmaxStableBatch(logits: Matrix): Matrix =
    val t0 = System.nanoTime()
    softmaxBatchCalls += 1
    if softmaxBatchEnabled then
      MetalNativeBridge.trySoftmaxBatch(logits) match
        case Right(out) =>
          softmaxBatchNanos += (System.nanoTime() - t0)
          out
        case Left(reason) =>
          disableOp("batchSoftmax", reason)
          val out = cpuFallback.softmaxStableBatch(logits)
          softmaxBatchNanos += (System.nanoTime() - t0)
          out
    else
      val out = cpuFallback.softmaxStableBatch(logits)
      softmaxBatchNanos += (System.nanoTime() - t0)
      out
  override def crossEntropy(probs: Vec, target: Int): Double = cpuFallback.crossEntropy(probs, target)
  override def crossEntropyBatch(probs: Matrix, targets: Vector[Int]): Vec =
    val t0 = System.nanoTime()
    ceBatchCalls += 1
    if ceBatchEnabled then
      MetalNativeBridge.tryCrossEntropyBatch(probs, targets) match
        case Right(out) =>
          ceBatchNanos += (System.nanoTime() - t0)
          out
        case Left(reason) =>
          disableOp("batchCrossEntropy", reason)
          val out = cpuFallback.crossEntropyBatch(probs, targets)
          ceBatchNanos += (System.nanoTime() - t0)
          out
    else
      val out = cpuFallback.crossEntropyBatch(probs, targets)
      ceBatchNanos += (System.nanoTime() - t0)
      out
  override def matMul(a: Matrix, b: Matrix): Matrix =
    val t0 = System.nanoTime()
    matMulCalls += 1
    if matMulEnabled then
      MetalNativeBridge.tryMatMul(a, b) match
        case Right(out) =>
          matMulNanos += (System.nanoTime() - t0)
          out
        case Left(reason) =>
          disableOp("batchMatMul", reason)
          val out = cpuFallback.matMul(a, b)
          matMulNanos += (System.nanoTime() - t0)
          out
    else
      val out = cpuFallback.matMul(a, b)
      matMulNanos += (System.nanoTime() - t0)
      out
  override def addRowBias(m: Matrix, bias: Vec): Matrix = cpuFallback.addRowBias(m, bias)
  override def reduceSumRows(m: Matrix): Vec = cpuFallback.reduceSumRows(m)
  override def linearActivationBatch(x: Matrix, wT: Matrix, b: Vec, activation: String): (Matrix, Matrix) =
    val t0 = System.nanoTime()
    linearBatchCalls += 1
    if linearBatchEnabled then
      MetalNativeBridge.tryLinearActivationBatch(x, wT, b, activation) match
        case Right(out) =>
          linearBatchNanos += (System.nanoTime() - t0)
          out
        case Left(reason) =>
          disableOp("batchLinearActivation", reason)
          val out = cpuFallback.linearActivationBatch(x, wT, b, activation)
          linearBatchNanos += (System.nanoTime() - t0)
          out
    else
      val out = cpuFallback.linearActivationBatch(x, wT, b, activation)
      linearBatchNanos += (System.nanoTime() - t0)
      out

  override def resetProfile(): Unit =
    matVecCalls = 0L
    outerCalls = 0L
    linearCalls = 0L
    matVecNanos = 0L
    outerNanos = 0L
    linearNanos = 0L
    matMulCalls = 0L
    linearBatchCalls = 0L
    softmaxBatchCalls = 0L
    ceBatchCalls = 0L
    matMulNanos = 0L
    linearBatchNanos = 0L
    softmaxBatchNanos = 0L
    ceBatchNanos = 0L

  override def profileSummary: String =
    def ms(nanos: Long): Double = nanos.toDouble / 1e6
    s"matVec: calls=$matVecCalls timeMs=${String.format(Locale.US, "%.2f", Double.box(ms(matVecNanos)))}, " +
      s"outer: calls=$outerCalls timeMs=${String.format(Locale.US, "%.2f", Double.box(ms(outerNanos)))}, " +
      s"linearActivation: calls=$linearCalls timeMs=${String.format(Locale.US, "%.2f", Double.box(ms(linearNanos)))}, " +
      s"matMul: calls=$matMulCalls timeMs=${String.format(Locale.US, "%.2f", Double.box(ms(matMulNanos)))}, " +
      s"linearBatch: calls=$linearBatchCalls timeMs=${String.format(Locale.US, "%.2f", Double.box(ms(linearBatchNanos)))}, " +
      s"softmaxBatch: calls=$softmaxBatchCalls timeMs=${String.format(Locale.US, "%.2f", Double.box(ms(softmaxBatchNanos)))}, " +
      s"ceBatch: calls=$ceBatchCalls timeMs=${String.format(Locale.US, "%.2f", Double.box(ms(ceBatchNanos)))}"

object MetalBackend:
  def create(precision: String): MetalBackend =
    val probe = MetalNativeBridge.probe()
    MetalBackend(precision = precision, probe = probe)
