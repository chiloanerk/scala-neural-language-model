package compute

import linalg.{Matrix, Vec}
import java.nio.file.{Files, Path}

object MetalNativeBridge:
  @volatile private var attemptedLoad = false
  @volatile private var loaded = false
  @volatile private var loadError: Option[String] = None
  @volatile private var cachedProbe: Option[MetalProbe] = None

  @native private def initNative(): Boolean
  @native private def isAvailableNative(): Boolean
  @native private def deviceNameNative(): String
  @native private def matVecMulNative(matrix: Array[Double], rows: Int, cols: Int, vec: Array[Double]): Array[Double]
  @native private def outerNative(a: Array[Double], b: Array[Double]): Array[Double]
  @native private def matMulNative(
      a: Array[Double],
      aRows: Int,
      aCols: Int,
      b: Array[Double],
      bRows: Int,
      bCols: Int
  ): Array[Double]
  @native private def linearActivationNative(
      matrix: Array[Double],
      rows: Int,
      cols: Int,
      x: Array[Double],
      bias: Array[Double],
      activationCode: Int
  ): Array[Double]
  @native private def linearActivationBatchNative(
      x: Array[Double],
      rows: Int,
      inCols: Int,
      wT: Array[Double],
      wRows: Int,
      wCols: Int,
      bias: Array[Double],
      activationCode: Int
  ): Array[Double]
  @native private def softmaxBatchNative(logits: Array[Double], rows: Int, cols: Int): Array[Double]
  @native private def crossEntropyBatchNative(probs: Array[Double], rows: Int, cols: Int, targets: Array[Int]): Array[Double]

  private def failSwitch(name: String): Boolean =
    sys.props.get(s"metal.simulate.fail.$name").exists(_.trim == "1")

  private def contiguous(m: Matrix): Matrix =
    if !m.transposed then m
    else Matrix.fromFunction(m.rows, m.cols)((r, c) => m.get(r, c))

  def ensureLoaded(): Either[String, Unit] = synchronized {
    if loaded then Right(())
    else
      if !attemptedLoad then
        attemptedLoad = true
        val candidate =
          sys.props
            .get("metal.jni.lib")
            .orElse(sys.env.get("METAL_JNI_LIB"))
            .getOrElse("metal-jni/build/libmetal_jni.dylib")
        try
          val p = Path.of(candidate)
          if Files.exists(p) then System.load(p.toAbsolutePath.toString)
          else System.loadLibrary("metal_jni")
          loaded = true
          loadError = None
        catch
          case t: Throwable =>
            loaded = false
            loadError = Some(t.getMessage)

      if loaded then Right(())
      else Left(loadError.getOrElse("unknown native load error"))
  }

  def probe(): MetalProbe =
    cachedProbe match
      case Some(p) => p
      case None =>
        val p =
          ensureLoaded() match
            case Left(err) => MetalProbe(loaded = false, available = false, deviceName = "n/a", error = Some(err))
            case Right(_) =>
              try
                val initOk = initNative()
                val available = initOk && isAvailableNative()
                val name = if available then Option(deviceNameNative()).getOrElse("Apple GPU") else "n/a"
                MetalProbe(loaded = true, available = available, deviceName = name, error = if available then None else Some("Metal unavailable"))
              catch
                case t: Throwable =>
                  val msg = Option(t.getMessage).getOrElse(t.getClass.getSimpleName)
                  MetalProbe(loaded = true, available = false, deviceName = "n/a", error = Some(s"${t.getClass.getSimpleName}: $msg"))
        cachedProbe = Some(p)
        p

  def tryMatVecMul(matrix: Matrix, vec: Vec): Either[String, Vec] =
    if failSwitch("matvec") then Left("SIM_FAIL:matvec")
    else
      val matrixC = contiguous(matrix)
      if matrixC.cols != vec.length then Left(s"shape mismatch matrix.cols=${matrixC.cols} vec.length=${vec.length}")
      else
        probe() match
          case MetalProbe(_, true, _, _) =>
            try
              val out = matVecMulNative(matrixC.data.toArray, matrixC.rows, matrixC.cols, vec.toArray)
              Right(out.toVector)
            catch
              case t: Throwable =>
                Left(s"${t.getClass.getSimpleName}: ${Option(t.getMessage).getOrElse("native matVecMul failed")}")
          case p =>
            Left(p.error.getOrElse("Metal unavailable"))

  def tryOuter(a: Vec, b: Vec): Either[String, Matrix] =
    if failSwitch("outer") then Left("SIM_FAIL:outer")
    else
      probe() match
        case MetalProbe(_, true, _, _) =>
          try
            val out = outerNative(a.toArray, b.toArray)
            Right(Matrix(out.toVector, a.length, b.length))
          catch
            case t: Throwable =>
              Left(s"${t.getClass.getSimpleName}: ${Option(t.getMessage).getOrElse("native outer failed")}")
        case p =>
          Left(p.error.getOrElse("Metal unavailable"))

  def tryMatMul(a: Matrix, b: Matrix): Either[String, Matrix] =
    if failSwitch("matmul") then Left("SIM_FAIL:matmul")
    else
      val aC = contiguous(a)
      val bC = contiguous(b)
      if aC.cols != bC.rows then Left(s"shape mismatch a.cols=${aC.cols} b.rows=${bC.rows}")
      else
        probe() match
          case MetalProbe(_, true, _, _) =>
            try
              val out = matMulNative(aC.data.toArray, aC.rows, aC.cols, bC.data.toArray, bC.rows, bC.cols)
              Right(Matrix(out.toVector, aC.rows, bC.cols))
            catch
              case t: Throwable =>
                Left(s"${t.getClass.getSimpleName}: ${Option(t.getMessage).getOrElse("native matMul failed")}")
          case p =>
            Left(p.error.getOrElse("Metal unavailable"))

  def tryLinearActivation(m: Matrix, x: Vec, b: Vec, activation: String): Either[String, (Vec, Vec)] =
    if failSwitch("linear") then Left("SIM_FAIL:linearActivation")
    else
      val mC = contiguous(m)
      if mC.cols != x.length then Left(s"shape mismatch m.cols=${mC.cols} x.length=${x.length}")
      else if mC.rows != b.length then Left(s"shape mismatch m.rows=${mC.rows} b.length=${b.length}")
      else
        val activationCode = activation.toLowerCase match
          case "relu" => 1
          case "tanh" => 0
          case _      => -1

        if activationCode == -1 then Left(s"unsupported activation: $activation")
        else
          probe() match
            case MetalProbe(_, true, _, _) =>
              try
                val packed = linearActivationNative(mC.data.toArray, mC.rows, mC.cols, x.toArray, b.toArray, activationCode)
                if packed.length != mC.rows * 2 then Left(s"native linearActivation returned wrong size=${packed.length}")
                else
                  val z = packed.slice(0, mC.rows).toVector
                  val a = packed.slice(mC.rows, mC.rows * 2).toVector
                  Right((z, a))
              catch
                case t: Throwable =>
                  Left(s"${t.getClass.getSimpleName}: ${Option(t.getMessage).getOrElse("native linearActivation failed")}")
            case p =>
              Left(p.error.getOrElse("Metal unavailable"))

  def tryLinearActivationBatch(x: Matrix, wT: Matrix, bias: Vec, activation: String): Either[String, (Matrix, Matrix)] =
    if failSwitch("linear_batch") then Left("SIM_FAIL:linear_batch")
    else
      val xC = contiguous(x)
      val wTC = contiguous(wT)
      if xC.cols != wTC.rows then Left(s"shape mismatch x.cols=${xC.cols} wT.rows=${wTC.rows}")
      else if wTC.cols != bias.length then Left(s"shape mismatch wT.cols=${wTC.cols} bias.length=${bias.length}")
      else
        val activationCode = activation.toLowerCase match
          case "relu" => 1
          case "tanh" => 0
          case _      => -1
        if activationCode == -1 then Left(s"unsupported activation: $activation")
        else
          probe() match
            case MetalProbe(_, true, _, _) =>
              try
                val packed = linearActivationBatchNative(
                  xC.data.toArray,
                  xC.rows,
                  xC.cols,
                  wTC.data.toArray,
                  wTC.rows,
                  wTC.cols,
                  bias.toArray,
                  activationCode
                )
                val expected = xC.rows * wTC.cols * 2
                if packed.length != expected then Left(s"native linearActivationBatch returned wrong size=${packed.length} expected=$expected")
                else
                  val zSize = xC.rows * wTC.cols
                  val z = Matrix(packed.slice(0, zSize).toVector, xC.rows, wTC.cols)
                  val a = Matrix(packed.slice(zSize, zSize * 2).toVector, xC.rows, wTC.cols)
                  Right((z, a))
              catch
                case t: Throwable =>
                  Left(s"${t.getClass.getSimpleName}: ${Option(t.getMessage).getOrElse("native linearActivationBatch failed")}")
            case p =>
              Left(p.error.getOrElse("Metal unavailable"))

  def trySoftmaxBatch(logits: Matrix): Either[String, Matrix] =
    if failSwitch("softmax_batch") then Left("SIM_FAIL:softmax_batch")
    else
      val logitsC = contiguous(logits)
      probe() match
        case MetalProbe(_, true, _, _) =>
          try
            val out = softmaxBatchNative(logitsC.data.toArray, logitsC.rows, logitsC.cols)
            Right(Matrix(out.toVector, logitsC.rows, logitsC.cols))
          catch
            case t: Throwable =>
              Left(s"${t.getClass.getSimpleName}: ${Option(t.getMessage).getOrElse("native softmaxBatch failed")}")
        case p =>
          Left(p.error.getOrElse("Metal unavailable"))

  def tryCrossEntropyBatch(probs: Matrix, targets: Vector[Int]): Either[String, Vec] =
    if failSwitch("ce_batch") then Left("SIM_FAIL:ce_batch")
    else
      val probsC = contiguous(probs)
      if probsC.rows != targets.length then Left(s"shape mismatch probs.rows=${probsC.rows} targets=${targets.length}")
      else
        probe() match
          case MetalProbe(_, true, _, _) =>
            try
              val out = crossEntropyBatchNative(probsC.data.toArray, probsC.rows, probsC.cols, targets.toArray)
              Right(out.toVector)
            catch
              case t: Throwable =>
                Left(s"${t.getClass.getSimpleName}: ${Option(t.getMessage).getOrElse("native crossEntropyBatch failed")}")
          case p =>
            Left(p.error.getOrElse("Metal unavailable"))

final case class MetalProbe(loaded: Boolean, available: Boolean, deviceName: String, error: Option[String])
