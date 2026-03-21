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
  @native private def linearActivationNative(
      matrix: Array[Double],
      rows: Int,
      cols: Int,
      x: Array[Double],
      bias: Array[Double],
      activationCode: Int
  ): Array[Double]

  private def failSwitch(name: String): Boolean =
    sys.props.get(s"metal.simulate.fail.$name").exists(_.trim == "1")

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
      if matrix.cols != vec.length then Left(s"shape mismatch matrix.cols=${matrix.cols} vec.length=${vec.length}")
      else
        probe() match
          case MetalProbe(_, true, _, _) =>
            try
              val out = matVecMulNative(matrix.data.toArray, matrix.rows, matrix.cols, vec.toArray)
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

  def tryLinearActivation(m: Matrix, x: Vec, b: Vec, activation: String): Either[String, (Vec, Vec)] =
    if failSwitch("linear") then Left("SIM_FAIL:linearActivation")
    else
      if m.cols != x.length then Left(s"shape mismatch m.cols=${m.cols} x.length=${x.length}")
      else if m.rows != b.length then Left(s"shape mismatch m.rows=${m.rows} b.length=${b.length}")
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
                val packed = linearActivationNative(m.data.toArray, m.rows, m.cols, x.toArray, b.toArray, activationCode)
                if packed.length != m.rows * 2 then Left(s"native linearActivation returned wrong size=${packed.length}")
                else
                  val z = packed.slice(0, m.rows).toVector
                  val a = packed.slice(m.rows, m.rows * 2).toVector
                  Right((z, a))
              catch
                case t: Throwable =>
                  Left(s"${t.getClass.getSimpleName}: ${Option(t.getMessage).getOrElse("native linearActivation failed")}")
            case p =>
              Left(p.error.getOrElse("Metal unavailable"))

final case class MetalProbe(loaded: Boolean, available: Boolean, deviceName: String, error: Option[String])
