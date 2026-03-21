package compute

object BackendSelector:
  def normalizeBackend(raw: String): String = raw.trim.toLowerCase match
    case "gpu" | "metal" => "gpu"
    case _                  => "cpu"

  def normalizePrecision(raw: String): String = raw.trim.toLowerCase match
    case "fp32" => "fp32"
    case _       => "fp64"

  def fromConfig(backend: String, precision: String, warn: String => Unit = _ => ()): ComputeBackend =
    val b = normalizeBackend(backend)
    val p = normalizePrecision(precision)

    b match
      case "cpu" => CpuBackend(p)
      case "gpu" =>
        val metal = MetalBackend.create(p)
        if !metal.isGpu then
          warn(s"GPU requested but unavailable. Falling back to CPU. ${metal.diagnostics}")
        metal

  def gpuInfo(precision: String = "fp64"): String =
    val backend = fromConfig("gpu", precision)
    backend.diagnostics
