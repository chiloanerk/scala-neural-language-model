package app

import compute.BackendSelector
import data.{TextPipeline, VocabIO}
import eval.Metrics
import linalg.LinearAlgebra
import nn.{LanguageModel, ModelConfig}
import observability.{BenchmarkCell, GcProbe, GpuHotPathMetrics, MemoryMetrics, MemoryProbe, PlatformInfo, RunMetrics, RunObservability, RunProfile}
import train.{CheckpointIO, ReplayBuffer, SaveDecision, TrainConfig, Trainer}

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}
import java.util.concurrent.atomic.{AtomicBoolean, AtomicReference}
import java.util.Properties
import java.util.Locale
import scala.io.StdIn
import scala.jdk.CollectionConverters._
import scala.util.Using
import scala.util.control.NoStackTrace

object Main:
  private case object BackToLauncher extends RuntimeException("back-to-launcher") with NoStackTrace
  private val ProjectRoot = Path.of(".").toAbsolutePath.normalize
  private val BenchmarkPreferenceOrder: Vector[(String, String)] =
    Vector(("gpu", "fp32"), ("gpu", "fp64"), ("cpu", "fp32"), ("cpu", "fp64"))
  private val LauncherCommands: Vector[String] =
    Vector("train", "predict", "benchmark", "chunk", "gpu-info", "help", "exit")

  private def displayPath(path: Path): String =
    val normalized = path.toAbsolutePath.normalize
    if normalized.startsWith(ProjectRoot) then
      val rel = ProjectRoot.relativize(normalized).toString
      if rel.isEmpty then "." else rel
    else path.toString

  // Persistent model paths
  val ModelPath = Path.of("data/models/latest.ckpt")
  val VocabPath = Path.of("data/models/latest.vocab")
  val ReplayPath = Path.of("data/models/latest.replay")
  val InterruptModelPath = Path.of("data/models/latest.interrupt.ckpt")
  val InterruptVocabPath = Path.of("data/models/latest.interrupt.vocab")
  val InterruptMetaPath = Path.of("data/models/latest.interrupt.meta")

  // Training presets
  case class Preset(
      name: String,
      epochs: Int,
      patience: Int,
      hiddenDim: Int,
      embedDim: Int,
      learningRate: Double,
      description: String
  )

  val presets = Vector(
    Preset("quick", 5, 3, 32, 16, 0.05, "Test (~30 sec)"),
    Preset("balanced", 20, 5, 64, 24, 0.02, "Standard (~2 min)"),
    Preset("thorough", 50, 10, 128, 48, 0.01, "Best quality (~10 min)")
  )

  private def readLineWithPrompt(prompt: String): String | Null =
    // sbt forked runs can buffer non-newline output; print prompt as its own line.
    println(prompt)
    Console.out.flush()
    StdIn.readLine()

  private def isBackToken(raw: String): Boolean =
    val t = raw.trim.toLowerCase
    t == "b" || t == "back"

  private enum MenuPick:
    case Choice(index: Int)
    case Back

  private def parseMenuPick(raw: String, optionCount: Int, defaultIndex: Int, allowBack: Boolean = false): Option[MenuPick] =
    if allowBack && isBackToken(raw) then Some(MenuPick.Back)
    else CliHelpers.parseMenuChoice(raw, optionCount, defaultIndex).map(MenuPick.Choice(_))

  private def promptYesNoOrBack(label: String, default: Boolean): Option[Boolean] =
    val defaultStr = if default then "Y/n" else "y/N"
    val raw = readLineWithPrompt(s"$label [$defaultStr, b=back]: ")
    if raw == null then
      println("\nNo interactive input available for confirmation; canceling for safety.")
      Some(false)
    else if isBackToken(raw) then None
    else Some(CliHelpers.parseYesNo(raw, default))

  private def defaultPrecisionForBackend(backend: String): String =
    "fp64"

  private def formatDecimal(value: Double, decimals: Int = 2): String =
    String.format(Locale.US, s"%.${math.max(0, decimals)}f", Double.box(value))

  private def deviceNameFromDiagnostics(diagnostics: String): String =
    val marker = "device="
    val idx = diagnostics.indexOf(marker)
    if idx < 0 then "unknown"
    else
      val start = idx + marker.length
      val tail = diagnostics.substring(start)
      val end = tail.indexWhere(ch => ch == ',' || ch == ')' || ch == ';')
      if end < 0 then tail.trim else tail.substring(0, end).trim

  private def platformInfoFromDiagnostics(diagnostics: String): PlatformInfo =
    PlatformInfo(
      osName = System.getProperty("os.name", "unknown"),
      osVersion = System.getProperty("os.version", "unknown"),
      arch = System.getProperty("os.arch", "unknown"),
      javaVersion = System.getProperty("java.version", "unknown"),
      deviceName = deviceNameFromDiagnostics(diagnostics)
    )

  private[app] def renderBenchmarkMetricsReport(
      runMetrics: RunMetrics,
      persisted: Option[RunObservability.PersistOutcome],
      displayPathFn: Path => String = p => p.toString
  ): String =
    val lines = scala.collection.mutable.ArrayBuffer.empty[String]
    val matrixSorted = runMetrics.benchmarkMatrix.sortBy(c => (c.backend, c.precision))
    val matrixByKey = runMetrics.benchmarkMatrix.map(c => (c.backend, c.precision) -> c).toMap
    val preferredCell =
      BenchmarkPreferenceOrder.collectFirst { case key if matrixByKey.contains(key) => matrixByKey(key) }
        .orElse(runMetrics.benchmarkMatrix.maxByOption(_.exPerSec))

    def speedupByKeys(numerator: (String, String), denominator: (String, String)): Option[Double] =
      for
        num <- matrixByKey.get(numerator)
        den <- matrixByKey.get(denominator)
        if den.exPerSec > 0.0
      yield num.exPerSec / den.exPerSec

    val relativeSpeedups = Vector(
      speedupByKeys(("gpu", "fp32"), ("cpu", "fp32")).map(v => s"gpu fp32 vs cpu fp32: ${formatDecimal(v)}x"),
      speedupByKeys(("gpu", "fp64"), ("cpu", "fp64")).map(v => s"gpu fp64 vs cpu fp64: ${formatDecimal(v)}x"),
      speedupByKeys(("gpu", "fp32"), ("gpu", "fp64")).map(v => s"fp32 vs fp64 on gpu: ${formatDecimal(v)}x"),
      speedupByKeys(("cpu", "fp32"), ("cpu", "fp64")).map(v => s"fp32 vs fp64 on cpu: ${formatDecimal(v)}x")
    ).flatten

    lines += "=== Metrics Report (benchmark --metrics) ==="
    lines += s"Run ID: ${runMetrics.runId}"
    lines += s"Platform: ${runMetrics.platform.osName} ${runMetrics.platform.osVersion} | arch=${runMetrics.platform.arch} | java=${runMetrics.platform.javaVersion} | device=${runMetrics.platform.deviceName}"
    lines += s"Throughput: ${formatDecimal(runMetrics.throughputExPerSec)} ex/s"
    lines += s"Total runtime: ${formatDecimal(runMetrics.profile.totalSeconds)} s"
    lines += s"Memory peak RSS: ${runMetrics.memoryPeak.rssBytes} bytes (${formatDecimal(runMetrics.memoryPeak.rssBytes.toDouble / (1024.0 * 1024.0))} MB)"
    lines += s"GC delta: count=${runMetrics.gcCountDelta} timeMs=${runMetrics.gcTimeMsDelta}"
    if runMetrics.profile.topStages.nonEmpty then
      val scope = preferredCell.map(c => s"${c.backend}/${c.precision}").getOrElse("n/a")
      lines += s"Top stages (preferred run: $scope):"
      runMetrics.profile.topStages.foreach { s =>
        lines += f"  ${s.stage}%-18s ${formatDecimal(s.timeMs)} ms (${formatDecimal(s.sharePct, 1)}%%)"
      }
    if relativeSpeedups.nonEmpty then
      lines += "Relative speedup:"
      relativeSpeedups.foreach(s => lines += s"  - $s")
    lines += "Benchmark matrix:"
    lines += "  backend  precision  ex/s      effective  fallback"
    matrixSorted.foreach { c =>
      val fb = if c.fallbackOps.isEmpty then "none" else c.fallbackOps.mkString("+")
      lines += f"  ${c.backend}%-7s ${c.precision}%-9s ${formatDecimal(c.exPerSec)}%8s  ${c.effectiveBackend}%-9s $fb"
    }
    persisted.foreach { out =>
      out.baseline.foreach { b =>
        lines += s"Baseline: ${b.runId} (${formatDecimal(b.throughputExPerSec)} ex/s)"
      }
      out.regression.foreach { r =>
        lines += s"Regression verdict: ${r.status} (${r.message})"
      }
      lines += s"Summary file: ${displayPathFn(out.latestSummaryPath)}"
      out.latestDiffSummaryPath.foreach(p => lines += s"Diff file: ${displayPathFn(p)}")
    }
    lines.mkString("\n")

  private def loadInterruptMeta(path: Path): Map[String, String] =
    if !Files.isRegularFile(path) then Map.empty
    else
      try
        val props = Properties()
        val in = Files.newInputStream(path)
        try props.load(in)
        finally in.close()
        props.stringPropertyNames().asScala.map(k => k -> props.getProperty(k)).toMap
      catch
        case _: Exception => Map.empty

  private def saveInterruptMeta(path: Path, values: Map[String, String]): Unit =
    val props = Properties()
    values.foreach { case (k, v) => props.setProperty(k, v) }
    Files.createDirectories(path.getParent)
    val out = Files.newOutputStream(path)
    try props.store(out, "interrupt training metadata")
    finally out.close()

  private def cleanupInterruptArtifacts(): Vector[Path] =
    Vector(InterruptModelPath, InterruptVocabPath, InterruptMetaPath).flatMap { p =>
      try
        if Files.deleteIfExists(p) then Some(p) else None
      catch
        case _: Exception => None
    }

  private def parseObservabilitySettings(flags: Map[String, String]): RunObservability.Settings =
    val recordMetrics = flags.get("recordMetrics").map(v => CliHelpers.parseYesNo(v, default = true)).getOrElse(true)
    val metricsDir = Path.of(flags.getOrElse("metricsDir", RunObservability.DefaultMetricsDir.toString))
    val runLabel = flags.get("runLabel").map(_.trim).filter(_.nonEmpty)
    val compareTo = flags.get("compareTo").map(_.trim).filter(_.nonEmpty)
    val regressionWarnPct = flags.get("regressionWarnPct").flatMap(_.toDoubleOption).getOrElse(5.0)
    RunObservability.Settings(
      recordMetrics = recordMetrics,
      metricsDir = metricsDir,
      runLabel = runLabel,
      compareTo = compareTo,
      regressionWarnPct = regressionWarnPct
    )

  private def peakMemory(start: MemoryMetrics, end: MemoryMetrics, epochHeap: Vector[Long], epochRss: Vector[Long]): MemoryMetrics =
    val heapPeak = (epochHeap :+ start.heapUsedBytes :+ end.heapUsedBytes).maxOption.getOrElse(start.heapUsedBytes)
    val rssPeak = (epochRss :+ start.rssBytes :+ end.rssBytes).maxOption.getOrElse(start.rssBytes)
    MemoryMetrics(
      timestampEpochMs = math.max(start.timestampEpochMs, end.timestampEpochMs),
      heapUsedBytes = heapPeak,
      heapCommittedBytes = math.max(start.heapCommittedBytes, end.heapCommittedBytes),
      nonHeapUsedBytes = math.max(start.nonHeapUsedBytes, end.nonHeapUsedBytes),
      rssBytes = rssPeak
    )

  private[app] def benchmarkMatrix(backendFlag: Option[String], precisionFlag: Option[String]): Vector[(String, String)] =
    val backends = backendFlag match
      case Some(b) => Vector(BackendSelector.normalizeBackend(b))
      case None    => Vector("cpu", "gpu")
    val precisions = precisionFlag match
      case Some(p) => Vector(BackendSelector.normalizePrecision(p))
      case None    => Vector("fp64", "fp32")
    for
      b <- backends
      p <- precisions
    yield (b, p)

  private[app] def benchmarkSelection(index: Int): (Option[String], Option[String]) =
    index match
      case 0 => (None, None)                 // CPU/GPU x fp64/fp32
      case 1 => (Some("cpu"), Some("fp64"))
      case 2 => (Some("cpu"), Some("fp32"))
      case 3 => (Some("gpu"), Some("fp64"))
      case 4 => (Some("gpu"), Some("fp32"))
      case 5 => (Some("cpu"), None)          // CPU x (fp64, fp32)
      case 6 => (Some("gpu"), None)          // GPU x (fp64, fp32)
      case 7 => (None, Some("fp64"))         // (CPU,GPU) x fp64
      case 8 => (None, Some("fp32"))         // (CPU,GPU) x fp32
      case _ => (None, None)

  private[app] def preferredBenchmarkKey(results: Iterable[((String, String), Double)]): Option[(String, String)] =
    val ranked = results.toVector
    if ranked.isEmpty then None
    else
      val keys = ranked.map(_._1).toSet
      BenchmarkPreferenceOrder.find(keys.contains)
        .orElse(ranked.sortBy { case (_, exPerSec) => -exPerSec }.headOption.map(_._1))

  private[app] def launcherCommandForSelection(index: Int): Option[String] =
    LauncherCommands.lift(index)

  private[app] def resolveLauncherCommand(raw: String): Option[String] =
    CliHelpers.parseMenuChoice(raw, optionCount = LauncherCommands.length, defaultIndex = 0)
      .flatMap(launcherCommandForSelection)

  private def readTrimmedRequired(prompt: String, field: String): String =
    Option(readLineWithPrompt(prompt)) match
      case Some(v) => v.trim
      case None =>
        println(s"\nNo interactive input available for '$field'.")
        println("Tip: run with explicit flags in non-interactive mode (add --yes to auto-confirm).")
        sys.exit(1)
        ""

  def main(args: Array[String]): Unit =
    if args.isEmpty then
      runInteractiveLauncher()
      return

    args(0) match
      case "train"   =>
        try runTrain(CliHelpers.parseArgs(args.drop(1)))
        catch case BackToLauncher => println("Canceled and returned.")
      case "predict" => runPredict(CliHelpers.parseArgs(args.drop(1)))
      case "chunk"   => runChunker(CliHelpers.parseArgs(args.drop(1)))
      case "gpu-info" => runGpuInfo(CliHelpers.parseArgs(args.drop(1)))
      case "benchmark" =>
        try runBenchmark(CliHelpers.parseArgs(args.drop(1)))
        catch case BackToLauncher => println("Canceled and returned.")
      case "test"    => TestRunner.main(Array.empty)
      case _ =>
        printUsage()
        sys.exit(1)

  private def runInteractiveLauncher(): Unit =
    val modelStatus = if Files.isRegularFile(ModelPath) then
      val size = Files.size(ModelPath) / 1024
      s"✓ Model exists (${size} KB)"
    else "✗ No model yet"

    var keepRunning = true
    while keepRunning do
      var resolved: Option[String] = None
      var canceled = false
      while resolved.isEmpty && !canceled do
        println(s"\n=== Scala Neural Language Model (NLM) ===")
        println(s"Status: $modelStatus\n")
        println("Tip: press Enter to accept the default in [brackets].")
        println("What would you like to do?")
        println("  1. Train model (guided)")
        println("  2. Predict next words")
        println("  3. Run benchmark")
        println("  4. Chunk text files")
        println("  5. Check GPU info")
        println("  6. Help / command reference")
        println("  7. Exit")
        println()
        Option(readLineWithPrompt("Select [1]: ")) match
          case None =>
            canceled = true
            println("\nNo interactive input available. Showing help instead.")
            printUsage()
          case Some(rawInput) =>
            resolveLauncherCommand(rawInput.trim) match
              case Some(command) => resolved = Some(command)
              case None =>
                println(s"Invalid selection. Enter a number from 1 to ${LauncherCommands.length}.")

      resolved match
        case Some("train") =>
          try runTrain(Map.empty)
          catch case BackToLauncher => println("Returning to main menu.")
          println("Quick actions: 1) Train  2) Predict  3) Benchmark  7) Exit")
        case Some("predict") =>
          try runPredict(Map.empty)
          catch case BackToLauncher => println("Returning to main menu.")
          println("Quick actions: 1) Train  2) Predict  3) Benchmark  7) Exit")
        case Some("benchmark") =>
          try runBenchmark(Map.empty)
          catch case BackToLauncher => println("Returning to main menu.")
          println("Quick actions: 1) Train  2) Predict  3) Benchmark  7) Exit")
        case Some("chunk") =>
          try runChunker(Map.empty)
          catch case BackToLauncher => println("Returning to main menu.")
          println("Quick actions: 1) Train  2) Predict  3) Benchmark  7) Exit")
        case Some("gpu-info") =>
          try runGpuInfo(Map.empty)
          catch case BackToLauncher => println("Returning to main menu.")
          println("Quick actions: 1) Train  2) Predict  3) Benchmark  7) Exit")
        case Some("help") =>
          printUsage()
          println("Quick actions: 1) Train  2) Predict  3) Benchmark  7) Exit")
        case Some("exit")      =>
          println("Exiting.")
          keepRunning = false
        case _ =>
          keepRunning = false

      if canceled then keepRunning = false

  private def runTrain(flags: Map[String, String]): Unit =
    println("\n=== Training ===\n")
    if flags.isEmpty then
      println("Tip: press Enter to accept defaults shown in [brackets]. Type 'b' to go back where available.\n")
    val observability = parseObservabilitySettings(flags)
    val runStartMs = System.currentTimeMillis()
    val memoryStart = MemoryProbe.snapshot(runStartMs)
    val gcStart = GcProbe.snapshot()

    val freshTraining = flags.get("fresh").exists(CliHelpers.isTruthy)
    val interruptAvailable = Files.isRegularFile(InterruptModelPath) && Files.isRegularFile(InterruptVocabPath)
    val interruptNewerThanLatest =
      if !interruptAvailable then false
      else if !Files.isRegularFile(ModelPath) then true
      else Files.getLastModifiedTime(InterruptModelPath).toMillis >= Files.getLastModifiedTime(ModelPath).toMillis
    val autoResumeInterrupt = !freshTraining && interruptAvailable && interruptNewerThanLatest
    val interruptMeta = if autoResumeInterrupt then loadInterruptMeta(InterruptMetaPath) else Map.empty
    val effectiveFlags =
      if autoResumeInterrupt then
        val merged = scala.collection.mutable.Map.from(flags ++ interruptMeta.filterNot { case (k, _) => flags.contains(k) })
        if flags.contains("input") then merged.remove("inputs")
        if flags.contains("inputs") then merged.remove("input")
        if flags.contains("input") || flags.contains("inputs") then
          if !flags.contains("inputWeights") then merged.remove("inputWeights")
        merged.toMap
      else flags

    val activeModelPath = if autoResumeInterrupt then InterruptModelPath else ModelPath
    val activeVocabPath = if autoResumeInterrupt then InterruptVocabPath else VocabPath
    if autoResumeInterrupt then
      println(s"Found interrupt snapshot newer than latest model.")
      println(s"Auto-resuming from: ${displayPath(activeModelPath)}")
      if interruptMeta.nonEmpty then println("Inherited prior run settings from interrupt metadata.")

    val hasExistingModel = Files.isRegularFile(activeModelPath)
    val inputFlag = effectiveFlags.get("input").map(_.trim).filter(_.nonEmpty)
    val inputsFlag = effectiveFlags.get("inputs").map(_.trim).filter(_.nonEmpty)

    require(!(inputFlag.isDefined && inputsFlag.isDefined), "Use only one of --input or --inputs.")

    val actuallyFresh = if hasExistingModel && !freshTraining && inputFlag.isEmpty && inputsFlag.isEmpty && !autoResumeInterrupt then
      println(s"Found existing model: ${displayPath(activeModelPath)} (${Files.size(activeModelPath) / 1024} KB)")
      println()
      println("How do you want to proceed?")
      println("  1. Continue training (default, keeps current model)")
      println("  2. Start fresh (new model from scratch)")
      println("  b. Back to main menu")
      println()
      var actionFresh = false
      var resolved = false
      while !resolved do
        val raw = readTrimmedRequired("Select [1]: ", "select action")
        parseMenuPick(raw, optionCount = 2, defaultIndex = 0, allowBack = true) match
          case Some(MenuPick.Choice(idx)) =>
            actionFresh = idx == 1
            resolved = true
          case Some(MenuPick.Back) =>
            println("Returning to main menu.")
            throw BackToLauncher
          case None =>
            println("Please choose 1 or 2, or type 'b' to go back.")
      actionFresh
    else if autoResumeInterrupt then
      println("Continuing from interrupt snapshot (auto-resume).\n")
      false
    else if freshTraining then
      println("Starting fresh training (ignoring any existing model).\n")
      true
    else if hasExistingModel then
      val size = Files.size(activeModelPath) / 1024
      println(s"Found existing model: ${displayPath(activeModelPath)} (${size} KB)")
      println("This will continue training from where you left off.\n")
      false
    else
      println("No existing model found. Starting fresh training.\n")
      false

    val inputPaths: Vector[Path] = inputsFlag match
      case Some(csv) => csv.split(",").toVector.map(_.trim).filter(_.nonEmpty).map(Path.of(_))
      case None =>
        inputFlag match
          case Some(single) => Vector(Path.of(single))
          case None         => Vector(selectTrainingInputInteractively())

    require(inputPaths.nonEmpty, "No input files provided.")
    inputPaths.foreach(p => require(Files.isRegularFile(p), s"Input file not found: $p"))

    val existingCfg = if hasExistingModel && !actuallyFresh && Files.isRegularFile(activeModelPath) then
      Some(CheckpointIO.load(activeModelPath)._2)
    else
      None

    val presetName = effectiveFlags.getOrElse("preset", {
      println("\nPick a training preset:")
      presets.zipWithIndex.foreach { case (p, idx) =>
        println(s"  ${idx + 1}. ${p.name.capitalize} - ${p.description} (${p.epochs} epochs, patience=${p.patience})")
      }
      println("  b. Back to main menu")
      var selectedPreset: Option[String] = None
      while selectedPreset.isEmpty do
        val raw = readTrimmedRequired("\nSelect preset [2]: ", "select preset")
        parseMenuPick(raw, optionCount = presets.length, defaultIndex = 1, allowBack = true) match
          case Some(MenuPick.Choice(idx)) =>
            selectedPreset = Some(presets(idx).name)
          case Some(MenuPick.Back) =>
            println("Returning to main menu.")
            throw BackToLauncher
          case None =>
            println(s"Please choose 1-${presets.length}, or type 'b' to go back.")
      selectedPreset.get
    })
    val preset = presets.find(_.name == presetName).getOrElse {
      println(s"Unknown preset '$presetName'. Using Balanced.")
      presets(1)
    }
    if !effectiveFlags.contains("preset") then println(s"Selected preset: ${preset.name.capitalize}")

    val autoConfirm = flags.get("yes").exists(CliHelpers.isTruthy)
    val backend = BackendSelector.normalizeBackend(effectiveFlags.getOrElse("backend", "gpu"))
    val precision = effectiveFlags.get("precision") match
      case Some(raw) => BackendSelector.normalizePrecision(raw)
      case None =>
        val defaultPrecision = defaultPrecisionForBackend(backend)
        if autoConfirm then defaultPrecision
        else
          println("\nChoose numeric precision:")
          println("  1. fp64 - most stable, slower")
          println("  2. fp32 - faster, lower memory (recommended on GPU)")
          println("  b. Back to main menu")
          val defaultIdx = if defaultPrecision == "fp32" then 1 else 0
          var selectedPrecision: Option[String] = None
          while selectedPrecision.isEmpty do
            val raw = readTrimmedRequired(s"Select precision [${defaultIdx + 1}]: ", "select precision")
            parseMenuPick(raw, optionCount = 2, defaultIndex = defaultIdx, allowBack = true) match
              case Some(MenuPick.Choice(idx)) =>
                selectedPrecision = Some(if idx == 1 then "fp32" else "fp64")
              case Some(MenuPick.Back) =>
                println("Returning to main menu.")
                throw BackToLauncher
              case None =>
                println("Please choose 1 or 2, or type 'b' to go back.")
          selectedPrecision.get
    if !effectiveFlags.contains("precision") then println(s"Selected precision: $precision")
    val learningRate = effectiveFlags.get("lr").flatMap(_.toDoubleOption).getOrElse(preset.learningRate)
    val lrDecay = effectiveFlags.get("lrDecay").flatMap(_.toDoubleOption).getOrElse(1.0)
    val batchSize = effectiveFlags.get("batchSize").flatMap(_.toIntOption).getOrElse(0)
    val prefetch = effectiveFlags.get("prefetch").flatMap(_.toIntOption).getOrElse(1)
    val profileGpu = effectiveFlags.get("profileGpu").exists(CliHelpers.isTruthy)
    val replayFlagsProvided = effectiveFlags.contains("replayRatio") || effectiveFlags.contains("replayBufferSize") || effectiveFlags.contains("replayBufferPath")
    val replayFileDefault = ReplayPath
    val replayFileExists = Files.isRegularFile(replayFileDefault)
    var replayRatio = effectiveFlags.get("replayRatio").flatMap(_.toDoubleOption).getOrElse(0.3)
    var replayBufferSize = effectiveFlags.get("replayBufferSize").flatMap(_.toIntOption).getOrElse(0)
    var replayBufferPath = effectiveFlags.get("replayBufferPath").map(Path.of(_))
    val ewcLambda = effectiveFlags.get("ewcLambda").flatMap(_.toDoubleOption).getOrElse(0.0)
    val ewcSamples = effectiveFlags.get("ewcSamples").flatMap(_.toIntOption).getOrElse(0)
    if effectiveFlags.get("gpuInfo").exists(CliHelpers.isTruthy) then
      println(s"GPU info: ${BackendSelector.gpuInfo(precision)}")

    if !replayFlagsProvided then
      if hasExistingModel && !actuallyFresh then
        replayRatio = 0.3
        if replayFileExists then replayBufferPath = Some(replayFileDefault)
        if replayBufferSize <= 0 then replayBufferSize = 10000

        if !autoConfirm then
          println("\nReplay settings:")
          if replayFileExists then println(s"  Found replay memory: ${displayPath(replayFileDefault)}")
          else println("  No replay memory file found yet. One can be created after this run.")
          println("  What replay does: mixes older examples into new training to reduce forgetting.")
          println("  Replay ratio guide:")
          println("    - lower (0.1-0.2): faster training, weaker retention")
          println("    - medium (0.3): balanced speed and retention (recommended)")
          println("    - higher (0.4-0.6): stronger retention, slower training")
          println("  1. Use replay defaults (recommended)")
          println("  2. Disable replay for this run (fastest, higher forgetting risk)")
          println("  3. Customize replay settings")
          println("  b. Back to main menu")
          var replayChosen = false
          while !replayChosen do
            val choiceRaw = readTrimmedRequired("Select [1]: ", "select replay option")
            parseMenuPick(choiceRaw, optionCount = 3, defaultIndex = 0, allowBack = true) match
              case Some(MenuPick.Choice(choiceIdx)) =>
                choiceIdx match
                  case 1 =>
                    replayRatio = 0.0
                    replayBufferSize = 0
                    replayBufferPath = None
                    println("Replay disabled for this run.")
                  case 2 =>
                    val ratioRaw = readTrimmedRequired(f"Replay ratio [${replayRatio}%.2f]: ", "replay ratio")
                    replayRatio = if ratioRaw.isEmpty then replayRatio else ratioRaw.toDoubleOption.getOrElse(replayRatio)
                    val sizeRaw = readTrimmedRequired(s"Replay buffer size [$replayBufferSize]: ", "replay buffer size")
                    replayBufferSize = if sizeRaw.isEmpty then replayBufferSize else sizeRaw.toIntOption.getOrElse(replayBufferSize)
                    if replayFileExists then
                      val useExisting = promptYesNo(s"Use existing replay file ($replayFileDefault)?", true)
                      replayBufferPath = if useExisting then Some(replayFileDefault) else None
                    println(f"Replay customized: ratio=$replayRatio%.2f, bufferSize=$replayBufferSize")
                  case _ =>
                    println(f"Replay defaults selected: ratio=$replayRatio%.2f, bufferSize=$replayBufferSize")
                replayChosen = true
              case Some(MenuPick.Back) =>
                println("Returning to main menu.")
                throw BackToLauncher
              case None =>
                println("Please choose 1-3, or type 'b' to go back.")
      else if actuallyFresh && !autoConfirm then
        val enableReplay = promptYesNo("Enable replay memory for this fresh run and future training?", true)
        if enableReplay then
          replayRatio = 0.3
          replayBufferSize = 10000
          replayBufferPath = Some(replayFileDefault)

    require(replayRatio >= 0.0 && replayRatio < 1.0, s"--replayRatio must be in [0,1), got $replayRatio")
    require(replayBufferSize >= 0, s"--replayBufferSize must be >=0, got $replayBufferSize")
    require(ewcLambda == 0.0, "--ewcLambda is not implemented yet. Keep it at 0.0 for now.")

    val contextSize = effectiveFlags.get("contextSize").map(_.toInt).getOrElse {
      existingCfg match
        case Some(cfg) =>
          println(s"\nUsing existing model architecture: context=${cfg.contextSize}")
          cfg.contextSize
        case None =>
          var selected: Option[Int] = None
          while selected.isEmpty do
            val raw = readTrimmedRequired("\nContext size (how many previous words to use) [3]: ", "context size")
            if isBackToken(raw) then
              println("Returning to main menu.")
              throw BackToLauncher
            selected = raw.toIntOption match
              case Some(v) if v >= 1 => Some(v)
              case _ if raw.isEmpty  => Some(3)
              case _ =>
                println("Please enter a number >= 1, or type 'b' to go back.")
                None
          println(s"Selected context size: ${selected.get}")
          selected.get
    }
    val maxVocab = effectiveFlags.get("maxVocab").map(_.toInt).getOrElse {
      existingCfg match
        case Some(cfg) =>
          println(s"Using existing model vocabulary size: ${cfg.vocabSize}")
          cfg.vocabSize
        case None =>
          var selected: Option[Int] = None
          while selected.isEmpty do
            val raw = readTrimmedRequired("Max vocabulary (how many unique words to keep) [3000]: ", "max vocabulary")
            if isBackToken(raw) then
              println("Returning to main menu.")
              throw BackToLauncher
            selected = raw.toIntOption match
              case Some(v) if v >= 100 => Some(v)
              case _ if raw.isEmpty    => Some(3000)
              case _ =>
                println("Please enter a number >= 100, or type 'b' to go back.")
                None
          println(s"Selected max vocabulary: ${selected.get}")
          selected.get
    }

    val rawInputWeights = effectiveFlags.get("inputWeights")
    val sanitizedInputWeights =
      if flags.contains("inputWeights") then rawInputWeights
      else
        rawInputWeights.filter { csv =>
          val count = csv.split(",").toVector.map(_.trim).count(_.nonEmpty)
          count == inputPaths.length
        }
    if autoResumeInterrupt && rawInputWeights.isDefined && sanitizedInputWeights.isEmpty then
      println("Ignoring incompatible inherited inputWeights from interrupt metadata.")
    val inputWeights = parseInputWeights(sanitizedInputWeights, inputPaths.length)
    Files.createDirectories(ModelPath.getParent)

    println(s"\n=== Summary ===")
    if inputPaths.length == 1 then println(s"  Input: ${displayPath(inputPaths.head)}")
    else
      println(s"  Inputs (${inputPaths.length}):")
      inputPaths.zipWithIndex.foreach { case (p, idx) =>
        println(f"    ${idx + 1}%2d. ${displayPath(p)} (weight=${inputWeights(idx)}%.3f)")
      }
    println(s"  Preset: ${preset.name.capitalize} (${preset.epochs} epochs, patience=${preset.patience})")
    println(s"  Context size: $contextSize")
    println(s"  Max vocab: $maxVocab")
    println(s"  Backend: $backend ($precision)")
    val replayFile = replayBufferPath.getOrElse(ReplayPath)
    val replayLoadRequested = replayBufferPath.isDefined && Files.isRegularFile(replayFile)
    val replayPersistenceEnabled = replayBufferSize > 0
    val replayRequested = replayRatio > 0.0 && (inputPaths.length > 1 || replayLoadRequested || replayPersistenceEnabled)
    println(f"  Learning rate: $learningRate%.4f (decay=$lrDecay%.4f)")
    if replayRequested || replayPersistenceEnabled then
      val mode =
        if replayPersistenceEnabled && replayLoadRequested then "load+persist"
        else if replayPersistenceEnabled then "persist"
        else if replayLoadRequested then "load-only"
        else "phased-only"
      println(f"  Replay: mode=$mode ratio=$replayRatio%.2f bufferSize=$replayBufferSize path=${displayPath(replayFile)}")
    else println("  Replay: disabled")
    if ewcLambda > 0 then println(f"  EWC: enabled (lambda=$ewcLambda%.4f, samples=$ewcSamples)")
    if batchSize > 0 then println(s"  Batch size: $batchSize")
    if profileGpu then println("  GPU profile: enabled")
    if actuallyFresh then println(s"  🆕 Starting fresh model (new architecture)")
    else if existingCfg.isDefined then println(s"  ✓ Continuing from existing model")
    else println(s"  🆕 Starting fresh model")
    println()

    val confirm = if autoConfirm then
      println("Auto-confirming (--yes flag)")
      true
    else
      promptYesNoOrBack("Start training now?", true) match
        case Some(v) => v
        case None =>
          println("Returning to main menu.")
          throw BackToLauncher

    if !confirm then
      println("Training canceled.")
      return

    println()

    val loadedInputs = inputPaths.map(path => readTextRobust(path, autoConfirm))
    val tokenizedInputs = loadedInputs.map { case (_, text) => TextPipeline.tokenize(text) }
    val vocab = if actuallyFresh || !Files.isRegularFile(activeVocabPath) then
      TextPipeline.buildVocab(tokenizedInputs.flatten.toVector, maxVocab)
    else
      VocabIO.load(activeVocabPath)

    val modelCfg = ModelConfig(
      contextSize = contextSize,
      embedDim = preset.embedDim,
      hiddenDim = preset.hiddenDim,
      vocabSize = vocab.size,
      activation = "tanh"
    )

    val params0 = if hasExistingModel && !actuallyFresh && Files.isRegularFile(activeModelPath) then
      val (existingParams, cfg) = CheckpointIO.load(activeModelPath)
      println(s"Loaded existing model (vocab: ${cfg.vocabSize} words, context: ${cfg.contextSize}, embed: ${cfg.embedDim}, hidden: ${cfg.hiddenDim})")
      existingParams
    else
      println("Initializing new model...")
      LanguageModel.initParams(modelCfg, 42)

    val phases = loadedInputs.zipWithIndex.map { case ((effectivePath, _), idx) =>
      val ids = TextPipeline.tokensToIds(tokenizedInputs(idx), vocab)
      val examples = TextPipeline.buildExamples(ids, contextSize)
      val (trainSet, valSet) = TextPipeline.splitDeterministic(examples, 0.9, 42 + idx)
      require(trainSet.nonEmpty, s"Training set is empty for $effectivePath. Provide more text or reduce contextSize.")
      require(valSet.nonEmpty, s"Validation set is empty for $effectivePath. Provide more text or reduce contextSize.")
      Trainer.TrainingPhase(effectivePath.getFileName.toString, trainSet, valSet, inputWeights(idx))
    }.toVector

    val vocabHash = vocab.idToToken.mkString("|").hashCode.toHexString
    val replayState =
      if replayPersistenceEnabled || replayLoadRequested then
        if Files.isRegularFile(replayFile) then
          Some(ReplayBuffer.load(replayFile, ReplayBuffer.Expected(contextSize, vocab.size, vocabHash)))
        else if replayPersistenceEnabled then
          Some(ReplayBuffer.initialize(contextSize, vocab.size, vocabHash))
        else
          throw new IllegalArgumentException(s"Replay buffer file not found for --replayBufferPath: $replayFile")
      else None

    val trainCfg = TrainConfig(
      epochs = preset.epochs,
      learningRate = learningRate,
      lrDecay = lrDecay,
      l2 = 0.0,
      clipNorm = None,
      shuffleEachEpoch = true,
      seed = 42,
      patience = preset.patience,
      activation = "tanh",
      backend = backend,
      precision = precision,
      batchSize = batchSize,
      prefetch = prefetch,
      profileGpu = profileGpu,
      inputWeights = inputWeights,
      replayRatio = replayRatio,
      replayBufferSize = replayBufferSize,
      replayBufferPath = replayBufferPath.map(_.toString),
      domainLabels = phases.map(_.label),
      mixedValWeights = inputWeights,
      ewcLambda = ewcLambda,
      ewcSamples = ewcSamples
    )
    val interruptMetaValues = Map(
      "preset" -> preset.name,
      "backend" -> backend,
      "precision" -> precision,
      "lr" -> learningRate.toString,
      "lrDecay" -> lrDecay.toString,
      "batchSize" -> batchSize.toString,
      "prefetch" -> prefetch.toString,
      "replayRatio" -> replayRatio.toString,
      "replayBufferSize" -> replayBufferSize.toString,
      "replayBufferPath" -> replayBufferPath.map(_.toString).getOrElse(""),
      "contextSize" -> contextSize.toString,
      "maxVocab" -> maxVocab.toString,
      "input" -> (if inputPaths.length == 1 then inputPaths.head.toString else ""),
      "inputs" -> (if inputPaths.length > 1 then inputPaths.map(_.toString).mkString(",") else ""),
      "inputWeights" -> inputWeights.map(w => f"$w%.6f").mkString(",")
    ).filter(_._2.nonEmpty)

    val usePhasedTraining =
      phases.length > 1 || replayRatio > 0.0 || replayState.nonEmpty || replayPersistenceEnabled

    val safePointParams = AtomicReference(params0)
    val trainingInProgress = AtomicBoolean(true)
    val shutdownHook = new Thread(() =>
      if trainingInProgress.get() then
        try
          CheckpointIO.save(safePointParams.get(), modelCfg, InterruptModelPath)
          VocabIO.save(vocab, InterruptVocabPath)
          saveInterruptMeta(InterruptMetaPath, interruptMetaValues)
          Console.err.println(s"[interrupt-save] Emergency checkpoint: ${displayPath(InterruptModelPath)}")
          Console.err.println(s"[interrupt-save] Emergency vocab: ${displayPath(InterruptVocabPath)}")
          Console.err.println(s"[interrupt-save] Emergency settings: ${displayPath(InterruptMetaPath)}")
        catch
          case _: Exception => ()
    )
    Runtime.getRuntime.addShutdownHook(shutdownHook)

    val result =
      try
        if usePhasedTraining then
          Trainer.trainPhased(
            params0,
            phases,
            trainCfg,
            replayState,
            onSafePoint = p => safePointParams.set(p)
          )
        else
          Trainer.train(
            params0,
            phases.head.trainSet,
            phases.head.valSet,
            trainCfg,
            onSafePoint = p => safePointParams.set(p)
          )
      finally
        trainingInProgress.set(false)
        try Runtime.getRuntime.removeShutdownHook(shutdownHook)
        catch case _: IllegalStateException => ()
    val runEndMs = System.currentTimeMillis()
    val memoryEnd = MemoryProbe.snapshot(runEndMs)
    val gcEnd = GcProbe.snapshot()

    val shouldSave = result.saveDecision != SaveDecision.Discard
    if shouldSave then
      CheckpointIO.save(result.params, modelCfg, ModelPath)
      VocabIO.save(vocab, VocabPath)
      if replayPersistenceEnabled then
        result.replayBuffer.foreach(_.save(replayFile))
      val cleanedInterruptArtifacts = cleanupInterruptArtifacts()
      if cleanedInterruptArtifacts.nonEmpty then
        println("  Interrupt recovery artifacts cleaned up.")

    println("\n=== Results ===")
    val bestMetric = result.history.minBy(_.valLoss)
    val finalMetric =
      result.saveDecision match
        case SaveDecision.SaveBest    => bestMetric
        case SaveDecision.SaveCurrent => result.history.last
        case SaveDecision.Discard     => result.history.last
    val restoredNote =
      result.saveDecision match
        case SaveDecision.SaveBest if trainCfg.patience > 0 => " (restored best)"
        case SaveDecision.SaveCurrent if result.interrupted  => " (interrupted: current)"
        case SaveDecision.Discard if result.interrupted      => " (interrupted: discarded)"
        case _                                               => ""
    result.history.foreach { m =>
      val indicator = if m.epoch == bestMetric.epoch then " ← best" else ""
      val phaseTag = m.phaseLabel.map(p => s" phase=$p").getOrElse("")
      println(
        f"  Epoch ${m.epoch}%2d:$phaseTag train=${m.trainLoss}%.4f val=${m.valLoss}%.4f ppl=${m.valPerplexity}%.1f status=${m.status.toString.toLowerCase}(${m.statusReason}) gap=${m.generalizationGap}%.3f delta=${m.bestDeltaPct}%.2f%%$indicator"
      )
      if m.perDomainValMetrics.nonEmpty then
        println("    per-domain: " + m.perDomainValMetrics.map(dm => f"${dm.domain}:${dm.valLoss}%.4f/${dm.valPerplexity}%.2f").mkString(", "))
      if m.retentionMetrics.nonEmpty then
        println("    retention:  " + m.retentionMetrics.map(r => f"${r.domain}:${r.retentionPct}%.1f%%(${r.status})").mkString(", "))
    }

    if shouldSave then
      println(s"\n✓ Model saved to ${displayPath(ModelPath)}")
      println(s"  Vocab saved to ${displayPath(VocabPath)}")
      if replayPersistenceEnabled then println(s"  Replay saved to ${displayPath(replayFile)}")
    else
      println("\nTraining output discarded; checkpoint, vocab, and replay were not saved.")
    println(f"  Final$restoredNote: val_loss=${finalMetric.valLoss}%.4f val_ppl=${finalMetric.valPerplexity}%.1f")
    if trainCfg.patience > 0 && result.history.nonEmpty then
      val last = result.history.last
      println(f"  Last epoch (pre-restore): val_loss=${last.valLoss}%.4f val_ppl=${last.valPerplexity}%.1f")

    val epochHeap = result.history.map(_.heapUsedBytes).filter(_ >= 0)
    val epochRss = result.history.map(_.rssBytes).filter(_ >= 0)
    val memoryPeak = peakMemory(memoryStart, memoryEnd, epochHeap, epochRss)
    val throughput = if result.history.isEmpty then 0.0 else result.history.map(_.examplesPerSec).sum / result.history.length.toDouble
    val finalEpoch = result.history.lastOption
    val finalDomainLosses = finalEpoch.map(_.perDomainValMetrics.map(m => m.domain -> m.valLoss)).getOrElse(Vector.empty)
    val finalRetention = finalEpoch.map(_.retentionMetrics.map(m => m.domain -> m.retentionPct)).getOrElse(Vector.empty)
    val runProfile = RunProfile(
      totalSeconds = (runEndMs - runStartMs).toDouble / 1000.0,
      epochSecondsTotal = result.history.map(_.epochSeconds).sum,
      epochSecondsAvg = if result.history.isEmpty then 0.0 else result.history.map(_.epochSeconds).sum / result.history.length.toDouble,
      batchMillisAvg = if result.history.isEmpty then 0.0 else result.history.map(_.avgBatchMillis).sum / result.history.length.toDouble,
      batchMillisP95 = if result.history.isEmpty then 0.0 else result.history.map(_.p95BatchMillis).sum / result.history.length.toDouble,
      backendProfileSummary = result.backendProfileSummary,
      topStages = RunObservability.topStagesFromProfile(result.backendProfileSummary)
    )
    val fallbackOps = RunObservability.fallbackOpsFromDiagnostics(result.requestedBackend, result.effectiveBackend, result.backendDiagnostics)
    val gpuMetrics = GpuHotPathMetrics(
      requestedBackend = result.requestedBackend,
      effectiveBackend = result.effectiveBackend,
      diagnostics = result.backendDiagnostics,
      enabledOps = result.gpuOpsEnabled,
      profileSummary = result.backendProfileSummary,
      fallbackDetected = result.requestedBackend == "gpu" && result.effectiveBackend != "gpu"
    )
    val runMetrics = RunMetrics(
      runId = RunObservability.buildRunId("train"),
      timestampEpochMs = runEndMs,
      mode = "train",
      runLabel = observability.runLabel,
      configFingerprint = RunObservability.configFingerprint(
        Vector(
          "preset" -> preset.name,
          "contextSize" -> contextSize.toString,
          "maxVocab" -> maxVocab.toString,
          "backend" -> backend,
          "precision" -> precision,
          "lr" -> learningRate.toString,
          "batchSize" -> batchSize.toString,
          "epochs" -> preset.epochs.toString,
          "replayRatio" -> replayRatio.toString
        )
      ),
      dataset = inputPaths.map(displayPath).mkString(","),
      requestedBackend = backend,
      precision = precision,
      throughputExPerSec = throughput,
      trainLoss = finalEpoch.map(_.trainLoss),
      valLoss = finalEpoch.map(_.valLoss),
      valPerplexity = finalEpoch.map(_.valPerplexity),
      epochs = result.history.length,
      domainValLosses = finalDomainLosses,
      retentionByDomainPct = finalRetention,
      profile = runProfile,
      gpu = gpuMetrics,
      platform = platformInfoFromDiagnostics(result.backendDiagnostics),
      fallbackOps = fallbackOps,
      memoryStart = memoryStart,
      memoryEnd = memoryEnd,
      memoryPeak = memoryPeak,
      gcCountDelta = math.max(0L, gcEnd.collectionCount - gcStart.collectionCount),
      gcTimeMsDelta = math.max(0L, gcEnd.collectionTimeMs - gcStart.collectionTimeMs),
      notes = Map(
        "saveDecision" -> result.saveDecision.toString,
        "interrupted" -> result.interrupted.toString,
        "modelSaved" -> shouldSave.toString
      )
    )
    RunObservability.persistAndReport(runMetrics, observability)
    println()
    println("Continue with: sbt \"run train --inputs <more1.txt,more2.txt>\"")
    println("Or predict:    sbt 'run predict --context \"your text\"'")

  private def selectTrainingInputInteractively(): Path =
    val candidates = discoverTextFiles(Path.of(".").toAbsolutePath.normalize, maxDepth = 6)
      .filter(p => !p.toString.contains("/chunks/"))
    val candidatesWithLines = candidates.map { p =>
      val lines = countLines(p)
      val size = Files.size(p) / 1024
      (p, lines, size)
    }.filter { case (_, lines, size) => lines > 0 || size > 0 }
    if candidatesWithLines.isEmpty then
      println("Error: No readable .txt files found. Specify --input <file.txt> or --inputs <a,b,c>")
      sys.exit(1)
    val (recommended, other) = CliHelpers.classifyTrainingFiles(candidatesWithLines.map(_._1))
    val orderedPaths = recommended ++ other
    val byPath = candidatesWithLines.map { case (p, lines, size) => p -> (lines, size) }.toMap
    println("Found text files you can train on (chunk files excluded):")
    println("Tip: press Enter to choose the default [1]. Type 'b' to go back.\n")
    if recommended.nonEmpty then println("  Recommended training files:")
    recommended.zipWithIndex.foreach { case (p, idx) =>
      val (lines, size) = byPath(p)
      val linesStr = if lines > 0 then s"$lines lines" else "unknown lines"
      println(s"    ${idx + 1}. ${displayPath(p)} ($linesStr, ${size} KB)")
    }
    if other.nonEmpty then println("  Other text files (likely derived; usually avoid):")
    other.zipWithIndex.foreach { case (p, idx) =>
      val (lines, size) = byPath(p)
      val linesStr = if lines > 0 then s"$lines lines" else "unknown lines"
      println(s"    ${recommended.length + idx + 1}. ${displayPath(p)} ($linesStr, ${size} KB)")
    }
    println()
    var selected: Option[Path] = None
    while selected.isEmpty do
      val raw = readTrimmedRequired(s"Choose file number [1]: ", "choose training file")
      parseMenuPick(raw, optionCount = orderedPaths.length, defaultIndex = 0, allowBack = true) match
        case Some(MenuPick.Choice(idx)) => selected = Some(orderedPaths(idx))
        case Some(MenuPick.Back) =>
          println("Returning to main menu.")
          throw BackToLauncher
        case None =>
          println(s"Please choose 1-${orderedPaths.length}, or type 'b' to go back.")
    println(s"Selected training file: ${displayPath(selected.get)}")
    selected.get

  private def parseInputWeights(raw: Option[String], count: Int): Vector[Double] =
    require(count >= 1, "at least one input is required")
    raw match
      case None => Vector.fill(count)(1.0 / count.toDouble)
      case Some(csv) =>
        val values = csv.split(",").toVector.map(_.trim).filter(_.nonEmpty).map(_.toDouble)
        require(values.length == count, s"--inputWeights count (${values.length}) must match number of inputs ($count)")
        val clamped = values.map(v => if v.isFinite && v > 0 then v else 0.0)
        val sum = clamped.sum
        require(sum > 0, "--inputWeights must include at least one positive value")
        clamped.map(_ / sum)

  private[app] def utf8CopyPath(path: Path): Path =
    val fileName = path.getFileName.toString
    val dot = fileName.lastIndexOf('.')
    val rewritten =
      if dot > 0 then
        val stem = fileName.substring(0, dot)
        val ext = fileName.substring(dot)
        s"${stem}.utf8$ext"
      else s"${fileName}.utf8.txt"
    val parent = Option(path.getParent).getOrElse(Path.of("."))
    parent.resolve(rewritten)

  private def readTextWithSystemDefault(path: Path): String =
    import java.io._
    val reader = new BufferedReader(new InputStreamReader(Files.newInputStream(path)))
    try
      val text = new StringBuilder
      var line = reader.readLine()
      while line != null do
        text.append(line).append("\n")
        line = reader.readLine()
      text.toString
    finally reader.close()

  private def maybeCreateUtf8Copy(path: Path): Option[Path] =
    println(s"UTF-8 read failed for ${displayPath(path)}.")
    val suggested = utf8CopyPath(path)
    Option(readLineWithPrompt("Create a UTF-8 normalized copy and use it for this run? [Y/n]: ")) match
      case None =>
        println("No interactive input available; using system default encoding for this run.")
        None
      case Some(raw) =>
        if !CliHelpers.parseYesNo(raw, default = true) then None
        else
          val target =
            if Files.isRegularFile(suggested) then
              println(s"Using existing UTF-8 copy: ${displayPath(suggested)}")
              suggested
            else
              println(s"Writing UTF-8 copy: ${displayPath(suggested)}")
              val decoded = readTextWithSystemDefault(path)
              Files.writeString(suggested, decoded, StandardCharsets.UTF_8)
              suggested
          Some(target)

  private def readTextRobust(path: Path, autoConfirm: Boolean): (Path, String) =
    try (path, Files.readString(path, StandardCharsets.UTF_8))
    catch
      case _: Exception =>
        val convertedPath =
          if autoConfirm then
            println(s"UTF-8 read failed for ${displayPath(path)}, trying with system default encoding...")
            None
          else maybeCreateUtf8Copy(path)
        convertedPath match
          case Some(p) =>
            println(s"Using UTF-8 input for training: ${displayPath(p)}")
            (p, Files.readString(p, StandardCharsets.UTF_8))
          case None =>
            (path, readTextWithSystemDefault(path))

  private def runChunker(flags: Map[String, String]): Unit =
    println("\n=== File Chunker ===\n")
    if flags.isEmpty then
      println("Tip: press Enter to accept defaults shown in [brackets].\n")

    // Get input file
    val inputPath = flags.get("input") match
      case Some(path) => Path.of(path)
      case None =>
        // Discover text files
        val candidates = discoverTextFiles(Path.of(".").toAbsolutePath.normalize, maxDepth = 6)
        if candidates.isEmpty then
          println("No .txt files found. Specify --input <file.txt>")
          sys.exit(1)
        
        println("Found text files:")
        candidates.zipWithIndex.foreach { case (p, idx) =>
          val lines = countLines(p)
          val size = Files.size(p) / 1024
          println(s"  ${idx + 1}. ${displayPath(p)} (${lines} lines, ${size} KB)")
        }
        println()
        
        var selected: Option[Path] = None
        while selected.isEmpty do
          val raw = readTrimmedRequired("Choose file number [1]: ", "choose file number")
          CliHelpers.parseMenuChoice(raw, optionCount = candidates.length, defaultIndex = 0) match
            case Some(idx) => selected = Some(candidates(idx))
            case None      => println(s"Please choose a number from 1 to ${candidates.length}.")
        selected.get

    require(Files.isRegularFile(inputPath), s"Input file not found: $inputPath")
    
    val totalLines = countLines(inputPath)
    val sizeKB = Files.size(inputPath) / 1024
    println(s"\nSelected: ${displayPath(inputPath)}")
    println(s"Size: ${totalLines} lines, ${sizeKB} KB\n")

    // Get chunk size
    val chunkSize = flags.get("lines").map(_.toInt).getOrElse {
      // Recommend based on file size
      val recommended = CliHelpers.recommendChunkSize(totalLines)
      println(s"Recommended chunk size: $recommended lines")
      println(s"  - This will create ${(totalLines + recommended - 1) / recommended} chunks\n")
      
      var size: Option[Int] = None
      while size.isEmpty do
        val raw = readTrimmedRequired(s"Lines per chunk [$recommended]: ", "lines per chunk")
        size = if raw.isEmpty then Some(recommended) else raw.toIntOption
        if size.isEmpty then println("Please enter a number.")
      size.get
    }

    // Get output directory
    val outputDir = flags.get("output") match
      case Some(path) => Path.of(path)
      case None =>
        val parent = Option(inputPath.getParent).getOrElse(Path.of("."))
        parent.resolve("chunks")

    Files.createDirectories(outputDir)

    // Get base name for chunks
    val baseName = flags.getOrElse("name", inputPath.getFileName.toString.replace(".txt", ""))

    println(s"\nChunking plan:")
    println(s"  Input: ${displayPath(inputPath)} (${totalLines} lines)")
    println(s"  Output: ${displayPath(outputDir)}/")
    println(s"  Chunk size: $chunkSize lines")
    val autoConfirm = flags.get("yes").exists(CliHelpers.isTruthy)
    
    println(s"  Files: ${baseName}-part1.txt, ${baseName}-part2.txt, ...")
    println()

    val confirm = if autoConfirm then
      println("Auto-confirming (--yes flag)")
      true
    else
      promptYesNo("Create chunks?", true)
      
    if !confirm then
      println("Canceled.")
      sys.exit(0)

    // Read and split
    val lines = Files.readAllLines(inputPath, StandardCharsets.UTF_8)
    val totalChunks = (lines.size() + chunkSize - 1) / chunkSize

    println(s"\nCreating $totalChunks chunks...")

    var chunkNum = 1
    var lineIdx = 0
    while lineIdx < lines.size() do
      val endIdx = math.min(lineIdx + chunkSize, lines.size())
      val chunkLines = lines.subList(lineIdx, endIdx)
      
      val chunkPath = outputDir.resolve(s"${baseName}-part$chunkNum.txt")
      Files.write(chunkPath, chunkLines, StandardCharsets.UTF_8)
      
      val chunkLinesCount = endIdx - lineIdx
      println(s"  ✓ ${chunkPath.getFileName} ($chunkLinesCount lines)")
      
      chunkNum += 1
      lineIdx = endIdx

    println(s"\n✓ Created $totalChunks chunks in ${displayPath(outputDir)}/")
    println()
    println("Next steps:")
    println(s"  sbt \"run train --input $outputDir/${baseName}-part1.txt --preset quick\"")
    println(s"  sbt \"run train --input $outputDir/${baseName}-part2.txt --preset quick\"")
    println("  ... continue with remaining parts")

  private def runPredict(flags: Map[String, String]): Unit =
    println("\n=== Prediction ===\n")

    // Check model exists
    if !Files.isRegularFile(ModelPath) then
      println("No model found. Train first:")
      println("  sbt \"run train --input data/corpus/text.txt\"")
      sys.exit(1)

    val (params, cfg) = CheckpointIO.load(ModelPath)
    val vocab = VocabIO.load(VocabPath)
    val backend = BackendSelector.fromConfig(
      flags.getOrElse("backend", "gpu"),
      flags.getOrElse("precision", defaultPrecisionForBackend(flags.getOrElse("backend", "gpu"))),
      warn = msg => println(s"[backend] $msg")
    )

    println(s"Model: ${vocab.size} words, context=${cfg.contextSize}")
    println()

    val topK = CliHelpers.boundedTopK(flags.get("topK").flatMap(_.toIntOption).getOrElse(5))
    def predictOnce(contextText: String): Unit =
      val tokens = TextPipeline.tokenize(contextText)
      val contextIds = adaptContext(tokens.map(vocab.toId), cfg.contextSize, vocab.unkId)
      val cache = LanguageModel.forward(params, contextIds, cfg.activation, backend)
      val top = LinearAlgebra.argTopK(cache.probs, topK)

      println("\nTop predictions:")
      top.zipWithIndex.foreach { case ((id, prob), idx) =>
        val word = vocab.toToken(id)
        val bar = "█" * ((prob * 20).toInt max 1)
        println(f"  ${idx + 1}. $word%-12s $bar%-20s ${prob * 100}%.1f%%")
      }

      top.headOption.foreach { case (id, prob) =>
        if id == vocab.unkId && prob >= 0.2 then
          println("  Tip: high <UNK> confidence. Try longer context, in-domain words, or retrain with larger --maxVocab.")
      }

    flags.get("context") match
      case Some(text) =>
        predictOnce(text)
      case None =>
        println("Enter text to predict next word (or 'quit'):")
        var keepRunning = true
        while keepRunning do
          val raw = StdIn.readLine("> ")
          if raw == null then
            keepRunning = false
          else
            val normalized = raw.trim
            if normalized.equalsIgnoreCase("quit") || normalized.equalsIgnoreCase("exit") then
              keepRunning = false
            else if normalized.nonEmpty then
              predictOnce(normalized)

  private def runGpuInfo(flags: Map[String, String]): Unit =
    val backendName = BackendSelector.normalizeBackend(flags.getOrElse("backend", "gpu"))
    val precision = BackendSelector.normalizePrecision(flags.getOrElse("precision", defaultPrecisionForBackend(backendName)))
    println(s"\nGPU probe ($precision): ${BackendSelector.gpuInfo(precision)}")

  private def runBenchmark(flags: Map[String, String]): Unit =
    println("\n=== Benchmark ===")
    val effectiveFlags =
      if flags.nonEmpty then flags
      else
        println("No benchmark flags provided. Starting guided benchmark setup.")
        println("Tip: press Enter at any prompt to accept the default value.")
        val defaultInput = "data/corpus/example-corpus.txt"
        val inputRaw = Option(readLineWithPrompt(s"Input file [$defaultInput]: ")).map(_.trim).getOrElse("")
        val input = if inputRaw.isEmpty then defaultInput else inputRaw

        def promptInt(label: String, default: Int, min: Int): Int =
          val raw = Option(readLineWithPrompt(s"$label [$default]: ")).map(_.trim).getOrElse("")
          val parsed =
            if raw.isEmpty then default
            else
              raw.toIntOption match
                case Some(v) => v
                case None =>
                  println(s"Invalid input '$raw'. Using default $default.")
                  default
          val chosen = math.max(min, parsed)
          if chosen != parsed then println(s"Adjusted $label to minimum $min.")
          println(s"Selected $label: $chosen")
          chosen

        val sample = promptInt("Sample examples", 2000, 1)
        val contextSize = promptInt("Context size", 3, 1)
        val maxVocab = promptInt("Max vocabulary", 3000, 100)
        val batchSize = promptInt("Batch size override (0=auto)", 0, 0)
        val activationRaw = Option(readLineWithPrompt("Activation [tanh]: ")).map(_.trim.toLowerCase).getOrElse("")
        val activation = if activationRaw == "relu" then "relu" else "tanh"
        println(s"Selected activation: $activation")

        println("\nChoose benchmark combinations:")
        println("  1. CPU+GPU x fp64+fp32 (full matrix)")
        println("  2. CPU fp64")
        println("  3. CPU fp32")
        println("  4. GPU fp64")
        println("  5. GPU fp32")
        println("  6. CPU x fp64+fp32")
        println("  7. GPU x fp64+fp32")
        println("  8. CPU+GPU x fp64")
        println("  9. CPU+GPU x fp32")
        println("  b. Back to main menu")
        val choiceRaw = Option(readLineWithPrompt("Select [1]: ")).map(_.trim).getOrElse("")
        val choiceIdx = parseMenuPick(choiceRaw, optionCount = 9, defaultIndex = 0, allowBack = true) match
          case Some(MenuPick.Choice(idx)) => idx
          case Some(MenuPick.Back) =>
            println("Returning to main menu.")
            throw BackToLauncher
          case None =>
            println("Invalid selection. Using default 1.")
            0
        val (backendOpt, precisionOpt) = benchmarkSelection(choiceIdx)
        val combosPreview = benchmarkMatrix(backendOpt, precisionOpt).map { case (b, p) => s"$b/$p" }.mkString(", ")
        println(s"Selected combinations: $combosPreview")

        val metricsEnabledPrompt = promptYesNo("Enable metrics logging and report output (--metrics)?", true)
        val runLabelRaw = Option(readLineWithPrompt("Run label (optional): ")).map(_.trim).getOrElse("")
        val compareDefault = "latest"
        val compareToRaw = Option(readLineWithPrompt(s"Compare to [$compareDefault]: ")).map(_.trim).getOrElse("")
        val compareTo = if compareToRaw.isEmpty then compareDefault else compareToRaw
        val regressionWarnPctRaw = Option(readLineWithPrompt("Regression warn threshold % [5.0]: ")).map(_.trim).getOrElse("")
        val regressionWarnPct = regressionWarnPctRaw.toDoubleOption.getOrElse(5.0)

        println("\nBenchmark plan:")
        println(s"  input=$input")
        println(s"  sample=$sample contextSize=$contextSize maxVocab=$maxVocab batchSize=$batchSize activation=$activation")
        println(s"  combinations=$combosPreview")
        println(s"  metrics=${if metricsEnabledPrompt then "on" else "off"}")
        if metricsEnabledPrompt then
          println(s"  compareTo=$compareTo regressionWarnPct=${formatDecimal(regressionWarnPct, 2)}")
        if runLabelRaw.nonEmpty then println(s"  runLabel=$runLabelRaw")
        promptYesNoOrBack("Start benchmark now?", true) match
          case None =>
            println("Returning to main menu.")
            throw BackToLauncher
          case Some(false) =>
            println("Benchmark canceled.")
            return
          case Some(true) => ()

        val mutable = scala.collection.mutable.LinkedHashMap(
          "input" -> input,
          "sample" -> sample.toString,
          "contextSize" -> contextSize.toString,
          "maxVocab" -> maxVocab.toString,
          "batchSize" -> batchSize.toString,
          "activation" -> activation
        )
        backendOpt.foreach(v => mutable += ("backend" -> v))
        precisionOpt.foreach(v => mutable += ("precision" -> v))
        if metricsEnabledPrompt then
          mutable += ("metrics" -> "true")
          mutable += ("compareTo" -> compareTo)
          mutable += ("regressionWarnPct" -> regressionWarnPct.toString)
        if runLabelRaw.nonEmpty then mutable += ("runLabel" -> runLabelRaw)
        mutable.toMap

    val metricsEnabled = effectiveFlags.get("metrics").exists(CliHelpers.isTruthy)
    val baseObservability = parseObservabilitySettings(effectiveFlags)
    val observability =
      if metricsEnabled then
        baseObservability.copy(
          recordMetrics = true,
          compareTo = baseObservability.compareTo.orElse(Some("latest"))
        )
      else baseObservability.copy(recordMetrics = false, compareTo = None)
    val runStartMs = System.currentTimeMillis()
    val memoryStart = MemoryProbe.snapshot(runStartMs)
    val gcStart = GcProbe.snapshot()
    val inputPath = Path.of(effectiveFlags.getOrElse("input", "data/corpus/example-corpus.txt"))
    if !Files.isRegularFile(inputPath) then
      println(s"Input file not found: $inputPath")
      sys.exit(1)

    val contextSize = effectiveFlags.get("contextSize").flatMap(_.toIntOption).getOrElse(3)
    val maxVocab = effectiveFlags.get("maxVocab").flatMap(_.toIntOption).getOrElse(3000)
    val sample = effectiveFlags.get("sample").flatMap(_.toIntOption).getOrElse(2000)
    val benchmarkBackendFlag = effectiveFlags.get("backend")
    val precisionFlag = effectiveFlags.get("precision")
    val activation = effectiveFlags.getOrElse("activation", "tanh")
    val batchSize = effectiveFlags.get("batchSize").flatMap(_.toIntOption).getOrElse(0)

    val raw = Files.readString(inputPath, StandardCharsets.UTF_8)
    val tokens = TextPipeline.tokenize(raw)
    val vocab = TextPipeline.buildVocab(tokens, maxVocab)
    val ids = TextPipeline.tokensToIds(tokens, vocab)
    val examples = TextPipeline.buildExamples(ids, contextSize).take(sample)
    if examples.isEmpty then
      println("No examples to benchmark. Adjust context/sample.")
      return

    val cfg = ModelConfig(contextSize = contextSize, embedDim = 24, hiddenDim = 64, vocabSize = vocab.size, activation = activation)
    val params = LanguageModel.initParams(cfg, seed = 42)

    final case class BenchResult(
        exPerSec: Double,
        estSec: Double,
        diagnostics: String,
        effectiveBackend: String,
        enabledOps: Vector[String],
        profileSummary: String
    )

    def bench(label: String, backendName: String, precision: String): BenchResult =
      val backend = BackendSelector.fromConfig(backendName, precision, warn = _ => ())
      backend.resetProfile()
      val trainCfg = TrainConfig(epochs = 1, activation = activation, backend = backendName, precision = precision, batchSize = batchSize, profileGpu = true)
      val sec = Trainer.estimateEpochSeconds(params, examples, trainCfg, backend, sampleSize = math.min(500, examples.length))
      val exPerSec = if sec <= 0 then 0.0 else examples.length / sec
      println(f"$label%-4s backend=${backend.diagnostics} | est=${sec}%.2fs for ${examples.length} ex | ${exPerSec}%.1f ex/s")
      if backend.isGpu then println(s"     profile=${backend.profileSummary}")
      BenchResult(
        exPerSec = exPerSec,
        estSec = sec,
        diagnostics = backend.diagnostics,
        effectiveBackend = backend.name,
        enabledOps = backend.gpuOpsEnabled.toVector.sorted,
        profileSummary = backend.profileSummary
      )

    val results = scala.collection.mutable.Map.empty[(String, String), BenchResult]
    val combos = benchmarkMatrix(benchmarkBackendFlag, precisionFlag)
    combos.foreach { case (backendName, precision) =>
      val label = s"${backendName.toUpperCase}[$precision]"
      val res = bench(label, backendName, precision)
      results((backendName, precision)) = res
    }

    Vector("fp64", "fp32").foreach { precision =>
      val cpu = results.get(("cpu", precision)).map(_.exPerSec)
      val gpu = results.get(("gpu", precision)).map(_.exPerSec)
      (cpu, gpu) match
        case (Some(c), Some(g)) if c > 0 && g > 0 =>
          println(f"Speedup (GPU/CPU, $precision): ${g / c}%.2fx")
        case _ => ()
    }

    val opPrecisions = precisionFlag.map(BackendSelector.normalizePrecision).map(Vector(_)).getOrElse(Vector("fp64", "fp32"))
    opPrecisions.foreach { precision =>
      println(s"\nPer-op timing (synthetic, $precision)")
      def benchOps(label: String, backendName: String): Unit =
        val backend = BackendSelector.fromConfig(backendName, precision, warn = _ => ())
        val rows = 256
        val cols = 256
        val m = linalg.Matrix.fromFunction(rows, cols)((r, c) => ((r + c) % 17) * 0.01)
        val x = Vector.tabulate(cols)(i => (i % 13) * 0.01)
        val b = Vector.tabulate(rows)(i => (i % 7) * 0.01)
        val a = Vector.tabulate(rows)(i => (i % 11) * 0.01)
        val o = Vector.tabulate(cols)(i => (i % 5) * 0.01)
        val iters = 100

        def timeMs(fn: => Unit): Double =
          val t0 = System.nanoTime()
          var i = 0
          while i < iters do
            fn
            i += 1
          (System.nanoTime() - t0).toDouble / 1e6

        val matVecMs = timeMs { backend.matVecMul(m, x); () }
        val outerMs = timeMs { backend.outer(a, o); () }
        val fusedMs = timeMs { backend.linearActivation(m, x, b, activation); () }
        println(f"$label%-4s matVec=${matVecMs / iters}%.3fms outer=${outerMs / iters}%.3fms fused=${fusedMs / iters}%.3fms")

      val opBackends = benchmarkBackendFlag.map(BackendSelector.normalizeBackend).map(Vector(_)).getOrElse(Vector("cpu", "gpu"))
      opBackends.foreach { b =>
        benchOps(b.toUpperCase, b)
      }
    }

    val runEndMs = System.currentTimeMillis()
    val memoryEnd = MemoryProbe.snapshot(runEndMs)
    val gcEnd = GcProbe.snapshot()
    val throughputValues = results.values.map(_.exPerSec).filter(_ > 0).toVector
    val throughput = if throughputValues.isEmpty then 0.0 else throughputValues.sum / throughputValues.length.toDouble
    val preferredKey = preferredBenchmarkKey(results.iterator.map { case (k, r) => (k, r.exPerSec) }.toVector)
    val preferredResult = preferredKey.flatMap(results.get)
    val memoryPeak = peakMemory(memoryStart, memoryEnd, Vector.empty, Vector.empty)
    val matrixRows = results.toVector.sortBy(_._1.toString).map { case ((b, p), r) =>
      val fallbackOps = RunObservability.fallbackOpsFromDiagnostics(b, r.effectiveBackend, r.diagnostics)
      BenchmarkCell(
        backend = b,
        precision = p,
        effectiveBackend = r.effectiveBackend,
        exPerSec = r.exPerSec,
        estSec = r.estSec,
        fallbackOps = fallbackOps,
        diagnostics = r.diagnostics
      )
    }
    val chosenFallbackOps = preferredResult
      .map(r => RunObservability.fallbackOpsFromDiagnostics(preferredKey.map(_._1).getOrElse(benchmarkBackendFlag.getOrElse("all")), r.effectiveBackend, r.diagnostics))
      .getOrElse(Vector.empty)

    val runMetrics = RunMetrics(
      runId = RunObservability.buildRunId("benchmark"),
      timestampEpochMs = runEndMs,
      mode = "benchmark",
      runLabel = observability.runLabel,
      configFingerprint = RunObservability.configFingerprint(
        Vector(
          "input" -> inputPath.toString,
          "contextSize" -> contextSize.toString,
          "maxVocab" -> maxVocab.toString,
          "sample" -> sample.toString,
          "backendFilter" -> benchmarkBackendFlag.getOrElse("all"),
          "precisionFilter" -> precisionFlag.getOrElse("all"),
          "batchSize" -> batchSize.toString
        )
      ),
      dataset = displayPath(inputPath),
      requestedBackend = benchmarkBackendFlag.getOrElse("all"),
      precision = precisionFlag.getOrElse("all"),
      throughputExPerSec = throughput,
      trainLoss = None,
      valLoss = None,
      valPerplexity = None,
      epochs = 0,
      domainValLosses = Vector.empty,
      retentionByDomainPct = Vector.empty,
      profile = RunProfile(
        totalSeconds = (runEndMs - runStartMs).toDouble / 1000.0,
        epochSecondsTotal = results.values.map(_.estSec).sum,
        epochSecondsAvg = if results.isEmpty then 0.0 else results.values.map(_.estSec).sum / results.size.toDouble,
        batchMillisAvg = 0.0,
        batchMillisP95 = 0.0,
        backendProfileSummary = preferredResult.map(_.profileSummary).getOrElse(""),
        topStages = RunObservability.topStagesFromProfile(preferredResult.map(_.profileSummary).getOrElse(""))
      ),
      gpu = GpuHotPathMetrics(
        requestedBackend = benchmarkBackendFlag.getOrElse("all"),
        effectiveBackend = preferredResult.map(_.effectiveBackend).getOrElse("cpu"),
        diagnostics = preferredResult.map(_.diagnostics).getOrElse("n/a"),
        enabledOps = preferredResult.map(_.enabledOps).getOrElse(Vector.empty),
        profileSummary = preferredResult.map(_.profileSummary).getOrElse(""),
        fallbackDetected = benchmarkBackendFlag.contains("gpu") && preferredResult.exists(_.effectiveBackend != "gpu")
      ),
      platform = platformInfoFromDiagnostics(preferredResult.map(_.diagnostics).getOrElse("")),
      fallbackOps = chosenFallbackOps,
      memoryStart = memoryStart,
      memoryEnd = memoryEnd,
      memoryPeak = memoryPeak,
      gcCountDelta = math.max(0L, gcEnd.collectionCount - gcStart.collectionCount),
      gcTimeMsDelta = math.max(0L, gcEnd.collectionTimeMs - gcStart.collectionTimeMs),
      benchmarkMatrix = matrixRows,
      notes = matrixRows.map { c =>
        s"benchmark.${c.backend}.${c.precision}.exPerSec" -> f"${c.exPerSec}%.4f"
      }.toMap
    )
    val persisted = RunObservability.persistAndReport(runMetrics, observability)
    if metricsEnabled then
      println()
      println(renderBenchmarkMetricsReport(runMetrics, persisted, displayPath))

  private def adaptContext(ids: Vector[Int], contextSize: Int, padId: Int): Vector[Int] =
    if ids.length >= contextSize then ids.takeRight(contextSize)
    else Vector.fill(contextSize - ids.length)(padId) ++ ids

  private def countLines(path: Path): Int =
    try
      val lines = Files.readAllLines(path, StandardCharsets.UTF_8)
      lines.size()
    catch
      case _: Exception =>
        try
          // Fallback to a byte-level newline count so we still produce a useful
          // estimate even when charset decoding fails.
          val in = Files.newInputStream(path)
          try
            val buffer = new Array[Byte](8192)
            var lineCount = 0
            var sawAnyByte = false
            var lastByte: Int = -1
            var read = in.read(buffer)
            while read != -1 do
              var i = 0
              while i < read do
                val b = buffer(i) & 0xff
                if b == '\n' then lineCount += 1
                sawAnyByte = true
                lastByte = b
                i += 1
              read = in.read(buffer)
            if !sawAnyByte then 0
            else if lastByte == '\n' then lineCount
            else lineCount + 1
          finally in.close()
        catch
          case _: Exception => 0

  private def discoverTextFiles(root: Path, maxDepth: Int = 6, maxResults: Int = 20): Vector[Path] =
    if !Files.exists(root) then Vector.empty
    else
      Using.resource(Files.walk(root, maxDepth)) { stream =>
        stream.iterator().asScala
          .filter(Files.isRegularFile(_))
          .filter(p => p.getFileName.toString.toLowerCase.endsWith(".txt"))
          .toVector
          .sortBy(_.toString)
          .take(maxResults)
      }

  private def promptYesNo(label: String, default: Boolean): Boolean =
    val defaultStr = if default then "Y/n" else "y/N"
    val raw = readLineWithPrompt(s"$label [$defaultStr]: ")
    val resolved = CliHelpers.resolveConfirmation(raw, default, autoConfirm = false)
    if raw == null then
      println("\nNo interactive input available for confirmation; canceling for safety.")
    resolved

  private def printUsage(): Unit =
    println(
      """Usage:
        |
        |  sbt "run"              Show this menu
        |  sbt "run train"        Train/continue training
        |  sbt "run predict"      Predict next words
        |  sbt "run chunk"        Split large files into chunks
        |  sbt "run gpu-info"     Show Metal/JNI status
        |  sbt "run benchmark"    Interactive benchmark setup + throughput estimate
        |
        |Training:
        |  --input FILE      Training text file (single-input mode)
        |  --inputs CSV      Multiple training files (e.g., a.txt,b.txt)
        |  --inputWeights CSV Optional per-input weights (normalized)
        |  --preset NAME     quick/balanced/thorough (default: balanced)
        |  --fresh           Start fresh (ignore existing model)
        |  --contextSize N   Words to look back (default: 3)
        |  --maxVocab N      Max unique words (default: 3000)
        |  --backend NAME    cpu|gpu (default: gpu)
        |  --precision MODE  fp64|fp32 (default: fp64)
        |  --lr VALUE        Learning rate override (default: preset-specific)
        |  --lrDecay VALUE   Learning-rate decay per epoch (default: 1.0)
        |  --batchSize N     Mini-batch size (default: 128 gpu / 32 cpu)
        |  --prefetch N      Prefetch window (1 or 2; default: 1)
        |  --profileGpu      Print per-op GPU profile summary after training
        |  --gpuInfo         Print GPU probe details before training
        |  --replayRatio V   Replay mix ratio in [0,1) (default: 0.3)
        |  --replayBufferSize N Replay memory size (0 disables persistence)
        |  --replayBufferPath FILE Optional replay file path
        |  --ewcLambda V     Optional EWC regularization strength (default: 0)
        |  --ewcSamples N    Optional EWC sample count
        |
        |Chunking:
        |  --input FILE      File to split (required, or select interactively)
        |  --lines N         Lines per chunk (default: auto-recommend)
        |  --output DIR      Output directory (default: <input>/chunks/)
        |  --name BASE       Base name for chunks (default: input filename)
        |
        |Prediction:
        |  --context TEXT    Words to continue from
        |  --topK N          Predictions to show (default: 5)
        |  --backend NAME    cpu|gpu (default: gpu)
        |  --precision MODE  fp64|fp32 (default: fp64)

        |GPU:
        |  sbt "run gpu-info --precision fp64"
        |
        |Benchmark:
        |  --input FILE      Corpus text path
        |  --backend NAME    cpu|gpu (default: run both)
        |  --contextSize N   Context size (default: 3)
        |  --maxVocab N      Max vocab (default: 3000)
        |  --sample N        Number of examples (default: 2000)
        |  --precision MODE  fp64|fp32 (default: run both)
        |  --batchSize N     Batch size override for benchmark estimate
        |  --metrics          Persist metrics and print complete benchmark report
        |  --metricsDir DIR  Metrics output dir (default: data/metrics; used with --metrics)
        |  --runLabel TEXT   Optional run label for grouping/comparison
        |  --compareTo REF   latest|runId|runLabel baseline selector (default with --metrics: latest)
        |  --regressionWarnPct N Warn threshold percent (default: 5.0)
        |
        |Examples:
        |  sbt "run train --input data/corpus/text.txt"
        |  sbt "run train --inputs data/a.txt,data/b.txt --inputWeights 0.7,0.3 --replayRatio 0.3 --replayBufferSize 10000"
        |  sbt "run chunk --input large-file.txt --lines 1000"
        |  sbt 'run predict --context "the cat sat" --topK 5'
|""".stripMargin)
