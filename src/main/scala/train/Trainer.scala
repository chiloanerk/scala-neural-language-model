package train

import compute.{BackendSelector, ComputeBackend}
import data.Example
import eval.Metrics
import nn.{LanguageModel, Params}
import java.util.concurrent.atomic.{AtomicBoolean, AtomicInteger}
import scala.util.Random
import scala.util.control.NonFatal
import sun.misc.{Signal, SignalHandler}

enum TrainingStatus:
  case Improving, Stalled, Regressing

enum SaveDecision:
  case SaveBest, SaveCurrent, Discard

final case class TrainConfig(
    epochs: Int = 10,
    learningRate: Double = 0.05,
    lrDecay: Double = 1.0,
    l2: Double = 0.0,
    clipNorm: Option[Double] = None,
    shuffleEachEpoch: Boolean = true,
    seed: Int = 42,
    patience: Int = 0,  // 0 = disabled, >0 = early stopping enabled
    activation: String = "tanh",
    backend: String = "gpu",
    precision: String = "fp64",
    batchSize: Int = 0,
    prefetch: Int = 1,
    profileGpu: Boolean = false
)

final case class EpochMetrics(
    epoch: Int,
    trainLoss: Double,
    valLoss: Double,
    valPerplexity: Double,
    learningRate: Double,
    epochSeconds: Double = 0.0,
    examplesPerSec: Double = 0.0,
    status: TrainingStatus = TrainingStatus.Stalled,
    statusReason: String = "n/a",
    bestDeltaPct: Double = 0.0,
    generalizationGap: Double = 0.0
)
final case class TrainResult(
    params: Params,
    history: Vector[EpochMetrics],
    interrupted: Boolean = false,
    saveDecision: SaveDecision = SaveDecision.SaveCurrent
)

object Trainer:

  private[train] final case class Trajectory(
      status: TrainingStatus,
      reason: String,
      bestDeltaPct: Double,
      generalizationGap: Double,
      valTrend3: Double
  )

  // Progress bar helper (package-private for testing)
  private[train] def createProgressBar(percent: Int, width: Int = 20): String =
    val clamped = math.max(0, math.min(100, percent))
    val filled = (clamped * width) / 100
    val empty = width - filled
    val bar = "█" * filled + "░" * empty
    s"[$bar]"

  private[train] def classifyTrajectory(history: Vector[EpochMetrics], trainLoss: Double, valLoss: Double): Trajectory =
    val safeVal = math.max(math.abs(valLoss), 1e-9)
    val gap = (valLoss - trainLoss) / safeVal

    val bestPrior = history.map(_.valLoss).reduceOption(_ min _)
    val bestDeltaPct = bestPrior match
      case Some(best) => ((best - valLoss) / math.max(best, 1e-9)) * 100.0
      case None       => 0.0

    val trendSeries = history.takeRight(2).map(_.valLoss) :+ valLoss
    val valTrend3 =
      if trendSeries.length >= 3 then
        (trendSeries.last - trendSeries.head) / (trendSeries.length - 1).toDouble
      else 0.0

    val prevGap = history.lastOption.map(_.generalizationGap)
    val gapWidening = prevGap.exists(g => gap > g + 0.01)
    val improvingTrend = valTrend3 < -1e-4
    val worseningTrend = valTrend3 > 1e-4

    if history.isEmpty then
      Trajectory(TrainingStatus.Improving, "baseline", bestDeltaPct, gap, valTrend3)
    else if bestDeltaPct >= 0.5 then
      Trajectory(TrainingStatus.Improving, "better val", bestDeltaPct, gap, valTrend3)
    else if improvingTrend && !gapWidening then
      Trajectory(TrainingStatus.Improving, "improving trend", bestDeltaPct, gap, valTrend3)
    else if bestDeltaPct < -0.5 then
      Trajectory(TrainingStatus.Regressing, "val regressed", bestDeltaPct, gap, valTrend3)
    else if worseningTrend && gapWidening then
      Trajectory(TrainingStatus.Regressing, "widening gap", bestDeltaPct, gap, valTrend3)
    else
      Trajectory(TrainingStatus.Stalled, "flat val", bestDeltaPct, gap, valTrend3)

  private[train] def resolveInterruptDecision(
      interactive: Boolean,
      readLine: String => String | Null = prompt => scala.io.StdIn.readLine(prompt)
  ): SaveDecision =
    if !interactive then SaveDecision.SaveBest
    else
      var decision: Option[SaveDecision] = None
      while decision.isEmpty do
        val raw = Option(readLine("\nInterrupted. Choose action [b=save best, c=save current, d=discard]: ")).getOrElse("").trim.toLowerCase
        raw match
          case "b" | "best" | "1"    => decision = Some(SaveDecision.SaveBest)
          case "c" | "current" | "2" => decision = Some(SaveDecision.SaveCurrent)
          case "d" | "discard" | "3" => decision = Some(SaveDecision.Discard)
          case _                         => println("Please choose b, c, or d.")
      decision.get

  private def installInterruptHandler(
      cancelRequested: AtomicBoolean,
      currentEpoch: AtomicInteger,
      display: TrainingDisplay
  ): Option[SignalHandler] =
    try
      val handler = new SignalHandler:
        override def handle(sig: Signal): Unit =
          if cancelRequested.get() then
            System.err.println("\nSecond interrupt received. Exiting immediately.")
            System.exit(130)
          else
            cancelRequested.set(true)
            display.onCancellationRequested(currentEpoch.get())
      Some(Signal.handle(new Signal("INT"), handler))
    catch
      case NonFatal(_) => None

  private def restoreInterruptHandler(previous: Option[SignalHandler]): Unit =
    previous.foreach { h =>
      try Signal.handle(new Signal("INT"), h)
      catch case NonFatal(_) => ()
    }

  def train(initial: Params, trainSet: Vector[Example], valSet: Vector[Example], cfg: TrainConfig): TrainResult =
    require(cfg.epochs >= 1, s"epochs must be >= 1, got ${cfg.epochs}")
    require(cfg.learningRate > 0.0, s"learningRate must be > 0, got ${cfg.learningRate}")
    require(cfg.lrDecay > 0.0, s"lrDecay must be > 0, got ${cfg.lrDecay}")
    require(cfg.patience >= 0, s"patience must be >= 0, got ${cfg.patience}")

    val display = TrainingDisplay.create()
    val interactive = TrainingDisplay.isInteractiveTerminal

    val backend = BackendSelector.fromConfig(cfg.backend, cfg.precision, warn = msg => println(s"  [backend] $msg"))
    println(s"Using backend: ${backend.diagnostics}")
    if backend.isGpu then
      val ops = if backend.gpuOpsEnabled.nonEmpty then backend.gpuOpsEnabled.toVector.sorted.mkString(", ") else "none"
      println(s"GPU ops enabled: $ops")
    backend.resetProfile()

    var params = initial
    var lr = cfg.learningRate
    var history = Vector.empty[EpochMetrics]

    // Early stopping state
    var bestParams = initial
    var bestValLoss = Double.MaxValue
    var bestEpoch = 0
    var patienceCounter = 0

    val cancelRequested = AtomicBoolean(false)
    val currentEpoch = AtomicInteger(0)
    val previousSignalHandler = installInterruptHandler(cancelRequested, currentEpoch, display)

    try
      val totalExamples = trainSet.length
      val effectiveBatchSize =
        if cfg.batchSize > 0 then cfg.batchSize
        else if backend.isGpu then 128 else 32
      println(s"Batch size: $effectiveBatchSize")
      // Show progress every ~10 checkpoints per epoch, but not too chatty on small sets
      val progressInterval = math.max(100, math.max(1, totalExamples / 10))
      val estimateSec = estimateEpochSeconds(params, trainSet, cfg, backend)
      display.onTrainingStart(estimateSec)

      var epoch = 1
      var stopTraining = false
      var interrupted = false
      var saveDecision = SaveDecision.SaveCurrent

      while epoch <= cfg.epochs && !stopTraining do
        currentEpoch.set(epoch)
        display.onEpochStart(epoch, cfg.epochs, lr, totalExamples)

        val startTime = System.currentTimeMillis()
        val rnd = Random(cfg.seed + epoch)
        val epochData = if cfg.shuffleEachEpoch then rnd.shuffle(trainSet) else trainSet
        val batches = epochData.grouped(effectiveBatchSize).toVector

        var lossSum = 0.0
        var seen = 0
        var nextReportAt = progressInterval
        var batchIndex = 0

        while batchIndex < batches.length && !stopTraining do
          val batch = batches(batchIndex)
          val (updated, loss) = LanguageModel.trainBatchStep(
            params,
            batch,
            lr,
            l2 = cfg.l2,
            clipNorm = cfg.clipNorm,
            activation = cfg.activation,
            backend = backend
          )
          params = updated
          lossSum += loss * batch.size
          seen += batch.size

          if seen >= nextReportAt || seen == totalExamples || cancelRequested.get() then
            val elapsed = (System.currentTimeMillis() - startTime) / 1000.0
            val progress = if totalExamples <= 0 then 100 else ((seen.toDouble / totalExamples) * 100).toInt
            val avgLoss = if seen == 0 then 0.0 else lossSum / seen
            val examplesPerSec = if elapsed > 0 then seen.toDouble / elapsed else 0.0
            val estimatedTotal = if examplesPerSec > 0 then totalExamples.toDouble / examplesPerSec else 0.0
            val remaining = math.max(0.0, estimatedTotal - elapsed)
            display.onBatchProgress(
              BatchProgress(
                epoch = epoch,
                totalEpochs = cfg.epochs,
                percent = progress,
                elapsedSec = elapsed,
                remainingSec = remaining,
                examplesPerSec = examplesPerSec,
                avgLoss = avgLoss
              )
            )
            while nextReportAt <= seen do
              nextReportAt += progressInterval

          if cancelRequested.get() then
            interrupted = true
            stopTraining = true

          batchIndex += 1

        val trainLoss = if seen == 0 then 0.0 else lossSum / seen.toDouble
        val epochTime = (System.currentTimeMillis() - startTime) / 1000.0

        val valLoss = Metrics.meanLoss(params, valSet, cfg.activation, backend, batchSize = effectiveBatchSize)
        val ppl = Metrics.perplexity(valLoss)

        val trajectory = classifyTrajectory(history, trainLoss, valLoss)
        val examplesPerSec = if epochTime > 0 then seen.toDouble / epochTime else 0.0

        val metrics = EpochMetrics(
          epoch = epoch,
          trainLoss = trainLoss,
          valLoss = valLoss,
          valPerplexity = ppl,
          learningRate = lr,
          epochSeconds = epochTime,
          examplesPerSec = examplesPerSec,
          status = trajectory.status,
          statusReason = trajectory.reason,
          bestDeltaPct = trajectory.bestDeltaPct,
          generalizationGap = trajectory.generalizationGap
        )

        history :+= metrics

        var isBest = false
        var earlyStopMessage: Option[String] = None
        if cfg.patience > 0 then
          if valLoss < bestValLoss then
            bestValLoss = valLoss
            bestParams = params
            bestEpoch = epoch
            patienceCounter = 0
            isBest = true
          else
            patienceCounter += 1
            if patienceCounter >= cfg.patience then
              earlyStopMessage = Some(f"Early stopping triggered at epoch $epoch (best: epoch $bestEpoch)")
              stopTraining = true
        else
          bestParams = params // No early stopping, keep latest
          bestValLoss = valLoss
          bestEpoch = epoch

        display.onEpochComplete(metrics, isBest = isBest, patienceCounter = patienceCounter, patience = cfg.patience)
        earlyStopMessage.foreach(println)

        if interrupted then
          saveDecision = resolveInterruptDecision(interactive)
          saveDecision match
            case SaveDecision.SaveBest =>
              if bestEpoch == 0 then
                bestParams = params
                bestEpoch = epoch
            case SaveDecision.SaveCurrent =>
              bestParams = params
            case SaveDecision.Discard =>
              ()
        else
          lr *= cfg.lrDecay
          epoch += 1

      if !interrupted then
        saveDecision = if cfg.patience > 0 then SaveDecision.SaveBest else SaveDecision.SaveCurrent

      if cfg.patience > 0 && bestEpoch > 0 && bestEpoch < history.length && saveDecision == SaveDecision.SaveBest then
        println(f"Restoring best model from epoch $bestEpoch (val_loss=$bestValLoss%.6f)")

      display.onTrainingComplete(interrupted)
      if cfg.profileGpu && backend.isGpu then
        println(s"GPU profile: ${backend.profileSummary}")

      val finalParams = saveDecision match
        case SaveDecision.SaveCurrent => params
        case SaveDecision.SaveBest    => bestParams
        case SaveDecision.Discard     => params

      TrainResult(params = finalParams, history = history, interrupted = interrupted, saveDecision = saveDecision)
    finally
      restoreInterruptHandler(previousSignalHandler)

  def estimateEpochSeconds(
      params: Params,
      trainSet: Vector[Example],
      cfg: TrainConfig,
      backend: ComputeBackend,
      sampleSize: Int = 200
  ): Double =
    if trainSet.isEmpty then 0.0
    else
      val n = math.min(sampleSize, trainSet.length)
      val sample = trainSet.take(n)
      val t0 = System.nanoTime()
      val batchSize =
        if cfg.batchSize > 0 then cfg.batchSize
        else if backend.isGpu then 128 else 32
      sample.grouped(batchSize).foreach { batch =>
        val contexts = batch.map(_.context).toVector
        val targets = batch.map(_.target).toVector
        val cache = LanguageModel.forwardBatch(params, contexts, cfg.activation, backend)
        LanguageModel.backwardBatch(params, cache, targets, cfg.activation, backend)
      }
      val elapsedSec = (System.nanoTime() - t0).toDouble / 1e9
      if elapsedSec <= 0.0 then 0.0
      else elapsedSec * trainSet.length.toDouble / n.toDouble
