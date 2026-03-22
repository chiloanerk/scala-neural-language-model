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

final case class DomainValMetrics(
    domain: String,
    valLoss: Double,
    valPerplexity: Double,
    weight: Double
)

final case class RetentionMetrics(
    domain: String,
    baselineValLoss: Double,
    currentValLoss: Double,
    retentionPct: Double,
    status: String
)

final case class TrainConfig(
    epochs: Int = 10,
    learningRate: Double = 0.05,
    lrDecay: Double = 1.0,
    l2: Double = 0.0,
    clipNorm: Option[Double] = None,
    shuffleEachEpoch: Boolean = true,
    seed: Int = 42,
    patience: Int = 0,
    activation: String = "tanh",
    backend: String = "gpu",
    precision: String = "fp64",
    batchSize: Int = 0,
    prefetch: Int = 1,
    profileGpu: Boolean = false,
    inputWeights: Vector[Double] = Vector.empty,
    replayRatio: Double = 0.0,
    replayBufferSize: Int = 0,
    replayBufferPath: Option[String] = None,
    domainLabels: Vector[String] = Vector.empty,
    mixedValWeights: Vector[Double] = Vector.empty,
    ewcLambda: Double = 0.0,
    ewcSamples: Int = 0
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
    generalizationGap: Double = 0.0,
    mixedValLoss: Double = Double.NaN,
    perDomainValMetrics: Vector[DomainValMetrics] = Vector.empty,
    retentionMetrics: Vector[RetentionMetrics] = Vector.empty,
    phaseLabel: Option[String] = None
)

final case class TrainResult(
    params: Params,
    history: Vector[EpochMetrics],
    interrupted: Boolean = false,
    saveDecision: SaveDecision = SaveDecision.SaveCurrent,
    replayBuffer: Option[ReplayBuffer] = None
)

object Trainer:
  private final case class InstalledSignalHandlers(previous: Map[String, SignalHandler])

  final case class TrainingPhase(
      label: String,
      trainSet: Vector[Example],
      valSet: Vector[Example],
      weight: Double
  )

  private[train] final case class Trajectory(
      status: TrainingStatus,
      reason: String,
      bestDeltaPct: Double,
      generalizationGap: Double,
      valTrend3: Double
  )

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
      if trendSeries.length >= 3 then (trendSeries.last - trendSeries.head) / (trendSeries.length - 1).toDouble
      else 0.0

    val prevGap = history.lastOption.map(_.generalizationGap)
    val gapWidening = prevGap.exists(g => gap > g + 0.01)
    val improvingTrend = valTrend3 < -1e-4
    val worseningTrend = valTrend3 > 1e-4
    val deltaDeadbandPct = 0.1
    val nearFlatDelta = math.abs(bestDeltaPct) < deltaDeadbandPct

    if history.isEmpty then
      Trajectory(TrainingStatus.Improving, "baseline", bestDeltaPct, gap, valTrend3)
    else if bestDeltaPct >= 0.5 then
      Trajectory(TrainingStatus.Improving, "better val", bestDeltaPct, gap, valTrend3)
    else if nearFlatDelta && improvingTrend && !gapWidening then
      Trajectory(TrainingStatus.Improving, "slow improvement", bestDeltaPct, gap, valTrend3)
    else if nearFlatDelta && (worseningTrend || gapWidening) then
      Trajectory(TrainingStatus.Stalled, "minor drift", bestDeltaPct, gap, valTrend3)
    else if nearFlatDelta then
      Trajectory(TrainingStatus.Stalled, "near plateau", bestDeltaPct, gap, valTrend3)
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
  ): Option[InstalledSignalHandlers] =
    try
      val handler = new SignalHandler:
        override def handle(sig: Signal): Unit =
          if cancelRequested.get() then
            System.err.println(s"\nSecond ${sig.getName} received. Exiting immediately.")
            System.exit(130)
          else
            cancelRequested.set(true)
            display.onCancellationRequested(currentEpoch.get())
      val previous = scala.collection.mutable.Map.empty[String, SignalHandler]
      Vector("INT", "TERM").foreach { name =>
        try
          previous(name) = Signal.handle(new Signal(name), handler)
        catch
          case NonFatal(_) => ()
      }
      if previous.nonEmpty then Some(InstalledSignalHandlers(previous.toMap)) else None
    catch
      case NonFatal(_) => None

  private def restoreInterruptHandler(previous: Option[InstalledSignalHandlers]): Unit =
    previous.foreach { installed =>
      installed.previous.foreach { case (name, handler) =>
        try Signal.handle(new Signal(name), handler)
        catch case NonFatal(_) => ()
      }
    }

  def train(initial: Params, trainSet: Vector[Example], valSet: Vector[Example], cfg: TrainConfig): TrainResult =
    val phase = TrainingPhase("domain-1", trainSet, valSet, weight = 1.0)
    trainPhased(initial, Vector(phase), cfg.copy(replayRatio = 0.0, replayBufferSize = 0), replayBuffer = None)
      .copy(replayBuffer = None)

  def trainPhased(
      initial: Params,
      phases: Vector[TrainingPhase],
      cfg: TrainConfig,
      replayBuffer: Option[ReplayBuffer] = None
  ): TrainResult =
    require(phases.nonEmpty, "phases cannot be empty")
    require(phases.forall(_.trainSet.nonEmpty), "all phase train sets must be non-empty")
    require(phases.forall(_.valSet.nonEmpty), "all phase validation sets must be non-empty")
    require(cfg.epochs >= 1, s"epochs must be >= 1, got ${cfg.epochs}")
    require(cfg.learningRate > 0.0, s"learningRate must be > 0, got ${cfg.learningRate}")
    require(cfg.lrDecay > 0.0, s"lrDecay must be > 0, got ${cfg.lrDecay}")
    require(cfg.patience >= 0, s"patience must be >= 0, got ${cfg.patience}")
    require(cfg.replayRatio >= 0.0 && cfg.replayRatio < 1.0, s"replayRatio must be in [0,1), got ${cfg.replayRatio}")
    require(cfg.replayBufferSize >= 0, s"replayBufferSize must be >= 0, got ${cfg.replayBufferSize}")

    val display = TrainingDisplay.create()
    val interactive = TrainingDisplay.isInteractiveTerminal

    val backend = BackendSelector.fromConfig(cfg.backend, cfg.precision, warn = msg => println(s"  [backend] $msg"))
    println(s"Using backend: ${backend.diagnostics}")
    if backend.isGpu then
      val ops = if backend.gpuOpsEnabled.nonEmpty then backend.gpuOpsEnabled.toVector.sorted.mkString(", ") else "none"
      println(s"GPU ops enabled: $ops")
    backend.resetProfile()

    val weights = normalizedWeights(phases.map(_.weight))
    val weightedPhases = phases.zip(weights).map { case (p, w) => p.copy(weight = w) }

    var params = initial
    var lr = cfg.learningRate
    var history = Vector.empty[EpochMetrics]

    var bestParams = initial
    var bestValLoss = Double.MaxValue
    var bestEpoch = 0
    var patienceCounter = 0

    var buffer = replayBuffer

    val cancelRequested = AtomicBoolean(false)
    val currentEpoch = AtomicInteger(0)
    val previousSignalHandler = installInterruptHandler(cancelRequested, currentEpoch, display)

    try
      val defaultBatchSize = if cfg.batchSize > 0 then cfg.batchSize else if backend.isGpu then 128 else 32
      println(s"Batch size: $defaultBatchSize")

      val phaseEstimate = weightedPhases.map(p => estimateEpochSeconds(params, p.trainSet, cfg, backend)).sum / weightedPhases.length.toDouble
      display.onTrainingStart(phaseEstimate)

      val totalEpochsPlanned = cfg.epochs * weightedPhases.length
      var globalEpoch = 1
      var stopTraining = false
      var interrupted = false
      var saveDecision = SaveDecision.SaveCurrent

      var phaseIndex = 0
      while phaseIndex < weightedPhases.length && !stopTraining do
        val phase = weightedPhases(phaseIndex)
        println(s"\nPhase ${phaseIndex + 1}/${weightedPhases.length}: ${phase.label}")

        val seenPhases = weightedPhases.take(phaseIndex + 1)
        val oldPhases = weightedPhases.take(phaseIndex)
        val retentionBaselines =
          oldPhases.map { p =>
            val baseline = Metrics.meanLoss(params, p.valSet, cfg.activation, backend, batchSize = defaultBatchSize)
            p.label -> baseline
          }.toMap

        var phaseEpoch = 1
        while phaseEpoch <= cfg.epochs && !stopTraining do
          currentEpoch.set(globalEpoch)

          val replayExamples = buildReplaySlice(
            currentTrainSize = phase.trainSet.length,
            replayRatio = cfg.replayRatio,
            seed = cfg.seed + globalEpoch,
            buffer = buffer
          )

          val combinedTrainSet =
            if replayExamples.isEmpty then phase.trainSet
            else phase.trainSet ++ replayExamples

          val epochData =
            if cfg.shuffleEachEpoch then Random(cfg.seed + globalEpoch).shuffle(combinedTrainSet)
            else combinedTrainSet

          val totalExamples = epochData.length
          val progressInterval = math.max(100, math.max(1, totalExamples / 10))
          display.onEpochStart(globalEpoch, totalEpochsPlanned, lr, totalExamples)

          val startTime = System.currentTimeMillis()
          val batches = epochData.grouped(defaultBatchSize).toVector

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
                  epoch = globalEpoch,
                  totalEpochs = totalEpochsPlanned,
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
          val examplesPerSec = if epochTime > 0 then seen.toDouble / epochTime else 0.0

          val domainMetrics = seenPhases.map { p =>
            val loss = Metrics.meanLoss(params, p.valSet, cfg.activation, backend, batchSize = defaultBatchSize)
            DomainValMetrics(
              domain = p.label,
              valLoss = loss,
              valPerplexity = Metrics.perplexity(loss),
              weight = p.weight
            )
          }

          val mixedValLoss = weightedLoss(domainMetrics)
          val mixedPpl = Metrics.perplexity(mixedValLoss)

          val retention = oldPhases.flatMap { p =>
            retentionBaselines.get(p.label).zip(domainMetrics.find(_.domain == p.label)).map { case (baseline, current) =>
              val ratio = if current.valLoss <= 0 then 1.0 else baseline / current.valLoss
              val retentionPct = ratio * 100.0
              val band =
                if retentionPct >= 95.0 then "retained"
                else if retentionPct >= 80.0 then "warning"
                else "significant forgetting"
              RetentionMetrics(
                domain = p.label,
                baselineValLoss = baseline,
                currentValLoss = current.valLoss,
                retentionPct = retentionPct,
                status = band
              )
            }
          }

          val trajectory = classifyTrajectory(history, trainLoss, mixedValLoss)
          val metrics = EpochMetrics(
            epoch = globalEpoch,
            trainLoss = trainLoss,
            valLoss = mixedValLoss,
            valPerplexity = mixedPpl,
            learningRate = lr,
            epochSeconds = epochTime,
            examplesPerSec = examplesPerSec,
            status = trajectory.status,
            statusReason = trajectory.reason,
            bestDeltaPct = trajectory.bestDeltaPct,
            generalizationGap = trajectory.generalizationGap,
            mixedValLoss = mixedValLoss,
            perDomainValMetrics = domainMetrics,
            retentionMetrics = retention,
            phaseLabel = Some(phase.label)
          )

          history :+= metrics

          var isBest = false
          var earlyStopMessage: Option[String] = None
          if cfg.patience > 0 then
            if mixedValLoss < bestValLoss then
              bestValLoss = mixedValLoss
              bestParams = params
              bestEpoch = globalEpoch
              patienceCounter = 0
              isBest = true
            else
              patienceCounter += 1
              if patienceCounter >= cfg.patience then
                earlyStopMessage = Some(f"Early stopping triggered at epoch $globalEpoch (best: epoch $bestEpoch)")
                stopTraining = true
          else
            bestParams = params
            bestValLoss = mixedValLoss
            bestEpoch = globalEpoch

          display.onEpochComplete(metrics, isBest = isBest, patienceCounter = patienceCounter, patience = cfg.patience)
          earlyStopMessage.foreach(println)

          if interrupted then
            saveDecision = resolveInterruptDecision(interactive)
            saveDecision match
              case SaveDecision.SaveBest =>
                if bestEpoch == 0 then
                  bestParams = params
                  bestEpoch = globalEpoch
              case SaveDecision.SaveCurrent =>
                bestParams = params
              case SaveDecision.Discard => ()
          else
            lr *= cfg.lrDecay
            globalEpoch += 1
            phaseEpoch += 1
        if cfg.replayBufferSize > 0 && phase.trainSet.nonEmpty then
          buffer = Some(
            buffer match
              case Some(existing) => existing.add(phase.trainSet, phase.label, cfg.replayBufferSize)
              case None           => ReplayBuffer.empty.add(phase.trainSet, phase.label, cfg.replayBufferSize)
          )

        phaseIndex += 1

      if !interrupted then
        saveDecision = if cfg.patience > 0 then SaveDecision.SaveBest else SaveDecision.SaveCurrent

      if cfg.patience > 0 && bestEpoch > 0 && bestEpoch < history.length && saveDecision == SaveDecision.SaveBest then
        println(f"Restoring best model from epoch $bestEpoch (mixed_val_loss=$bestValLoss%.6f)")

      display.onTrainingComplete(interrupted)
      if cfg.profileGpu && backend.isGpu then
        println(s"GPU profile: ${backend.profileSummary}")

      val finalParams = saveDecision match
        case SaveDecision.SaveCurrent => params
        case SaveDecision.SaveBest    => bestParams
        case SaveDecision.Discard     => params

      TrainResult(
        params = finalParams,
        history = history,
        interrupted = interrupted,
        saveDecision = saveDecision,
        replayBuffer = buffer
      )
    finally
      restoreInterruptHandler(previousSignalHandler)

  private def weightedLoss(metrics: Vector[DomainValMetrics]): Double =
    if metrics.isEmpty then 0.0
    else
      val sumW = metrics.map(_.weight).sum
      if sumW <= 0 then metrics.map(_.valLoss).sum / metrics.length.toDouble
      else metrics.map(m => m.valLoss * m.weight).sum / sumW

  private def normalizedWeights(raw: Vector[Double]): Vector[Double] =
    if raw.isEmpty then Vector.empty
    else
      val clamped = raw.map(w => if w.isFinite && w > 0 then w else 0.0)
      val sum = clamped.sum
      if sum <= 0 then Vector.fill(raw.length)(1.0 / raw.length.toDouble)
      else clamped.map(_ / sum)

  private def buildReplaySlice(
      currentTrainSize: Int,
      replayRatio: Double,
      seed: Int,
      buffer: Option[ReplayBuffer]
  ): Vector[Example] =
    if replayRatio <= 0.0 || currentTrainSize <= 0 then Vector.empty
    else
      buffer match
        case None => Vector.empty
        case Some(b) if b.examples.isEmpty => Vector.empty
        case Some(b) =>
          val replayCount = math.round((currentTrainSize * replayRatio) / (1.0 - replayRatio)).toInt
          b.sample(replayCount, seed)

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
      val batchSize = if cfg.batchSize > 0 then cfg.batchSize else if backend.isGpu then 128 else 32
      sample.grouped(batchSize).foreach { batch =>
        val contexts = batch.map(_.context).toVector
        val targets = batch.map(_.target).toVector
        val cache = LanguageModel.forwardBatch(params, contexts, cfg.activation, backend)
        LanguageModel.backwardBatch(params, cache, targets, cfg.activation, backend)
      }
      val elapsedSec = (System.nanoTime() - t0).toDouble / 1e9
      if elapsedSec <= 0.0 then 0.0 else elapsedSec * trainSet.length.toDouble / n.toDouble
