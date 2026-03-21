package train

import compute.{BackendSelector, ComputeBackend}
import data.Example
import eval.Metrics
import nn.{LanguageModel, Params}
import scala.util.Random

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
    precision: String = "fp32",
    batchSize: Int = 0,
    prefetch: Int = 1,
    profileGpu: Boolean = false
)

final case class EpochMetrics(epoch: Int, trainLoss: Double, valLoss: Double, valPerplexity: Double, learningRate: Double)
final case class TrainResult(params: Params, history: Vector[EpochMetrics])

object Trainer:

  // Progress bar helper (package-private for testing)
  private[train] def createProgressBar(percent: Int, width: Int = 20): String =
    val filled = (percent * width) / 100
    val empty = width - filled
    val bar = "█" * filled + "░" * empty
    s"[$bar]"

  def train(initial: Params, trainSet: Vector[Example], valSet: Vector[Example], cfg: TrainConfig): TrainResult =
    require(cfg.epochs >= 1, s"epochs must be >= 1, got ${cfg.epochs}")
    require(cfg.learningRate > 0.0, s"learningRate must be > 0, got ${cfg.learningRate}")
    require(cfg.lrDecay > 0.0, s"lrDecay must be > 0, got ${cfg.lrDecay}")
    require(cfg.patience >= 0, s"patience must be >= 0, got ${cfg.patience}")

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

    val totalExamples = trainSet.length
    val effectiveBatchSize =
      if cfg.batchSize > 0 then cfg.batchSize
      else if backend.isGpu then 128 else 32
    println(s"Batch size: $effectiveBatchSize")
    // Show progress every 10% or every 100 examples, whichever is smaller
    val progressInterval = math.max(100, totalExamples / 10)
    val estimateSec = estimateEpochSeconds(params, trainSet, cfg, backend)
    if estimateSec > 0 then
      println(f"Estimated epoch time: ${estimateSec / 60.0}%.1f min (${estimateSec}%.0f sec)")
      if estimateSec > 120 then
        println("  Tip: long epoch detected. Consider chunking input, contextSize<=3, and lower maxVocab for faster iteration.")

    var epoch = 1
    while epoch <= cfg.epochs do
      println(f"Epoch $epoch/$cfg.epochs starting (lr=$lr%.4f, examples=$totalExamples)...")

      val startTime = System.currentTimeMillis()
      val rnd = Random(cfg.seed + epoch)
      val epochData = if cfg.shuffleEachEpoch then rnd.shuffle(trainSet) else trainSet
      val batches = epochData.grouped(effectiveBatchSize).toVector

      var lossSum = 0.0
      var seen = 0
      var nextReportAt = progressInterval
      batches.foreach { batch =>
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

        if seen >= nextReportAt || seen == totalExamples then
          val elapsed = (System.currentTimeMillis() - startTime) / 1000.0
          val progress = (seen.toDouble / totalExamples) * 100
          val avgLoss = lossSum / seen
          val examplesPerSec = if elapsed > 0 then seen.toDouble / elapsed else 0.0
          val estimatedTotal = if examplesPerSec > 0 then totalExamples.toDouble / examplesPerSec else 0.0
          val remaining = math.max(0.0, estimatedTotal - elapsed)
          val bar = createProgressBar(progress.toInt)
          println(f"  $bar $progress%3.0f%% | ${elapsed}%.1fs/${remaining}%.1fs | ${examplesPerSec}%.0f ex/s | loss=$avgLoss%.4f")
          while nextReportAt <= seen do
            nextReportAt += progressInterval
      }

      val trainLoss = if seen == 0 then 0.0 else lossSum / seen.toDouble
      val epochTime = (System.currentTimeMillis() - startTime) / 1000.0

      println(f"  Epoch $epoch complete in ${epochTime}%.1fs - train_loss=$trainLoss%.6f")
      println(f"  Computing validation metrics...")

      val valLoss = Metrics.meanLoss(params, valSet, cfg.activation, backend, batchSize = effectiveBatchSize)
      val ppl = Metrics.perplexity(valLoss)
      println(f"  val_loss=$valLoss%.6f val_ppl=$ppl%.4f")

      history :+= EpochMetrics(epoch, trainLoss, valLoss, ppl, lr)

      // Early stopping check
      if cfg.patience > 0 then
        if valLoss < bestValLoss then
          bestValLoss = valLoss
          bestParams = params
          bestEpoch = epoch
          patienceCounter = 0
          println(f"  New best model! val_loss=$bestValLoss%.6f")
        else
          patienceCounter += 1
          println(f"  No improvement (patience: ${patienceCounter}/${cfg.patience})")
          if patienceCounter >= cfg.patience then
            println(f"  Early stopping triggered at epoch $epoch (best: epoch $bestEpoch)")
            epoch = cfg.epochs + 1 // Exit loop
      else
        bestParams = params // No early stopping, use final params

      lr *= cfg.lrDecay
      epoch += 1

    if cfg.patience > 0 && bestEpoch < history.length then
      println(f"Restoring best model from epoch $bestEpoch (val_loss=$bestValLoss%.6f)")

    println("Training complete!")
    if cfg.profileGpu && backend.isGpu then
      println(s"GPU profile: ${backend.profileSummary}")
    TrainResult(params = bestParams, history = history)

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
