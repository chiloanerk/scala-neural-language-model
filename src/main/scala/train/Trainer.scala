package train

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
    seed: Int = 42
)

final case class EpochMetrics(epoch: Int, trainLoss: Double, valLoss: Double, valPerplexity: Double, learningRate: Double)
final case class TrainResult(params: Params, history: Vector[EpochMetrics])

object Trainer:

  def train(initial: Params, trainSet: Vector[Example], valSet: Vector[Example], cfg: TrainConfig): TrainResult =
    require(cfg.epochs >= 1, s"epochs must be >= 1, got ${cfg.epochs}")
    require(cfg.learningRate > 0.0, s"learningRate must be > 0, got ${cfg.learningRate}")
    require(cfg.lrDecay > 0.0, s"lrDecay must be > 0, got ${cfg.lrDecay}")

    var params = initial
    var lr = cfg.learningRate
    var history = Vector.empty[EpochMetrics]

    val totalExamples = trainSet.length
    val progressInterval = math.max(1000, totalExamples / 20) // Print every 5% or 1000 examples

    var epoch = 1
    while epoch <= cfg.epochs do
      println(f"Epoch $epoch/$cfg.epochs starting (lr=$lr%.4f, examples=$totalExamples)...")
      
      val startTime = System.currentTimeMillis()
      val rnd = Random(cfg.seed + epoch)
      val epochData = if cfg.shuffleEachEpoch then rnd.shuffle(trainSet) else trainSet

      var lossSum = 0.0
      var seen = 0
      epochData.foreach { ex =>
        val (updated, loss) = LanguageModel.trainStep(params, ex, lr, l2 = cfg.l2, clipNorm = cfg.clipNorm)
        params = updated
        lossSum += loss
        seen += 1
        
        if seen % progressInterval == 0 then
          val elapsed = (System.currentTimeMillis() - startTime) / 1000.0
          val progress = (seen.toDouble / totalExamples) * 100
          val avgLoss = lossSum / seen
          println(f"  Progress: $progress%.0f%% ($seen/$totalExamples examples, ${elapsed}%.1fs, avg_loss=$avgLoss%.4f)")
      }

      val trainLoss = if seen == 0 then 0.0 else lossSum / seen.toDouble
      val epochTime = (System.currentTimeMillis() - startTime) / 1000.0
      
      println(f"  Epoch $epoch complete in ${epochTime}%.1fs - train_loss=$trainLoss%.6f")
      println(f"  Computing validation metrics...")
      
      val valLoss = Metrics.meanLoss(params, valSet)
      val ppl = Metrics.perplexity(valLoss)
      println(f"  val_loss=$valLoss%.6f val_ppl=$ppl%.4f")
      
      history :+= EpochMetrics(epoch, trainLoss, valLoss, ppl, lr)

      lr *= cfg.lrDecay
      epoch += 1

    println("Training complete!")
    TrainResult(params = params, history = history)
