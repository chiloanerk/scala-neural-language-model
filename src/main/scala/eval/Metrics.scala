package eval

import compute.{ComputeBackend, CpuBackend}
import data.Example
import nn.{LanguageModel, Params}

object Metrics:
  def meanLoss(
      params: Params,
      examples: Vector[Example],
      activation: String = "tanh",
      backend: ComputeBackend = CpuBackend.Default,
      batchSize: Int = 64
  ): Double =
    if examples.isEmpty then 0.0
    else
      val effectiveBatch = math.max(1, batchSize)
      val total = examples.grouped(effectiveBatch).foldLeft(0.0) { (acc, batch) =>
        val contexts = batch.map(_.context).toVector
        val targets = batch.map(_.target).toVector
        val cache = LanguageModel.forwardBatch(params, contexts, activation, backend)
        acc + LanguageModel.lossFromBatchCache(cache, targets, backend).sum
      }
      total / examples.length.toDouble

  def perplexity(meanLoss: Double): Double = math.exp(meanLoss)
