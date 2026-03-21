package eval

import compute.{ComputeBackend, CpuBackend}
import data.Example
import nn.{LanguageModel, Params}

object Metrics:
  def meanLoss(
      params: Params,
      examples: Vector[Example],
      activation: String = "tanh",
      backend: ComputeBackend = CpuBackend.Default
  ): Double =
    if examples.isEmpty then 0.0
    else
      val total = examples.foldLeft(0.0) { (acc, ex) =>
        val cache = LanguageModel.forward(params, ex.context, activation, backend)
        acc + LanguageModel.lossFromCache(cache, ex.target, backend)
      }
      total / examples.length.toDouble

  def perplexity(meanLoss: Double): Double = math.exp(meanLoss)
