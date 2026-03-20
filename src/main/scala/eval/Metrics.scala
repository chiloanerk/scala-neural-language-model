package eval

import data.Example
import nn.{LanguageModel, Params}

object Metrics:
  def meanLoss(params: Params, examples: Vector[Example], activation: String = "tanh"): Double =
    if examples.isEmpty then 0.0
    else
      val total = examples.foldLeft(0.0) { (acc, ex) =>
        val cache = LanguageModel.forward(params, ex.context, activation)
        acc + LanguageModel.lossFromCache(cache, ex.target)
      }
      total / examples.length.toDouble

  def perplexity(meanLoss: Double): Double = math.exp(meanLoss)
