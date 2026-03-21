package eval

import munit.FunSuite
import data.Example
import nn.{LanguageModel, ModelConfig}

class MetricsSuite extends FunSuite:
  test("meanLoss returns 0 on empty") {
    val cfg = ModelConfig(2, 3, 4, 5)
    val p = LanguageModel.initParams(cfg, 1)
    assertEquals(Metrics.meanLoss(p, Vector.empty), 0.0)
  }

  test("meanLoss and perplexity are finite") {
    val cfg = ModelConfig(2, 3, 4, 6)
    val p = LanguageModel.initParams(cfg, 2)
    val ex = Vector(Example(Vector(0, 1), 2), Example(Vector(1, 2), 3))
    val loss = Metrics.meanLoss(p, ex)
    val ppl = Metrics.perplexity(loss)
    assert(loss.isFinite)
    assert(ppl.isFinite)
    assert(ppl > 0.0)
  }
