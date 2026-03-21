package nn

import munit.FunSuite
import data.Example
import linalg.LinearAlgebra

class LanguageModelSuite extends FunSuite:
  private def cfg = ModelConfig(contextSize = 2, embedDim = 3, hiddenDim = 4, vocabSize = 7, activation = "tanh")

  test("initParams has expected shapes") {
    val p = LanguageModel.initParams(cfg, seed = 1)
    assertEquals(p.E.rows, cfg.vocabSize)
    assertEquals(p.E.cols, cfg.embedDim)
    assertEquals(p.W1.rows, cfg.hiddenDim)
    assertEquals(p.W1.cols, cfg.contextSize * cfg.embedDim)
    assertEquals(p.W2.rows, cfg.vocabSize)
    assertEquals(p.W2.cols, cfg.hiddenDim)
  }

  test("forward returns valid probabilities for tanh and relu") {
    val p = LanguageModel.initParams(cfg, seed = 2)
    val tanh = LanguageModel.forward(p, Vector(1, 2), "tanh")
    val relu = LanguageModel.forward(p, Vector(1, 2), "relu")
    assert(math.abs(tanh.probs.sum - 1.0) < 1e-9)
    assert(math.abs(relu.probs.sum - 1.0) < 1e-9)
    assert(tanh.probs.forall(p => p >= 0.0 && p.isFinite))
    assert(relu.probs.forall(p => p >= 0.0 && p.isFinite))
  }

  test("unknown activation throws") {
    val p = LanguageModel.initParams(cfg, seed = 3)
    intercept[IllegalArgumentException] {
      LanguageModel.forward(p, Vector(1, 2), "swish")
    }
  }

  test("lossFromCache is non-negative") {
    val p = LanguageModel.initParams(cfg, seed = 4)
    val c = LanguageModel.forward(p, Vector(1, 2))
    val loss = LanguageModel.lossFromCache(c, 3)
    assert(loss >= 0.0)
  }

  test("backward returns gradients with matching shapes") {
    val p = LanguageModel.initParams(cfg, seed = 5)
    val c = LanguageModel.forward(p, Vector(1, 2), "relu")
    val g = LanguageModel.backward(p, c, target = 3, activation = "relu")

    assertEquals(g.dE.rows, p.E.rows)
    assertEquals(g.dE.cols, p.E.cols)
    assertEquals(g.dW1.rows, p.W1.rows)
    assertEquals(g.dW1.cols, p.W1.cols)
    assertEquals(g.db1.length, p.b1.length)
    assertEquals(g.dW2.rows, p.W2.rows)
    assertEquals(g.dW2.cols, p.W2.cols)
    assertEquals(g.db2.length, p.b2.length)
  }

  test("update changes parameters") {
    val p = LanguageModel.initParams(cfg, seed = 6)
    val c = LanguageModel.forward(p, Vector(1, 2))
    val g = LanguageModel.backward(p, c, 3)
    val updated = LanguageModel.update(p, g, lr = 0.01)
    assertNotEquals(updated.W1.data, p.W1.data)
  }

  test("trainStep returns new params and finite loss") {
    val p = LanguageModel.initParams(cfg, seed = 7)
    val (updated, loss) = LanguageModel.trainStep(p, Example(Vector(1, 2), 3), lr = 0.01, activation = "tanh")
    assert(loss.isFinite)
    assertNotEquals(updated.W2.data, p.W2.data)
  }

  test("forwardBatch returns valid probabilities") {
    val p = LanguageModel.initParams(cfg, seed = 8)
    val batch = Vector(Vector(1, 2), Vector(2, 3), Vector(3, 4))
    val cache = LanguageModel.forwardBatch(p, batch, "tanh")
    assertEquals(cache.probs.rows, batch.length)
    assertEquals(cache.probs.cols, cfg.vocabSize)
    assert((0 until cache.probs.rows).forall(r => math.abs(cache.probs.rowSlice(r).sum - 1.0) < 1e-9))
  }

  test("backwardBatch returns gradients with matching shapes") {
    val p = LanguageModel.initParams(cfg, seed = 9)
    val batch = Vector(Vector(1, 2), Vector(2, 3))
    val targets = Vector(3, 4)
    val cache = LanguageModel.forwardBatch(p, batch, "relu")
    val g = LanguageModel.backwardBatch(p, cache, targets, activation = "relu")
    assertEquals(g.dE.rows, p.E.rows)
    assertEquals(g.dE.cols, p.E.cols)
    assertEquals(g.dW1.rows, p.W1.rows)
    assertEquals(g.dW1.cols, p.W1.cols)
    assertEquals(g.dW2.rows, p.W2.rows)
    assertEquals(g.dW2.cols, p.W2.cols)
  }

  test("trainBatchStep with batchSize=1 matches trainStep") {
    val p = LanguageModel.initParams(cfg, seed = 10)
    val ex = Example(Vector(1, 2), 3)
    val (u1, l1) = LanguageModel.trainStep(p, ex, lr = 0.01, activation = "tanh")
    val (u2, l2) = LanguageModel.trainBatchStep(p, Vector(ex), lr = 0.01, activation = "tanh")
    assert(math.abs(l1 - l2) < 1e-9)
    val maxDiff = u1.W2.data.zip(u2.W2.data).map { case (a, b) => math.abs(a - b) }.max
    assert(maxDiff < 1e-9)
  }

  test("model output supports top-2 and top-3 predictions with sorted valid ids") {
    val p = LanguageModel.initParams(cfg, seed = 11)
    val cache = LanguageModel.forward(p, Vector(1, 2), "tanh")

    val top2 = LinearAlgebra.argTopK(cache.probs, 2)
    val top3 = LinearAlgebra.argTopK(cache.probs, 3)

    assertEquals(top2.length, 2)
    assertEquals(top3.length, 3)
    assert(top2.forall { case (id, prob) => id >= 0 && id < cfg.vocabSize && prob.isFinite && prob >= 0.0 })
    assert(top3.forall { case (id, prob) => id >= 0 && id < cfg.vocabSize && prob.isFinite && prob >= 0.0 })
    assert(top2(0)._2 >= top2(1)._2)
    assert(top3(0)._2 >= top3(1)._2 && top3(1)._2 >= top3(2)._2)
    assert(top3.map(_._1).distinct.length == 3)
  }
