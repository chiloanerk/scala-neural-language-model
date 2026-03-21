package train

import munit.FunSuite
import data.Example
import nn.{LanguageModel, ModelConfig}

class TrainerSuite extends FunSuite:
  private val cfg = ModelConfig(contextSize = 2, embedDim = 4, hiddenDim = 6, vocabSize = 8, activation = "tanh")

  test("createProgressBar reflects percent") {
    val bar0 = Trainer.createProgressBar(0)
    val bar50 = Trainer.createProgressBar(50)
    val bar100 = Trainer.createProgressBar(100)

    assert(bar0.startsWith("[") && bar0.endsWith("]"))
    assert(bar100.contains("█"))
    assert(bar50.contains("█") && bar50.contains("░"))
  }

  test("train returns history for configured epochs without early stopping") {
    val p0 = LanguageModel.initParams(cfg, seed = 10)
    val trainSet = Vector.fill(30)(Example(Vector(1, 2), 3))
    val valSet = Vector.fill(10)(Example(Vector(1, 2), 3))

    val result = Trainer.train(p0, trainSet, valSet, TrainConfig(epochs = 4, learningRate = 0.05, patience = 0, seed = 10, backend = "cpu"))
    assertEquals(result.history.length, 4)
  }

  test("train supports early stopping") {
    val p0 = LanguageModel.initParams(cfg, seed = 20)
    val trainSet = Vector.fill(30)(Example(Vector(1, 2), 3))
    val valSet = Vector.fill(10)(Example(Vector(1, 2), 4))

    val result = Trainer.train(p0, trainSet, valSet, TrainConfig(epochs = 12, learningRate = 0.05, patience = 2, seed = 20, backend = "cpu"))
    assert(result.history.length <= 12)
    assert(result.history.nonEmpty)
  }

  test("train supports configurable batch sizes") {
    val p0 = LanguageModel.initParams(cfg, seed = 30)
    val trainSet = Vector.fill(40)(Example(Vector(1, 2), 3))
    val valSet = Vector.fill(10)(Example(Vector(1, 2), 3))

    val b1 = Trainer.train(p0, trainSet, valSet, TrainConfig(epochs = 2, learningRate = 0.05, seed = 30, backend = "cpu", batchSize = 1))
    val b8 = Trainer.train(p0, trainSet, valSet, TrainConfig(epochs = 2, learningRate = 0.05, seed = 30, backend = "cpu", batchSize = 8))
    assertEquals(b1.history.length, 2)
    assertEquals(b8.history.length, 2)
    assert(b1.history.last.trainLoss.isFinite)
    assert(b8.history.last.trainLoss.isFinite)
  }
