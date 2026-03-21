package train

import munit.FunSuite
import data.Example
import eval.Metrics
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

  test("early stopping restores best params by validation loss") {
    val p0 = LanguageModel.initParams(cfg, seed = 21)
    val trainSet = Vector.fill(30)(Example(Vector(1, 2), 3))
    val valSet = Vector.fill(10)(Example(Vector(1, 2), 4))

    val result = Trainer.train(
      p0,
      trainSet,
      valSet,
      TrainConfig(epochs = 10, learningRate = 0.05, patience = 2, seed = 21, backend = "cpu", precision = "fp64")
    )

    val bestHistoryVal = result.history.map(_.valLoss).min
    val restoredVal = Metrics.meanLoss(result.params, valSet, activation = "tanh", batchSize = 8)
    assert(math.abs(restoredVal - bestHistoryVal) < 1e-6)
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

  test("trajectory classifier marks improving when val meaningfully improves") {
    val history = Vector(
      EpochMetrics(epoch = 1, trainLoss = 2.0, valLoss = 2.0, valPerplexity = 7.4, learningRate = 0.05, generalizationGap = 0.0)
    )
    val t = Trainer.classifyTrajectory(history, trainLoss = 1.8, valLoss = 1.95)
    assertEquals(t.status, TrainingStatus.Improving)
  }

  test("trajectory classifier marks stalled on flat val") {
    val history = Vector(
      EpochMetrics(epoch = 1, trainLoss = 1.2, valLoss = 1.0, valPerplexity = 2.7, learningRate = 0.05, generalizationGap = -0.2)
    )
    val t = Trainer.classifyTrajectory(history, trainLoss = 1.22, valLoss = 1.003)
    assertEquals(t.status, TrainingStatus.Stalled)
  }

  test("trajectory classifier marks regressing when val gets worse") {
    val history = Vector(
      EpochMetrics(epoch = 1, trainLoss = 0.8, valLoss = 1.0, valPerplexity = 2.7, learningRate = 0.05, generalizationGap = 0.2),
      EpochMetrics(epoch = 2, trainLoss = 0.7, valLoss = 1.02, valPerplexity = 2.8, learningRate = 0.05, generalizationGap = 0.3)
    )
    val t = Trainer.classifyTrajectory(history, trainLoss = 0.65, valLoss = 1.08)
    assertEquals(t.status, TrainingStatus.Regressing)
  }

  test("trajectory classifier treats tiny delta as plateau instead of regression noise") {
    val history = Vector(
      EpochMetrics(epoch = 1, trainLoss = 4.1, valLoss = 4.20, valPerplexity = 66.7, learningRate = 0.01, generalizationGap = 0.02),
      EpochMetrics(epoch = 2, trainLoss = 4.0, valLoss = 4.19, valPerplexity = 66.0, learningRate = 0.01, generalizationGap = 0.03)
    )
    val t = Trainer.classifyTrajectory(history, trainLoss = 3.99, valLoss = 4.194)
    assertEquals(t.status, TrainingStatus.Stalled)
  }

  test("interrupt decision defaults to save best in non-interactive mode") {
    val got = Trainer.resolveInterruptDecision(interactive = false, _ => "d")
    assertEquals(got, SaveDecision.SaveBest)
  }

  test("interrupt decision loops until valid input in interactive mode") {
    val inputs = scala.collection.mutable.Queue("x", "current")
    val got = Trainer.resolveInterruptDecision(interactive = true, _ => inputs.dequeue())
    assertEquals(got, SaveDecision.SaveCurrent)
  }
