package train

import munit.FunSuite

class TrainingDisplaySuite extends FunSuite:
  test("ansi display updates the current line in-place and finalizes one epoch line") {
    val writes = new StringBuilder
    val logs = scala.collection.mutable.ArrayBuffer.empty[String]
    val d = AnsiEpochBoardDisplay(write = s => writes.append(s), log = s => logs += s)

    d.onEpochStart(epoch = 1, totalEpochs = 3, lr = 0.02, totalExamples = 100)
    d.onBatchProgress(BatchProgress(epoch = 1, totalEpochs = 3, percent = 50, elapsedSec = 2.0, remainingSec = 2.0, examplesPerSec = 25.0, avgLoss = 1.23))
    d.onEpochComplete(
      EpochMetrics(
        epoch = 1,
        trainLoss = 1.2,
        valLoss = 1.1,
        valPerplexity = 3.0,
        learningRate = 0.02,
        epochSeconds = 4.0,
        status = TrainingStatus.Improving,
        statusReason = "better val",
        bestDeltaPct = 1.1,
        generalizationGap = -0.09
      ),
      isBest = true,
      patienceCounter = 0,
      patience = 2
    )

    val out = writes.toString
    assert(out.contains("\u001b[2K"))
    assert(out.contains("Epoch  1/ 3"))
    assert(out.contains("Epoch  1 done"))
    assert(out.endsWith("\n"))
    assertEquals(logs.size, 0)
  }

  test("plain display emits simple single-pass log lines without ANSI codes") {
    val logs = scala.collection.mutable.ArrayBuffer.empty[String]
    val d = PlainLogDisplay(log = s => logs += s)
    d.onEpochStart(epoch = 2, totalEpochs = 4, lr = 0.01, totalExamples = 200)
    d.onBatchProgress(BatchProgress(epoch = 2, totalEpochs = 4, percent = 25, elapsedSec = 1.0, remainingSec = 3.0, examplesPerSec = 50.0, avgLoss = 0.99))
    d.onTrainingComplete(interrupted = false)

    val merged = logs.mkString("\n")
    assert(merged.contains("Epoch 2/4 starting"))
    assert(merged.contains("25%"))
    assert(merged.contains("Training complete!"))
    assert(!merged.contains("\u001b["))
  }
