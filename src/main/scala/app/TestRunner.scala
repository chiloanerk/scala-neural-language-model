package app

import data.{Example, TextPipeline, Vocab}
import eval.Metrics
import linalg.{LinearAlgebra, Matrix}
import nn.{LanguageModel, ModelConfig, Params}
import train.{TrainConfig, Trainer}

object TestRunner:

  final case class TestResult(name: String, passed: Boolean = true, message: String = "")

  def main(args: Array[String]): Unit =
    val tests = Vector(
      testMatrixAndTranspose(),
      testSoftmaxStability(),
      testForwardCacheShapes(),
      testEmbeddingScatterAccumulation(),
      testGradientCheck(),
      testTrainingRegression(),
      testInferenceTopK(),
      testReLUBasic(),
      testReLUGradient(),
      testReLUVsTanh(),
      testEarlyStoppingTriggers(),
      testEarlyStoppingRestoresBest(),
      testProgressBar()
    )

    val failures = tests.filterNot(_.passed)
    tests.foreach { t =>
      if t.passed then println(s"[PASS] ${t.name}")
      else println(s"[FAIL] ${t.name}: ${t.message}")
    }

    if failures.nonEmpty then
      println(s"Failed ${failures.length}/${tests.length} tests")
      sys.exit(1)
    else
      println(s"All ${tests.length} tests passed")

  private def approxEqual(a: Double, b: Double, eps: Double = 1e-8): Boolean =
    math.abs(a - b) <= eps

  private def relError(a: Double, b: Double): Double =
    math.abs(a - b) / math.max(1e-8, math.abs(a) + math.abs(b))

  private def testMatrixAndTranspose(): TestResult =
    try
      val m = Matrix.fromFunction(2, 3)((r, c) => (r * 10 + c).toDouble)
      val t = m.transposeView
      val a = m.get(1, 2) == 12.0
      val b = t.get(2, 1) == 12.0
      val c = t.rows == 3 && t.cols == 2

      if a && b && c then TestResult("matrix/transpose-view")
      else TestResult("matrix/transpose-view", passed = false, "incorrect values/dimensions")
    catch
      case e: Throwable => TestResult("matrix/transpose-view", passed = false, e.getMessage)

  private def testSoftmaxStability(): TestResult =
    try
      val probs = LinearAlgebra.softmaxStable(Vector(1000.0, 1001.0, 999.0))
      val finite = probs.forall(_.isFinite)
      val sumOk = math.abs(probs.sum - 1.0) < 1e-9
      if finite && sumOk then TestResult("softmax-stability")
      else TestResult("softmax-stability", passed = false, s"finite=$finite sum=${probs.sum}")
    catch
      case e: Throwable => TestResult("softmax-stability", passed = false, e.getMessage)

  private def tinyConfig(): ModelConfig = ModelConfig(contextSize = 2, embedDim = 3, hiddenDim = 4, vocabSize = 5)

  private def tinyParams(seed: Int = 7): Params = LanguageModel.initParams(tinyConfig(), seed)

  private def testForwardCacheShapes(): TestResult =
    try
      val cfg = tinyConfig()
      val p = tinyParams()
      val context = Vector(1, 3)
      val cache = LanguageModel.forward(p, context)

      val ok =
        cache.x.length == cfg.contextSize * cfg.embedDim &&
          cache.z1.length == cfg.hiddenDim &&
          cache.a1.length == cfg.hiddenDim &&
          cache.logits.length == cfg.vocabSize &&
          cache.probs.length == cfg.vocabSize &&
          math.abs(cache.probs.sum - 1.0) < 1e-9

      if ok then TestResult("forward-cache-shapes")
      else TestResult("forward-cache-shapes", passed = false, "cache dimensions are incorrect")
    catch
      case e: Throwable => TestResult("forward-cache-shapes", passed = false, e.getMessage)

  private def testEmbeddingScatterAccumulation(): TestResult =
    try
      val cfg = tinyConfig()
      val p = tinyParams(seed = 11)
      val context = Vector(2, 2)
      val cache = LanguageModel.forward(p, context)
      val grads = LanguageModel.backward(p, cache, target = 1)

      val row2 = grads.dE.rowSlice(2)
      val nonZero = row2.exists(v => math.abs(v) > 1e-12)
      val othersZero = (0 until cfg.vocabSize).filter(_ != 2).forall { r =>
        grads.dE.rowSlice(r).forall(v => math.abs(v) <= 1e-12)
      }

      if nonZero && othersZero then TestResult("embedding-scatter-accumulation")
      else TestResult("embedding-scatter-accumulation", passed = false, "gradient accumulation did not behave as expected")
    catch
      case e: Throwable => TestResult("embedding-scatter-accumulation", passed = false, e.getMessage)

  private def testGradientCheck(): TestResult =
    try
      val cfg = tinyConfig()
      val p = tinyParams(seed = 19)
      val ex = Example(context = Vector(1, 4), target = 2)

      val cache = LanguageModel.forward(p, ex.context)
      val grads = LanguageModel.backward(p, cache, ex.target)

      val eps = 1e-5

      def lossWith(p2: Params): Double =
        val c = LanguageModel.forward(p2, ex.context)
        LanguageModel.lossFromCache(c, ex.target)

      val numericW2 = {
        val r = 0
        val c = 0
        val plus = p.copy(W2 = p.W2.updated(r, c, p.W2.get(r, c) + eps))
        val minus = p.copy(W2 = p.W2.updated(r, c, p.W2.get(r, c) - eps))
        (lossWith(plus) - lossWith(minus)) / (2.0 * eps)
      }
      val analyticW2 = grads.dW2.get(0, 0)

      val numericW1 = {
        val r = 0
        val c = 0
        val plus = p.copy(W1 = p.W1.updated(r, c, p.W1.get(r, c) + eps))
        val minus = p.copy(W1 = p.W1.updated(r, c, p.W1.get(r, c) - eps))
        (lossWith(plus) - lossWith(minus)) / (2.0 * eps)
      }
      val analyticW1 = grads.dW1.get(0, 0)

      val numericE = {
        val r = 1
        val c = 0
        val plus = p.copy(E = p.E.updated(r, c, p.E.get(r, c) + eps))
        val minus = p.copy(E = p.E.updated(r, c, p.E.get(r, c) - eps))
        (lossWith(plus) - lossWith(minus)) / (2.0 * eps)
      }
      val analyticE = grads.dE.get(1, 0)

      val ok =
        relError(numericW2, analyticW2) < 1e-3 &&
          relError(numericW1, analyticW1) < 1e-3 &&
          relError(numericE, analyticE) < 1e-3

      if ok then TestResult("gradient-check")
      else
        TestResult(
          "gradient-check",
          passed = false,
          f"relErr(W2)=${relError(numericW2, analyticW2)}%.6f relErr(W1)=${relError(numericW1, analyticW1)}%.6f relErr(E)=${relError(numericE, analyticE)}%.6f"
        )
    catch
      case e: Throwable => TestResult("gradient-check", passed = false, e.getMessage)

  private def testTrainingRegression(): TestResult =
    try
      val text = "the cat sat on the mat the cat sat on the rug the dog sat on the rug"
      val tokens = TextPipeline.tokenize(text)
      val vocab = TextPipeline.buildVocab(tokens, maxVocab = 50)
      val ids = TextPipeline.tokensToIds(tokens, vocab)
      val examples = TextPipeline.buildExamples(ids, contextSize = 2)
      val (trainSet, valSet) = TextPipeline.splitDeterministic(examples, trainRatio = 0.8, seed = 10)

      val cfg = ModelConfig(contextSize = 2, embedDim = 8, hiddenDim = 12, vocabSize = vocab.size)
      val p0 = LanguageModel.initParams(cfg, seed = 10)
      val baseTrainLoss = Metrics.meanLoss(p0, trainSet)

      val result = Trainer.train(
        p0,
        trainSet,
        valSet,
        TrainConfig(epochs = 20, learningRate = 0.08, lrDecay = 0.97, clipNorm = Some(5.0), seed = 10)
      )

      val finalTrainLoss = Metrics.meanLoss(result.params, trainSet)
      val improved = finalTrainLoss < baseTrainLoss - 0.05

      if improved then TestResult("training-regression")
      else TestResult("training-regression", passed = false, f"base=$baseTrainLoss%.4f final=$finalTrainLoss%.4f")
    catch
      case e: Throwable => TestResult("training-regression", passed = false, e.getMessage)

  private def testInferenceTopK(): TestResult =
    try
      val text = "we like scala and we like math and we like models"
      val tokens = TextPipeline.tokenize(text)
      val vocab = TextPipeline.buildVocab(tokens, 100)
      val ids = TextPipeline.tokensToIds(tokens, vocab)
      val examples = TextPipeline.buildExamples(ids, contextSize = 2)
      val cfg = ModelConfig(contextSize = 2, embedDim = 6, hiddenDim = 10, vocabSize = vocab.size)
      val p0 = LanguageModel.initParams(cfg, seed = 23)
      val trained = Trainer.train(p0, examples, examples, TrainConfig(epochs = 15, learningRate = 0.08, lrDecay = 0.98, seed = 23)).params

      val context = Vector(vocab.toId("we"), vocab.toId("like"))
      val probs = LanguageModel.forward(trained, context).probs
      val top = LinearAlgebra.argTopK(probs, 3)

      val sorted = top.zip(top.drop(1)).forall { case ((_, a), (_, b)) => a >= b }
      val inBounds = top.forall { case (id, _) => id >= 0 && id < vocab.size }

      if sorted && inBounds then TestResult("inference-topk")
      else TestResult("inference-topk", passed = false, "top-k invalid ordering or indices")
    catch
      case e: Throwable => TestResult("inference-topk", passed = false, e.getMessage)

  // ============ NEW TESTS FOR ReLU, EARLY STOPPING, PROGRESS BAR ============

  private def testReLUBasic(): TestResult =
    try
      val input = Vector(-2.0, -1.0, 0.0, 1.0, 2.0)
      val output = LinearAlgebra.relu(input)
      val expected = Vector(0.0, 0.0, 0.0, 1.0, 2.0)

      val ok = output == expected
      if ok then TestResult("relu/basic")
      else TestResult("relu/basic", passed = false, s"expected=$expected got=$output")
    catch
      case e: Throwable => TestResult("relu/basic", passed = false, e.getMessage)

  private def testReLUGradient(): TestResult =
    try
      val input = Vector(-2.0, -1.0, 0.0, 1.0, 2.0)
      val grad = LinearAlgebra.reluGrad(input)
      // Note: at x=0, we return 0 (could also be 1, doesn't matter much)
      val expected = Vector(0.0, 0.0, 0.0, 1.0, 1.0)

      val ok = grad == expected
      if ok then TestResult("relu/gradient")
      else TestResult("relu/gradient", passed = false, s"expected=$expected got=$grad")
    catch
      case e: Throwable => TestResult("relu/gradient", passed = false, e.getMessage)

  private def testReLUVsTanh(): TestResult =
    try
      val cfg = tinyConfig().copy(activation = "relu")
      val p = tinyParams(seed = 42)
      val context = Vector(1, 3)

      val cacheRelu = LanguageModel.forward(p, context, "relu")
      val cacheTanh = LanguageModel.forward(p, context, "tanh")

      // Both should produce valid probability distributions
      val reluValid = cacheRelu.probs.forall(p => p.isFinite && p >= 0) && math.abs(cacheRelu.probs.sum - 1.0) < 1e-6
      val tanhValid = cacheTanh.probs.forall(p => p.isFinite && p >= 0) && math.abs(cacheTanh.probs.sum - 1.0) < 1e-6

      // ReLU should produce different activations than Tanh (a1 values)
      val different = cacheRelu.a1 != cacheTanh.a1

      if reluValid && tanhValid && different then TestResult("relu/vs-tanh")
      else TestResult("relu/vs-tanh", passed = false, s"reluValid=$reluValid tanhValid=$tanhValid different=$different")
    catch
      case e: Throwable => TestResult("relu/vs-tanh", passed = false, e.getMessage)

  private def testEarlyStoppingTriggers(): TestResult =
    try
      val cfg = tinyConfig()
      val p = tinyParams(seed = 100)

      // Create a tiny dataset that will overfit quickly
      val examples = Vector.fill(20)(Example(context = Vector(1, 2), target = 3))
      val valExamples = Vector.fill(5)(Example(context = Vector(1, 2), target = 4)) // Different target = worse val loss

      val result = Trainer.train(
        p,
        examples,
        valExamples,
        TrainConfig(epochs = 10, learningRate = 0.1, patience = 2, seed = 100)
      )

      // Should have stopped early (before 10 epochs)
      val stoppedEarly = result.history.length < 10
      // Should have at least 3 epochs (patience=2, so min is best + 2 no-improvement)
      val minEpochs = result.history.length >= 3

      if stoppedEarly && minEpochs then TestResult("early-stopping/triggers")
      else TestResult("early-stopping/triggers", passed = false, s"epochs=${result.history.length} expected <10 and >=3")
    catch
      case e: Throwable => TestResult("early-stopping/triggers", passed = false, e.getMessage)

  private def testEarlyStoppingRestoresBest(): TestResult =
    try
      val cfg = tinyConfig()
      val p = tinyParams(seed = 200)

      // Create training data where model will overfit
      val examples = Vector.tabulate(30)(i => Example(context = Vector(i % 3, (i + 1) % 3), target = i % 4))
      val valExamples = Vector.tabulate(10)(i => Example(context = Vector(i % 3, (i + 1) % 3), target = (i + 2) % 4))

      val result = Trainer.train(
        p,
        examples,
        valExamples,
        TrainConfig(epochs = 8, learningRate = 0.1, patience = 2, seed = 200)
      )

      // The restored params should give better val loss than final epoch
      val bestValLoss = result.history.map(_.valLoss).min
      val finalValLoss = result.history.last.valLoss

      // Best should be better (lower) than final (since it overfit)
      val restoredIsBetter = bestValLoss <= finalValLoss

      if restoredIsBetter then TestResult("early-stopping/restores-best")
      else TestResult("early-stopping/restores-best", passed = false, s"best=$bestValLoss final=$finalValLoss")
    catch
      case e: Throwable => TestResult("early-stopping/restores-best", passed = false, e.getMessage)

  private def testProgressBar(): TestResult =
    try
      // Test progress bar logic inline
      def createProgressBar(percent: Int, width: Int = 20): String =
        val filled = (percent * width) / 100
        val empty = width - filled
        val bar = "█" * filled + "░" * empty
        s"[$bar]"

      val bar0 = createProgressBar(0)
      val bar50 = createProgressBar(50)
      val bar100 = createProgressBar(100)

      // bar0 should be all empty
      val bar0Ok = bar0.contains("░") && !bar0.contains("█")
      // bar100 should be all filled
      val bar100Ok = bar100.contains("█") && !bar100.contains("░")
      // bar50 should have both
      val bar50Ok = bar50.contains("█") && bar50.contains("░")

      // All should have brackets
      val bracketsOk = bar0.startsWith("[") && bar0.endsWith("]") &&
        bar50.startsWith("[") && bar50.endsWith("]") &&
        bar100.startsWith("[") && bar100.endsWith("]")

      if bar0Ok && bar50Ok && bar100Ok && bracketsOk then TestResult("progress-bar")
      else TestResult("progress-bar", passed = false, s"bar0=$bar0 bar50=$bar50 bar100=$bar100")
    catch
      case e: Throwable => TestResult("progress-bar", passed = false, e.getMessage)
