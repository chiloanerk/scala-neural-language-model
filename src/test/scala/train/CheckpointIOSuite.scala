package train

import munit.FunSuite
import nn.{LanguageModel, ModelConfig}
import java.nio.file.Files

class CheckpointIOSuite extends FunSuite:
  test("save/load checkpoint roundtrip") {
    val cfg = ModelConfig(contextSize = 2, embedDim = 4, hiddenDim = 5, vocabSize = 6, activation = "tanh")
    val params = LanguageModel.initParams(cfg, seed = 1)

    val tmp = Files.createTempFile("model", ".ckpt")
    try
      CheckpointIO.save(params, cfg, tmp)
      val (loadedParams, loadedCfg) = CheckpointIO.load(tmp)

      assertEquals(loadedCfg.contextSize, cfg.contextSize)
      assertEquals(loadedCfg.embedDim, cfg.embedDim)
      assertEquals(loadedCfg.hiddenDim, cfg.hiddenDim)
      assertEquals(loadedCfg.vocabSize, cfg.vocabSize)

      assertEquals(loadedParams.E.rows, params.E.rows)
      assertEquals(loadedParams.W1.cols, params.W1.cols)
      assertEquals(loadedParams.b2.length, params.b2.length)
    finally
      Files.deleteIfExists(tmp)
  }
