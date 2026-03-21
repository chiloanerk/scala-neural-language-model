package data

import munit.FunSuite
import java.nio.file.Files

class VocabIOSuite extends FunSuite:
  test("save/load vocab roundtrip") {
    val vocab = TextPipeline.buildVocab(Vector("alpha", "beta", "alpha"), maxVocab = 10)
    val tmp = Files.createTempFile("vocab-io", ".txt")
    try
      VocabIO.save(vocab, tmp)
      val loaded = VocabIO.load(tmp)
      assertEquals(loaded.idToToken, vocab.idToToken)
      assertEquals(loaded.tokenToId, vocab.tokenToId)
    finally
      Files.deleteIfExists(tmp)
  }
