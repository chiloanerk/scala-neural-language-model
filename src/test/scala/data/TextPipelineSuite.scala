package data

import munit.FunSuite

class TextPipelineSuite extends FunSuite:
  test("tokenize lowercases and splits") {
    val t = TextPipeline.tokenize("Hello, WORLD! It's me.")
    assertEquals(t, Vector("hello", "world", "it's", "me"))
  }

  test("buildVocab keeps unk and max size") {
    val tokens = Vector("a", "b", "a", "c")
    val vocab = TextPipeline.buildVocab(tokens, maxVocab = 3)
    assertEquals(vocab.idToToken.head, "<UNK>")
    assertEquals(vocab.size, 3)
  }

  test("vocab toId and toToken") {
    val vocab = TextPipeline.buildVocab(Vector("cat", "dog"), maxVocab = 5)
    val catId = vocab.toId("cat")
    assertEquals(vocab.toToken(catId), "cat")
    assertEquals(vocab.toId("missing"), vocab.unkId)
  }

  test("tokensToIds and buildExamples") {
    val vocab = TextPipeline.buildVocab(Vector("a", "b", "c", "d"), 10)
    val ids = TextPipeline.tokensToIds(Vector("a", "b", "c", "d"), vocab)
    val ex = TextPipeline.buildExamples(ids, contextSize = 2)
    assertEquals(ex.length, 2)
    assertEquals(ex.head.context.length, 2)
  }

  test("splitDeterministic is stable by seed") {
    val items = Vector.range(0, 20)
    val a = TextPipeline.splitDeterministic(items, 0.8, 42)
    val b = TextPipeline.splitDeterministic(items, 0.8, 42)
    assertEquals(a, b)
  }
