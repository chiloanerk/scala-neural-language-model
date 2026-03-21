package train

import data.Example
import munit.FunSuite

import java.nio.file.Files

class ReplayBufferSuite extends FunSuite:

  test("save/load replay buffer roundtrip with metadata compatibility") {
    val tmp = Files.createTempFile("replay-buffer", ".replay")
    try
      val buffer0 = ReplayBuffer.initialize(contextSize = 3, vocabSize = 10, vocabHash = "abc123")
      val buffer = buffer0
        .add(Vector(Example(Vector(1, 2, 3), 4)), domain = "a.txt", capacity = 10)
        .add(Vector(Example(Vector(2, 3, 4), 5)), domain = "b.txt", capacity = 10)
      buffer.save(tmp)

      val loaded = ReplayBuffer.load(tmp, ReplayBuffer.Expected(contextSize = 3, vocabSize = 10, vocabHash = "abc123"))
      assertEquals(loaded.examples, buffer.examples)
      assertEquals(loaded.domainLabels.toSet, Set("a.txt", "b.txt"))
    finally
      Files.deleteIfExists(tmp)
  }

  test("load fails on metadata mismatch") {
    val tmp = Files.createTempFile("replay-buffer-mismatch", ".replay")
    try
      val buffer = ReplayBuffer.initialize(contextSize = 3, vocabSize = 10, vocabHash = "abc123")
        .add(Vector(Example(Vector(1, 2, 3), 4)), domain = "a.txt", capacity = 10)
      buffer.save(tmp)

      intercept[IllegalArgumentException] {
        ReplayBuffer.load(tmp, ReplayBuffer.Expected(contextSize = 4, vocabSize = 10, vocabHash = "abc123"))
      }
    finally
      Files.deleteIfExists(tmp)
  }

  test("sample is deterministic for the same seed") {
    val buffer = ReplayBuffer.initialize(contextSize = 2, vocabSize = 20, vocabHash = "v")
      .add(
        Vector(
          Example(Vector(1, 2), 3),
          Example(Vector(2, 3), 4),
          Example(Vector(3, 4), 5)
        ),
        domain = "a.txt",
        capacity = 50
      )

    val a = buffer.sample(5, seed = 99)
    val b = buffer.sample(5, seed = 99)
    assertEquals(a, b)
  }
