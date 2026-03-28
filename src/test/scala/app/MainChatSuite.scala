package app

import munit.FunSuite

import scala.util.Random

class MainChatSuite extends FunSuite:
  test("isExitCommand recognizes chat/predict exit aliases") {
    assert(Main.isExitCommand("quit"))
    assert(Main.isExitCommand("EXIT"))
    assert(Main.isExitCommand("  Quit  "))
    assert(!Main.isExitCommand("continue"))
  }

  test("chatDecodeCandidates applies topK and topP and renormalizes") {
    val probs = Vector(0.4, 0.3, 0.2, 0.1)
    val got = Main.chatDecodeCandidates(probs, temperature = 1.0, topP = 0.7, topK = 3)

    assertEquals(got.map(_._1), Vector(0, 1))
    val sum = got.map(_._2).sum
    assertEqualsDouble(sum, 1.0, 1e-9)
    assert(got.forall { case (_, p) => p > 0.0 })
  }

  test("chatDecodeCandidates falls back safely for degenerate distributions") {
    val got = Main.chatDecodeCandidates(Vector(0.0, 0.0, 0.0), temperature = 1.0, topP = 0.9, topK = 40)
    assertEquals(got.length, 1)
    assertEquals(got.head._2, 1.0)
  }

  test("chatSampleToken is deterministic for one-hot candidates") {
    val rnd = Random(123)
    val candidates = Vector((5, 1.0))
    val draws = Vector.fill(10)(Main.chatSampleToken(candidates, rnd))
    assertEquals(draws.distinct, Vector(5))
  }

  test("suppressUnkProbability zeros unk when alternatives exist") {
    val probs = Vector(0.2, 0.5, 0.3)
    val got = Main.suppressUnkProbability(probs, unkId = 0, banUnk = true)
    assertEqualsDouble(got(0), 0.0, 1e-12)
    assertEqualsDouble(got(1), 0.5, 1e-12)
    assertEqualsDouble(got(2), 0.3, 1e-12)
  }

  test("suppressUnkProbability keeps unk when it is the only mass") {
    val probs = Vector(1.0, 0.0, 0.0)
    val got = Main.suppressUnkProbability(probs, unkId = 0, banUnk = true)
    assertEquals(got, probs)
  }

  test("predictionTopForDisplay suppresses unk in non-debug mode") {
    val probs = Vector(0.6, 0.3, 0.1) // unk at 0
    val got = Main.predictionTopForDisplay(probs, topK = 2, unkId = 0, debugMode = false)
    assertEquals(got.map(_._1), Vector(1, 2))
  }

  test("predictionTopForDisplay keeps unk in debug mode") {
    val probs = Vector(0.6, 0.3, 0.1) // unk at 0
    val got = Main.predictionTopForDisplay(probs, topK = 2, unkId = 0, debugMode = true)
    assertEquals(got.map(_._1), Vector(0, 1))
  }
