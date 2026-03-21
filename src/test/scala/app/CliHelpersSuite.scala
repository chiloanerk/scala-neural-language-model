package app

import munit.FunSuite
import java.nio.file.Path

class CliHelpersSuite extends FunSuite:
  test("trimOrEmpty handles null and whitespace") {
    assertEquals(CliHelpers.trimOrEmpty(null), "")
    assertEquals(CliHelpers.trimOrEmpty("  hi  "), "hi")
  }

  test("isTruthy handles common true values") {
    assert(CliHelpers.isTruthy("true"))
    assert(CliHelpers.isTruthy("YES"))
    assert(CliHelpers.isTruthy("1"))
    assert(!CliHelpers.isTruthy(null))
    assert(!CliHelpers.isTruthy("no"))
  }

  test("parseMenuChoice uses default when empty") {
    val got = CliHelpers.parseMenuChoice("", optionCount = 3, defaultIndex = 1)
    assertEquals(got, Some(1))
  }

  test("parseMenuChoice maps 1-based input to index") {
    val got = CliHelpers.parseMenuChoice("2", optionCount = 3, defaultIndex = 0)
    assertEquals(got, Some(1))
  }

  test("parseMenuChoice returns None for out-of-range") {
    val got = CliHelpers.parseMenuChoice("9", optionCount = 2, defaultIndex = 0)
    assertEquals(got, None)
  }

  test("parseYesNo handles explicit values and default fallback") {
    assert(CliHelpers.parseYesNo("y", default = false))
    assert(!CliHelpers.parseYesNo("n", default = true))
    assert(CliHelpers.parseYesNo("", default = true))
    assert(!CliHelpers.parseYesNo("", default = false))
    assert(CliHelpers.parseYesNo("maybe", default = true))
  }

  test("parseArgs supports boolean flags") {
    val args = Array("--yes", "--fresh", "--gpuInfo")
    val got = CliHelpers.parseArgs(args)
    assertEquals(got.get("yes"), Some("true"))
    assertEquals(got.get("fresh"), Some("true"))
    assertEquals(got.get("gpuInfo"), Some("true"))
  }

  test("parseArgs supports mixed key/value and boolean flags") {
    val args = Array("--input", "data.txt", "--preset", "quick", "--batchSize", "128", "--yes")
    val got = CliHelpers.parseArgs(args)
    assertEquals(got.get("input"), Some("data.txt"))
    assertEquals(got.get("preset"), Some("quick"))
    assertEquals(got.get("batchSize"), Some("128"))
    assertEquals(got.get("yes"), Some("true"))
  }

  test("parseArgs supports replay and multi-input flags") {
    val args = Array(
      "--inputs",
      "a.txt,b.txt",
      "--inputWeights",
      "0.7,0.3",
      "--replayRatio",
      "0.3",
      "--replayBufferSize",
      "5000"
    )
    val got = CliHelpers.parseArgs(args)
    assertEquals(got.get("inputs"), Some("a.txt,b.txt"))
    assertEquals(got.get("inputWeights"), Some("0.7,0.3"))
    assertEquals(got.get("replayRatio"), Some("0.3"))
    assertEquals(got.get("replayBufferSize"), Some("5000"))
  }

  test("parseArgs uses last-wins for repeated flags") {
    val args = Array("--yes", "false", "--yes", "--backend", "cpu", "--backend", "gpu")
    val got = CliHelpers.parseArgs(args)
    assertEquals(got.get("yes"), Some("true"))
    assertEquals(got.get("backend"), Some("gpu"))
  }

  test("parseArgs ignores stray tokens and keeps valid flags") {
    val args = Array("train", "--input", "a.txt", "junk", "--yes")
    val got = CliHelpers.parseArgs(args)
    assertEquals(got.get("input"), Some("a.txt"))
    assertEquals(got.get("yes"), Some("true"))
    assert(!got.contains("train"))
    assert(!got.contains("junk"))
  }

  test("resolveConfirmation handles auto confirm and EOF safely") {
    assert(CliHelpers.resolveConfirmation(null, default = true, autoConfirm = true))
    assert(!CliHelpers.resolveConfirmation(null, default = true, autoConfirm = false))
    assert(!CliHelpers.resolveConfirmation("n", default = true, autoConfirm = false))
    assert(CliHelpers.resolveConfirmation("y", default = false, autoConfirm = false))
  }

  test("looksLikeDerivedTextFile detects vocab and chunks") {
    assert(CliHelpers.looksLikeDerivedTextFile(Path.of("vocab.txt")))
    assert(CliHelpers.looksLikeDerivedTextFile(Path.of("train-part1.txt")))
    assert(!CliHelpers.looksLikeDerivedTextFile(Path.of("novel.txt")))
  }

  test("classifyTrainingFiles separates recommended and derived") {
    val files = Vector(Path.of("story.txt"), Path.of("vocab.txt"), Path.of("news.txt"))
    val (recommended, other) = CliHelpers.classifyTrainingFiles(files)
    assertEquals(recommended.map(_.toString), Vector("story.txt", "news.txt"))
    assertEquals(other.map(_.toString), Vector("vocab.txt"))
  }

  test("recommendChunkSize follows thresholds") {
    assertEquals(CliHelpers.recommendChunkSize(100), 500)
    assertEquals(CliHelpers.recommendChunkSize(7000), 1000)
    assertEquals(CliHelpers.recommendChunkSize(20000), 2000)
  }

  test("boundedTopK clamps bounds and defaults invalid") {
    assertEquals(CliHelpers.boundedTopK(0), 5)
    assertEquals(CliHelpers.boundedTopK(3), 3)
    assertEquals(CliHelpers.boundedTopK(999), 50)
  }
