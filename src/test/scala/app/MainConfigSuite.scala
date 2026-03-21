package app

import munit.FunSuite
import train.TrainConfig

class MainConfigSuite extends FunSuite:
  test("training presets use conservative learning rates for longer runs") {
    val byName = Main.presets.map(p => p.name -> p).toMap
    assertEqualsDouble(byName("quick").learningRate, 0.05, 1e-12)
    assertEqualsDouble(byName("balanced").learningRate, 0.02, 1e-12)
    assertEqualsDouble(byName("thorough").learningRate, 0.01, 1e-12)
  }

  test("TrainConfig default precision is fp64") {
    val cfg = TrainConfig()
    assertEquals(cfg.precision, "fp64")
    assertEquals(cfg.replayRatio, 0.0)
    assertEquals(cfg.replayBufferSize, 0)
    assertEquals(cfg.ewcLambda, 0.0)
  }

  test("benchmarkMatrix defaults to cpu/gpu x fp64/fp32") {
    val combos = Main.benchmarkMatrix(None, None).toSet
    assertEquals(
      combos,
      Set(
        ("cpu", "fp64"),
        ("cpu", "fp32"),
        ("gpu", "fp64"),
        ("gpu", "fp32")
      )
    )
  }

  test("benchmarkMatrix respects explicit backend and precision") {
    val combos = Main.benchmarkMatrix(Some("gpu"), Some("fp32"))
    assertEquals(combos, Vector(("gpu", "fp32")))
  }

  test("utf8CopyPath inserts .utf8 before extension") {
    val got = Main.utf8CopyPath(java.nio.file.Path.of("data/corpus/bbc-all.txt"))
    assertEquals(got.toString, "data/corpus/bbc-all.utf8.txt")
  }

  test("utf8CopyPath appends .utf8.txt when no extension") {
    val got = Main.utf8CopyPath(java.nio.file.Path.of("data/corpus/bbc-all"))
    assertEquals(got.toString, "data/corpus/bbc-all.utf8.txt")
  }
