package train

import nn.{ModelConfig, Params}
import linalg.Matrix

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}

object CheckpointIO:

  def save(params: Params, config: ModelConfig, path: Path): Unit =
    val sb = new StringBuilder
    sb.append(s"config.contextSize=${config.contextSize}\n")
    sb.append(s"config.embedDim=${config.embedDim}\n")
    sb.append(s"config.hiddenDim=${config.hiddenDim}\n")
    sb.append(s"config.vocabSize=${config.vocabSize}\n")

    def appendMatrix(name: String, m: Matrix): Unit =
      sb.append(s"$name.rows=${m.rows}\n")
      sb.append(s"$name.cols=${m.cols}\n")
      sb.append(s"$name.data=${m.data.mkString(",")}\n")

    def appendVec(name: String, v: Vector[Double]): Unit =
      sb.append(s"$name.size=${v.length}\n")
      sb.append(s"$name.data=${v.mkString(",")}\n")

    appendMatrix("E", params.E)
    appendMatrix("W1", params.W1)
    appendVec("b1", params.b1)
    appendMatrix("W2", params.W2)
    appendVec("b2", params.b2)

    Files.write(path, sb.result().getBytes(StandardCharsets.UTF_8))

  def load(path: Path): (Params, ModelConfig) =
    val raw = Files.readAllLines(path, StandardCharsets.UTF_8)
    val entries = raw.toArray(new Array[String](raw.size())).toVector
      .filter(_.contains("="))
      .map { line =>
        val Array(k, v) = line.split("=", 2)
        k.trim -> v.trim
      }
      .toMap

    def getInt(key: String): Int = entries.getOrElse(key, throw new IllegalArgumentException(s"Missing key: $key")).toInt
    def getDoubles(key: String): Vector[Double] =
      val rawData = entries.getOrElse(key, throw new IllegalArgumentException(s"Missing key: $key"))
      if rawData.isEmpty then Vector.empty
      else rawData.split(",").toVector.map(_.toDouble)

    def parseMatrix(name: String): Matrix =
      val rows = getInt(s"$name.rows")
      val cols = getInt(s"$name.cols")
      val data = getDoubles(s"$name.data")
      Matrix(data, rows, cols)

    def parseVec(name: String): Vector[Double] =
      val size = getInt(s"$name.size")
      val data = getDoubles(s"$name.data")
      require(data.length == size, s"$name size mismatch: expected $size got ${data.length}")
      data

    val cfg = ModelConfig(
      contextSize = getInt("config.contextSize"),
      embedDim = getInt("config.embedDim"),
      hiddenDim = getInt("config.hiddenDim"),
      vocabSize = getInt("config.vocabSize")
    )

    val params = Params(
      E = parseMatrix("E"),
      W1 = parseMatrix("W1"),
      b1 = parseVec("b1"),
      W2 = parseMatrix("W2"),
      b2 = parseVec("b2")
    )

    (params, cfg)
