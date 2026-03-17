package data

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}

object VocabIO:
  def save(vocab: Vocab, path: Path): Unit =
    val lines = vocab.idToToken.zipWithIndex.sortBy(_._2).map(_._1)
    Files.write(path, lines.mkString("\n").getBytes(StandardCharsets.UTF_8))

  def load(path: Path, unkToken: String = "<UNK>"): Vocab =
    val lines = Files.readAllLines(path, StandardCharsets.UTF_8)
    val tokens = lines.toArray(new Array[String](lines.size())).toVector
    val withUnk = if tokens.contains(unkToken) then tokens else unkToken +: tokens
    val deduped = withUnk.distinct
    val tokenToId = deduped.zipWithIndex.toMap
    Vocab(tokenToId = tokenToId, idToToken = deduped, unkToken = unkToken)
