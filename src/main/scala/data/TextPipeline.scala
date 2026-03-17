package data

import scala.util.Random

final case class Example(context: Vector[Int], target: Int)

final case class Vocab(tokenToId: Map[String, Int], idToToken: Vector[String], unkToken: String = "<UNK>"):
  def size: Int = idToToken.length
  def unkId: Int = tokenToId(unkToken)

  def toId(token: String): Int = tokenToId.getOrElse(token, unkId)
  def toToken(id: Int): String =
    if id >= 0 && id < idToToken.length then idToToken(id) else unkToken

object TextPipeline:
  private val splitRegex = "[^a-z0-9']+".r

  def tokenize(text: String): Vector[String] =
    splitRegex
      .split(text.toLowerCase)
      .toVector
      .map(_.trim)
      .filter(_.nonEmpty)

  def buildVocab(tokens: Vector[String], maxVocab: Int, unkToken: String = "<UNK>"): Vocab =
    require(maxVocab >= 2, s"maxVocab must be >= 2, got $maxVocab")

    val freq = tokens.groupMapReduce(identity)(_ => 1)(_ + _)
    val sorted = freq.toVector.sortBy { case (tok, count) => (-count, tok) }
    val selected = sorted.take(maxVocab - 1).map(_._1)
    val idToToken = (unkToken +: selected).distinct.toVector
    val tokenToId = idToToken.zipWithIndex.map { case (t, idx) => t -> idx }.toMap

    Vocab(tokenToId = tokenToId, idToToken = idToToken, unkToken = unkToken)

  def tokensToIds(tokens: Vector[String], vocab: Vocab): Vector[Int] =
    tokens.map(vocab.toId)

  def buildExamples(ids: Vector[Int], contextSize: Int): Vector[Example] =
    require(contextSize >= 1, s"contextSize must be >= 1, got $contextSize")
    if ids.length <= contextSize then Vector.empty
    else
      Vector.tabulate(ids.length - contextSize) { i =>
        val ctx = ids.slice(i, i + contextSize)
        val tgt = ids(i + contextSize)
        Example(ctx, tgt)
      }

  def splitDeterministic[A](items: Vector[A], trainRatio: Double, seed: Int): (Vector[A], Vector[A]) =
    require(trainRatio > 0.0 && trainRatio < 1.0, s"trainRatio must be in (0,1), got $trainRatio")
    val rnd = Random(seed)
    val shuffled = rnd.shuffle(items)
    val trainSize = math.max(1, (shuffled.length * trainRatio).toInt)
    (shuffled.take(trainSize), shuffled.drop(trainSize))
