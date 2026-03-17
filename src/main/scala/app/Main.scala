package app

import data.{TextPipeline, VocabIO}
import eval.Metrics
import linalg.LinearAlgebra
import nn.{LanguageModel, ModelConfig}
import train.{CheckpointIO, TrainConfig, Trainer}

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}

object Main:

  def main(args: Array[String]): Unit =
    if args.isEmpty then
      printUsage()
      sys.exit(1)

    args(0) match
      case "train"   => runTrain(parseArgs(args.drop(1)))
      case "predict" => runPredict(parseArgs(args.drop(1)))
      case "test"    => TestRunner.main(Array.empty)
      case _ =>
        printUsage()
        sys.exit(1)

  private def runTrain(flags: Map[String, String]): Unit =
    val inputPath = Path.of(required(flags, "input"))
    val modelPath = Path.of(flags.getOrElse("model", "model.ckpt"))
    val vocabPath = Path.of(flags.getOrElse("vocab", "vocab.txt"))

    val contextSize = flags.get("contextSize").map(_.toInt).getOrElse(3)
    val embedDim = flags.get("embedDim").map(_.toInt).getOrElse(24)
    val hiddenDim = flags.get("hiddenDim").map(_.toInt).getOrElse(64)
    val maxVocab = flags.get("maxVocab").map(_.toInt).getOrElse(3000)
    val epochs = flags.get("epochs").map(_.toInt).getOrElse(10)
    val lr = flags.get("lr").map(_.toDouble).getOrElse(0.05)
    val lrDecay = flags.get("lrDecay").map(_.toDouble).getOrElse(1.0)
    val l2 = flags.get("l2").map(_.toDouble).getOrElse(0.0)
    val clipNorm = flags.get("clipNorm").map(_.toDouble)
    val seed = flags.get("seed").map(_.toInt).getOrElse(42)
    val trainRatio = flags.get("trainRatio").map(_.toDouble).getOrElse(0.9)

    val rawText = Files.readString(inputPath, StandardCharsets.UTF_8)
    val tokens = TextPipeline.tokenize(rawText)
    val vocab = TextPipeline.buildVocab(tokens, maxVocab)
    val ids = TextPipeline.tokensToIds(tokens, vocab)
    val examples = TextPipeline.buildExamples(ids, contextSize)
    val (trainSet, valSet) = TextPipeline.splitDeterministic(examples, trainRatio, seed)

    require(trainSet.nonEmpty, "Training set is empty. Provide more text or reduce contextSize.")

    val cfg = ModelConfig(contextSize = contextSize, embedDim = embedDim, hiddenDim = hiddenDim, vocabSize = vocab.size)
    val params0 = LanguageModel.initParams(cfg, seed)

    val trainCfg = TrainConfig(
      epochs = epochs,
      learningRate = lr,
      lrDecay = lrDecay,
      l2 = l2,
      clipNorm = clipNorm,
      shuffleEachEpoch = true,
      seed = seed
    )

    val result = Trainer.train(params0, trainSet, valSet, trainCfg)
    result.history.foreach { m =>
      println(f"epoch=${m.epoch}%d lr=${m.learningRate}%.5f train_loss=${m.trainLoss}%.6f val_loss=${m.valLoss}%.6f val_ppl=${m.valPerplexity}%.4f")
    }

    CheckpointIO.save(result.params, cfg, modelPath)
    VocabIO.save(vocab, vocabPath)

    val finalValLoss = Metrics.meanLoss(result.params, valSet)
    val finalPpl = Metrics.perplexity(finalValLoss)
    println(f"saved model=${modelPath.toString} vocab=${vocabPath.toString} final_val_loss=${finalValLoss}%.6f final_val_ppl=${finalPpl}%.4f")

  private def runPredict(flags: Map[String, String]): Unit =
    val modelPath = Path.of(required(flags, "model"))
    val vocabPath = Path.of(required(flags, "vocab"))
    val contextText = required(flags, "context")
    val topK = flags.get("topK").map(_.toInt).getOrElse(5)

    val (params, cfg) = CheckpointIO.load(modelPath)
    val vocab = VocabIO.load(vocabPath)

    require(vocab.size == cfg.vocabSize, s"vocab size ${vocab.size} does not match model vocab size ${cfg.vocabSize}")

    val contextTokens = TextPipeline.tokenize(contextText)
    val contextIds = adaptContext(contextTokens.map(vocab.toId), cfg.contextSize, vocab.unkId)

    val cache = LanguageModel.forward(params, contextIds)
    val top = LinearAlgebra.argTopK(cache.probs, topK)

    println(s"context_ids=${contextIds.mkString("[", ",", "]")}")
    top.foreach { case (id, prob) =>
      println(f"${vocab.toToken(id)}%-20s id=${id}%d prob=${prob}%.6f")
    }

  private def adaptContext(ids: Vector[Int], contextSize: Int, padId: Int): Vector[Int] =
    if ids.length >= contextSize then ids.takeRight(contextSize)
    else Vector.fill(contextSize - ids.length)(padId) ++ ids

  private def parseArgs(args: Array[String]): Map[String, String] =
    args.toVector
      .sliding(2, 2)
      .collect {
        case Vector(k, v) if k.startsWith("--") => k.stripPrefix("--") -> v
      }
      .toMap

  private def required(flags: Map[String, String], key: String): String =
    flags.getOrElse(key, throw new IllegalArgumentException(s"Missing required --$key"))

  private def printUsage(): Unit =
    println(
      """Usage:
        |  sbt \"run train --input <text.txt> [--model model.ckpt] [--vocab vocab.txt] [--contextSize 3] [--embedDim 24] [--hiddenDim 64] [--maxVocab 3000] [--epochs 10] [--lr 0.05] [--lrDecay 1.0] [--l2 0.0] [--clipNorm 1.0] [--seed 42] [--trainRatio 0.9]\"
        |  sbt \"run predict --model <model.ckpt> --vocab <vocab.txt> --context 'your context words' [--topK 5]\"
        |  sbt \"run test\"
        |""".stripMargin
    )
