package train

import data.Example

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}
import scala.util.Random

final case class ReplayBuffer(
    schemaVersion: Int,
    contextSize: Int,
    vocabSize: Int,
    vocabHash: String,
    domainLabels: Vector[String],
    entries: Vector[ReplayBuffer.Entry]
):

  def examples: Vector[Example] = entries.map(_.example)

  def add(examples: Vector[Example], domain: String, capacity: Int): ReplayBuffer =
    if capacity <= 0 || examples.isEmpty then this
    else
      val newEntries = examples.map(ex => ReplayBuffer.Entry(example = ex, domain = domain))
      val merged = entries ++ newEntries
      val trimmed = if merged.length <= capacity then merged else merged.takeRight(capacity)
      val labels = (domainLabels ++ Vector(domain)).distinct
      copy(domainLabels = labels, entries = trimmed)

  def sample(count: Int, seed: Int): Vector[Example] =
    if count <= 0 || entries.isEmpty then Vector.empty
    else
      val rnd = Random(seed)
      Vector.fill(count)(entries(rnd.nextInt(entries.length)).example)

  def save(path: Path): Unit =
    val lines = Vector(ReplayBuffer.encodeMetadata(this)) ++ entries.map(ReplayBuffer.encodeEntry)
    Files.createDirectories(path.toAbsolutePath.normalize.getParent)
    Files.write(path, lines.mkString("\n").getBytes(StandardCharsets.UTF_8))

object ReplayBuffer:
  final case class Entry(example: Example, domain: String)

  final case class Expected(
      contextSize: Int,
      vocabSize: Int,
      vocabHash: String
  )

  private val CurrentSchemaVersion = 1

  val empty: ReplayBuffer =
    ReplayBuffer(
      schemaVersion = CurrentSchemaVersion,
      contextSize = 0,
      vocabSize = 0,
      vocabHash = "",
      domainLabels = Vector.empty,
      entries = Vector.empty
    )

  def initialize(contextSize: Int, vocabSize: Int, vocabHash: String): ReplayBuffer =
    ReplayBuffer(
      schemaVersion = CurrentSchemaVersion,
      contextSize = contextSize,
      vocabSize = vocabSize,
      vocabHash = vocabHash,
      domainLabels = Vector.empty,
      entries = Vector.empty
    )

  def load(path: Path, expected: Expected): ReplayBuffer =
    val lines = Files.readAllLines(path, StandardCharsets.UTF_8)
    require(!lines.isEmpty, s"Replay buffer is empty: $path")

    val metadata = parseMetadata(lines.get(0), path)
    require(metadata.schemaVersion == CurrentSchemaVersion, s"Replay schema mismatch: expected $CurrentSchemaVersion got ${metadata.schemaVersion}")
    require(metadata.contextSize == expected.contextSize, s"Replay contextSize mismatch: expected ${expected.contextSize} got ${metadata.contextSize}")
    require(metadata.vocabSize == expected.vocabSize, s"Replay vocabSize mismatch: expected ${expected.vocabSize} got ${metadata.vocabSize}")
    require(metadata.vocabHash == expected.vocabHash, s"Replay vocabHash mismatch: expected ${expected.vocabHash} got ${metadata.vocabHash}")

    val entries =
      lines.subList(1, lines.size()).toArray(new Array[String](math.max(0, lines.size() - 1))).toVector
        .map(_.trim)
        .filter(_.nonEmpty)
        .map(parseEntry(_, path))

    metadata.copy(entries = entries)

  private def parseMetadata(line: String, path: Path): ReplayBuffer =
    val normalized = line.trim
    require(normalized.contains("\"type\":\"metadata\""), s"Replay metadata header missing in $path")

    val schemaVersion = intField(normalized, "schemaVersion", path)
    val contextSize = intField(normalized, "contextSize", path)
    val vocabSize = intField(normalized, "vocabSize", path)
    val vocabHash = stringField(normalized, "vocabHash", path)
    val domains = stringArrayField(normalized, "domains")

    ReplayBuffer(
      schemaVersion = schemaVersion,
      contextSize = contextSize,
      vocabSize = vocabSize,
      vocabHash = vocabHash,
      domainLabels = domains,
      entries = Vector.empty
    )

  private def parseEntry(line: String, path: Path): Entry =
    val normalized = line.trim
    require(normalized.contains("\"type\":\"example\""), s"Invalid replay entry record in $path: $line")

    val context = intArrayField(normalized, "context")
    val target = intField(normalized, "target", path)
    val domain = optionalStringField(normalized, "domain").getOrElse("unknown")
    Entry(Example(context, target), domain)

  private def encodeMetadata(buffer: ReplayBuffer): String =
    val domains = buffer.domainLabels.map(d => s"\"${escape(d)}\"").mkString(",")
    s"{" +
      s"\"type\":\"metadata\"," +
      s"\"schemaVersion\":${buffer.schemaVersion}," +
      s"\"contextSize\":${buffer.contextSize}," +
      s"\"vocabSize\":${buffer.vocabSize}," +
      s"\"vocabHash\":\"${escape(buffer.vocabHash)}\"," +
      s"\"domains\":[${domains}]" +
      s"}"

  private def encodeEntry(entry: Entry): String =
    val context = entry.example.context.mkString(",")
    s"{" +
      s"\"type\":\"example\"," +
      s"\"context\":[${context}]," +
      s"\"target\":${entry.example.target}," +
      s"\"domain\":\"${escape(entry.domain)}\"" +
      s"}"

  private def intField(line: String, key: String, path: Path): Int =
    val pattern = ("\"" + java.util.regex.Pattern.quote(key) + "\":(-?[0-9]+)").r
    pattern.findFirstMatchIn(line).map(_.group(1).toInt).getOrElse(
      throw new IllegalArgumentException(s"Missing integer field '$key' in replay file $path")
    )

  private def stringField(line: String, key: String, path: Path): String =
    optionalStringField(line, key).getOrElse(
      throw new IllegalArgumentException(s"Missing string field '$key' in replay file $path")
    )

  private def optionalStringField(line: String, key: String): Option[String] =
    val pattern = ("\"" + java.util.regex.Pattern.quote(key) + "\":\"([^\"]*)\"").r
    pattern.findFirstMatchIn(line).map(m => unescape(m.group(1)))

  private def intArrayField(line: String, key: String): Vector[Int] =
    val pattern = ("\"" + java.util.regex.Pattern.quote(key) + "\":\\[([^\\]]*)\\]").r
    pattern.findFirstMatchIn(line) match
      case None => Vector.empty
      case Some(m) =>
        val raw = m.group(1).trim
        if raw.isEmpty then Vector.empty
        else raw.split(",").toVector.map(_.trim).filter(_.nonEmpty).map(_.toInt)

  private def stringArrayField(line: String, key: String): Vector[String] =
    val pattern = ("\"" + java.util.regex.Pattern.quote(key) + "\":\\[([^\\]]*)\\]").r
    pattern.findFirstMatchIn(line) match
      case None => Vector.empty
      case Some(m) =>
        val raw = m.group(1).trim
        if raw.isEmpty then Vector.empty
        else
          raw
            .split(",")
            .toVector
            .map(_.trim)
            .filter(token => token.startsWith("\"") && token.endsWith("\""))
            .map(s => unescape(s.substring(1, s.length - 1)))

  private def escape(s: String): String =
    s.replace("\\", "\\\\").replace("\"", "\\\"")

  private def unescape(s: String): String =
    s.replace("\\\"", "\"").replace("\\\\", "\\")
