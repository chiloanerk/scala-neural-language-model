ThisBuild / scalaVersion := "3.3.3"
ThisBuild / organization := "local"
ThisBuild / version := "0.1.0"
Global / excludeLintKeys += Compile / runMain / connectInput

lazy val root = (project in file("."))
  .settings(
    name := "scala-neural-language-model-nlm",
    Compile / mainClass := Some("app.Main"),
    libraryDependencies += "org.scalameta" %% "munit" % "1.0.2" % Test,
    Compile / run / fork := true,
    Compile / run / connectInput := true,
    Compile / runMain / connectInput := true,
    Compile / run / outputStrategy := Some(StdoutOutput),
    Compile / run / javaOptions += s"-Dmetal.jni.lib=${(ThisBuild / baseDirectory).value.getAbsolutePath}/metal-jni/build/libmetal_jni.dylib",
    Test / javaOptions += s"-Dmetal.jni.lib=${(ThisBuild / baseDirectory).value.getAbsolutePath}/metal-jni/build/libmetal_jni.dylib",
    Test / parallelExecution := false
  )
