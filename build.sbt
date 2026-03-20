ThisBuild / scalaVersion := "3.3.3"
ThisBuild / organization := "local"
ThisBuild / version := "0.1.0"

lazy val root = (project in file("."))
  .settings(
    name := "scala-next-word-from-scratch",
    libraryDependencies += "org.scalameta" %% "munit" % "1.0.2" % Test
  )
