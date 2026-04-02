package sicfun.holdem.types

/** Applies scoped runtime-property overrides for tests without mutating global JVM properties. */
object TestSystemPropertyScope:
  def withSystemProperties[A](updates: Seq[(String, Option[String])])(thunk: => A): A =
    ScopedRuntimeProperties.withOverrides(updates)(thunk)
