package sicfun.holdem.types

/**
  * Test-facing facade for [[ScopedRuntimeProperties]].
  *
  * This object provides a more intuitive name (`withSystemProperties`) for use in test
  * code, hiding the implementation detail that overrides are thread-local snapshots
  * rather than actual JVM system property mutations.
  *
  * Usage in tests:
  * {{{
  * TestSystemPropertyScope.withSystemProperties(Seq(
  *   "sicfun.gpu.enabled" -> Some("false"),
  *   "sicfun.debug.level" -> None  // clears the property
  * )) {
  *   // code under test sees the overridden values
  * }
  * // overrides are automatically cleaned up here
  * }}}
  *
  * This approach is safe for parallel test execution because each test thread gets its
  * own isolated override scope via `InheritableThreadLocal`, unlike `System.setProperty`
  * which mutates process-global state.
  */
object TestSystemPropertyScope:
  /** Executes `thunk` with the given system property overrides active.
    *
    * @param updates sequence of (property-name, optional-value) pairs;
    *                `Some(v)` sets the property, `None` clears it
    * @param thunk   the test code to execute with overrides active
    * @tparam A      the return type of the thunk
    * @return the result of evaluating `thunk`
    */
  def withSystemProperties[A](updates: Seq[(String, Option[String])])(thunk: => A): A =
    ScopedRuntimeProperties.withOverrides(updates)(thunk)
