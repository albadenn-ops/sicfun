package sicfun.holdem.types

/**
  * Thread-safe, scoped runtime-property overrides for tests and local tooling.
  *
  * This mechanism replaces direct mutation of `System.setProperty` / `System.clearProperty`
  * in test code, which is inherently unsafe in parallel test execution because JVM system
  * properties are process-global mutable state.
  *
  * Instead, overrides are stored in an `InheritableThreadLocal` stack of snapshots.
  * Each call to [[withOverrides]] pushes a new merged snapshot onto the stack and pops it
  * on exit (via try/finally), providing deterministic cleanup even when exceptions occur.
  * Because the thread-local is inheritable, child threads (e.g. Fork-Join pool workers)
  * automatically see the parent's overrides without explicit propagation.
  *
  * The two-level `Option[Option[String]]` returned by [[get]] distinguishes three states:
  *   - `None` -- no override exists; callers should fall back to `System.getProperty`
  *   - `Some(Some(value))` -- the property is overridden to `value`
  *   - `Some(None)` -- the property is explicitly cleared (simulates `System.clearProperty`)
  *
  * Used by [[TestSystemPropertyScope]] (test helper) and runtime configuration gates
  * that need safe, per-test property isolation.
  */
object ScopedRuntimeProperties:

  /** Thread-local stack of property override snapshots.
    *
    * Each element in the `List` is a snapshot map. The head of the list is the
    * currently active set of overrides. The list structure enables nesting:
    * inner `withOverrides` calls push onto the stack without losing the outer scope.
    *
    * `InheritableThreadLocal` ensures child threads created within a scope
    * inherit the parent's snapshot stack.
    */
  private val overridesRef = new InheritableThreadLocal[List[Map[String, Option[String]]]]:
    override def initialValue(): List[Map[String, Option[String]]] = Nil
    override def childValue(parentValue: List[Map[String, Option[String]]]): List[Map[String, Option[String]]] =
      parentValue

  /** Executes `thunk` with the given property overrides active, then restores the previous state.
    *
    * Overrides are merged into the current snapshot (if any), so nested calls accumulate.
    * A value of `Some(v)` sets the property; `None` clears it (simulates removal).
    *
    * @param updates sequence of (property-name, optional-value) pairs to override
    * @param thunk   the code block to execute with overrides active
    * @tparam A      the return type of the thunk
    * @return the result of evaluating `thunk`
    */
  def withOverrides[A](updates: Seq[(String, Option[String])])(thunk: => A): A =
    val previous = overridesRef.get()
    // Merge updates into the current top-of-stack snapshot (or an empty map if no scope is active)
    val merged = updates.foldLeft(previous.headOption.getOrElse(Map.empty[String, Option[String]])) {
      case (acc, (key, value)) => acc.updated(key, value)
    }
    // Push the merged snapshot onto the stack
    overridesRef.set(merged :: previous)
    try thunk
    finally
      // Pop: restore the previous stack (or remove the thread-local entirely if empty)
      previous match
        case Nil => overridesRef.remove()
        case _ => overridesRef.set(previous)

  /** Looks up a property override in the current scope.
    *
    * @param property the property name to look up
    * @return `None` if no override exists in any active scope;
    *         `Some(Some(value))` if the property is overridden to `value`;
    *         `Some(None)` if the property is explicitly cleared
    */
  def get(property: String): Option[Option[String]] =
    overridesRef.get().headOption.flatMap(_.get(property))
