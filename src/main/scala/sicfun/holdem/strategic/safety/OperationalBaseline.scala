package sicfun.holdem.strategic

final case class DeploymentBeliefSummary(
    beliefEntropy: Double,
    exploitabilitySnapshot: Ev,
    timestamp: Long
)

final case class EmpiricalDeploymentSet(
    entries: Vector[DeploymentBeliefSummary],
    maxSize: Int = 50
):
  def add(summary: DeploymentBeliefSummary): EmpiricalDeploymentSet =
    val updated = entries :+ summary
    if updated.size > maxSize then
      copy(entries = updated.drop(updated.size - maxSize))
    else
      copy(entries = updated)

  def deploymentExploitability: Ev =
    if entries.isEmpty then Ev.Zero
    else entries.map(_.exploitabilitySnapshot).reduce((a, b) => if a >= b then a else b)

final case class OperationalBaseline(
    epsilonBase: Double,
    deploymentSet: EmpiricalDeploymentSet,
    description: String
)
