package sicfun.holdem.runtime.agent

import sicfun.holdem.runtime.protocol.*
import sicfun.holdem.types.PokerAction
import scala.util.Random

enum AbstractAction:
  case Fold, Passive, HalfPot, Pot, AllIn

object AbstractAction:
  val ordered: Vector[AbstractAction] =
    Vector(Fold, Passive, HalfPot, Pot, AllIn)

final class BlueprintOnlyAgent(
    override val seatId: SeatId,
    store: BlueprintStore,
    hasher: InfostateHasher,
    rngSeed: Long
) extends SeatAgent:
  private val rng = new Random(rngSeed)

  def decide(snapshot: TableSnapshot, legalActions: Set[PokerAction]): PokerAction =
    val hash = hasher.hashFor(snapshot, seatId)
    val dist = store.lookup(hash)
    val idx = sampleIndex(dist.probs)
    val abstractPick = AbstractAction.ordered(idx)
    translateToLegal(abstractPick, legalActions)

  private def sampleIndex(probs: Array[Float]): Int =
    val r = rng.nextFloat()
    var acc = 0f
    var i = 0
    while i < probs.length do
      acc += probs(i)
      if r <= acc then return i
      i += 1
    probs.length - 1

  private def translateToLegal(a: AbstractAction, legal: Set[PokerAction]): PokerAction =
    a match
      case AbstractAction.Fold =>
        if legal.contains(PokerAction.Fold) then PokerAction.Fold
        else translateToLegal(AbstractAction.Passive, legal)
      case AbstractAction.Passive =>
        if legal.contains(PokerAction.Check) then PokerAction.Check
        else if legal.contains(PokerAction.Call) then PokerAction.Call
        else PokerAction.Fold
      case AbstractAction.HalfPot | AbstractAction.Pot | AbstractAction.AllIn =>
        val raises = legal.collect { case r: PokerAction.Raise => r }.toVector
        if raises.isEmpty then translateToLegal(AbstractAction.Passive, legal)
        else
          val target: Double = a match
            case AbstractAction.HalfPot => raises.map(_.amount).min
            case AbstractAction.Pot     => median(raises.map(_.amount))
            case AbstractAction.AllIn   => raises.map(_.amount).max
            case _                      => raises.head.amount
          raises.minBy(r => math.abs(math.log(r.amount) - math.log(target)))

  private def median(xs: Vector[Double]): Double =
    val sorted = xs.sorted
    sorted(sorted.size / 2)
