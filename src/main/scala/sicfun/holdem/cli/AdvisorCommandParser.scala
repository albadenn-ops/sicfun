package sicfun.holdem.cli
import sicfun.holdem.types.*

import sicfun.core.Card

/**
  * Command parser for the interactive poker advisor REPL.
  *
  * This file defines the [[AdvisorCommand]] ADT (algebraic data type) representing all
  * possible commands the user can issue in the advisor REPL, and the [[AdvisorCommandParser]]
  * that transforms raw text input into typed command values.
  *
  * The parser is designed for a conversational poker coaching workflow:
  *   1. Start a new hand (`new`)
  *   2. Set hero's hole cards (`h AcKh` or just `AcKh`)
  *   3. Record actions for hero and villain (`h raise 6`, `v call`)
  *   4. Deal community cards (`board Ts9h8d`)
  *   5. Ask for advice (`?` or `advise`)
  *   6. Review the hand after showdown (`review`)
  *
  * The parser is intentionally forgiving: it accepts multiple synonyms for each command
  * (e.g. "quit"/"exit"/"q"), supports both concatenated and spaced card notation,
  * and treats "bet" as an alias for "raise" since casual players use both interchangeably.
  *
  * Invalid or unrecognized input always produces [[AdvisorCommand.Unknown]] with a
  * descriptive reason string, rather than throwing exceptions. This keeps the REPL loop
  * simple and lets the caller display helpful error messages.
  */

/** Typed commands for the interactive poker advisor REPL.
  *
  * Each variant represents a distinct user intent. The [[Unknown]] variant captures
  * unrecognized input together with a reason string for error reporting.
  */
enum AdvisorCommand:
  /** Start a new hand, resetting the current hand state. */
  case NewHand
  /** Set hero's private hole cards for the current hand. */
  case HeroCards(cards: sicfun.holdem.types.HoleCards)
  /** Record an action taken by the hero (fold, check, call, or raise). */
  case HeroAction(action: PokerAction)
  /** Record an action taken by the villain (fold, check, call, or raise). */
  case VillainAction(action: PokerAction)
  /** Deal community cards to the board (flop, turn, or river cards). */
  case DealBoard(cards: Vector[Card])
  /** Request a strategic recommendation from the advisor engine. */
  case Advise
  /** Review the completed hand with post-hoc analysis. */
  case Review
  /** Display cumulative session statistics (hands played, win rate, etc.). */
  case SessionStats
  /** Undo the most recent action or state change. */
  case Undo
  /** Display help text listing all available commands. */
  case Help
  /** Reveal villain's hole cards at showdown for post-hand analysis. */
  case VillainShowdown(cards: sicfun.holdem.types.HoleCards)
  /** Exit the advisor REPL session. */
  case Quit
  /** Unrecognized input, with the original text and a descriptive reason for rejection. */
  case Unknown(input: String, reason: String)

/** Parses raw REPL input lines into typed [[AdvisorCommand]] values.
  *
  * Grammar:
  *   new                              → NewHand
  *   hero AcKh / h AcKh              → HeroCards
  *   h raise 6 / h call / h fold     → HeroAction
  *   v bet 8 / v call / v raise 20   → VillainAction (bet = raise)
  *   v show QhQs / v show Qh Qs     → VillainShowdown
  *   board Ts9h8d / board Ts 9h 8d   → DealBoard
  *   ? / advise                       → Advise
  *   review                           → Review
  *   session / stats                  → SessionStats
  *   undo                             → Undo
  *   help                             → Help
  *   quit / exit / q                  → Quit
  */
object AdvisorCommandParser:

  /** Parses a raw REPL input line into a typed [[AdvisorCommand]].
    *
    * The input is trimmed, split on whitespace, and the first token is matched
    * case-insensitively against known command prefixes. Sub-tokens are dispatched
    * to specialized parsers for hero actions, villain actions, and board cards.
    *
    * As a convenience, bare 4-character tokens (e.g. "AcKh") are interpreted as
    * hero hole cards, allowing the user to skip the "h" prefix.
    *
    * @param line the raw input string from the REPL
    * @return a typed [[AdvisorCommand]]; never throws
    */
  def parse(line: String): AdvisorCommand =
    val trimmed = line.trim
    if trimmed.isEmpty then AdvisorCommand.Unknown("", "empty input")
    else
      val parts = trimmed.split("\\s+").toVector
      val cmd = parts.head.toLowerCase
      cmd match
        case "new"                                       => AdvisorCommand.NewHand
        case "hero" | "h" if parts.length >= 2           => parseHeroSub(parts.tail)
        case "villain" | "v" if parts.length >= 2        => parseVillainSub(parts.tail)
        case "board" | "b" if parts.length >= 2          => parseBoardSub(parts.tail)
        case "?" | "advise" | "advice"                   => AdvisorCommand.Advise
        case "review"                                    => AdvisorCommand.Review
        case "session" | "stats"                         => AdvisorCommand.SessionStats
        case "undo"                                      => AdvisorCommand.Undo
        case "help"                                      => AdvisorCommand.Help
        case "quit" | "exit" | "q"                       => AdvisorCommand.Quit
        case _ =>
          // Try interpreting the whole line as a 4-char hole card token
          if trimmed.length == 4 then
            tryParseHoleCards(trimmed) match
              case Some(hc) => AdvisorCommand.HeroCards(hc)
              case None     => AdvisorCommand.Unknown(trimmed, "unrecognized command")
          else AdvisorCommand.Unknown(trimmed, "unrecognized command")

  // ---- Hero sub-commands ----

  /** Parses the tokens following "h" or "hero", dispatching to action or hole-card parsing.
    *
    * Supports: "h raise 6", "h call", "h fold", "h check", "h AcKh", "h Ac Kh".
    */
  private def parseHeroSub(tokens: Vector[String]): AdvisorCommand =
    val first = tokens.head.toLowerCase
    first match
      case "raise" | "bet" => parseRaiseAmount(tokens.tail, isHero = true)
      case "call"          => AdvisorCommand.HeroAction(PokerAction.Call)
      case "fold"          => AdvisorCommand.HeroAction(PokerAction.Fold)
      case "check"         => AdvisorCommand.HeroAction(PokerAction.Check)
      case _ =>
        // Try as hole cards: "h AcKh" or "h Ac Kh"
        if tokens.length == 1 && tokens.head.length == 4 then
          tryParseHoleCards(tokens.head) match
            case Some(hc) => AdvisorCommand.HeroCards(hc)
            case None     => AdvisorCommand.Unknown(s"h ${tokens.mkString(" ")}", s"invalid hole cards: ${tokens.head}")
        else if tokens.length == 2 && tokens(0).length == 2 && tokens(1).length == 2 then
          tryParseHoleCards(tokens(0) + tokens(1)) match
            case Some(hc) => AdvisorCommand.HeroCards(hc)
            case None     => AdvisorCommand.Unknown(s"h ${tokens.mkString(" ")}", s"invalid hole cards: ${tokens.mkString(" ")}")
        else
          AdvisorCommand.Unknown(s"h ${tokens.mkString(" ")}", "expected: raise|bet|call|fold|check or hole cards")

  // ---- Villain sub-commands ----

  /** Parses the tokens following "v" or "villain", dispatching to action or showdown parsing.
    *
    * Supports: "v raise 20", "v call", "v fold", "v check", "v show QhQs", "v show Qh Qs".
    * "bet" is accepted as a synonym for "raise" since casual players use both.
    */
  private def parseVillainSub(tokens: Vector[String]): AdvisorCommand =
    val first = tokens.head.toLowerCase
    first match
      case "raise" | "bet" => parseRaiseAmount(tokens.tail, isHero = false)
      case "call"          => AdvisorCommand.VillainAction(PokerAction.Call)
      case "fold"          => AdvisorCommand.VillainAction(PokerAction.Fold)
      case "check"         => AdvisorCommand.VillainAction(PokerAction.Check)
      case "show" | "shows" | "showdown" =>
        if tokens.tail.isEmpty then
          AdvisorCommand.Unknown(s"v ${tokens.mkString(" ")}", "showdown requires cards (e.g., v show QhQs)")
        else
          val cardStr = tokens.tail.mkString("")
          tryParseHoleCards(cardStr) match
            case Some(hc) => AdvisorCommand.VillainShowdown(hc)
            case None => AdvisorCommand.Unknown(s"v ${tokens.mkString(" ")}", s"invalid hole cards: ${tokens.tail.mkString(" ")}")
      case _ =>
        AdvisorCommand.Unknown(s"v ${tokens.mkString(" ")}", "expected: raise|bet|call|fold|check|show")

  // ---- Board sub-command ----

  /** Parses board card tokens into a [[AdvisorCommand.DealBoard]].
    *
    * Accepts two notations:
    *   - Concatenated: "Ts9h8d" (a single token, split into 2-character chunks)
    *   - Spaced: "Ts 9h 8d" (each card as a separate whitespace-delimited token)
    *
    * Validates each card token via `Card.parse` and reports invalid cards by name.
    */
  private def parseBoardSub(tokens: Vector[String]): AdvisorCommand =
    // Support: "board Ts9h8d" (one token, split into 2-char chunks)
    //      or: "board Ts 9h 8d" (separate tokens)
    val cardTokens =
      if tokens.length == 1 && tokens.head.length > 2 && tokens.head.length % 2 == 0 then
        tokens.head.grouped(2).toVector
      else
        tokens

    val parsed = cardTokens.map(Card.parse)
    if parsed.exists(_.isEmpty) then
      val bad = cardTokens.zip(parsed).collect { case (t, None) => t }
      AdvisorCommand.Unknown(s"board ${tokens.mkString(" ")}", s"invalid card(s): ${bad.mkString(", ")}")
    else
      AdvisorCommand.DealBoard(parsed.flatten)

  // ---- Helpers ----

  /** Parses a raise/bet amount from the remaining tokens.
    *
    * Validates that the amount is present, numeric, and positive.
    * Returns a HeroAction or VillainAction depending on `isHero`.
    *
    * @param tokens the remaining tokens after "raise" or "bet"
    * @param isHero true if the raise belongs to the hero, false for villain
    */
  private def parseRaiseAmount(tokens: Vector[String], isHero: Boolean): AdvisorCommand =
    if tokens.isEmpty then
      val who = if isHero then "h" else "v"
      AdvisorCommand.Unknown(s"$who raise", "raise requires an amount (e.g., raise 6)")
    else
      tokens.head.toDoubleOption match
        case Some(amount) if amount > 0.0 =>
          val action = PokerAction.Raise(amount)
          if isHero then AdvisorCommand.HeroAction(action) else AdvisorCommand.VillainAction(action)
        case Some(_) =>
          AdvisorCommand.Unknown(s"raise ${tokens.head}", "raise amount must be positive")
        case None =>
          AdvisorCommand.Unknown(s"raise ${tokens.head}", s"invalid number: ${tokens.head}")

  /** Attempts to parse a 4-character token as canonical hole cards.
    *
    * Delegates to [[CliHelpers.parseHoleCards]] and catches any exception,
    * returning `None` for invalid input rather than propagating the error.
    *
    * @param token a 4-character string like "AcKh"
    * @return `Some(holeCards)` if valid, `None` otherwise
    */
  private def tryParseHoleCards(token: String): Option[HoleCards] =
    try Some(CliHelpers.parseHoleCards(token))
    catch case _: Exception => None
