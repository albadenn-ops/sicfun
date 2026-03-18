package sicfun.holdem.types

/** Decision mode for hero play: adaptive (Bayesian engine) or GTO (CFR solver). */
enum HeroMode:
  case Adaptive
  case Gto
