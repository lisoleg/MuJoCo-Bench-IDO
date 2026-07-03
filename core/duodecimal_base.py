"""
Duodecimal Base-12 Periodic Domain Preferred Representation Base
=================================================================

Implements the Duodecimal (Base-12) preferred representation base for
TOMAS EML hypergraph physical quantity encoding, as specified by the
IDO/TOMAS framework and Bian's fractional geometry.

Key Insight:
    The periodic domain Q_circ = {1/2, 1/3, 1/4, 1/6, 1/12} has
    denominators whose prime factors are {2, 3} ⊆ prime(12) = {2, 3}.
    This means all periodic quantities (angles, time, frequency ratios)
    have ZERO-RESIDUAL representation in Base-12, while Base-10 produces
    truncation loops that pollute IEEE-754 calculations of Mao Rui d_sem.

Theorem (Duodecimal Minimum Semantic Residual for Periodic Domain):
    For all r ∈ Q_circ, prime(denom(r)) ⊆ {2,3} ⊆ prime(12)
    ⇒ Base-12 provides exact representation (zero semantic residual)
    ⇒ Base-10 requires truncation ⇒ IEEE-754 pollutes d_sem computation

Engineering Implication:
    EML-Lite KB physical quantities should use duodecimal_fixed(12)
    or ExactRational(a//b, b|12) encoding to prevent floating-point
    noise from corrupting Mao Rui d_sem calculations, especially
    T_Shield / Dead-Zero threshold comparisons.

Reference:
    七零七光之科技 (2023-2026). 泰格坦高维十二进制数学.
    Zhang, F. (2026). EML-SemZip与卞氏折叠饱和阈值.
    复合体理学 WeChat: mp.weixin.qq.com/s/2Jtk_WAqU0joCG39cTqLzg

Author: MuJoCo-Bench-IDO v0.6.4 — Duodecimal Base Module
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from fractions import Fraction
import math

IDO_DUODECIMAL_BASE_VERSION: str = "v0.1.0"

# ── Duodecimal digit map ──
DUO_DIGITS = "0123456789XE"  # X=10, E=11

# ── Periodic domain Q_circ ──
# These fractions have zero-residual representation in Base-12
PERIODIC_DOMAIN = {
    Fraction(1, 2): "0.6",
    Fraction(1, 3): "0.4",
    Fraction(1, 4): "0.3",
    Fraction(1, 6): "0.2",
    Fraction(1, 12): "0.1",
}

# ── Non-preferred fractions (require additional digits in Base-12) ──
NON_PREFERRED = {
    Fraction(1, 8): "0.16",    # 2³ not fully in prime(12)
    Fraction(1, 9): "0.14",    # 3² okay but longer representation
    Fraction(1, 5): "0.2497",  # 5 not in prime(12) → infinite loop
    Fraction(1, 7): "0.186X35", # 7 not in prime(12) → infinite loop
    Fraction(1, 10): "0.12497", # 5 not in prime(12) → infinite loop
}


@dataclass
class DuodecimalConfig:
    """Configuration for duodecimal computation.

    Attributes:
        precision: Number of duodecimal digits after point for non-exact.
        enforce_exact: Whether to reject non-exact conversions.
    """
    precision: int = 12
    enforce_exact: bool = False


class DuodecimalBase:
    """Duodecimal (Base-12) preferred representation base for EML encoding.

    Provides zero-residual encoding for periodic domain quantities,
    preventing IEEE-754 truncation noise from corrupting Mao Rui
    d_sem calculations in TOMAS EML hypergraph.
    """

    VERSION: str = IDO_DUODECIMAL_BASE_VERSION

    def __init__(self, config: Optional[DuodecimalConfig] = None) -> None:
        """Initialize duodecimal base with configuration."""
        self.config = config or DuodecimalConfig()

    def to_duodecimal(self, value: float) -> str:
        """Convert a float value to duodecimal representation.

        Args:
            value: Float value to convert.

        Returns:
            Duodecimal string representation.
        """
        if value == 0:
            return "0"

        sign = "-" if value < 0 else ""
        value = abs(value)

        # Integer part
        int_part = int(value)
        frac_part = value - int_part

        int_str = self._int_to_duo(int_part)
        if frac_part == 0:
            return sign + int_str

        # Fractional part
        frac_str = self._frac_to_duo(frac_part)
        return sign + int_str + "." + frac_str

    def _int_to_duo(self, n: int) -> str:
        """Convert integer to duodecimal.

        Args:
            n: Positive integer.

        Returns:
            Duodecimal string.
        """
        if n == 0:
            return "0"
        digits = []
        while n > 0:
            digits.append(DUO_DIGITS[n % 12])
            n //= 12
        return "".join(reversed(digits))

    def _frac_to_duo(self, frac: float) -> str:
        """Convert fractional part to duodecimal.

        Args:
            frac: Fractional part (0 < frac < 1).

        Returns:
            Duodecimal fractional string.
        """
        digits = []
        for _ in range(self.config.precision):
            frac *= 12
            digit = int(frac)
            digits.append(DUO_DIGITS[digit])
            frac -= digit
            if frac == 0:
                break  # Exact representation
        return "".join(digits)

    def from_fraction(self, frac: Fraction) -> str:
        """Convert a rational fraction to duodecimal (exact if in Q_circ).

        Args:
            frac: Fraction to convert.

        Returns:
            Duodecimal representation string.
        """
        # Check if it's in the periodic domain (exact representation)
        if frac in PERIODIC_DOMAIN:
            return PERIODIC_DOMAIN[frac]

        if frac in NON_PREFERRED:
            return NON_PREFERRED[frac]

        # General conversion
        return self.to_duodecimal(float(frac))

    def exact_values(self) -> Dict[str, str]:
        """Return all Q_circ periodic domain exact duodecimal values.

        Returns:
            Dict mapping fraction strings to duodecimal representations.
        """
        result: Dict[str, str] = {}
        for frac, duo_str in PERIODIC_DOMAIN.items():
            result[str(frac)] = duo_str
        return result

    def is_exact_in_base12(self, frac: Fraction) -> bool:
        """Check if a fraction has exact (zero-residual) representation in Base-12.

        A fraction a/b is exact in Base-12 if prime(b) ⊆ prime(12) = {2, 3}.

        Args:
            frac: Fraction to check.

        Returns:
            True if exact representation exists in Base-12.
        """
        denom = frac.denominator
        # Factor out 2s and 3s
        while denom % 2 == 0:
            denom //= 2
        while denom % 3 == 0:
            denom //= 3
        return denom == 1

    def semantic_residual(self, value: float, base: int = 10) -> float:
        """Compute semantic residual (encoding error) for a value in given base.

        For periodic domain values:
        - Base-12: zero residual (exact representation)
        - Base-10: positive residual (truncation/rounding error)

        This residual corrupts Mao Rui d_sem calculations.

        Args:
            value: Value to compute residual for.
            base: Representation base (default 10).

        Returns:
            Semantic encoding residual.
        """
        # Convert value to fraction if possible
        frac = Fraction(value).limit_denominator(1000)

        if base == 12:
            if self.is_exact_in_base12(frac):
                return 0.0  # Zero residual!
            else:
                return abs(value - float(frac))
        elif base == 10:
            if frac.denominator == 1 or set(
                self._prime_factors(frac.denominator)
            ) <= {2, 5}:  # prime(10) = {2, 5}
                return 0.0
            else:
                # Truncation error from IEEE-754
                duo_repr = self.to_duodecimal(value)
                return abs(value - self.from_duodecimal(duo_repr))
        else:
            return abs(value - float(frac))

    def _prime_factors(self, n: int) -> set:
        """Compute prime factors of n.

        Args:
            n: Integer to factorize.

        Returns:
            Set of prime factors.
        """
        factors = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.add(d)
                n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors

    def from_duodecimal(self, duo_str: str) -> float:
        """Convert duodecimal string back to float.

        Args:
            duo_str: Duodecimal string (e.g., "0.4" for 1/3).

        Returns:
            Float value.
        """
        duo_str = duo_str.strip()
        if duo_str.startswith("-"):
            sign = -1
            duo_str = duo_str[1:]
        else:
            sign = 1

        if "." in duo_str:
            int_part, frac_part = duo_str.split(".", 1)
        else:
            int_part = duo_str
            frac_part = ""

        # Integer part
        int_val = 0
        for ch in int_part:
            int_val = int_val * 12 + DUO_DIGITS.index(ch)

        # Fractional part
        frac_val = 0.0
        for i, ch in enumerate(frac_part):
            frac_val += DUO_DIGITS.index(ch) / (12 ** (i + 1))

        return sign * (int_val + frac_val)

    def angle_partition(self, circle_divisions: int = 12) -> Dict[str, float]:
        """Partition a circle into duodecimal divisions.

        EML-Lite KB should use this for angle encoding:
        angle_partition(circle, n=12) → exact duodecimal angles

        Args:
            circle_divisions: Number of divisions (default 12).

        Returns:
            Dict mapping angle names to exact duodecimal values.
        """
        result: Dict[str, float] = {}
        for i in range(circle_divisions):
            frac = Fraction(i, circle_divisions)
            duo_str = self.from_fraction(frac)
            angle_deg = 360 * i / circle_divisions
            result[f"{angle_deg}°"] = duo_str
        return result

    def compare_bases(self, frac: Fraction) -> Dict:
        """Compare Base-10 vs Base-12 representation for a fraction.

        Shows why Base-12 is preferred for periodic domain quantities.

        Args:
            frac: Fraction to compare.

        Returns:
            Dict with base-10, base-12 representations and residuals.
        """
        base10_str = str(float(frac))
        base12_str = self.from_fraction(frac)
        residual_10 = self.semantic_residual(float(frac), base=10)
        residual_12 = self.semantic_residual(float(frac), base=12)

        return {
            "fraction": str(frac),
            "base10": base10_str,
            "base12": base12_str,
            "residual_base10": residual_10,
            "residual_base12": residual_12,
            "is_exact_base12": residual_12 == 0.0,
            "preferred_base": "12" if residual_12 < residual_10 else "10",
        }
