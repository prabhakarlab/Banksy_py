"""banksy_ilgwg: an optimized, vectorized BANKSY backend.

This backend is intentionally NOT compatible with the original ``banksy``
package's API.
"""

from __future__ import annotations

from banksy_ilgwg.matrix import build_banksy_matrix

__all__ = ["build_banksy_matrix"]
