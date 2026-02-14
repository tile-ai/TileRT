"""Named indices for DSA temporary variables.

Mirrors the C++ ``DsaTempVars`` constants defined in
``include/lib/models/deepseek_v3_2/helper.hpp`` so that Python code can
reference temp_vars by name instead of magic numbers.

Usage::

    from tilert.models.deepseek_v3_2.temp_var_indices import Idx

    token_out = intermediates[Idx.TOKEN_OUT]  # equivalent to intermediates[25]
"""

from enum import IntEnum


class DsaTempVarIdx(IntEnum):
    """Index constants for DSA temp_vars, mirroring C++ DsaTempVars."""

    Q = 0
    KV = 1
    KI = 2
    Q_NOPE_DOWN = 3
    Q_PE = 4
    IQ = 5
    IQ_RT = 6
    IDX_SCORES = 7
    IDX_LOGITS = 8
    IDX_SELECTS = 9
    Q_NOPE = 10
    O = 11  # noqa: E741 — mirrors C++ DsaTempVars::O
    O_ACC = 12
    O_LSE = 13
    O_LSE_ACC = 14
    PROJ_O = 15
    UNPROJ_O = 16
    SCORES = 17
    X_MLP_IN = 18
    UP_GATE = 19
    SEL_PROBS = 20
    SEL_INDICES = 21
    EXP_OUT = 22
    X_RMSNORM = 23
    LOGITS_OUT = 24
    TOKEN_OUT = 25
    EMBEDDING_RMSNORM = 26
    HIDDEN_RMSNORM = 27
    EH_PROJ = 28
    X_TENSOR = 29
    ROPE_FREQS = 30
    CUR_POS = 31
    TOKEN_ID = 32
    LAST_HIDDEN_STATES = 33
    DRAFT_TOKENS = 34
    PREDICTED_TOKENS = 35
    PREDICTED_HIDDEN = 36
    ACCEPTED_TOKENS = 37
    NEXT_DRAFT_TOKENS = 38
    X_QUANT = 39
    X_SCALE = 40
    MOE_UP_GATE = 41
    IDX_SEL_WS = 42
    MTP0_TOKEN_OUT = 43
    MTP1_TOKEN_OUT = 44
    MTP0_EXP_OUT = 45
    SAMPLING_SEED = 46
    SAMPLING_POSITIONS = 47
    SAMPLING_CONFIG = 48
    TOP_P_SCORES = 49
    TOP_P_DEBUG = 50


# Sentinel: total number of temp vars. Must equal C++ DsaTempVars::temp_vars_size.
TEMP_VARS_SIZE = 51

# Short alias for convenient access
Idx = DsaTempVarIdx


def validate_temp_vars_layout() -> None:
    """Validate that the Python enum matches the C++ DsaTempVars layout.

    Checks:
    1. Enum member count equals TEMP_VARS_SIZE.
    2. Indices are contiguous 0..TEMP_VARS_SIZE-1 with no gaps or duplicates.
    3. (If libtilert.so is loaded) C++ temp_vars_size matches Python TEMP_VARS_SIZE.

    Raises:
        RuntimeError: If any validation check fails.
    """
    members = list(DsaTempVarIdx)

    # Check member count
    if len(members) != TEMP_VARS_SIZE:
        raise RuntimeError(
            f"DsaTempVarIdx has {len(members)} members but TEMP_VARS_SIZE={TEMP_VARS_SIZE}"
        )

    # Check contiguous indices
    indices = sorted(m.value for m in members)
    expected = list(range(TEMP_VARS_SIZE))
    if indices != expected:
        missing = set(expected) - set(indices)
        dupes = [i for i in indices if indices.count(i) > 1]
        raise RuntimeError(
            f"DsaTempVarIdx indices are not contiguous 0..{TEMP_VARS_SIZE - 1}. "
            f"Missing: {missing}, Duplicates: {set(dupes)}"
        )

    # Check against C++ if the library is loaded
    try:
        import torch

        cpp_size = torch.ops.tilert.dsa_temp_vars_size()
        if cpp_size != TEMP_VARS_SIZE:
            raise RuntimeError(
                f"Python TEMP_VARS_SIZE={TEMP_VARS_SIZE} != "
                f"C++ DsaTempVars::temp_vars_size={cpp_size}"
            )
    except (AttributeError, RuntimeError):
        # Library not loaded or op not available — skip C++ check
        pass
