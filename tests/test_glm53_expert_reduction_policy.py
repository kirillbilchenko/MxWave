"""Check the arithmetic requirement that removes global Stream-K partitions."""

from __future__ import annotations

import math

import pytest

from integrations.glm53_mxfp4.expert_reduction_policy import block_quantum, capacity_for


@pytest.mark.parametrize("n,tile_n,sms,bps", [
    (2048, 128, 48, 3), (4096, 128, 48, 4), (2048, 128, 48, 1),
    (4096, 256, 132, 2),
])
def test_every_padded_dispatch_has_whole_launch_waves(
    n: int, tile_n: int, sms: int, bps: int,
) -> None:
    quantum = block_quantum(n, tile_n, sms, bps)
    grid, columns = sms * bps, n // tile_n
    assert all((q * columns) % grid for q in range(1, quantum))
    for consumed_blocks in range(1, 289):
        padded = capacity_for(consumed_blocks * 8, 8, quantum) // 8
        assert padded >= consumed_blocks and padded - consumed_blocks < quantum
        assert padded * columns % grid == 0
        # The kernel's tail then has one full launch wave: K/grid is not split.
        tail = padded * columns % grid or grid
        assert math.ceil(64 * tail / grid) == 64


def test_known_eight_row_case_and_worst_case_capacity() -> None:
    q = block_quantum(2048, 128, 48, 3)
    assert q == 9
    assert capacity_for(440, 8, q) == capacity_for(448, 8, q) == 504
    assert capacity_for(512, 8, q) == 576


@pytest.mark.parametrize("args", [(0, 8, 3), (63, 8, 3), (64, 0, 3), (64, 8, 0)])
def test_bad_capacity_is_rejected(args: tuple[int, int, int]) -> None:
    with pytest.raises(ValueError):
        capacity_for(*args)
