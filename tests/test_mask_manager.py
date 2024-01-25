import os

import numpy as np
import pytest
from returns.result import Success

from maskmanager import MaskManager


def test_mask_manager_init() -> None:
    index = np.arange(14)

    manager = MaskManager()
    manager.init("bus", index)

    assert manager.masks["bus"][0]["action"] == "init"
    assert manager.masks["bus"][0]["action_id"] == 0
    assert np.array_equal(manager.masks["bus"][0]["data"], index)

    assert np.array_equal(manager.get_index("bus"), index)

    with pytest.raises(ValueError):
        manager.init("bus", index)

    with pytest.raises(ValueError):
        manager.get_index("line", index)

    with pytest.raises(ValueError):
        manager.init("trafo", np.array([0, 0, 0, 1]))

    assert manager.traceback("bus", 0) == Success(("bus", "init", 0, None))
    assert manager.traceback("bus", 100).failure()


def test_mask_manager_add() -> None:
    index = np.arange(14)

    manager = MaskManager()
    manager.init("bus", index)

    with pytest.raises(ValueError):
        manager.add("bus", np.array([0, 1, 2]))

    with pytest.raises(ValueError):
        manager.add("bus", np.array([0, 24, 25]))

    manager.add("bus", np.array([23, 24, 25]))

    assert np.array_equal(
        manager.get_index("bus"), np.concatenate([index, np.array([23, 24, 25])])
    )

    assert manager.traceback("bus", 0) == Success(("bus", "init", 0, None))
    assert manager.traceback("bus", 23) == Success(("bus", "add", 23, None))
    assert manager.traceback("bus", 15).failure()


def test_mask_manager_remove() -> None:
    index = np.arange(14)

    manager = MaskManager()
    manager.init("bus", index)

    with pytest.raises(ValueError):
        manager.remove("bus", np.array([26, 27, 28]))

    manager.remove("bus", np.array([0, 1, 2]))

    assert manager.traceback("bus", 3) == Success(("bus", "init", 3, None))
    assert manager.traceback("bus", 0).failure()

    assert np.array_equal(
        manager.get_index("bus"), np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    )

    with pytest.raises(ValueError):
        manager.remove("bus", np.array([0, 1, 2]))

    manager.remove("bus", np.array([3, 4, 5]))

    assert np.array_equal(
        manager.get_index("bus"), np.array([6, 7, 8, 9, 10, 11, 12, 13])
    )

    assert manager.traceback("bus", 3).failure()
    assert manager.traceback("bus", 6) == Success(("bus", "init", 6, None))

    # Re-add items
    manager.add("bus", np.array([0, 1, 2]))

    assert np.array_equal(
        manager.get_index("bus"), np.array([6, 7, 8, 9, 10, 11, 12, 13, 0, 1, 2])
    )
    assert manager.traceback("bus", 0) == Success(("bus", "add", 0, None))


def test_mask_manager_convert() -> None:
    manager = MaskManager()
    manager.init("line", np.arange(16, dtype=int))
    manager.init("trafo", np.arange(4, dtype=int))
    manager.init("branch", np.array([], dtype=int))

    manager.convert("line", np.array([1, 2]), "branch", np.array([0, 1]))

    assert np.array_equal(
        manager.get_index("line"),
        np.array([0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
    )
    assert np.array_equal(manager.get_index("branch"), np.array([0, 1]))

    assert manager.traceback("branch", 0) == Success(("line", "init", 1, None))
    assert manager.traceback("branch", 2).failure()
    assert manager.traceback("line", 1).failure()
    assert manager.traceback("line", 0) == Success(("line", "init", 0, None))

    # Origin indices don't exist anymore
    with pytest.raises(ValueError):
        manager.convert("line", np.array([1, 2]), "branch", np.array([2, 3]))

    # Destination indices already exist
    with pytest.raises(ValueError):
        manager.convert("line", np.array([3, 4]), "branch", np.array([0, 1]))

    # Shape mismatch
    with pytest.raises(ValueError):
        manager.convert("trafo", np.arange(4), "branch", np.arange(5) + 2)

    # Element type doesn't exist
    with pytest.raises(ValueError):
        manager.convert("line", np.arange(4), "bus", np.arange(4) + 2)

    manager.convert("trafo", np.arange(4), "branch", np.arange(4) + 2)

    assert np.array_equal(manager.get_index("trafo"), np.array([]))

    assert np.array_equal(manager.get_index("branch"), np.array([0, 1, 2, 3, 4, 5]))

    manager.add("line", np.array([1]))
    assert manager.traceback("line", 1) == Success(("line", "add", 1, None))

    manager.convert("line", np.array([0, 1]), "trafo", np.array([999, 1000]))
    manager.convert("trafo", np.array([999, 1000]), "branch", np.array([999, 1000]))
    manager.convert("branch", np.array([999, 1000]), "line", np.array([1, 2]))

    assert manager.traceback("line", 2) == Success(("line", "add", 1, None))
    assert manager.traceback("line", 1) == Success(("line", "init", 0, None))


def test_mask_manager_reindex() -> None:
    index = np.arange(14, dtype=int)
    manager = MaskManager()
    manager.init("bus", index)

    shuffled = index.copy()
    np.random.shuffle(shuffled)

    with pytest.raises(ValueError):
        manager.reindex("bus", shuffled[1:])

    with pytest.raises(ValueError):
        manager.reindex("bus", np.ones_like(shuffled))

    manager.reindex("bus", shuffled)

    assert np.array_equal(manager.get_index("bus"), shuffled)

    with pytest.raises(ValueError):
        manager.reindex("line", shuffled)

    for i in range(len(index)):
        assert manager.traceback("bus", shuffled[i]) == Success(
            ("bus", "init", i, None)
        )

    assert manager.traceback("bus", 100).failure()


def test_mask_manager_save(tmp_path: str) -> None:
    manager = MaskManager()
    manager.init("bus", np.arange(14, dtype=int))
    manager.add("bus", np.array([20, 21, 22]))
    manager.init("line", np.arange(16, dtype=int))
    manager.init("trafo", np.arange(4, dtype=int))
    manager.init("branch", np.array([], dtype=int))
    manager.convert("line", np.array([1, 2]), "branch", np.array([0, 1]))
    manager.convert("trafo", np.arange(4), "branch", np.arange(4) + 2)
    manager.add("line", np.array([1]))
    manager.convert("branch", np.array([0]), "trafo", np.array([999]))

    manager.save(os.path.join(tmp_path, "test_mask_manager.pkl"))

    manager2 = MaskManager.from_file(os.path.join(tmp_path, "test_mask_manager.pkl"))

    for element in ["bus", "line", "trafo", "branch"]:
        assert np.array_equal(manager.get_index(element), manager2.get_index(element))
        first_idx = manager.get_index(element)[0]
        assert manager.traceback(element, first_idx) == manager2.traceback(
            element, first_idx
        )

        for i in range(len(manager.masks[element])):
            assert (
                manager.masks[element][i]["action"]
                == manager2.masks[element][i]["action"]
            )
            assert (
                manager.masks[element][i]["action_id"]
                == manager2.masks[element][i]["action_id"]
            )
            assert np.array_equal(
                manager.masks[element][i]["data"], manager2.masks[element][i]["data"]
            )
