"""The mask manager is a central instance to handle masks for different pre-processing steps and
allows to trace back indices to their origin."""

import pickle
from typing import Optional

import numpy as np
from returns.result import Failure, Result, Success


def forward_init(_mask: np.ndarray, event: dict) -> np.ndarray:
    """Apply the init event to a mask"""
    return event["data"]


def forward_add(mask: np.ndarray, event: dict) -> np.ndarray:
    """Apply the add event to a mask"""
    return np.concatenate([mask, event["data"]])


def forward_remove(mask: np.ndarray, event: dict) -> np.ndarray:
    """Apply the remove event to a mask"""
    return mask[~np.in1d(mask, event["data"])]


def forward_reindex(mask: np.ndarray, event: dict) -> np.ndarray:
    """Apply the reindex event to a mask"""
    return mask[event["data"]]


def forward_convert_from(mask: np.ndarray, event: dict) -> np.ndarray:
    """Apply the convert from event to a mask"""
    return mask[~np.in1d(mask, event["data"])]


def forward_convert_to(mask: np.ndarray, event: dict) -> np.ndarray:
    """Apply the convert to event to a mask"""
    return np.concatenate([mask, event["data"]])


forward_hooks = {
    "init": forward_init,
    "add": forward_add,
    "remove": forward_remove,
    "reindex": forward_reindex,
    "convert_from": forward_convert_from,
    "convert_to": forward_convert_to,
}


class MaskManager:
    """The mask manager is a central instance to handle masks for different pre-processing steps
    The aim is that steps performed in preprocessing can be easily traced back to the original index
    of the data.

    Indices are tracked for different types of elements (e.g. busbars, lines, shunts, ...). Index
    operations are represented with a few primitives:
    - Init: Initialize a new mask for a given element type
    - Add: Add indices to an existing element type that came from undefined sources
    - Convert: Convert indices from one element type to another
    - Remove: Remove indices from an element type
    - Reindex: Reorder indices of an element type

    Furthermore the manager can save and load masks to/from disk.
    """

    def __init__(self):
        self.masks = {}
        self.action_id = 0

    def init(
        self, element: str, indices: np.ndarray, comment: Optional[str] = None
    ) -> int:
        """Initialize a new mask for a given element type

        Args:
            element (str): Element type
            indices (np.ndarray): Indices of the element type
            comment (Optional[str], optional): Comment for the init action. Defaults to None.

        Returns:
            int: Action id of the init action
        """
        if element in self.masks:
            raise ValueError("Element type already initialized")
        if np.unique(indices).size != indices.size:
            raise ValueError("Indices must be unique")

        cur_action_id = self.action_id
        self.masks[element] = [
            {
                "action": "init",
                "action_id": cur_action_id,
                "data": np.array(indices, copy=True),
                "comment": comment,
            }
        ]
        self.action_id += 1
        return cur_action_id

    def add(
        self, element: str, new_indices: np.ndarray, comment: Optional[str] = None
    ) -> int:
        """Adds elements with a given index at the end of the element type array

        This will throw an error if the indices were occupied previously
        Note that this assumes appending at the end of the masks array. If this is not the case,
        use reindex to reorder the indices after adding.

        Args:
            element (str): Element type
            new_indices (np.ndarray): Indices of the added elements
            comment (Optional[str], optional): Comment for the add action. Defaults to None.

        Returns:
            int: Action id of the add action
        """
        if element not in self.masks:
            raise ValueError("Element type not initialized")
        if np.intersect1d(self.get_index(element), new_indices).size != 0:
            raise ValueError("Indices already occupied")

        cur_action_id = self.action_id

        self.masks[element].append(
            {
                "action": "add",
                "action_id": cur_action_id,
                "data": np.array(new_indices, copy=True),
                "comment": comment,
            }
        )
        self.action_id += 1

        return cur_action_id

    def convert(
        self,
        old_element: str,
        old_indices: np.ndarray,
        new_element: str,
        new_indices: np.ndarray,
        comment: Optional[str] = None,
    ) -> int:
        """Converts indices from one element type to another

        Args:
            old_element (str): Old element type
            old_indices (np.ndarray): Indices of the old element type
            new_element (str): New element type
            new_indices (np.ndarray): Indices of the new element type
            comment (Optional[str], optional): Comment for the convert action. Defaults to None.

        Returns:
            int: Action id of the convert action
        """
        if old_indices.shape != new_indices.shape or len(old_indices.shape) != 1:
            raise ValueError("Old and new element type must have the same length")
        if old_element not in self.masks or new_element not in self.masks:
            raise ValueError("Old and new element type must be initialized")
        if np.intersect1d(self.get_index(new_element), new_indices).size != 0:
            raise ValueError("New indices already occupied")
        if (
            np.intersect1d(self.get_index(old_element), old_indices).size
            != old_indices.size
        ):
            raise ValueError("Old indices not occupied")

        cur_action_id = self.action_id

        self.masks[old_element].append(
            {
                "action": "convert_from",
                "action_id": cur_action_id,
                "new_element": new_element,
                "data": np.array(old_indices, copy=True),
                "new_indices": np.array(new_indices, copy=True),
                "comment": comment,
            }
        )
        self.masks[new_element].append(
            {
                "action": "convert_to",
                "action_id": cur_action_id,
                "old_element": old_element,
                "old_indices": np.array(old_indices, copy=True),
                "data": np.array(new_indices, copy=True),
                "comment": comment,
            }
        )

        self.action_id += 1
        return cur_action_id

    def remove(
        self, element: str, indices: np.ndarray, comment: Optional[str] = None
    ) -> int:
        """Removes elements at the given position

        This will throw an error if the indices were not occupied previously

        Args:
            element (str): Element type
            indices (np.ndarray): Indices of the removed elements
        """
        if element not in self.masks:
            raise ValueError("Element type not initialized")
        if np.intersect1d(self.get_index(element), indices).size != indices.size:
            raise ValueError("Indices not occupied")

        cur_action_id = self.action_id
        self.masks[element].append(
            {
                "action": "remove",
                "action_id": cur_action_id,
                "data": np.array(indices, copy=True),
                "comment": comment,
            }
        )

        self.action_id += 1
        return cur_action_id

    def reindex(
        self, element: str, new_indices: np.ndarray, comment: Optional[str] = None
    ) -> int:
        """Reorders the indices of the given element type

        Args:
            element (str): Element type
            new_indices (np.ndarray): New indices of the element type
            comment (Optional[str], optional): Comment for the reindex action. Defaults to None.

        Returns:
            int: Action id of the reindexing action
        """
        if element not in self.masks:
            raise ValueError("Element type not initialized")
        if np.unique(new_indices).size != new_indices.size:
            raise ValueError("New indices must be unique")

        cur_action_id = self.action_id
        current_index = self.get_index(element)
        if current_index.size != new_indices.size:
            raise ValueError("New indices must have same length as old indices")
        self.masks[element].append(
            {
                "action": "reindex",
                "action_id": cur_action_id,
                "data": np.array(new_indices, copy=True),
                "old_index": np.array(current_index, copy=True),
                "comment": comment,
            }
        )

        self.action_id += 1
        return cur_action_id

    def get_index(self, element: str, action_id: Optional[int] = None) -> np.ndarray:
        """Returns the indices of the given element type

        Args:
            element (str): Element type
            action_id (Optional[int], optional): If passed, returns the indices after the given
                action id. Defaults to None, which returns the latest state.

        Returns:
            np.ndarray: Indices of the element type
        """
        if element not in self.masks:
            raise ValueError("Element type not initialized")

        mask = np.array([], dtype=int)
        for event in self.masks[element]:
            if action_id is not None and event["action_id"] > action_id:
                break
            if event["action"] not in forward_hooks:
                raise RuntimeError(f"Unknown action stored: {event['action']}")
            mask = forward_hooks[event["action"]](mask, event)

        return mask

    def traceback(
        self, element: str, index: int, action_id: Optional[int] = None
    ) -> Result[tuple[str, str, int, Optional[str]], str]:
        """Returns the origin of the given index

        Currently, only a single index can be traced back at a time.

        Args:
            element (str): Element type of the final index
            index (int): Index to trace back
            action_id (Optional[int], optional): If passed, ignores all events after the given
                action id. Defaults to None, which returns the origin from the latest state.

        Returns:
            Result[tuple[str, str, int, Optional[str]], str]: Returns Success with a tuple of
                element type, action (either add or init), index and comment if the index could be
                traced back, else returns Failure with an error message.
        """
        if element not in self.masks:
            raise ValueError("Element type not initialized")

        for event in self.masks[element][::-1]:
            # Forward to the last event that happened before the given action id
            if action_id is not None and event["action_id"] > action_id:
                continue

            # Init has to finalize the search
            if event["action"] == "init":
                if np.isin(index, event["data"]).any():
                    return Success((element, "init", index, event["comment"]))
                return Failure("Index could not be traced back, it was not initialized")

            # Add can finalize the search if the item was added
            elif event["action"] == "add" and np.isin(index, event["data"]).any():
                return Success((element, "add", index, event["comment"]))

            # Remove fails the search if the item was removed
            elif event["action"] == "remove" and np.isin(index, event["data"]).any():
                return Failure("Index could not be traced back, it was removed")

            # Reindex can fail the search if the item was not reindexed, else it transforms the index
            elif event["action"] == "reindex":
                ind_pos = np.flatnonzero(event["data"] == index)
                if ind_pos.size == 0:
                    return Failure(
                        "Index could not be traced back, it was not reindexed"
                    )

                index = event["old_index"][ind_pos.item()]

            # Convert to can fail the search if the item was not converted, else it recurses and
            # calls the traceback function on the old element, starting from the action id one
            # before the current action id
            elif (
                event["action"] == "convert_to" and np.isin(index, event["data"]).any()
            ):
                ind_pos = np.flatnonzero(event["data"] == index)
                if ind_pos.size == 0:
                    return Failure(
                        "Index could not be traced back, it was not converted"
                    )

                index = event["old_indices"][ind_pos.item()]
                element = event["old_element"]
                return self.traceback(element, index, event["action_id"] - 1)

            # Convert from can fail the search if the item was converted
            elif (
                event["action"] == "convert_from"
                and np.isin(index, event["data"]).any()
            ):
                return Failure("Index could not be traced back, it was converted")
        raise RuntimeError("No init event found")

    def save(self, path: str) -> None:
        """Saves the mask-manager to a file

        Args:
            path (str): Path to the file
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def from_file(path: str) -> "MaskManager":
        """Loads the mask-manager from a file

        Args:
            path (str): Path to the file

        Returns:
            MaskManager: Mask manager loaded from the file
        """
        with open(path, "rb") as f:
            return pickle.load(f)
