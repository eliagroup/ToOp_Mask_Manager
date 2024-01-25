# Mask Manager

This is a small utility to organize different masks in a lengthy preprocessing procedure that removes, adds, converts or reindexes elements.

# Usage

First, create a MaskManager and initialize some element types with their original index. An index assigns a unique number to each element of a category.

```
manager = MaskManager()
manager.init("line", np.arange(16, dtype=int))
manager.init("trafo", np.array([0, 4, 5, 9]))
manager.init("bus", np.arange(14, dtype=int))
```

Then, you can modify the indices, which should mirror the preprocessing operations you perform on that data.

```
manager.convert("line", np.array([1, 2]), "trafo", np.array([1, 2]))
manager.remove("line", np.array([0]), comment="remove-faulty-lines")
manager.reindex("trafo", np.arange(6))
```

Finally, you can obtain the final masks with 
```
manager.get_index("trafo")  # Returns np.array([0, 1, 2, 3, 4, 5])
```

And you can also traceback an individual index until the point it was introduced to the manager
```
manager.traceback("trafo", 4)  # Returns Success("line", "init", 1, None)
```
