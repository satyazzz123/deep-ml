from typing import Dict, FrozenSet, Iterable, List, Set, Optional


def apriori(transactions: List[Set], min_support: float = 0.5, max_length: Optional[int] = None) -> Dict[frozenset, float]:

    # transactions = list(transactions)

    if len(transactions)==0:
        raise ValueError("min_support must be between 0 and 1")

    transaction_count = len(transactions)

    # If min_support is None, use 0 as the minimum support.
    if min_support is None:
        min_support = 0.0

    if not 0 <= min_support <= 1:
        raise ValueError("min_support must be between 0 and 1")

    if max_length is not None and max_length < 1:
        raise ValueError("max_length must be at least 1 or None")

    def support(itemset: frozenset) -> float:
        count = sum(
            itemset.issubset(transaction)
            for transaction in transactions
        )
        return count / transaction_count

    # -------------------------
    # L1: frequent 1-itemsets
    # -------------------------

    items = set()

    for transaction in transactions:
        items.update(transaction)

    current = {
        frozenset([item])
        for item in items
    }

    current = {
        itemset: support(itemset)
        for itemset in current
        if support(itemset) >= min_support
    }

    # Store ALL frequent itemsets
    all_frequent = dict(current)

    # -------------------------
    # Generate larger itemsets
    # -------------------------

    while current:

        current_length = len(next(iter(current)))

        if max_length is not None and current_length >= max_length:
            break

        # Generate candidates by joining frequent itemsets
        previous_itemsets = list(current.keys())

        candidates = set()

        for i in range(len(previous_itemsets)):
            for j in range(i + 1, len(previous_itemsets)):

                union = previous_itemsets[i] | previous_itemsets[j]

                if len(union) == current_length + 1:
                    candidates.add(union)

        # Count/support candidates
        next_level = {}

        for candidate in candidates:
            candidate_support = support(candidate)

            if candidate_support >= min_support:
                next_level[candidate] = candidate_support

        # No more frequent itemsets
        if not next_level:
            break

        all_frequent.update(next_level)

        current = next_level

    return all_frequent