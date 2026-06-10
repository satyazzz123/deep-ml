import torch


def OSA(source: str, target: str) -> torch.Tensor:
    m = len(source)
    n = len(target)

    dp = torch.zeros((m + 1, n + 1), dtype=torch.int32)

    # Base cases
    for i in range(m + 1):
        dp[i][0] = i

    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):

            # Cost of substitution
            if source[i - 1] == target[j - 1]:
                cost = 0
            else:
                cost = 1

            # Standard Levenshtein operations
            dp[i][j] = min(
                dp[i - 1][j] + 1,        # deletion
                dp[i][j - 1] + 1,        # insertion
                dp[i - 1][j - 1] + cost  # substitution/match
            )

            # OSA transposition
            if (
                i > 1
                and j > 1
                and source[i - 1] == target[j - 2]
                and source[i - 2] == target[j - 1]
            ):
                dp[i][j] = min(
                    dp[i][j],
                    dp[i - 2][j - 2] + 1
                )

    return dp[m][n]