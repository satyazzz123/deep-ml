SELECT user_id
FROM (
    SELECT user_id,
           EXTRACT(MONTH FROM purchase_date) AS month
    FROM purchases
    WHERE purchase_date >= '2024-01-01'
      AND purchase_date < '2025-01-01'
    GROUP BY user_id, EXTRACT(MONTH FROM purchase_date)
    HAVING COUNT(*) >= 2
) monthly
GROUP BY user_id
HAVING COUNT(*) = 12
ORDER BY user_id;