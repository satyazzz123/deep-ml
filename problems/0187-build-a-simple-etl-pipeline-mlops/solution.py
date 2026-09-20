def run_etl(csv_text: str) -> list[tuple[str, float]]:
    lines = csv_text.splitlines()

    totals = {}

    for line in lines[1:]:
        if not line.strip():
            continue

        parts = line.split(",")

        user_id = parts[0].strip()
        event_type = parts[1].strip()
        value_text = parts[2].strip()

        if event_type != "purchase":
            continue

        try:
            value = float(value_text)
        except ValueError:
            continue

        if user_id not in totals:
            totals[user_id] = 0.0

        totals[user_id] += value

    return sorted(totals.items())