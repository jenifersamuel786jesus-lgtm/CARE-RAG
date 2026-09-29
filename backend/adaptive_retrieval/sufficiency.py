def is_sufficient(coverage: float, threshold: float, aspect_count: int, selected_count: int) -> bool:
    return coverage >= threshold or (coverage >= .66 and aspect_count == 1 and selected_count >= 1)
