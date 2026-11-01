import warnings


def resolve_max_tokens(
    params: dict,
    generation_params: dict,
    prioritize_new_tokens: bool = False,
) -> dict:
    """Resolve ``max_tokens`` and ``max_new_tokens`` without ambiguity.

    Args:
        params: Dictionary containing user-provided parameters
        generation_params: Dictionary containing generation-specific parameters
        prioritize_new_tokens: If True, max_new_tokens takes precedence over max_tokens
                             If False, max_tokens takes precedence (default behavior)

    Returns:
        Updated generation_params dictionary
    """

    def get_token_params(param_dict: dict) -> tuple:
        """Extract max_tokens and max_new_tokens from a parameter dictionary."""
        return (
            param_dict.pop("max_tokens", None),
            param_dict.pop("max_new_tokens", None),
        )

    def resolve_tokens(max_tokens: int, max_new_tokens: int) -> int:
        """
        Resolve between max_tokens and max_new_tokens values based on priority.
        Returns the resolved token value or None if no valid value found.
        """
        if max_tokens is None:
            return max_new_tokens
        if max_new_tokens is None:
            return max_tokens

        if max_tokens != max_new_tokens:
            if prioritize_new_tokens:
                warnings.warn(
                    f"max_tokens ({max_tokens}) and max_new_tokens ({max_new_tokens}) "
                    f"are different. Using max_new_tokens value as it has priority."
                )
                return max_new_tokens
            warnings.warn(
                f"max_tokens ({max_tokens}) and max_new_tokens ({max_new_tokens}) "
                "differ; using max_tokens."
            )
            return max_tokens

        return max_tokens

    max_tokens, max_new_tokens = get_token_params(params)
    final_max_tokens = resolve_tokens(max_tokens, max_new_tokens)

    if final_max_tokens is None:
        max_tokens, max_new_tokens = get_token_params(generation_params)
        final_max_tokens = resolve_tokens(max_tokens, max_new_tokens)

    generation_params.pop("max_new_tokens", None)
    generation_params.pop("max_tokens", None)
    if final_max_tokens is not None:
        if prioritize_new_tokens:
            generation_params["max_new_tokens"] = final_max_tokens
        else:
            generation_params["max_tokens"] = final_max_tokens
    return generation_params
