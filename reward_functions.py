def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    correct_answer = answer[0] if answer else ""

    print(
        f"{'-'*20}\n"
        f"Question:\n{q}\n"
        f"Answer:\n{correct_answer}\n"
        f"Response:\n{responses[0]}\n"
        f"Extracted:\n{extracted_responses[0]}"
    )

    return [2.0 if r == correct_answer else 0.0 for r in extracted_responses]
