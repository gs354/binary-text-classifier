from binary_text_classifier.text_processing_functions import remove_currency_from_text


def test_remove_currency():
    replace_str = "currency"
    data = ["£200 is a bargain", "Win $ 400 when you text givemecash to 90888."]

    result_no_replacement = [" is a bargain", "win  when you text givemecash to 90888."]

    result_with_replacement = [
        f"{replace_str} is a bargain",
        f"win {replace_str} when you text givemecash to 90888.",
    ]

    assert (
        remove_currency_from_text(data=data, replace_str=None) == result_no_replacement
    )

    assert (
        remove_currency_from_text(data=data, replace_str=replace_str)
        == result_with_replacement
    )
