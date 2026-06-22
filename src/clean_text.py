import string
import pandas as pd


def clean_text(txt):
    """
    Clean a single text token by removing whitespace and noise punctuation.

    Keeps characters important for entities:
      @ . - /   (emails, phones, websites)

    Args:
        txt: Any value (converted to string internally).

    Returns:
        str: Cleaned lowercase string. Empty string '' if nothing remains.
    """
    # Convert to string first (handles None, integers, floats, etc.)
    text = str(txt)

    # Lowercase everything so "JOHN" and "john" are treated the same
    text = text.lower()

    # ── Remove whitespace ──────────────────────────────────────────────────────
    # string.whitespace = ' \t\n\r\x0b\x0c'  (space, tab, newline, etc.)
    whitespace_table = str.maketrans('', '', string.whitespace)
    text = text.translate(whitespace_table)

    # ── Remove noisy punctuation ───────────────────────────────────────────────
    # We intentionally EXCLUDE @ . - / from this list
    punctuation_to_remove = "!#$%&\'()*+:;<=>?[\\]^`{|}~"
    punctuation_table = str.maketrans('', '', punctuation_to_remove)
    text = text.translate(punctuation_table)

    return text


def clean_dataframe(df):
    """
    Apply clean_text() to the 'text' column of a DataFrame.
    Removes rows where text becomes empty after cleaning.

    Args:
        df (pandas.DataFrame): Must have a 'text' column.

    Returns:
        pandas.DataFrame: Cleaned copy with empty rows removed.
    """
    df = df.copy()   # never modify the original - work on a copy

    df['text'] = df['text'].apply(clean_text)

    # Drop rows where cleaning left nothing (e.g., rows that were just spaces)
    df = df[df['text'] != '']

    df.reset_index(drop=True, inplace=True)
    return df


if __name__ == '__main__':
    test_cases = [
        ("Hello!",           "hello"),
        ("JOHN",             "john"),
        ("  Smith  ",        "smith"),
        ("john@email.com",   "john@email.com"),
        ("123-456-7890",     "123-456-7890"),
        ("www.company.com",  "www.company.com"),
        ("  ",               ""),
        ("ABC#$%Company",    "abccompany"),
        (None,               "none"),
        (12345,              "12345"),
    ]

    print("Testing clean_text()")
    print(f"{'Input':<25}  {'Expected':<20}  {'Got':<20}  {'PASS?'}")
    print("-" * 75)

    all_passed = True
    for raw, expected in test_cases:
        got  = clean_text(raw)
        ok   = got == expected
        mark = "PASS" if ok else "FAIL"
        if not ok:
            all_passed = False
        print(f"{str(raw):<25}  {expected:<20}  {got:<20}  {mark}")

    print("-" * 75)
    print("All tests passed!" if all_passed else "Some tests FAILED - check your code.")