from src.core.indexer import _chunk_plain_text


def test_chunk_plain_text_respects_chunk_size_and_overlap():
    text = "one two three four five six seven eight nine ten"

    chunks = _chunk_plain_text(
        text=text,
        chunk_size=4,
        chunk_overlap=1,
    )

    assert chunks == [
        "one two three four",
        "four five six seven",
        "seven eight nine ten",
    ]