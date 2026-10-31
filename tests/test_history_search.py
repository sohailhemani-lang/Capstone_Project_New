from __future__ import annotations

from ui.history_search import search_history


def test_search_matches_message_content_case_insensitively():
    history = [
        {"role": "user", "content": "Who owns Payment-Service?"},
        {"role": "assistant", "content": "The Billing team owns it."},
    ]

    matches = search_history(history, "PAYMENT-SERVICE")

    assert len(matches) == 1
    assert matches[0].entry_index == 0
    assert matches[0].role == "user"
    assert [(field.name, field.value) for field in matches[0].fields] == [
        ("content", "Who owns Payment-Service?")
    ]


def test_search_finds_nested_response_and_citation_fields():
    history = [
        {
            "role": "assistant",
            "content": "Auth-DB is owned by Infrastructure.",
            "data": {
                "citations": [{"title": "Ownership policy", "doc_id": "doc-17"}],
                "route": "guardrails",
            },
        }
    ]

    matches = search_history(history, "own")

    assert len(matches) == 1
    assert {field.name for field in matches[0].fields} == {
        "content",
        "data.citations[0].title",
    }


def test_search_returns_no_results_for_blank_or_unmatched_queries():
    history = [{"role": "user", "content": "Hello"}]

    assert search_history(history, "  ") == []
    assert search_history(history, "missing") == []
