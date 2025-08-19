import pytest

part1_queries = [
    "Under what circumstances does Neo see a white rabbit?",
    "How did Trinity and Neo first meet?", 
    "Why is there no sunlight in the future?",
    "Who needs solar power to survive?",
    "Why do the Agents want to capture Morpheus?",
    "Describe the Nebuchadnezzar.",
    "What is Nebuchadnezzar's crew made up of?"
]


@pytest.mark.parametrize("query", part1_queries)
def test_part1_queries(client, query):
    """
    Test Part 1 queries against the RAG service.
    """
    response = client.post("/agent/query", json={"query": query})
    assert response.status_code == 200
    json_response = response.json()
    assert "answer" in json_response
    assert json_response["confidence"] > 0.5 and json_response["confidence"] <= 1.0
