import pytest
import logging
from unittest.mock import patch, MagicMock

part2_queries = [
    "How many times does Morpheus mention that Neo is the One?",
    "Why are humans similar to a virus? And who says that?",
    "Describe Cypher's personality.",
    "What does Cypher offer to the Agents, and in exchange for what?",
    "What is the purpose of the human fields, and who created them?"
]


@pytest.mark.parametrize("query", part2_queries)
def test_part2_queries(client, query, caplog):
    """
    Test Part 2 queries against the advanced version of the RAG service.
    """
    caplog.set_level(logging.INFO)
    
    response = client.post("/agent/query", json={"query": query})
    assert response.status_code == 200
    json_response = response.json()
    
    assert "answer" in json_response
    assert json_response["confidence"] > 0.0 and json_response["confidence"] <= 1.0
    assert "Detected complex query - using advanced reasoning approach" in caplog.text
