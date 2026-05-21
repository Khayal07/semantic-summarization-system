"""
Pytest configuration and fixtures.
"""

import pytest


@pytest.fixture
def sample_document():
    """Provide sample document for testing."""
    return """
    Machine learning has emerged as a transformative technology across industries.
    From healthcare to finance, the applications are vast and growing. Deep learning,
    a subset of machine learning, has achieved remarkable results in computer vision,
    natural language processing, and other domains. However, building effective ML systems
    requires not just powerful algorithms but also high-quality data, careful validation,
    and robust deployment strategies.
    """


@pytest.fixture
def sample_documents():
    """Provide multiple sample documents."""
    return [
        "Document one about machine learning and AI applications.",
        "Document two discussing neural networks and deep learning.",
        "Document three covering natural language processing advances.",
    ]
