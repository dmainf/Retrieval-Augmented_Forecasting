"""Chronos-Bolt backbone wiring.

- ``ChronosBoltModelForForecasting`` is re-exported from the installed
  ``chronos`` package (compatible with current ``transformers``) and used by the
  ``none`` and ``raf`` methods.
- ``ChronosBoltModelForForecastingWithRetrieval`` is our Cross-RAG subclass used
  by the ``cross_raf`` method.
"""
from chronos.chronos_bolt import ChronosBoltModelForForecasting, ChronosBoltOutput

from .retrieval_model import ChronosBoltModelForForecastingWithRetrieval

__all__ = [
    "ChronosBoltOutput",
    "ChronosBoltModelForForecasting",
    "ChronosBoltModelForForecastingWithRetrieval",
]
