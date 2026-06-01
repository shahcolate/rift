"""Rift's self-hosted control plane — continuous drift monitoring.

The CLI answers "did this upgrade break anything?" *once*, on a laptop, when a
human runs it. This package turns that into a standing service: define a
*monitor* (a suite + baseline + challenger + a schedule), and Rift re-runs the
comparison on a cron, stores the history, surfaces it on a dashboard, and fires
an alert the moment a model silently regresses.

It is a thin orchestration + persistence layer over the existing engine
(:func:`rift.runner.run_suite`, :func:`rift.comparator.compare_runs`,
:mod:`rift.observability`) — no eval logic is reimplemented here.

Everything in this package lives behind the optional ``[server]`` extra so the
core ``pip install rift-eval`` stays lean. Import :func:`create_app` to build
the FastAPI application (see :mod:`rift.server.app`).
"""

from __future__ import annotations


def create_app(*args, **kwargs):
    """Lazy re-export of :func:`rift.server.app.create_app`.

    Deferred so importing the package name doesn't require FastAPI to be
    installed until an app is actually built.
    """
    from .app import create_app as _create_app

    return _create_app(*args, **kwargs)


__all__ = ["create_app"]
