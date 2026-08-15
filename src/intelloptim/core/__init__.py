"""Pure domain types for intelloptim.

Modules here must not import ``httpx``, ``fastapi``, ``celery``, or
``optuna.storages`` — they form the framework-agnostic kernel that every
other component (client, server, worker) builds on.
"""
