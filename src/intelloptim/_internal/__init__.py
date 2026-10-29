"""Internal modules shared by the client, the worker and the server.

Nothing here is a public API: names and behaviour can change between releases
without notice. Import these modules from their own path, for example
``from intelloptim._internal.schemas import AskRequest``.

Submodules:

* :mod:`~intelloptim._internal.schemas` — request and response models of the
  REST API, and the protocol version.
* :mod:`~intelloptim._internal.hashing` — canonical serialisation and the
  hashes that give studies their identity.
* :mod:`~intelloptim._internal.logging` — logging configuration under the
  ``intelloptim`` logger.
* :mod:`~intelloptim._internal.audit` — audit records of security-relevant
  events.
"""

__all__ = ["audit", "hashing", "logging", "schemas"]
