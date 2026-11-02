Authentication and token handling
=================================

Every request to an intelloptim server carries an **API token**, a secret that
identifies who is calling. The token is sent as a bearer token in the
``Authorization`` header, never in the URL, so it does not end up in access logs
or browser histories.

How the client finds the token
------------------------------

:class:`intelloptim.client.Client` resolves the token from the first of these
sources that yields a non-empty value:

1. the ``token=`` argument,
2. the ``INTELLOPTIM_TOKEN`` environment variable,
3. the file ``~/.config/intelloptim/token``.

Surrounding whitespace is stripped. If none provides a token,
:class:`~intelloptim.client.errors.MissingTokenError` is raised when the client is
created, so a missing token fails immediately and not on the first request.

.. code-block:: python

   import intelloptim as intopt

   # token from INTELLOPTIM_TOKEN or the token file
   client = intopt.client.Client(url="https://intelloptim.example.org")

How the token file is protected
-------------------------------

On POSIX systems the client reads the token file only if it is safe: it must be
a regular file, owned by the current user, with no permissions for the group or
for others. A symbolic link, or a file that other users can read, is refused with
:class:`~intelloptim.client.errors.InsecureTokenFileError`. Create the file like
this:

.. code-block:: console

   mkdir -p ~/.config/intelloptim
   touch ~/.config/intelloptim/token
   chmod 600 ~/.config/intelloptim/token
   $EDITOR ~/.config/intelloptim/token      # paste the token, then save

For containers and secret managers that do not offer such a file, use the
environment variable.

Good practice
-------------

* **Treat the token as a password.** Do not put it in source control, in
  command-line arguments (they show up in process lists), in study
  specifications, in objective code or in logs.
* **Use HTTPS** for any server that is not on your own machine. The client
  verifies the server's certificate by default. Pass ``verify=False`` only for a
  self-signed certificate on a local test stack.
* **Inject the token from the platform:** a secret manager, a CI secret or a
  protected file on the cluster (see :doc:`cluster`), rather than copying it by
  hand.
* **Rotate tokens** according to the policy of the server you use, and replace a
  token that may have leaked as soon as possible.

What a token stands for
-----------------------

The server maps each token to a **user or a group**, and that identity is part of
the identity of every study: two different users who declare the same study get
two different studies, while all the processes that act as the same user or
group work on the same one. The token itself is never part of the
hash that identifies a study, so rotating a token does not change your studies.

Alongside the token, every request carries a digest of the study configuration,
which the server compares with the stored one. This is a consistency check, not
authentication, but it protects the results: a process with another configuration
is refused (``409``) rather than silently mixing results.

When authentication or limits fail
----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - What you see
     - Meaning
   * - :class:`~intelloptim.client.errors.MissingTokenError`
     - No token was found in any of the three sources.
   * - :class:`~intelloptim.client.errors.InsecureTokenFileError`
     - The token file exists but is not safe to read. Fix its permissions or use
       ``INTELLOPTIM_TOKEN``.
   * - :class:`~intelloptim.client.errors.IntelloptimAPIError` with
       ``status_code == 401``
     - The server does not accept the token: it is wrong or no longer valid.
   * - :class:`~intelloptim.client.errors.IntelloptimAPIError` with
       ``status_code == 429``
     - The call quota is exhausted on the server.
   * - :class:`~intelloptim.client.errors.InsufficientQuotaError`
     - ``optimize`` found, before sending anything, that ``n_trials`` exceeds the
       calls you have left.
   * - :class:`~intelloptim.client.errors.StudyLimitExceededError`
     - You have reached the cap on active studies.

How tokens are issued, rotated and revoked, and what quotas apply, is decided by
whoever deploys the server. Ask them, or see the server's own documentation.
