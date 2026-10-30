Authentication and token handling
---------------------------------

Every request carries an API bearer token. :class:`intelloptim.client.Client`
resolves it in this order: an explicit ``token=`` argument, the
``INTELLOPTIM_TOKEN`` environment variable, then
``~/.config/intelloptim/token``.

For production, prefer a secret manager or an environment variable injected by
the execution platform. On POSIX hosts, the token file must be a regular file
owned by the current user and inaccessible to group and other users. Symlinks
and insecure permissions are rejected.

Use HTTPS for any non-local server. Treat the token as a password: never put it
in source control, command-line arguments, study specifications, or logs.
