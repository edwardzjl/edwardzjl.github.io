---
slug: install-postgres-on-osx
authors: [jlzhou]
tags: [postgres, osx]
---

# Install postgres on OSX

If you installed Postgres from homebrew, the default user `postgres` isn't automatically created, you need to run following command in your terminal:

<!-- truncate -->

```sh
/Applications/Postgres.app/Contents/Versions/9.*/bin/createuser -s postgres
```
