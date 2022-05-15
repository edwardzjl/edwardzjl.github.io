---
slug: install-postgres-on-osx
title: Install postgres on OSX
authors: [jlzhou]
tags: [postgres, osx]
---

If you installed Postgres from homebrew, the default user `postgres` isn't automatically created, you need to run following command in your terminal:

```sh
/Applications/Postgres.app/Contents/Versions/9.*/bin/createuser -s postgres
```
