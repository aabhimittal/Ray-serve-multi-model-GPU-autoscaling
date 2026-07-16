## Summary

<!-- What does this PR change and why? -->

## Changes

<!-- Bullet the notable changes. -->
-

## How to test

<!-- Commands / steps a reviewer can run. -->
```bash
make install-dev
make check          # lint + type-check + unit tests
make test-int       # end-to-end Ray Serve test (optional, needs ray)
```

## Checklist

- [ ] `make lint` passes
- [ ] `make test` passes
- [ ] Docs/README updated if behavior changed
- [ ] Config changes reflected in `config/models.yaml` and `config/serve_config.yaml`
