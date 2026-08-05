# Changelog

## [0.2.0](https://github.com/powabase-ai/powabase-ai/compare/v0.1.3...v0.2.0) (2026-08-05)


### Features

* rate-limit handling for Firecrawl/Exa with pacing, Retry-After re-queue, and sources.error_code ([7b46bae](https://github.com/powabase-ai/powabase-ai/commit/7b46baeb71ff54a0e6110ecb0074c3cb5957baa6))


### Bug Fixes

* **ci:** re-run the title check when the head moves ([#33](https://github.com/powabase-ai/powabase-ai/issues/33)) ([23d3cad](https://github.com/powabase-ai/powabase-ai/commit/23d3cad7c18aa209107bdbcf13bfa2aae8e9f33b))


### Documentation

* add a contributing guide ([#31](https://github.com/powabase-ai/powabase-ai/issues/31)) ([e3d4547](https://github.com/powabase-ai/powabase-ai/commit/e3d4547d85f2916f0660136ff37231dfebb228a9))

## [0.1.3](https://github.com/powabase-ai/powabase-ai/compare/v0.1.2...v0.1.3) (2026-08-05)


### Bug Fixes

* **deps:** powabase-agentic 0.2.0 ([#27](https://github.com/powabase-ai/powabase-ai/issues/27)) ([2b61809](https://github.com/powabase-ai/powabase-ai/commit/2b61809aa55fa29127c4f4e8b9a1c615c1f79608))

## [0.1.2](https://github.com/powabase-ai/powabase-ai/compare/v0.1.1...v0.1.2) (2026-08-05)


### Bug Fixes

* **deps:** pystemmer 3.1.0, which has a linux/arm64 wheel ([#24](https://github.com/powabase-ai/powabase-ai/issues/24)) ([78d3235](https://github.com/powabase-ai/powabase-ai/commit/78d3235056cd8e0c368dfe6204c3a88eda551a18))

## [0.1.1](https://github.com/powabase-ai/powabase-ai/compare/v0.1.0...v0.1.1) (2026-08-05)


### Bug Fixes

* **build:** take the agentic version from uv.lock, not a build-arg ([#21](https://github.com/powabase-ai/powabase-ai/issues/21)) ([558520d](https://github.com/powabase-ai/powabase-ai/commit/558520d4b83c82cd4c00ecee9454f244cac438a1))
* **deps:** bump powabase-agentic from 0.1.0rc2 to 0.1.0rc3 ([#14](https://github.com/powabase-ai/powabase-ai/issues/14)) ([df067bf](https://github.com/powabase-ai/powabase-ai/commit/df067bf8933cae876178e59eaf0c54e8ae89b4b5))
* **deps:** declare dev tooling as a PEP 735 dependency-group ([#5](https://github.com/powabase-ai/powabase-ai/issues/5)) ([dad12fd](https://github.com/powabase-ai/powabase-ai/commit/dad12fd3c738554df7b81c89fb30d7c19d9f136c))
* **deps:** request the rerankers extra so the lock carries zeroentropy ([#19](https://github.com/powabase-ai/powabase-ai/issues/19)) ([5fbc4d2](https://github.com/powabase-ai/powabase-ai/commit/5fbc4d2ed560bddb97fe5af663d44063650ca7a0))
* flow orchestration-hook + compaction-settings fixes; pin powabase-agentic 0.1.0rc2 ([#1](https://github.com/powabase-ai/powabase-ai/issues/1)) ([c899c9f](https://github.com/powabase-ai/powabase-ai/commit/c899c9f2a3ce1c5a335a169c28f7f22aceb0546b))


### Documentation

* correct published image name + two dead API routes; clarify the stack pulls this image ([5564337](https://github.com/powabase-ai/powabase-ai/commit/5564337ce31b9a3b52ccdf8a9bb8ed73704e8ba6))
