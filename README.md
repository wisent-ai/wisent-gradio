<!-- wisent-banner:start -->
<p align="center">
  <img src="assets/readme-banner.webp" alt="wisent-gradio by Wisent" width="100%">
</p>
<!-- wisent-banner:end -->

# wisent-gradio (superseded)

This repository held `wisent-gradio`, a Gradio interface that grouped the
Python `wisent` CLI's commands into tabs and ran each one through
`wisent.app.core.runner`. It imported the `wisent` package (`wisent.core…`),
which the Rust cutover replaced with [Ster](https://github.com/wisent-ai/ster),
so none of it could start any longer; the fleet holds no Python. The package,
its release manifest, its release-surface tool and its publishing workflows
were removed. The versions already on PyPI remain as published.

The interface also decided with numbers nobody stated: onboarding bounds and
a request wait, a pair chunk copied from the extractors, a Hugging Face socket
wait and pool size, a legacy extraction worker count, a token cap and a
contrastive split. Its replacement takes each of them from the run that
needs it or refuses by name.

| `wisent-gradio` | Ster |
|---|---|
| the Gradio tabs | [Ster Desktop](https://github.com/wisent-ai/ster-desktop), which runs each workflow as one `ster request` process |
| a command of the `wisent` CLI | its Ster verb, mapped one by one in [Ster's README](https://github.com/wisent-ai/ster#readme) |
| the hyphenated verbs (`generate-pairs`, `optimize-steering`, `diagnose-vectors`, …) | `ster pairs …`, `ster vector …`, `ster generate`, `ster extract` |
| the `synthetic` alias | none: `ster pairs synthesize` is the one name |

[Desktop requests](https://github.com/wisent-ai/ster/blob/main/docs/guide/desktop-requests.md)
documents the request body, the events and the exit statuses Ster Desktop
reads.
