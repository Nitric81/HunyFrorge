# Security Policy

## Supported versions

HunyForge is pre-1.0 and under active development. Security fixes are applied to `main`; there are no backported release branches yet. Run the latest commit on `main`.

## Reporting a vulnerability

Please do **not** open a public issue for a suspected vulnerability.

Report privately via [GitHub Security Advisories](https://github.com/Nitric81/HunyFrorge/security/advisories/new) ("Report a vulnerability"). Include:

- The affected component (API, worker, pipeline, packaging scripts, Docker config, frontend).
- Steps to reproduce or a proof of concept.
- The impact you believe is possible.

You will get an acknowledgement as soon as possible. We will coordinate a fix and disclosure timeline with you.

## Scope notes

HunyForge is a **single-user local tool** with a specific threat model — please read it before reporting:

- The API has **no authentication or authorization by design**, and is only safe because it binds to `127.0.0.1`. "The API is unauthenticated" is a known property, not a vulnerability. Misconfiguring your deployment to listen on a public interface is a deployment issue.
- In scope: path traversal outside the job/data directories, command injection from user-controlled input, unsafe deserialization, committed secrets, dependency vulnerabilities with a practical exploit path, CORS/origin bypass, and anything that lets a remote party reach a correctly-configured localhost deployment.
- The bundled `setup-unreal.ps1` / `HunyForgeUnrealSetup.py` run inside the user's own Unreal Editor install by design; report only cases where they execute attacker-controlled content from another user.

## Security expectations for self-hosting

- Keep the published ports bound to `127.0.0.1`. If you front the API with a reverse proxy or expose it on a LAN, add your own authentication (e.g. a proxy-level bearer token or mTLS) first.
- Do not commit `.env` files, model paths containing credentials, or job data — `.gitignore` covers these; keep it that way.
- Model weights are user-supplied and downloaded over your own Hugging Face account; their licenses and integrity are your responsibility.
