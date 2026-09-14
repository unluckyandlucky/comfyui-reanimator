# Reanimator Bridge

Run [Reanimator](https://reanimator.app) generations on your own GPU, with your own models.

> **Experimental.** Local generation needs ComfyUI running on this computer plus the
> models each preset uses. It is the newest half of Reanimator and the half that depends
> on your machine, so expect rough edges. Cloud generation is unaffected: the editor
> switches back at any time.

**Your media never leaves this computer.** Source video, keyframes and results stay on your
disk. The only things that reach Reanimator's servers are your login, the template catalogue,
and which GPU you have.

Local generation is free on every plan.

## Install

ComfyUI Manager does not list the bridge yet (pending registry review). Until it does, pick one:

**Without a terminal.** Download the
[ZIP](https://github.com/unluckyandlucky/comfyui-reanimator/archive/refs/heads/main.zip), open it
and drag the `comfyui-reanimator-main` folder into `ComfyUI/custom_nodes`. Restart ComfyUI.
Avoid Windows' *Extract All*: it nests the folder one level too deep and ComfyUI won't find it.

**With git:**

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/unluckyandlucky/comfyui-reanimator
pip install cryptography Pillow
```

## Use

1. Open [reanimator.app](https://reanimator.app) and pick **Local GPU** in the editor.
2. Click **Connect this device**. Your browser will ask for permission to reach this computer —
   allow it.
3. ComfyUI shows an approval dialog with your account e-mail. Check it is yours, click **Allow**.
4. In the ComfyUI *Reanimator Bridge* panel, choose a **project folder**. That folder is the only
   place the editor can read media from.

Chrome and Edge are supported. Firefox is untested but expected to work. Safari cannot connect to
local servers from an HTTPS page.

## What it exposes

The bridge listens on **127.0.0.1 only** — never on your network — and accepts requests only from
`https://reanimator.app`, only with a token issued when you clicked Allow, and only for the
commands it implements. It will not run arbitrary workflows sent from the cloud: the server may
only name a template already installed here, with parameters validated against that template's
schema.

Revoke access at any time from the *Paired browsers* list in the ComfyUI panel.

## Configuration

Nothing to configure. The bridge picks port 8771 (or 8772/8773 if taken) and finds ComfyUI by
itself. **You do not need `--enable-cors-header`** — that flag exists for browsers talking
directly to ComfyUI, and nothing here does that.

State lives in `ComfyUI/user/reanimator/bridge.json`.

## Development

```bash
python tests/test_bridge.py                       # 41 security tests
python tools/generate_keypair.py --kid prod-2026-07
REANIMATOR_BRIDGE_DEV=1                           # also accept localhost:3000
```

Architecture and threat model: [`docs/plan-local-bridge.md`](../docs/plan-local-bridge.md).
