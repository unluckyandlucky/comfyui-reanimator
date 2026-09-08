"""Security tests for the Reanimator Bridge.

Every case here corresponds to an attack the design claims to stop. Run with:

    python -m unittest discover -s comfyui-reanimator/tests -v

stdlib unittest deliberately: ComfyUI's Python environment cannot be assumed to
have pytest.
"""

from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives import serialization
except ImportError:  # pragma: no cover
    raise SystemExit("These tests need 'cryptography': pip install cryptography")

from reanimator import config, jws, keys, media, pairing, projects  # noqa: E402
from reanimator.comfy import runner  # noqa: E402
from reanimator.workflow import benchmarks, binder, geometry, templates, validate  # noqa: E402


def b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def make_jws(private: Ed25519PrivateKey, claims: dict, *, kid="test", alg="EdDSA") -> str:
    header = b64url(json.dumps({"alg": alg, "typ": "JWT", "kid": kid}).encode())
    payload = b64url(json.dumps(claims).encode())
    signature = private.sign(f"{header}.{payload}".encode("ascii"))
    return f"{header}.{payload}.{b64url(signature)}"


class BridgeTestCase(unittest.TestCase):
    """Isolates config in a temp dir so tests never touch a real install."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self._orig_user_dir = config._comfy_user_dir
        config._comfy_user_dir = lambda: root  # type: ignore[assignment]
        config.reset_cache()  # o el config del test anterior se cuela aqui

        self.private = Ed25519PrivateKey.generate()
        public = self.private.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )
        self._orig_keys = keys.TRUSTED_KEYS
        keys.TRUSTED_KEYS = (keys.PublicKey(key_id="test", raw=public),)

        self.nonces = pairing.NonceStore()
        pairing._jtis = pairing._JtiStore()

    def tearDown(self) -> None:
        config._comfy_user_dir = self._orig_user_dir  # type: ignore[assignment]
        config.reset_cache()
        keys.TRUSTED_KEYS = self._orig_keys
        self._tmp.cleanup()

    def claims(self, **overrides) -> dict:
        nonce, _ = self.nonces.issue()
        now = int(time.time())
        base = {
            "iss": pairing.ISSUER,
            "aud": config.audience(),
            "sub": "usr_123",
            "email": "user@example.com",
            "origin": config.ALLOWED_ORIGIN,
            "nonce": nonce,
            "iat": now,
            "exp": now + 120,
            "jti": base64.b64encode(os.urandom(9)).decode(),
        }
        base.update(overrides)
        return base


class TestAssertion(BridgeTestCase):
    def test_valid_assertion_is_accepted(self):
        token = make_jws(self.private, self.claims())
        verified = pairing.verify_assertion(token, self.nonces)
        self.assertEqual(verified["sub"], "usr_123")

    def test_nonce_is_single_use(self):
        claims = self.claims()
        pairing.verify_assertion(make_jws(self.private, claims), self.nonces)
        # Same nonce, fresh jti: only the nonce store can stop this.
        replay = dict(claims, jti="different")
        with self.assertRaises(pairing.PairingError) as ctx:
            pairing.verify_assertion(make_jws(self.private, replay), self.nonces)
        self.assertEqual(ctx.exception.code, "bad_nonce")

    def test_assertion_replay_is_rejected_by_jti(self):
        token = make_jws(self.private, self.claims())
        pairing.verify_assertion(token, self.nonces)
        with self.assertRaises(pairing.PairingError):
            pairing.verify_assertion(token, self.nonces)

    def test_wrong_audience_is_rejected(self):
        token = make_jws(
            self.private, self.claims(aud="reanimator-bridge:someone-elses-machine")
        )
        with self.assertRaises(pairing.PairingError) as ctx:
            pairing.verify_assertion(token, self.nonces)
        self.assertEqual(ctx.exception.code, "bad_audience")

    def test_expired_assertion_is_rejected(self):
        now = int(time.time())
        token = make_jws(self.private, self.claims(iat=now - 600, exp=now - 300))
        with self.assertRaises(pairing.PairingError) as ctx:
            pairing.verify_assertion(token, self.nonces)
        self.assertEqual(ctx.exception.code, "expired")

    def test_overlong_lifetime_is_rejected(self):
        now = int(time.time())
        token = make_jws(self.private, self.claims(iat=now, exp=now + 86400))
        with self.assertRaises(pairing.PairingError):
            pairing.verify_assertion(token, self.nonces)

    def test_origin_mismatch_is_rejected(self):
        token = make_jws(self.private, self.claims(origin="https://evil.example"))
        with self.assertRaises(pairing.PairingError) as ctx:
            pairing.verify_assertion(token, self.nonces)
        self.assertEqual(ctx.exception.code, "bad_origin")

    def test_wrong_issuer_is_rejected(self):
        token = make_jws(self.private, self.claims(iss="https://evil.example"))
        with self.assertRaises(pairing.PairingError) as ctx:
            pairing.verify_assertion(token, self.nonces)
        self.assertEqual(ctx.exception.code, "bad_issuer")

    def test_signature_from_another_key_is_rejected(self):
        attacker = Ed25519PrivateKey.generate()
        token = make_jws(attacker, self.claims())
        with self.assertRaises(pairing.PairingError) as ctx:
            pairing.verify_assertion(token, self.nonces)
        self.assertEqual(ctx.exception.code, "bad_signature")

    def test_alg_none_is_rejected(self):
        claims = self.claims()
        header = b64url(json.dumps({"alg": "none", "typ": "JWT", "kid": "test"}).encode())
        payload = b64url(json.dumps(claims).encode())
        with self.assertRaises(pairing.PairingError):
            pairing.verify_assertion(f"{header}.{payload}.", self.nonces)

    def test_algorithm_confusion_is_rejected(self):
        token = make_jws(self.private, self.claims(), alg="HS256")
        with self.assertRaises(pairing.PairingError):
            pairing.verify_assertion(token, self.nonces)

    def test_unknown_kid_is_rejected(self):
        token = make_jws(self.private, self.claims(), kid="not-a-real-key")
        with self.assertRaises(pairing.PairingError) as ctx:
            pairing.verify_assertion(token, self.nonces)
        self.assertEqual(ctx.exception.code, "unknown_key")

    def test_placeholder_key_refuses_to_pair(self):
        keys.TRUSTED_KEYS = (keys.PublicKey(key_id="test", raw=b"\x00" * 32),)
        token = make_jws(self.private, self.claims())
        with self.assertRaises(pairing.PairingError) as ctx:
            pairing.verify_assertion(token, self.nonces)
        self.assertEqual(ctx.exception.code, "placeholder_key")

    def test_failed_verification_does_not_burn_the_nonce(self):
        """A bad attempt must not consume the nonce the real browser holds."""
        claims = self.claims()
        attacker = Ed25519PrivateKey.generate()
        with self.assertRaises(pairing.PairingError):
            pairing.verify_assertion(make_jws(attacker, claims), self.nonces)
        # The legitimate assertion, same nonce, still works.
        good = pairing.verify_assertion(make_jws(self.private, claims), self.nonces)
        self.assertEqual(good["sub"], "usr_123")


class TestTokens(BridgeTestCase):
    def _issue(self) -> str:
        claims = self.claims()
        pairing.verify_assertion(make_jws(self.private, claims), self.nonces)
        approval = pairing.approvals.create(claims, "Test PC")
        pairing.approvals.resolve(approval.request_id, True)
        return pairing.tokens.issue(approval, "test browser").token

    def test_valid_token_is_accepted(self):
        token = self._issue()
        entry = pairing.tokens.validate(token, config.ALLOWED_ORIGIN)
        self.assertEqual(entry.email, "user@example.com")

    def test_revoked_token_is_rejected(self):
        token = self._issue()
        self.assertTrue(pairing.tokens.revoke(token=token))
        with self.assertRaises(pairing.PairingError) as ctx:
            pairing.tokens.validate(token, config.ALLOWED_ORIGIN)
        self.assertEqual(ctx.exception.code, "unauthenticated")

    def test_token_bound_to_origin(self):
        token = self._issue()
        with self.assertRaises(pairing.PairingError) as ctx:
            pairing.tokens.validate(token, "https://evil.example")
        # Its own code, not bad_origin: the bridge may well accept this origin,
        # the token simply belongs to another one. Conflating the two makes the
        # editor tell the user to change a setting when it should tell them to
        # pair again.
        self.assertEqual(ctx.exception.code, "token_origin_mismatch")

    def test_missing_token_is_rejected(self):
        with self.assertRaises(pairing.PairingError):
            pairing.tokens.validate(None, config.ALLOWED_ORIGIN)

    def test_listing_never_leaks_the_token(self):
        self._issue()
        for row in pairing.tokens.list():
            self.assertNotIn("token", row)

    def test_rejected_approval_yields_no_token(self):
        claims = self.claims()
        approval = pairing.approvals.create(claims, "Test PC")
        pairing.approvals.resolve(approval.request_id, False)
        self.assertEqual(approval.state, "rejected")
        with self.assertRaises(pairing.PairingError):
            pairing.approvals.resolve(approval.request_id, True)


class TestConfigRobustness(BridgeTestCase):
    def test_config_with_bom_still_loads(self):
        """PowerShell's Out-File and Notepad both add a BOM. Reading as plain
        utf-8 would discard the whole file and silently reset the device id and
        every pairing."""
        config.update(dev_origins=True, device_label="Test PC")
        path = config.config_path()
        raw = path.read_text(encoding="utf-8")
        path.write_bytes(b"\xef\xbb\xbf" + raw.encode("utf-8"))
        config.reset_cache()

        data = config.load()
        self.assertTrue(data["dev_origins"])
        self.assertEqual(data["device_label"], "Test PC")

    def test_corrupt_config_is_kept_not_discarded(self):
        config.update(device_label="Test PC")
        path = config.config_path()
        path.write_text("{ this is not json", encoding="utf-8")
        config.reset_cache()

        config.load()  # must not raise
        self.assertTrue(path.with_suffix(".json.bad").exists())


class TestMediaConfinement(BridgeTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.root = Path(self._tmp.name) / "projects"
        (self.root / "sub").mkdir(parents=True)
        self.video = self.root / "sub" / "clip.mp4"
        self.video.write_bytes(b"0123456789" * 100)
        self.secret = Path(self._tmp.name) / "secret.mp4"
        self.secret.write_bytes(b"private")

    def test_relative_reference_resolves(self):
        path = media.resolve_within(self.root, "sub/clip.mp4")
        self.assertEqual(path, self.video.resolve())

    def test_path_traversal_is_rejected(self):
        for ref in ("../secret.mp4", "sub/../../secret.mp4", "..\\secret.mp4"):
            with self.subTest(ref=ref), self.assertRaises(media.MediaError):
                media.resolve_within(self.root, ref)

    def test_absolute_path_is_rejected(self):
        for ref in (str(self.secret), "/etc/passwd", "C:\\Windows\\win.ini"):
            with self.subTest(ref=ref), self.assertRaises(media.MediaError):
                media.resolve_within(self.root, ref)

    @unittest.skipUnless(
        hasattr(os, "symlink") and sys.platform != "win32",
        "symlink creation needs privileges on Windows",
    )
    def test_symlink_escape_is_rejected(self):
        link = self.root / "escape.mp4"
        link.symlink_to(self.secret)
        with self.assertRaises(media.MediaError):
            media.resolve_within(self.root, "escape.mp4")

    def test_disallowed_extension_is_rejected(self):
        script = self.root / "payload.py"
        script.write_text("print('hi')")
        with self.assertRaises(media.MediaError):
            media.resolve_within(self.root, "payload.py")

    def test_listing_exposes_no_absolute_paths(self):
        for item in media.list_media(self.root):
            self.assertNotIn(str(self.root), item["ref"])
            self.assertFalse(Path(item["ref"]).is_absolute())


class TestCapabilities(BridgeTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.root = Path(self._tmp.name) / "projects"
        self.root.mkdir()
        self.video = self.root / "clip.mp4"
        self.video.write_bytes(bytes(range(256)) * 4)
        self.store = media.CapabilityStore()

    def test_capability_is_opaque_and_scoped(self):
        cap = self.store.grant(self.video.resolve())
        self.assertNotIn("clip", cap.id)
        self.assertNotIn(str(self.root), cap.id)
        self.assertGreaterEqual(len(cap.id), 32)

    def test_unknown_capability_is_rejected(self):
        with self.assertRaises(media.MediaError) as ctx:
            self.store.resolve("made-up-identifier")
        self.assertEqual(ctx.exception.status, 404)

    def test_revoked_capability_stops_working(self):
        cap = self.store.grant(self.video.resolve())
        self.assertTrue(self.store.revoke(cap.id))
        with self.assertRaises(media.MediaError):
            self.store.resolve(cap.id)

    def test_expired_capability_stops_working(self):
        cap = self.store.grant(self.video.resolve())
        self.store._items[cap.id] = media.Capability(
            id=cap.id, path=cap.path, mime=cap.mime, size=cap.size,
            expires_at=int(time.time()) - 1,
        )
        with self.assertRaises(media.MediaError):
            self.store.resolve(cap.id)


class TestRangeParsing(BridgeTestCase):
    SIZE = 1000

    def test_normal_range(self):
        self.assertEqual(media.parse_range("bytes=0-499", self.SIZE), (0, 499))

    def test_open_ended_range(self):
        self.assertEqual(media.parse_range("bytes=500-", self.SIZE), (500, 999))

    def test_suffix_range(self):
        self.assertEqual(media.parse_range("bytes=-200", self.SIZE), (800, 999))

    def test_end_is_clamped_to_file_size(self):
        self.assertEqual(media.parse_range("bytes=900-99999", self.SIZE), (900, 999))

    def test_start_beyond_end_of_file_is_416(self):
        with self.assertRaises(media.MediaError) as ctx:
            media.parse_range("bytes=1000-1500", self.SIZE)
        self.assertEqual(ctx.exception.status, 416)

    def test_inverted_range_is_416(self):
        with self.assertRaises(media.MediaError) as ctx:
            media.parse_range("bytes=500-100", self.SIZE)
        self.assertEqual(ctx.exception.status, 416)

    def test_zero_length_suffix_is_416(self):
        with self.assertRaises(media.MediaError) as ctx:
            media.parse_range("bytes=-0", self.SIZE)
        self.assertEqual(ctx.exception.status, 416)

    def test_malformed_range_falls_back_to_whole_file(self):
        for header in ("bytes=abc-def", "items=0-10", "bytes=0-10, 20-30", ""):
            with self.subTest(header=header):
                self.assertIsNone(media.parse_range(header, self.SIZE))


class TestJws(BridgeTestCase):
    def test_tampered_payload_fails(self):
        token = make_jws(self.private, self.claims())
        header, payload, signature = token.split(".")
        forged = json.loads(jws.b64url_decode(payload))
        forged["email"] = "attacker@example.com"
        tampered = f"{header}.{b64url(json.dumps(forged).encode())}.{signature}"
        public = self.private.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )
        with self.assertRaises(jws.JwsError):
            jws.verify(tampered, public)

    def test_wrong_segment_count_fails(self):
        public = self.private.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )
        for token in ("a.b", "a.b.c.d", "", "not-a-token"):
            with self.subTest(token=token), self.assertRaises(jws.JwsError):
                jws.verify(token, public)

    def test_crit_header_is_refused(self):
        claims = self.claims()
        header = b64url(
            json.dumps({"alg": "EdDSA", "typ": "JWT", "kid": "test", "crit": ["x"]}).encode()
        )
        payload = b64url(json.dumps(claims).encode())
        signature = self.private.sign(f"{header}.{payload}".encode("ascii"))
        public = self.private.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )
        with self.assertRaises(jws.JwsError):
            jws.verify(f"{header}.{payload}.{b64url(signature)}", public)


# ==========================================================================
# Phase 2 -- templates, binding, validation, running
# ==========================================================================

TEXT_WIDGETS = set(binder.TEXT_WIDGETS)


# Inputs the real nodes declare optional. It matters: check_after_pruning only
# complains about a missing REQUIRED input, so a fake that calls everything
# required would report a correctly pruned workflow as broken.
# TextEncodeQwenImageEditPlus really does declare these optional --
# comfy_extras/nodes_qwen.py:62-65.
OPTIONAL_INPUTS = {
    ("TextEncodeQwenImageEditPlus", "vae"),
    ("TextEncodeQwenImageEditPlus", "image1"),
    ("TextEncodeQwenImageEditPlus", "image2"),
    ("TextEncodeQwenImageEditPlus", "image3"),
}


def fake_object_info(graph, *, files=None, drop_classes=(), rename_widgets=None):
    """A stand-in for ComfyUI's /object_info that accepts exactly this graph.

    Built from the graph rather than hand-written so a template change cannot
    quietly make the validator tests pass against a machine that no longer
    matches reality.
    """
    files = set(files) if files is not None else None       # None = every file present
    rename_widgets = rename_widgets or {}
    info: dict[str, dict] = {}

    for node_id, node in graph.items():
        class_type = node["class_type"]
        if class_type in drop_classes:
            continue
        block = info.setdefault(class_type, {"input": {"required": {}, "optional": {}}})["input"]
        for widget, value in node.get("inputs", {}).items():
            optional = (class_type, widget) in OPTIONAL_INPUTS
            required = block["optional"] if optional else block["required"]
            if binder.is_link(value):
                required.setdefault(widget, ["*"])
                continue
            widget = rename_widgets.get(widget, widget)
            if widget in TEXT_WIDGETS and isinstance(value, str):
                required[widget] = ["STRING", {"multiline": True}]
            elif isinstance(value, bool):
                required[widget] = ["BOOLEAN", {}]
            elif isinstance(value, int):
                required[widget] = ["INT", {"min": 0, "max": 2**53}]
            elif isinstance(value, float):
                required[widget] = ["FLOAT", {"min": 0.0, "max": 100.0}]
            elif isinstance(value, str):
                present = files is None or value in files
                options = required.get(widget, [[]])[0]
                options = list(options) if isinstance(options, list) else []
                if present and value not in options:
                    options.append(value)
                required[widget] = [options, {}]
    return info


def tiny_png(width=64, height=36, colour=(40, 80, 120)) -> bytes:
    """A real image. The bridge decodes every upload to measure it, so fake
    bytes are now correctly refused -- see geometry.measure."""
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), colour).save(buffer, format="PNG")
    return buffer.getvalue()


def code_without_comments(path):
    """Executable source only: no comments, no docstrings, no string literals."""
    import io
    import tokenize

    with path.open("rb") as handle:
        tokens = list(tokenize.tokenize(handle.readline))
    return " ".join(
        token.string
        for token in tokens
        if token.type not in (tokenize.COMMENT, tokenize.STRING)
    )


def merge_object_info(*infos):
    """Union of several fakes. A plain dict merge would replace SaveImage's
    choice list with the last template's, so the other template's perfectly
    valid filename_prefix would read as an invalid choice."""
    out: dict[str, dict] = {}
    for info in infos:
        for class_type, spec in info.items():
            target = out.setdefault(class_type, {"input": {"required": {}, "optional": {}}})
            for section in ("required", "optional"):
                for widget, entry in (spec.get("input", {}).get(section) or {}).items():
                    existing = target["input"][section].get(widget)
                    if (isinstance(entry, list) and entry and isinstance(entry[0], list)
                            and isinstance(existing, list) and existing
                            and isinstance(existing[0], list)):
                        merged = list(existing[0])
                        merged += [o for o in entry[0] if o not in merged]
                        target["input"][section][widget] = [merged, {}]
                    elif existing is None:
                        target["input"][section][widget] = entry
    return out


def with_checkpoints(info, template):
    """Offer every declared checkpoint variant, not just the one baked into the
    graph. $CHECKPOINT binds a different file to that widget, and a real
    ComfyUI with both installed lists both."""
    choices = info["UNETLoader"]["input"]["required"]["unet_name"][0]
    for entry in (template.checkpoints.get("variants") or {}).values():
        if entry["file"] not in choices:
            choices.append(entry["file"])
    return info


def renumber(graph):
    """Re-export the same workflow with completely different node ids.

    This is what ComfyUI does every time the user touches a subgraph, and it is
    the entire reason binding goes by title.
    """
    mapping = {old: f"n{i * 7 + 3}" for i, old in enumerate(graph)}
    out = {}
    for old, node in graph.items():
        clone = copy.deepcopy(node)
        for widget, value in clone.get("inputs", {}).items():
            if binder.is_link(value):
                clone["inputs"][widget] = [mapping[str(value[0])], value[1]]
        out[mapping[old]] = clone
    return out


def strip_titles(graph):
    out = copy.deepcopy(graph)
    for node in out.values():
        meta = node.get("_meta") or {}
        node["_meta"] = {"title": meta.get("original_title") or node["class_type"]}
    return out


class TemplateTestCase(BridgeTestCase):
    def setUp(self) -> None:
        super().setUp()
        templates.reset()
        self.template = templates.get("qwen-image-edit")
        self.graph = self.template.graph

    def tearDown(self) -> None:
        templates.reset()
        super().tearDown()


class TestTemplateRegistry(TemplateTestCase):
    def test_shipped_template_loads(self):
        self.assertEqual(self.template.kind, "image")
        self.assertEqual(self.template.images, 1)
        self.assertTrue(self.template.hash.startswith("sha256:"))

    def test_unknown_template_is_rejected(self):
        with self.assertRaises(templates.TemplateError) as ctx:
            templates.get("does-not-exist")
        self.assertEqual(ctx.exception.code, "unknown_template")

    def test_template_id_can_never_be_a_path(self):
        """An id is a dict key, never a path component -- but the slug check
        makes that true by construction instead of by careful reading."""
        for bad in ("../../etc/passwd", "a/b", "C:\\x", "..", "", "Qwen"):
            with self.subTest(bad=bad), self.assertRaises(templates.TemplateError):
                templates.get(bad)

    def test_hash_covers_the_manifest_too(self):
        """Hashing only the graph would let an edit to the declared models --
        the thing a pinned hash is meant to protect -- slip through."""
        directory = Path(self._tmp.name) / "tpl"
        directory.mkdir()
        graph = {"1": {"class_type": "SaveImage", "inputs": {}, "_meta": {"title": "$OUTPUT"}}}
        manifest = {"id": "tpl", "kind": "image", "slots": {"$OUTPUT": {"required": True}}}
        (directory / "tpl.json").write_text(json.dumps(graph), encoding="utf-8")
        (directory / "tpl.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        first = templates.load_all(directory)["tpl"].hash

        manifest["requires"] = {"models": [{"folder": "vae", "file": "other.safetensors"}]}
        (directory / "tpl.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        self.assertNotEqual(first, templates.load_all(directory)["tpl"].hash)


class TestBinder(TemplateTestCase):
    def test_all_declared_slots_are_found_by_title(self):
        slots = binder.find_slots(self.graph, self.template.slots)
        self.assertEqual(set(slots), set(self.template.slots))
        for slot in slots.values():
            self.assertEqual(slot.matched_by, "title")

    def test_binding_survives_a_re_export_with_new_node_ids(self):
        """The whole reason binding is by title. ComfyUI renumbers on every
        export -- subgraph ids like '102:38' change shape entirely -- so any
        lookup by id works exactly until the user edits their workflow."""
        original = binder.find_slots(self.graph, self.template.slots)
        moved = binder.find_slots(renumber(self.graph), self.template.slots)
        self.assertEqual(set(original), set(moved))
        self.assertNotEqual(
            {s.node_id for s in original.values()}, {s.node_id for s in moved.values()}
        )
        for name, slot in moved.items():
            self.assertEqual(slot.class_type, original[name].class_type)
            self.assertEqual(slot.widget, original[name].widget)

    def test_explicit_widget_in_the_title_is_honoured(self):
        slots = binder.find_slots(self.graph, self.template.slots)
        self.assertEqual(slots["$SEED"].widget, "seed")
        self.assertEqual(slots["$SEED"].class_type, "KSampler")

    def test_prompt_binds_to_the_widget_the_node_actually_has(self):
        """TextEncodeQwenImageEdit calls it 'prompt', not 'text'. Guessing
        'text' would bind nothing and generate from an empty instruction."""
        slots = binder.find_slots(self.graph, self.template.slots)
        self.assertEqual(slots["$PROMPT"].widget, "prompt")
        self.assertEqual(slots["$NEGATIVE"].widget, "prompt")
        self.assertNotEqual(slots["$PROMPT"].node_id, slots["$NEGATIVE"].node_id)

    def test_output_slot_has_no_widget(self):
        slots = binder.find_slots(self.graph, self.template.slots)
        self.assertIsNone(slots["$OUTPUT"].widget)
        self.assertEqual(binder.output_node_ids(slots), [slots["$OUTPUT"].node_id])

    def test_bind_writes_the_values(self):
        bound = binder.fill(
            self.graph, {"$PROMPT": "make it rain", "$IMAGE_1": "rb_abc123ef.png", "$SEED": 7}
        )
        slots = binder.find_slots(self.graph, self.template.slots)
        self.assertEqual(bound[slots["$PROMPT"].node_id]["inputs"]["prompt"], "make it rain")
        self.assertEqual(bound[slots["$IMAGE_1"].node_id]["inputs"]["image"], "rb_abc123ef.png")
        self.assertEqual(bound[slots["$SEED"].node_id]["inputs"]["seed"], 7)

    def test_bind_never_mutates_the_template(self):
        """Templates are loaded once and cached, so an in-place edit would leak
        one user's prompt into the next run."""
        before = copy.deepcopy(self.graph)
        binder.fill(self.graph, {"$PROMPT": "mutated"})
        self.assertEqual(self.graph, before)

    def test_unknown_slot_is_refused(self):
        with self.assertRaises(binder.BindError) as ctx:
            binder.fill(self.graph, {"$NOT_A_SLOT": "x"})
        self.assertEqual(ctx.exception.code, "unknown_slot")

    def test_non_scalar_value_is_refused(self):
        """A value must never be able to become a wire, a node or a structure --
        that is what keeps 'parameters only, never a graph' true."""
        for value in ([["78", 0]], {"class_type": "Evil"}, ["a", 0], None):
            with self.subTest(value=value), self.assertRaises(binder.BindError) as ctx:
                binder.fill(self.graph, {"$PROMPT": value})
            self.assertEqual(ctx.exception.code, "bad_value")

    def test_wired_input_is_never_overwritten(self):
        """KSampler.steps is driven by a switch node here. Writing a literal
        would disconnect it and the workflow would run happily, producing
        something the user never asked for."""
        slots = binder.find_slots(self.graph, self.template.slots)
        wired = binder.Slot(
            name="$STEPS", node_id=slots["$SEED"].node_id, class_type="KSampler",
            widget="steps", family="value", matched_by="title",
        )
        with self.assertRaises(binder.BindError) as ctx:
            binder.fill(self.graph, {"$STEPS": 4}, {"$STEPS": wired})
        self.assertEqual(ctx.exception.code, "slot_is_wired")

    def test_readonly_output_slot_cannot_be_written(self):
        with self.assertRaises(binder.BindError) as ctx:
            binder.fill(self.graph, {"$OUTPUT": "x"})
        self.assertEqual(ctx.exception.code, "readonly_slot")

    def test_duplicate_slot_title_is_refused(self):
        graph = copy.deepcopy(self.graph)
        for node in graph.values():
            if node["class_type"] == "ImageScaleToTotalPixels":
                node["_meta"]["title"] = "$PROMPT"
        with self.assertRaises(binder.BindError) as ctx:
            binder.find_slots(graph, self.template.slots)
        self.assertEqual(ctx.exception.code, "duplicate_slot")


class TestBinderDuckTyping(TemplateTestCase):
    """With every title stripped, shape alone has to find the slots."""

    def setUp(self) -> None:
        super().setUp()
        self.plain = strip_titles(self.graph)
        self.slots = binder.find_slots(self.plain, self.template.slots)

    def test_everything_is_found_without_a_single_title(self):
        self.assertEqual(set(self.slots), set(self.template.slots))
        for slot in self.slots.values():
            self.assertEqual(slot.matched_by, "duck")

    def test_save_image_is_the_output(self):
        self.assertEqual(self.slots["$OUTPUT"].class_type, "SaveImage")

    def test_load_image_is_the_first_image_input(self):
        self.assertEqual(self.slots["$IMAGE_1"].class_type, "LoadImage")
        self.assertEqual(self.slots["$IMAGE_1"].widget, "image")

    def test_prompt_and_negative_follow_the_sampler_wires(self):
        titled = binder.find_slots(self.graph, self.template.slots)
        by_id = {n: i for i, n in enumerate(self.plain)}
        original = {n: i for i, n in enumerate(self.graph)}
        self.assertEqual(by_id[self.slots["$PROMPT"].node_id], original[titled["$PROMPT"].node_id])
        self.assertEqual(
            by_id[self.slots["$NEGATIVE"].node_id], original[titled["$NEGATIVE"].node_id]
        )

    def test_seed_lands_on_the_sampler(self):
        self.assertEqual(self.slots["$SEED"].class_type, "KSampler")
        self.assertEqual(self.slots["$SEED"].widget, "seed")

    def test_sequencer_is_found_by_its_insert_frame_1_input(self):
        """The video templates in Phase 4 bind by this shape; keeping it here
        means the fallback is already proven when they arrive."""
        graph = {
            "1": {"class_type": "Whatever", "inputs": {"num_images": 2, "insert_frame_1": 0}},
            "2": {"class_type": "SaveImage", "inputs": {}},
        }
        slots = binder.find_slots(graph, ["$SEQUENCER", "$OUTPUT"])
        self.assertEqual(slots["$SEQUENCER"].node_id, "1")


def keyframe_graph(capacity=4):
    """A loader that takes many filenames and a sequencer that times them.

    Shaped after LTXSequencer + MultiImageLoader, which is what LTX-2.3 uses,
    but deliberately built here rather than loaded from the template: these
    tests are about the CONTRACT -- a list of images and a list of timings --
    and it has to hold for whatever node the next local video model brings.
    """
    timing = {}
    for i in range(1, capacity + 1):
        timing[f"insert_frame_{i}"] = 0
        timing[f"insert_second_{i}"] = 0.0
        timing[f"strength_{i}"] = 1.0
    return {
        "1": {
            "class_type": "MultiImageLoader",
            "_meta": {"title": "$IMAGE_PATHS"},
            "inputs": {"image_paths": "", "width": 0, "height": 0},
        },
        "2": {
            "class_type": "LTXSequencer",
            "_meta": {"title": "$SEQUENCER"},
            "inputs": {
                "num_images": 1, "insert_mode": "frames", "frame_rate": 24,
                "multi_input": ["1", 0], **timing,
            },
        },
        "3": {"class_type": "SaveVideo", "_meta": {"title": "$OUTPUT"}, "inputs": {}},
    }


class TestKeyframeListBinding(BridgeTestCase):
    """Many images into one widget, without ever naming a path.

    The editor sends `[{frame, strength}, ...]` and a list of uploaded images.
    Which node that becomes is the bridge's business -- that separation is the
    point, so that a second local video model does not reach the editor.
    """

    def setUp(self) -> None:
        super().setUp()
        self.graph = keyframe_graph()
        self.names = ["rb_00000001.png", "rb_00000002.png", "rb_00000003.png"]

    def bound(self, images=None, timings=None):
        values = {}
        if images is not None:
            values["$IMAGE_PATHS"] = images
        if timings is not None:
            values["$SEQUENCER"] = timings
        return binder.fill(self.graph, values)

    def test_images_become_one_newline_separated_widget(self):
        bound = self.bound(images=self.names)
        self.assertEqual(
            bound["1"]["inputs"]["image_paths"], "\n".join(self.names)
        )

    def test_timings_are_written_one_widget_per_keyframe(self):
        bound = self.bound(
            images=self.names,
            timings=[{"frame": 0}, {"frame": 18, "strength": 0.8}, {"frame": 37}],
        )
        inputs = bound["2"]["inputs"]
        self.assertEqual(inputs["num_images"], 3)
        self.assertEqual(
            [inputs["insert_frame_1"], inputs["insert_frame_2"], inputs["insert_frame_3"]],
            [0, 18, 37],
        )
        self.assertEqual(inputs["strength_2"], 0.8)
        self.assertEqual(inputs["strength_1"], 1.0)

    def test_unused_keyframe_widgets_are_left_alone(self):
        """The sequencer reads only the first `num_images` of them, so a stale
        value further down is harmless -- but writing one would say this run
        had a fourth key, and the next reader of the graph would believe it."""
        bound = self.bound(images=self.names[:2], timings=[{"frame": 0}, {"frame": 12}])
        self.assertEqual(bound["2"]["inputs"]["num_images"], 2)
        self.assertEqual(bound["2"]["inputs"]["insert_frame_3"], 0)
        self.assertEqual(bound["2"]["inputs"]["insert_frame_4"], 0)

    def test_insert_mode_is_forced_to_frames(self):
        """An animator's timing is in frames. If a request could switch this to
        seconds, the same list of numbers would silently mean something else --
        frame 18 becoming 18 seconds is a shot 25 times too long."""
        graph = keyframe_graph()
        graph["2"]["inputs"]["insert_mode"] = "seconds"
        bound = binder.fill(graph, {"$SEQUENCER": [{"frame": 0}, {"frame": 18}]})
        self.assertEqual(bound["2"]["inputs"]["insert_mode"], "frames")

    def test_a_path_is_refused_where_a_filename_belongs(self):
        """MultiImageLoader tries the string as an absolute path BEFORE looking
        in the input folder, so an unchecked value here reads any file on the
        disk. Names arriving here come from the bridge's own upload registry,
        which is precisely why the check cannot live only at the caller."""
        for evil in ("../../secrets.png", "C:/Users/me/passwords.png",
                     "/etc/passwd", "a.png\nC:/evil.png", "sub/dir.png"):
            with self.subTest(evil=evil), self.assertRaises(binder.BindError) as ctx:
                self.bound(images=[evil])
            self.assertEqual(ctx.exception.code, "bad_value")

    def test_more_keyframes_than_the_template_holds_is_refused(self):
        """Not truncated. The sequencer would have run happily on the first
        four and produced a video missing the fifth key, and nothing on screen
        would have said so."""
        with self.assertRaises(binder.BindError) as ctx:
            self.bound(timings=[{"frame": f} for f in (0, 10, 20, 30, 40)])
        self.assertEqual(ctx.exception.code, "too_many_keyframes")

    def test_nonsense_timings_are_refused(self):
        for bad in ([{"frame": -1}], [{"frame": 1.5}], [{"frame": True}],
                    [{"frame": 0, "strength": 2}], [{"frame": 0, "strength": "hard"}],
                    [{}], ["frame 0"], []):
            with self.subTest(bad=bad), self.assertRaises(binder.BindError) as ctx:
                self.bound(timings=bad)
            self.assertIn(ctx.exception.code, ("bad_value",))

    def test_a_list_value_still_cannot_become_a_wire(self):
        """The two list-shaped slots are the only loosening of 'scalars only',
        and they must not reopen the hole it was closing."""
        for value in ([["1", 0]], [{"frame": ["1", 0]}], [{"class_type": "Evil"}]):
            with self.subTest(value=value), self.assertRaises(binder.BindError):
                binder.fill(self.graph, {"$SEQUENCER": value})
        with self.assertRaises(binder.BindError):
            binder.fill(self.graph, {"$IMAGE_PATHS": [["1", 0]]})

    def test_a_wired_widget_is_still_never_overwritten(self):
        graph = keyframe_graph()
        graph["2"]["inputs"]["insert_frame_2"] = ["9", 0]
        with self.assertRaises(binder.BindError) as ctx:
            binder.fill(graph, {"$SEQUENCER": [{"frame": 0}, {"frame": 18}]})
        self.assertEqual(ctx.exception.code, "slot_is_wired")

    def test_binding_survives_stripped_titles(self):
        """Duck-typing has to find both nodes, or a user's own LTX workflow
        only works after they have retitled it."""
        graph = strip_titles(keyframe_graph())
        slots = binder.find_slots(graph, ["$IMAGE_PATHS", "$SEQUENCER", "$OUTPUT"])
        self.assertEqual(slots["$IMAGE_PATHS"].node_id, "1")
        self.assertEqual(slots["$SEQUENCER"].node_id, "2")
        self.assertEqual(slots["$IMAGE_PATHS"].matched_by, "duck")

    def test_bind_never_mutates_the_template(self):
        before = copy.deepcopy(self.graph)
        self.bound(images=self.names, timings=[{"frame": 0}])
        self.assertEqual(self.graph, before)


class TestMultiStageSequencers(BridgeTestCase):
    """LTX-2.3 samples the shot three times, one sequencer per stage.

    All three must carry the same timing. A stage left on the template's own
    shipped numbers re-times the shot halfway through the render, and the only
    evidence is a finished video that does not match the timeline.
    """

    def staged_graph(self):
        graph = keyframe_graph()
        for n, title in (("2", "$SEQUENCER"), ("4", "$SEQUENCER_2"), ("5", "$SEQUENCER_3")):
            if n not in graph:
                graph[n] = copy.deepcopy(graph["2"])
            graph[n]["_meta"] = {"title": title}
        return graph

    STAGES = ("$SEQUENCER", "$SEQUENCER_2", "$SEQUENCER_3")

    def stage_nodes(self, graph):
        slots = binder.find_slots(graph, list(self.STAGES))
        return {slots[name].node_id for name in self.STAGES}

    def test_each_stage_gets_its_own_node(self):
        self.assertEqual(len(self.stage_nodes(self.staged_graph())), 3)

    def test_duck_typing_does_not_hand_every_stage_the_same_node(self):
        self.assertEqual(len(self.stage_nodes(strip_titles(self.staged_graph()))), 3)

    def test_all_stages_receive_the_same_timing(self):
        timings = [{"frame": 0}, {"frame": 18}, {"frame": 37}]
        bound = binder.fill(
            self.staged_graph(),
            {"$SEQUENCER": timings, "$SEQUENCER_2": timings, "$SEQUENCER_3": timings},
        )
        seen = [
            [node["inputs"][f"insert_frame_{i}"] for i in (1, 2, 3)]
            for node in bound.values()
            if node.get("class_type") == "LTXSequencer"
        ]
        self.assertEqual(len(seen), 3)
        self.assertTrue(all(row == [0, 18, 37] for row in seen), seen)


# Nodes that only work when a browser is on ComfyUI's websocket. They read
# PromptServer.last_node_id, which execution.py sets ONLY for a prompt queued
# with a client_id -- and the bridge queues without one on purpose (see
# comfy/runner.py). Headless they do not degrade, they raise, and the raise
# happens inside the sampler callback, so it takes the whole run with it.
NEEDS_A_WATCHING_BROWSER = ("LTX2SamplingPreviewOverride",)


class TestShippedTemplatesRunHeadless(BridgeTestCase):
    """No shipped preset may depend on somebody looking at ComfyUI's tab.

    Found in use: the LTX preset carried KJNodes' live-preview override. It
    changes nothing about the render -- it only draws thumbnails while the
    sampler works -- and the same workflow finished fine in ComfyUI's own tab.
    Through the bridge it died 122 seconds in on "'NoneType' object has no
    attribute 'encode'". Reachability is what matters here, not presence: a
    node nothing walks back to never runs.
    """

    def setUp(self) -> None:
        super().setUp()
        templates.reset()

    def tearDown(self) -> None:
        templates.reset()
        super().tearDown()

    def test_no_preset_reaches_a_node_that_needs_a_client_id(self):
        for template in templates.load_all().values():
            with self.subTest(template=template.id):
                slots = binder.find_slots(template.graph, template.slots.keys())
                live = binder.reachable(template.graph, binder.output_node_ids(slots))
                reached = {
                    template.graph[node_id].get("class_type") for node_id in live
                }
                self.assertFalse(
                    reached & set(NEEDS_A_WATCHING_BROWSER),
                    f"{template.id} reaches {reached & set(NEEDS_A_WATCHING_BROWSER)}",
                )


class TestKeyframesAreBroughtToOneSize(TemplateTestCase):
    """Every picture in a collected role has to leave here the same size.

    Found in the log of a real run: MultiImageLoader batches the whole list
    into one tensor, and mixed shapes cannot be batched. It does not fail --
    multi_image_loader.py:171 prints a console warning and substitutes
    torch.zeros((1, 64, 64, 3)). That one 64x64 black frame is what reaches
    the sequencers AND the node the output resolution is derived from, so the
    run finishes nine minutes later as a black clip with none of the keyframes
    in it, reported as a success.
    """

    def setUp(self) -> None:
        super().setUp()
        from reanimator import server

        self.server = server
        self.template = templates.get("ltx-23-keyframes")
        self.inputs = Path(self._tmp.name) / "input"
        self.inputs.mkdir()

        class FakeFolderPaths:
            get_input_directory = staticmethod(lambda: str(self.inputs))

        self._orig = runner._folder_paths
        runner._folder_paths = lambda: FakeFolderPaths     # type: ignore[assignment]
        server._inputs.clear()

    def tearDown(self) -> None:
        runner._folder_paths = self._orig                  # type: ignore[assignment]
        self.server._inputs.clear()
        super().tearDown()

    def keyframe(self, width, height) -> str:
        """An upload, registered the way put_input registers one."""
        name = runner.write_input(tiny_png(width, height), ".png")
        self.server._inputs[f"id{len(self.server._inputs)}"] = {
            "file": name, "width": width, "height": height,
        }
        return name

    def match(self, sizes, mutate=True):
        values = {"$IMAGE_PATHS": [self.keyframe(*size) for size in sizes]}
        replaced, notes = self.server._match_collected_sizes(
            self.template, values, mutate=mutate
        )
        return values["$IMAGE_PATHS"], replaced, notes

    def sizes_on_disk(self, names):
        return [geometry.measure(runner.read_input(n)) for n in names]

    def test_every_key_ends_at_the_majority_size(self):
        names, replaced, notes = self.match([(848, 480), (848, 480), (640, 360)])
        self.assertEqual(self.sizes_on_disk(names), [(848, 480)] * 3)
        self.assertEqual(len(replaced), 1, "only the odd one out is rewritten")
        self.assertEqual([n["keyframe"] for n in notes], [3])

    def test_the_majority_wins_not_the_largest(self):
        """Otherwise one key exported wrong drags the whole render up to its
        resolution -- VRAM and minutes for a shot nobody asked to enlarge."""
        names, _, _ = self.match([(640, 360), (640, 360), (1920, 1080)])
        self.assertEqual(self.sizes_on_disk(names), [(640, 360)] * 3)

    def test_a_near_matching_aspect_is_not_called_a_letterbox(self):
        """848x480 is not exactly sixteen-ninths, so 640x360 into it really
        does leave bars -- three pixels of them. Calling that letterboxed is
        true and useless: real frame sizes are rarely exact, so an exact test
        flags nearly every resize and the flag stops carrying information."""
        _, _, notes = self.match([(848, 480), (848, 480), (640, 360)])
        self.assertEqual((notes[0]["from"], notes[0]["to"]), ("640x360", "848x480"))
        self.assertEqual(notes[0]["bars"], [0, 3])
        self.assertFalse(notes[0]["letterboxed"], notes)

    def test_the_bars_are_reported_in_pixels(self):
        """A number the caller can act on. 'letterboxed: true' alone does not
        say whether it is three pixels or a third of the picture."""
        _, _, notes = self.match([(848, 480), (848, 480), (480, 480)])
        self.assertEqual(notes[0]["bars"], [368, 0])

    def test_a_different_shape_is_reported_as_letterboxed(self):
        """Bars on one key and not on the rest are visible in the finished
        clip, so this is the one case the caller has to see coming."""
        _, _, notes = self.match([(848, 480), (848, 480), (480, 480)])
        self.assertTrue(notes[0]["letterboxed"], notes)

    def test_nothing_is_cropped_or_stretched(self):
        """fit_into scales and centres. A square key keeps its whole picture,
        so the 480 of height maps to 480 of width inside an 848 canvas."""
        names, _, _ = self.match([(848, 480), (848, 480), (480, 480)])
        self.assertEqual(self.sizes_on_disk(names)[2], (848, 480))

    def test_keys_that_already_agree_are_left_alone(self):
        names, replaced, notes = self.match([(848, 480), (848, 480)])
        self.assertEqual((replaced, notes), ([], []))
        self.assertEqual(self.sizes_on_disk(names), [(848, 480)] * 2)

    def test_a_dry_run_reports_without_consuming_the_uploads(self):
        """/validate runs this too. Rewriting the uploads there consumed them,
        and the /run that followed could not find the files it had just been
        told were fine."""
        before = sorted(p.name for p in self.inputs.iterdir())
        names, replaced, notes = self.match(
            [(848, 480), (848, 480), (640, 360)], mutate=False
        )
        self.assertEqual(replaced, [])
        self.assertEqual(len(notes), 1, "it still says what it would do")
        self.assertEqual(self.sizes_on_disk(names), [(848, 480), (848, 480), (640, 360)])
        self.assertEqual(
            sorted(p.name for p in self.inputs.iterdir()),
            sorted(before + [Path(n).name for n in names]),
        )

    def test_the_order_is_never_disturbed(self):
        """Keyframe N is timed by keyframe N. A resize that reordered the list
        would re-time the shot."""
        values = {"$IMAGE_PATHS": [self.keyframe(*s) for s in
                                   [(848, 480), (640, 360), (848, 480)]]}
        first, last = values["$IMAGE_PATHS"][0], values["$IMAGE_PATHS"][2]
        self.server._match_collected_sizes(self.template, values, mutate=True)
        names = values["$IMAGE_PATHS"]
        self.assertEqual((names[0], names[2]), (first, last))
        self.assertNotEqual(names[1], "the middle one is the one that moved")


class TestKeyframePresetPassesPreflight(BridgeTestCase):
    """A keyframe workflow has to survive /validate, not just the binder.

    Found in use: every LTX run died on "$SEQUENCER is on node 376
    (LTXSequencer), but that node has no editable field to receive the value."
    Nothing was wrong with the preset. A sequencer HAS no single widget -- its
    timing is a family of numbered ones that the binder fills in itself -- and
    the pre-flight only exempted $OUTPUT from needing one. The binder tests all
    passed, because none of them went through the validator.
    """

    MANIFEST = {
        "id": "kf",
        "kind": "video",
        "slots": {
            "$IMAGE_PATHS": {"required": True},
            "$SEQUENCER": {"required": True},
            "$OUTPUT": {"required": True},
        },
    }

    def setUp(self) -> None:
        super().setUp()
        self.graph = keyframe_graph()
        self.template = templates.Template(
            id="kf", manifest=self.MANIFEST, graph=self.graph, hash="sha256:x"
        )
        self.values = {
            "$IMAGE_PATHS": ["rb_0123abcd.png", "rb_0123abce.png"],
            "$SEQUENCER": [{"frame": 0}, {"frame": 24}],
        }

    def report(self, values=None):
        return validate.validate(
            self.template, values or self.values, fake_object_info(self.graph)
        )

    def test_a_sequencer_without_one_widget_is_not_an_error(self):
        report = self.report()
        self.assertTrue(report.ok, [e.as_dict() for e in report.errors])

    def test_the_timing_reaches_the_graph_that_would_be_queued(self):
        """report.graph is what gets queued, so that is where to look: passing
        the check while writing nothing would be the same bug, quieter."""
        node = self.report().graph["2"]["inputs"]
        self.assertEqual([node["insert_frame_1"], node["insert_frame_2"]], [0, 24])

    def test_a_slot_that_really_has_no_widget_is_still_an_error(self):
        """The exemption is for sequencers, not for every missing widget: a
        $PROMPT on a node with no text field must still be refused."""
        graph = copy.deepcopy(self.graph)
        graph["4"] = {
            "class_type": "ConditioningZeroOut",
            "_meta": {"title": "$PROMPT"},
            "inputs": {"conditioning": ["1", 0]},
        }
        template = templates.Template(
            id="kf",
            manifest={**self.MANIFEST, "slots": {**self.MANIFEST["slots"], "$PROMPT": {}}},
            graph=graph,
            hash="sha256:x",
        )
        report = validate.validate(template, self.values, fake_object_info(graph))
        self.assertIn("unknown_widget", {e.code for e in report.errors})


class TestBlankPaperPrompt(BridgeTestCase):
    """Drawing on paper and editing a frame are opposite requests.

    Found in use: a keyframe drawn on blank paper came back with "only some
    colours changed". Nothing had failed. The drawing travels as Image 1, so
    from the bridge's side it looked like an edit with no strokes, and the edit
    preamble says "Apply the user's instruction while PRESERVING character
    design, framing, background and lighting." Over somebody's pencil lines the
    most obedient thing a model can do with that sentence is hand them back.

    So these tests are mostly about words, and that is not a soft target: the
    preamble is the instruction. A wrong one produces a run that succeeds, a
    result that looks deliberate, and a user who blames the model.
    """

    def setUp(self) -> None:
        super().setUp()
        self.template = templates.load_all()["qwen-image-edit-2511-base"]

    def prompt(self, roles, intent):
        return validate.compose_prompt(self.template, "he raises his arm", set(roles), intent)

    def test_paper_is_never_told_to_preserve_picture_one(self):
        text = self.prompt({"cleanFrame", "contextFrames"}, "create_from_drawing")
        self.assertNotIn("preserving character design", text)
        self.assertNotIn("original clean video frame", text)

    def test_paper_is_told_it_is_a_sketch_to_realise(self):
        text = self.prompt({"cleanFrame"}, "create_from_drawing")
        self.assertIn("hand-drawn sketch", text)
        self.assertIn("photorealistic", text)
        # The sketch must not survive into the result, and saying so is the
        # only thing standing between a render and a tidied-up drawing.
        self.assertIn("No pencil lines", text)

    def test_picture_one_is_described_first(self):
        """Reading order is the model's order. The context line landed above
        the subject line purely because of where it sat in the manifest."""
        text = self.prompt({"cleanFrame", "contextFrames"}, "create_from_drawing")
        self.assertLess(text.index("Picture 1"), text.index("Picture 2"))

    def test_the_context_frame_is_numbered_by_what_was_actually_sent(self):
        """With no annotated frame the context IS Picture 2. Calling it
        Picture 3 points the model at an image nobody connected, and a model
        asked about a picture that is not there invents one."""
        with_drawing = self.prompt(
            {"cleanFrame", "annotatedFrame", "contextFrames"}, "edit_pose")
        self.assertIn("Picture 3 is a reference", with_drawing)

        without = self.prompt({"cleanFrame", "contextFrames"}, "edit_keyframe")
        self.assertIn("Picture 2 is a reference", without)
        self.assertNotIn("Picture 3", without)

    def test_the_editing_path_reads_exactly_as_intended(self):
        """The whole preamble, word for word.

        Pinned rather than sampled: these sentences ARE the instruction, so a
        clause that drifts is a behaviour change with no diff anyone would
        think to read. When it changes on purpose, this line has to be edited
        deliberately -- which is exactly the point of pinning it.
        """
        text = self.prompt({"cleanFrame", "annotatedFrame", "contextFrames"}, "edit_pose")
        self.assertEqual(
            text.split("\n")[0],
            "Picture 1 is the original clean video frame. "
            "Picture 2 is the same frame with the user's drawn pose/action guidance. "
            "The strokes are guidance only and must not appear in the result. "
            "Picture 3 is a reference to LOOK THINGS UP IN, never a second scene to "
            "blend, merge or collage with this one — it is either a nearby approved "
            "keyframe or a model sheet, and it never appears in the result as a "
            "picture. Take from it identity, design, materials and lighting, and — if "
            "it shows this same set — what is behind the subject, so background that "
            "this frame hides and the new pose uncovers is painted the way that "
            "reference shows it rather than invented. Never its pose, its layout or "
            "its framing, and never a person or object that is not already in this "
            "frame. "
            "Apply the user's instruction while preserving "
            "character design, framing, background and lighting.",
        )

    def test_a_reference_is_never_described_as_the_frame_itself(self):
        """A model sheet and a neighbouring key arrive in the same slot, so the
        sentence has to be true of both -- and both carry the same prohibition:
        take the design from it, never the pose or the framing. That is the
        whole difference between a reference and a base image."""
        for roles, intent in (
            ({"cleanFrame", "annotatedFrame", "contextFrames"}, "edit_pose"),
            ({"cleanFrame", "contextFrames"}, "create_from_drawing"),
        ):
            with self.subTest(intent=intent):
                text = self.prompt(roles, intent)
                self.assertIn("Never its pose, its layout or its framing", text)

    def test_a_reference_is_not_a_second_scene_to_mix_in(self):
        """What "reference" has to mean, spelled out for a model that was
        trained to compose several pictures into one.

        Left to itself, 2511 reads three pictures as three ingredients and
        hands back a blend of them -- which is what the animator sees: the
        neighbouring key mixed into the frame instead of consulted. Saying
        "consistency" does not stop it; saying it is never blended, merged or
        collaged, and never appears as a picture, is what stops it."""
        text = self.prompt(
            {"cleanFrame", "annotatedFrame", "contextFrames"}, "edit_pose")
        for clause in ("never a second scene to blend, merge or collage",
                       "never appears in the result as a picture",
                       "never a person or object that is not already in this frame"):
            self.assertIn(clause, text)

    def test_the_reference_is_where_uncovered_background_comes_from(self):
        """The one thing a neighbouring key is FOR, beyond identity.

        Change the pose and the body stops hiding part of the set. That
        background exists -- it is visible in another approved key -- so
        inventing it is a worse answer than looking it up, and the prompt has
        to say the reference is where to look."""
        text = self.prompt(
            {"cleanFrame", "annotatedFrame", "contextFrames"}, "edit_pose")
        self.assertIn("what is behind the subject", text)
        self.assertIn("the new pose uncovers", text)

    def test_the_new_intent_is_offered_to_the_editor(self):
        self.assertIn("create_from_drawing", self.template.summary()["intents"])


class TestDenoiseSlot(BridgeTestCase):
    """The knob behind "drawing fidelity".

    It is a real widget on the sampler, not an invented slider: how much of
    Picture 1's own pixels survive. On a paper creation Picture 1 IS the
    drawing, so that is fidelity to it.
    """

    def setUp(self) -> None:
        super().setUp()
        self.template = templates.load_all()["qwen-image-edit-2511-base"]

    def test_one_title_can_declare_two_slots(self):
        """The sampler carries the seed AND the denoise. Titling it for one and
        letting shape find the other works until a second sampler exists, at
        which point the untitled one binds to whichever comes first."""
        slots = binder.find_slots(self.template.graph, self.template.slots)
        self.assertEqual(slots["$SEED"].widget, "seed")
        self.assertEqual(slots["$DENOISE"].widget, "denoise")
        self.assertEqual(slots["$SEED"].node_id, slots["$DENOISE"].node_id)
        # Both by title: neither is left to inference.
        self.assertEqual(slots["$SEED"].matched_by, "title")
        self.assertEqual(slots["$DENOISE"].matched_by, "title")

    def test_both_values_are_written(self):
        bound = binder.fill(self.template.graph, {"$SEED": 7, "$DENOISE": 0.72})
        node = bound[binder.find_slots(self.template.graph, ["$SEED"])["$SEED"].node_id]
        self.assertEqual(node["inputs"]["seed"], 7)
        self.assertEqual(node["inputs"]["denoise"], 0.72)

    def test_a_title_with_junk_in_it_still_finds_the_slots(self):
        graph = copy.deepcopy(self.template.graph)
        for node in graph.values():
            if node.get("class_type") == "KSampler":
                node["_meta"]["title"] = "sampler $SEED:seed  notes $DENOISE:denoise"
        slots = binder.find_slots(graph, ["$SEED", "$DENOISE"])
        self.assertEqual(slots["$DENOISE"].widget, "denoise")

    def test_out_of_range_is_refused_against_the_node_itself(self):
        """The range is the NODE's, as ComfyUI reports it -- not a number
        copied into the manifest, which can drift away from the node it
        describes without anything noticing."""
        info = fake_object_info(self.template.graph)
        info["KSampler"]["input"]["required"]["denoise"] = ["FLOAT", {"min": 0.0, "max": 1.0}]
        report = validate.validate(
            self.template,
            {"$PROMPT": "x", "$IMAGE_1": "rb_0123abcd.png", "$DENOISE": 4.0},
            info,
            lambda *a: True,
        )
        self.assertFalse(report.ok)
        self.assertIn("out_of_range", [e.code for e in report.errors])

    def test_a_value_inside_the_range_is_accepted(self):
        info = fake_object_info(self.template.graph)
        info["KSampler"]["input"]["required"]["denoise"] = ["FLOAT", {"min": 0.0, "max": 1.0}]
        report = validate.validate(
            self.template,
            {"$PROMPT": "x", "$IMAGE_1": "rb_0123abcd.png", "$DENOISE": 0.72},
            info,
            lambda *a: True,
        )
        self.assertNotIn("out_of_range", [e.code for e in report.errors])


class TestWhenClauses(BridgeTestCase):
    """The grammar the prompt lines are selected with."""

    def holds(self, cond, roles=(), intent=None):
        return validate._when_holds(cond, set(roles), intent)

    def test_a_role_name_matches_a_supplied_role(self):
        self.assertTrue(self.holds("cleanFrame", roles=["cleanFrame"]))
        self.assertFalse(self.holds("cleanFrame", roles=["annotatedFrame"]))

    def test_always_and_none_are_true(self):
        self.assertTrue(self.holds("always"))
        self.assertTrue(self.holds(None))

    def test_negation(self):
        self.assertTrue(self.holds("!annotatedFrame", roles=["cleanFrame"]))
        self.assertFalse(self.holds("!cleanFrame", roles=["cleanFrame"]))

    def test_intent(self):
        self.assertTrue(self.holds("intent:edit_pose", intent="edit_pose"))
        self.assertFalse(self.holds("intent:edit_pose", intent="edit_keyframe"))
        self.assertFalse(self.holds("intent:edit_pose", intent=None))

    def test_a_list_needs_every_clause(self):
        self.assertTrue(self.holds(["cleanFrame", "!annotatedFrame"], roles=["cleanFrame"]))
        self.assertFalse(
            self.holds(["cleanFrame", "!annotatedFrame"],
                       roles=["cleanFrame", "annotatedFrame"]))

    def test_an_unknown_clause_is_false_not_an_error(self):
        """A manifest typo must drop one line, not take down the run."""
        self.assertFalse(self.holds("cleenFrame", roles=["cleanFrame"]))


class TestLegalFrameCount(BridgeTestCase):
    """The 8n+1 rule is LTX's, so it lives in LTX's manifest.

    Putting it in the editor would bake one model's arithmetic into a contract
    written to outlive that model -- the next local backend rounds differently
    and every caller would have to learn both.
    """

    def preset(self, **rule):
        block = {"slot": "$FRAMES", "min": 9, "step": 8, "offset": 1}
        block.update(rule)
        return templates.Template(
            id="t", manifest={"frames": block}, graph={}, hash="x"
        )

    def test_rounds_up_to_the_next_legal_length(self):
        t = self.preset()
        self.assertEqual([t.legal_frame_count(n) for n in (57, 58, 61, 65)],
                         [57, 65, 65, 65])

    def test_never_rounds_down(self):
        """Down would drop the tail: the last key could fall outside the video,
        or land on the final frame and be gone in an instant."""
        t = self.preset()
        for wanted in range(9, 90):
            self.assertGreaterEqual(t.legal_frame_count(wanted), wanted)

    def test_a_minimum_is_honoured(self):
        self.assertEqual(self.preset().legal_frame_count(2), 9)

    def test_a_preset_without_the_rule_is_left_alone(self):
        plain = templates.Template(id="t", manifest={}, graph={}, hash="x")
        self.assertEqual(plain.legal_frame_count(61), 61)


class TestValidator(TemplateTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.values = {"$PROMPT": "remove the UI text", "$IMAGE_1": "rb_0123abcd.png"}
        self.info = fake_object_info(self.graph)

    def report(self, values=None, info=None):
        return validate.validate(self.template, values or self.values, info or self.info)

    def codes(self, report):
        return {issue.code for issue in report.errors}

    def test_happy_path(self):
        report = self.report()
        self.assertTrue(report.ok, [e.as_dict() for e in report.errors])

    def test_missing_node_class_is_an_error(self):
        info = fake_object_info(self.graph, drop_classes=("TextEncodeQwenImageEdit",))
        report = self.report(info=info)
        self.assertIn("missing_node", self.codes(report))

    def test_missing_required_model_is_an_error(self):
        """The point of validating before queueing: say 'you are missing this
        .safetensors' now, not after four minutes of GPU time."""
        present = {
            v for node in self.graph.values() for v in node["inputs"].values()
            if isinstance(v, str)
        } - {"qwen_image_edit_fp8_e4m3fn.safetensors"}
        report = self.report(info=fake_object_info(self.graph, files=present))
        self.assertIn("missing_model", self.codes(report))
        self.assertIn("qwen_image_edit_fp8_e4m3fn", report.errors[0].message)

    def test_missing_optional_model_is_only_a_warning(self):
        """The Lightning LoRA ships switched off. Refusing to run because an
        unused file is absent would block a workflow that works perfectly."""
        lora = "Qwen-Image-Edit-Lightning-4steps-V1.0-bf16.safetensors"
        present = {
            v for node in self.graph.values() for v in node["inputs"].values()
            if isinstance(v, str)
        } - {lora}
        report = self.report(info=fake_object_info(self.graph, files=present))
        self.assertTrue(report.ok, [e.as_dict() for e in report.errors])
        self.assertIn("missing_model", {w.code for w in report.warnings})

    def test_renamed_widget_is_an_error(self):
        """A node update that renames a field must not turn into a silent
        no-op generation from an empty prompt."""
        info = fake_object_info(self.graph, rename_widgets={"prompt": "text"})
        report = self.report(info=info)
        self.assertIn("unknown_widget", self.codes(report))

    def test_out_of_range_number_is_an_error(self):
        info = copy.deepcopy(self.info)
        info["KSampler"]["input"]["required"]["seed"] = ["INT", {"min": 0, "max": 10}]
        report = self.report(values={**self.values, "$SEED": 9_999_999}, info=info)
        self.assertIn("out_of_range", self.codes(report))

    def test_invalid_choice_is_reported_as_such_not_as_a_missing_model(self):
        info = copy.deepcopy(self.info)
        info["KSampler"]["input"]["required"]["sampler_name"] = [["dpmpp_2m"], {}]
        report = self.report(info=info)
        self.assertIn("invalid_choice", self.codes(report))
        self.assertNotIn("missing_model", self.codes(report))

    def test_image_slot_refuses_anything_that_is_not_a_bridge_filename(self):
        for bad in (
            "../../../etc/passwd", "C:\\Windows\\win.ini", "/etc/shadow",
            "sub/rb_0123abcd.png", "anything.png", "rb_0123abcd.py", 12,
        ):
            with self.subTest(bad=bad):
                report = self.report(values={**self.values, "$IMAGE_1": bad})
                self.assertIn("bad_input_reference", self.codes(report))

    def test_a_freshly_written_input_is_not_an_invalid_choice(self):
        """LoadImage's choices are a listing of input/ taken when object_info
        was built, which can predate the frame we wrote milliseconds ago.
        Checking against it would reject every single generation."""
        report = self.report()
        self.assertTrue(report.ok, [e.as_dict() for e in report.errors])
        self.assertNotIn("invalid_choice", {w.code for w in report.warnings})

    def test_parameter_for_an_undeclared_slot_is_refused(self):
        report = self.report(values={**self.values, "$SOMETHING": "x"})
        self.assertIn("unknown_slot", self.codes(report))

    def test_missing_required_slot_is_an_error(self):
        graph = copy.deepcopy(self.graph)
        for node in graph.values():
            if node["_meta"].get("title") == "$OUTPUT":
                node["_meta"]["title"] = "Save Image"
                node["class_type"] = "NotAnOutput"
        template = templates.Template(
            id="t", manifest=self.template.manifest, graph=graph, hash="sha256:x"
        )
        report = validate.validate(template, self.values, fake_object_info(graph))
        self.assertIn("missing_slot", {e.code for e in report.errors})

    def test_prompt_length_is_capped(self):
        report = self.report(values={**self.values, "$PROMPT": "x" * 50000})
        self.assertIn("bad_value", self.codes(report))


class TestPruning(BridgeTestCase):
    """detachWhenAbsent: the only place the bridge changes a workflow's shape.

    Approved under five conditions, one test each.
    """

    def setUp(self) -> None:
        super().setUp()
        templates.reset()
        self.template = templates.get("qwen-image-edit-2511-base")
        self.graph = self.template.graph
        self.slots = binder.find_slots(self.graph, self.template.slots)
        self.info = fake_object_info(self.graph)
        self.outputs = binder.output_node_ids(self.slots)

    def tearDown(self) -> None:
        templates.reset()
        super().tearDown()

    def test_only_manifest_authorised_links_are_cut(self):
        """The request says 'no context frame'. It can never name a wire."""
        plan = self.template.detach_plan({"$IMAGE_1", "$IMAGE_2"})
        self.assertEqual(
            {(r["slot"], r["input"]) for r in plan},
            {("$PROMPT", "image3"), ("$NEGATIVE", "image3")},
        )
        # Nothing in the manifest authorises touching image1.
        self.assertNotIn(
            ("$PROMPT", "image1"), {(r["slot"], r["input"]) for r in plan}
        )

    def test_pruning_adds_no_nodes_and_changes_no_types(self):
        pruned, _ = binder.detach(
            self.graph, self.template.detach_plan({"$IMAGE_1"}), self.slots
        )
        self.assertEqual(set(pruned), set(self.graph))
        for node_id, node in pruned.items():
            self.assertEqual(node["class_type"], self.graph[node_id]["class_type"])

    def test_pruned_workflow_is_revalidated(self):
        pruned, _ = binder.detach(
            self.graph, self.template.detach_plan({"$IMAGE_1"}), self.slots
        )
        report = validate.check_after_pruning(pruned, self.outputs, self.info)
        self.assertTrue(report.ok, [i.as_dict() for i in report.errors])

    def test_the_output_must_stay_reachable(self):
        """Unplug the wrong wire and the run finishes having executed nothing
        of consequence -- with no error anywhere."""
        sabotaged = copy.deepcopy(self.graph)
        del sabotaged[self.slots["$OUTPUT"].node_id]["inputs"]["images"]
        report = validate.check_after_pruning(sabotaged, self.outputs, self.info)
        self.assertFalse(report.ok)
        self.assertIn("missing_input", {i.code for i in report.errors})

        report = validate.check_after_pruning(sabotaged, [], self.info)
        self.assertIn("no_output", {i.code for i in report.errors})

    def test_removed_links_are_reported(self):
        _, removed = binder.detach(
            self.graph, self.template.detach_plan({"$IMAGE_1"}), self.slots
        )
        self.assertTrue(removed)
        for entry in removed:
            self.assertIn(entry["input"], ("image2", "image3"))
            self.assertIn(entry["slot"], ("$PROMPT", "$NEGATIVE"))
            self.assertTrue(entry["node"])

    def test_unplugged_loader_stops_being_reachable(self):
        """Nodes are left in place on purpose: ComfyUI walks back from the
        outputs, so an unplugged LoadImage simply never runs, and the node table
        stays identical to the template the user installed."""
        loader = self.slots["$IMAGE_3"].node_id
        self.assertIn(loader, binder.reachable(self.graph, self.outputs))

        pruned, _ = binder.detach(
            self.graph, self.template.detach_plan({"$IMAGE_1", "$IMAGE_2"}), self.slots
        )
        self.assertIn(loader, pruned)                                   # still there
        self.assertNotIn(loader, binder.reachable(pruned, self.outputs))  # never runs

    def test_supplying_every_image_prunes_nothing(self):
        plan = self.template.detach_plan({"$IMAGE_1", "$IMAGE_2", "$IMAGE_3"})
        self.assertEqual(plan, [])

    def test_a_dangling_link_is_caught(self):
        broken = copy.deepcopy(self.graph)
        broken[self.slots["$PROMPT"].node_id]["inputs"]["image2"] = ["does-not-exist", 0]
        report = validate.check_after_pruning(broken, self.outputs, self.info)
        self.assertIn("dangling_link", {i.code for i in report.errors})


class TestQualityFallback(BridgeTestCase):
    """Preview selects the Lightning branch. If that file is absent, queueing
    anyway produces a run we already know will fail, so the quality steps down
    and says so. A warning followed by a doomed execution is not an answer."""

    def setUp(self) -> None:
        super().setUp()
        templates.reset()
        self.template = templates.get("qwen-image-edit-2511-base")
        self.lightning = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"

    def tearDown(self) -> None:
        templates.reset()
        super().tearDown()

    def test_preview_runs_when_lightning_is_installed(self):
        quality, fallback = self.template.resolve_quality("preview", lambda f, n: True)
        self.assertEqual(quality, "preview")
        self.assertIsNone(fallback)
        self.assertIs(self.template.quality_value("preview"), True)

    def test_preview_falls_back_to_final_without_lightning(self):
        quality, fallback = self.template.resolve_quality(
            "preview", lambda f, n: n != self.lightning
        )
        self.assertEqual(quality, "final")
        self.assertEqual(fallback["from"], "preview")
        self.assertEqual(fallback["to"], "final")
        self.assertIn("Lightning", fallback["reason"])
        self.assertIs(self.template.quality_value("final"), False)

    def test_final_is_unaffected_by_a_missing_lightning(self):
        quality, fallback = self.template.resolve_quality(
            "final", lambda f, n: n != self.lightning
        )
        self.assertEqual(quality, "final")
        self.assertIsNone(fallback)

    def test_final_still_works_when_every_file_is_missing(self):
        """Final loads no conditional model, so it stays runnable. A missing
        *required* model is a different failure and is reported separately."""
        quality, fallback = self.template.resolve_quality("preview", lambda f, n: False)
        self.assertEqual(quality, "final")

    def test_when_no_quality_can_run_it_says_so_instead_of_choosing_one(self):
        stub = templates.Template(
            id="stub", graph={}, hash="sha256:x",
            manifest={
                "id": "stub",
                "quality": {
                    "slot": "$PREVIEW",
                    "preview": {"value": True},
                    "final": {"value": False},
                },
                "requires": {"conditionalModels": [
                    {"folder": "loras", "file": "a.safetensors", "requiredFor": ["preview"]},
                    {"folder": "loras", "file": "b.safetensors", "requiredFor": ["final"]},
                ]},
            },
        )
        quality, fallback = stub.resolve_quality("preview", lambda f, n: False)
        self.assertIsNone(fallback["to"])
        self.assertIn("a.safetensors", fallback["reason"])


class TestValidationFollowsTheLiveGraph(BridgeTestCase):
    def setUp(self) -> None:
        super().setUp()
        templates.reset()
        self.template = templates.get("qwen-image-edit-2511-base")

    def tearDown(self) -> None:
        templates.reset()
        super().tearDown()

    def test_an_unplugged_loaders_placeholder_does_not_block_the_run(self):
        """The regression that broke case B on the real GPU. Validation ran
        before pruning, so it rejected the placeholder filename on a LoadImage
        that was about to be unplugged and would never have executed."""
        info = fake_object_info(self.template.graph)
        # LoadImage only offers files that exist in input/; the placeholder does not.
        info["LoadImage"]["input"]["required"]["image"] = [["rb_0123abcd.png"], {}]

        report = validate.validate(
            self.template,
            {"$PROMPT": "x", "$IMAGE_1": "rb_0123abcd.png"},   # no image 2 or 3
            info,
        )
        self.assertTrue(report.ok, [i.as_dict() for i in report.errors])
        self.assertTrue(report.pruned, "the unused image inputs should have been unplugged")

    def test_the_report_carries_the_graph_that_will_be_queued(self):
        report = validate.validate(
            self.template,
            {"$PROMPT": "hello", "$IMAGE_1": "rb_0123abcd.png"},
            fake_object_info(self.template.graph),
        )
        slots = binder.find_slots(self.template.graph, self.template.slots)
        self.assertEqual(report.graph[slots["$PROMPT"].node_id]["inputs"]["prompt"], "hello")
        self.assertNotIn("image3", report.graph[slots["$PROMPT"].node_id]["inputs"])
        self.assertNotIn(
            slots["$IMAGE_3"].node_id,
            binder.reachable(report.graph, binder.output_node_ids(slots)),
        )


class TestIntentRules(BridgeTestCase):
    """edit_pose needs a drawing; edit_keyframe does not."""

    def setUp(self) -> None:
        super().setUp()
        templates.reset()
        self.t = templates.get("qwen-image-edit-2511-base")

    def tearDown(self) -> None:
        templates.reset()
        super().tearDown()

    def test_edit_pose_requires_the_annotated_frame(self):
        report = validate.check_intent(self.t, "edit_pose", {"cleanFrame"})
        self.assertFalse(report.ok)
        self.assertEqual(report.errors[0].code, "missing_role")
        self.assertIn("annotatedFrame", report.errors[0].message)

    def test_edit_pose_is_happy_with_clean_plus_annotated(self):
        self.assertTrue(
            validate.check_intent(self.t, "edit_pose", {"cleanFrame", "annotatedFrame"}).ok
        )

    def test_edit_keyframe_runs_on_the_clean_frame_alone(self):
        self.assertTrue(validate.check_intent(self.t, "edit_keyframe", {"cleanFrame"}).ok)

    def test_context_is_optional_for_both(self):
        for intent in ("edit_pose", "edit_keyframe"):
            with self.subTest(intent=intent):
                self.assertNotIn(
                    "contextFrames", self.t.required_roles(intent)
                )

    def test_an_unsupported_intent_is_refused(self):
        report = validate.check_intent(self.t, "make_video", {"cleanFrame"})
        self.assertEqual(report.errors[0].code, "intent_not_supported")


class TestPromptComposition(BridgeTestCase):
    """The prompt and the wires come from the same decision."""

    def setUp(self) -> None:
        super().setUp()
        templates.reset()
        self.t = templates.get("qwen-image-edit-2511-base")

    def tearDown(self) -> None:
        templates.reset()
        super().tearDown()

    def test_only_connected_pictures_are_described(self):
        one = validate.compose_prompt(self.t, "raise the arm", {"cleanFrame"})
        self.assertIn("Picture 1", one)
        self.assertNotIn("Picture 2", one)
        self.assertNotIn("Picture 3", one)

        two = validate.compose_prompt(
            self.t, "raise the arm", {"cleanFrame", "annotatedFrame"}
        )
        self.assertIn("Picture 2", two)
        self.assertNotIn("Picture 3", two)

        three = validate.compose_prompt(
            self.t, "raise the arm", {"cleanFrame", "annotatedFrame", "contextFrames"}
        )
        self.assertIn("Picture 3", three)

    def test_the_instruction_survives_and_comes_last(self):
        text = validate.compose_prompt(self.t, "raise the arm", {"cleanFrame"})
        self.assertTrue(text.rstrip().endswith("raise the arm"))

    def test_the_always_line_is_always_there(self):
        for roles in ({"cleanFrame"}, {"cleanFrame", "annotatedFrame"}):
            self.assertIn("preserving character design", validate.compose_prompt(
                self.t, "x", roles))


class TestTemplateSecurityLayer(BridgeTestCase):
    """Layer 1: every node, including the ones nothing reaches."""

    def setUp(self) -> None:
        super().setUp()
        templates.reset()
        self.t = templates.get("qwen-image-edit-2511-base")

    def tearDown(self) -> None:
        templates.reset()
        super().tearDown()

    def test_the_shipped_templates_declare_every_class_they_use(self):
        for tid in ("qwen-image-edit", "qwen-image-edit-2511-base"):
            with self.subTest(template=tid):
                t = templates.get(tid)
                used = {n["class_type"] for n in t.graph.values()}
                self.assertEqual(used - set(t.requires["nodes"]), set())

    def test_an_undeclared_class_is_refused_even_when_disconnected(self):
        """A node nothing reaches today is one pruning rule away from running
        tomorrow, and a ComfyUI node is arbitrary Python."""
        graph = copy.deepcopy(self.t.graph)
        graph["999"] = {"class_type": "SomeoneElsesNode", "inputs": {}}   # wired to nothing
        tampered = templates.Template(
            id=self.t.id, graph=graph, manifest=self.t.manifest, hash="sha256:x"
        )
        report = validate.check_template_security(
            tampered, fake_object_info(graph)
        )
        self.assertFalse(report.ok)
        self.assertEqual(report.errors[0].code, "node_not_allowed")

    def test_the_security_layer_is_separate_from_the_executable_one(self):
        """They answer different questions and must not be collapsed: one
        guards against a tampered template, the other against a broken run."""
        self.assertTrue(validate.check_template_security(
            self.t, fake_object_info(self.t.graph)).ok)
        slots = binder.find_slots(self.t.graph, self.t.slots)
        self.assertTrue(validate.check_after_pruning(
            self.t.graph, binder.output_node_ids(slots),
            fake_object_info(self.t.graph)).ok)


class TestCheckpointVariants(BridgeTestCase):
    """Static preference. Deliberately not a benchmark-driven selector."""

    INT8 = "qwen_image_edit_2511_int8_convrot.safetensors"
    FP8 = "qwen_image_edit_2511_fp8mixed.safetensors"

    def setUp(self) -> None:
        super().setUp()
        templates.reset()
        benchmarks.reset_cache()
        self.t = templates.get("qwen-image-edit-2511-base")

    def tearDown(self) -> None:
        templates.reset()
        benchmarks.reset_cache()
        super().tearDown()

    def test_the_preferred_variant_wins_when_installed(self):
        chosen = self.t.select_checkpoint(file_exists=lambda f, n: True)
        self.assertEqual(chosen["variant"], "int8_convrot")
        self.assertEqual(chosen["file"], self.INT8)
        self.assertEqual(chosen["selectedBy"], "preferred")

    def test_an_alternative_is_used_when_the_preferred_is_absent(self):
        chosen = self.t.select_checkpoint(file_exists=lambda f, n: n != self.INT8)
        self.assertEqual(chosen["variant"], "fp8mixed")
        self.assertEqual(chosen["selectedBy"], "alternative")

    def test_a_user_override_wins(self):
        chosen = self.t.select_checkpoint(
            file_exists=lambda f, n: True, override="fp8mixed"
        )
        self.assertEqual(chosen["selectedBy"], "user-override")

    def test_telemetry_does_not_change_the_choice(self):
        """Recorded for diagnosis, never for control. Ranking by these rows
        once demoted the checkpoint that had just proved faster, because a cold
        run's total seconds got compared against a warm run's per-step time."""
        benchmarks.record({
            "templateId": self.t.id, "checkpointVariant": "fp8mixed",
            "secondsPerStep": 0.01, "success": True,
        })
        chosen = self.t.select_checkpoint(file_exists=lambda f, n: True)
        self.assertEqual(chosen["variant"], "int8_convrot")
        self.assertEqual(chosen["selectedBy"], "preferred")

    def test_the_reference_measurement_is_labelled_as_such(self):
        """Information, not a claim about the machine about to run."""
        chosen = self.t.select_checkpoint(file_exists=lambda f, n: True)
        reference = chosen["referenceMeasurement"]
        self.assertIn("RTX 3090", reference["machine"])
        self.assertEqual(reference["secondsPerStep"], 4.17)
        self.assertNotIn("measuredOnThisMachine", chosen)

    def test_fp8mixed_is_kept_as_an_alternative(self):
        self.assertIn("fp8mixed", self.t.checkpoints["variants"])
        self.assertIn("fp8mixed", self.t.checkpoints["alternatives"])

    # ----------------------------------------------------------------
    # Compatibility with what this ComfyUI can actually read.
    #
    # "The file is on disk" was never the same question as "this build can
    # load it". A real 3090 install running ComfyUI 0.22.3 had both variants
    # downloaded, picked the preferred one because the file existed, and died
    # a minute later with `UNETLoader: 'int8_tensorwise'` -- a format that
    # build simply does not have. The manifest already knew: the variant
    # declares requiresNativeOps and the alternative was right there.
    # ----------------------------------------------------------------

    # What a ComfyUI that understands int8 reports, versus one that does not.
    OPS_WITH_INT8 = {"float8_e4m3fn", "float8_e5m2", "int8_tensorwise", "convrot_w4a4"}
    OPS_WITHOUT_INT8 = {"float8_e4m3fn", "float8_e5m2", "nvfp4"}

    def test_preferred_is_used_when_this_build_supports_it(self):
        chosen = self.t.select_checkpoint(
            file_exists=lambda f, n: True, supported_ops=self.OPS_WITH_INT8
        )
        self.assertEqual(chosen["variant"], "int8_convrot")
        self.assertEqual(chosen["selectedBy"], "preferred")
        self.assertTrue(chosen["compatible"])
        self.assertNotIn("fallbackFrom", chosen)

    def test_an_unreadable_preferred_falls_back_to_the_declared_alternative(self):
        """The bug this whole block exists for."""
        chosen = self.t.select_checkpoint(
            file_exists=lambda f, n: True, supported_ops=self.OPS_WITHOUT_INT8
        )
        self.assertEqual(chosen["variant"], "fp8mixed")
        self.assertEqual(chosen["selectedBy"], "alternative")
        self.assertTrue(chosen["compatible"])

    def test_the_fallback_says_what_it_fell_back_from_and_why(self):
        """Loading something other than the manifest's first choice, and never
        saying so, is its own bug."""
        chosen = self.t.select_checkpoint(
            file_exists=lambda f, n: True, supported_ops=self.OPS_WITHOUT_INT8
        )
        origin = chosen["fallbackFrom"]
        self.assertEqual(origin["variant"], "int8_convrot")
        self.assertEqual(origin["code"], "unsupported_quantization")
        self.assertIn("int8_tensorwise", origin["missingOps"])

    def test_a_missing_preferred_still_falls_back_on_the_file_alone(self):
        """The older reason to fall back has to keep working."""
        chosen = self.t.select_checkpoint(
            file_exists=lambda f, n: n != self.INT8, supported_ops=self.OPS_WITH_INT8
        )
        self.assertEqual(chosen["variant"], "fp8mixed")
        self.assertEqual(chosen["fallbackFrom"]["code"], "missing_model")

    def test_nothing_readable_is_reported_not_guessed(self):
        """It still comes back with the preferred variant -- naming a file and
        its download URL beats a bare "no checkpoint" -- but flagged
        incompatible, and carrying why EVERY variant was rejected."""
        chosen = self.t.select_checkpoint(
            file_exists=lambda f, n: True, supported_ops={"nvfp4"}
        )
        self.assertFalse(chosen["compatible"])
        self.assertEqual(chosen["variant"], "int8_convrot")
        self.assertIn("int8_tensorwise", chosen["unsupportedOps"])
        self.assertEqual(
            {s["variant"] for s in chosen["skipped"]}, {"int8_convrot", "fp8mixed"}
        )
        self.assertEqual({s["code"] for s in chosen["skipped"]},
                         {"unsupported_quantization"})

    def test_an_unknown_build_screens_nothing_out(self):
        """None means nobody could tell -- an older ComfyUI, or the bridge
        imported outside one. Refusing everything there would break installs
        that work today, so the behaviour is exactly what it was before."""
        chosen = self.t.select_checkpoint(
            file_exists=lambda f, n: True, supported_ops=None
        )
        self.assertEqual(chosen["variant"], "int8_convrot")
        self.assertTrue(chosen["compatible"])

    def test_emulated_is_not_incompatible(self):
        """No regression for the machine this was found on. ComfyUI keeps a
        separate _disabled set for formats it knows but cannot run natively;
        those are emulated -- slower, and they DO produce an image. fp8mixed on
        an Ampere card is exactly that, and screening it out would refuse the
        one checkpoint that works."""
        chosen = self.t.select_checkpoint(
            file_exists=lambda f, n: n == self.FP8, supported_ops=self.OPS_WITHOUT_INT8
        )
        self.assertEqual(chosen["variant"], "fp8mixed")
        self.assertTrue(chosen["compatible"])

    def test_the_declared_ops_come_from_the_manifest_not_from_code(self):
        """No build number anywhere in the selector: the variant says what it
        needs, the install says what it has."""
        variants = self.t.checkpoints["variants"]
        self.assertIn("int8_tensorwise", variants["int8_convrot"]["requiresNativeOps"])
        self.assertIn("float8_e4m3fn", variants["fp8mixed"]["requiresNativeOps"])
        source = code_without_comments(
            Path(__file__).resolve().parents[1] / "reanimator/workflow/templates.py"
        )
        for hardcoded in ("0.22", "0.29", "int8_tensorwise", "convrot_w4a4"):
            self.assertNotIn(hardcoded, source)

    def test_an_override_that_cannot_load_is_still_reported_incompatible(self):
        """"I know what I am doing" does not make the format readable."""
        chosen = self.t.select_checkpoint(
            file_exists=lambda f, n: True, override="int8_convrot",
            supported_ops=self.OPS_WITHOUT_INT8,
        )
        self.assertEqual(chosen["selectedBy"], "user-override")
        self.assertFalse(chosen["compatible"])

    def test_the_tier_and_its_vram_floor_are_declared(self):
        self.assertEqual(self.t.tier, "full")
        self.assertEqual(self.t.minimum_vram_gb, 24)

    def test_a_card_below_the_floor_is_pointed_at_cloud(self):
        report = validate.check_device_fit(self.t, {"vramTotalMb": 12288})
        self.assertFalse(report.ok)
        self.assertEqual(report.errors[0].code, "insufficient_vram")
        self.assertEqual(report.errors[0].detail["recommend"], "cloud")

    def test_a_24gb_card_clears_the_floor(self):
        """24576 MB is 24.0 GB; a naive comparison reads it as 23.99 and
        refuses the exact card the tier was written for."""
        self.assertTrue(validate.check_device_fit(self.t, {"vramTotalMb": 24576}).ok)

    def test_an_unknown_gpu_is_not_refused(self):
        for profile in (None, {}, {"vramTotalMb": None}):
            with self.subTest(profile=profile):
                self.assertTrue(validate.check_device_fit(self.t, profile).ok)

    def test_only_the_selected_variant_is_required_on_disk(self):
        """Reporting every variant missing would tell the user to download
        20 GB they do not need."""
        chosen = self.t.select_checkpoint(file_exists=lambda f, n: True)
        report = validate.validate(
            self.t, {"$PROMPT": "x", "$IMAGE_1": "rb_0123abcd.png"},
            fake_object_info(self.t.graph),
            file_exists=lambda f, n: n != self.FP8,      # fp8 absent
            selected_checkpoint=chosen,
        )
        self.assertTrue(report.ok, [e.as_dict() for e in report.errors])

    def test_the_checkpoint_slot_exists_and_is_bindable(self):
        slots = binder.find_slots(self.t.graph, self.t.slots)
        self.assertEqual(slots["$CHECKPOINT"].class_type, "UNETLoader")
        self.assertEqual(slots["$CHECKPOINT"].widget, "unet_name")
        bound = binder.fill(self.t.graph, {"$CHECKPOINT": self.INT8}, slots)
        self.assertEqual(
            bound[slots["$CHECKPOINT"].node_id]["inputs"]["unet_name"], self.INT8
        )


class TestPresetResolution(BridgeTestCase):
    """Choosing the preset is the bridge's job, not the browser's.

    The client knows what the user is doing; only the bridge knows what is
    installed and what the GPU is. A client-side selector is a second
    implementation of this decision, and it drifted the first time a preset was
    added -- local-gpu.js preferred a template by literal id and kept choosing
    the older one, reporting the mismatch as a malformed request.
    """

    RIG = {"device": "NVIDIA GeForce RTX 3090", "computeCapability": "8.6",
           "vramTotalMb": 24576}

    def setUp(self) -> None:
        super().setUp()
        templates.reset()

    def tearDown(self) -> None:
        templates.reset()
        super().tearDown()

    def test_edit_pose_resolves_to_the_2511_preset(self):
        template, fallback = templates.resolve_for_intent(
            "edit_pose", self.RIG, lambda f, n: True
        )
        self.assertEqual(template.id, "qwen-image-edit-2511-base")
        self.assertIsNone(fallback)

    def test_priority_decides_not_the_id(self):
        for intent in ("edit_pose", "edit_keyframe"):
            with self.subTest(intent=intent):
                template, _ = templates.resolve_for_intent(
                    intent, self.RIG, lambda f, n: True
                )
                self.assertEqual(
                    template.manifest.get("priority"),
                    max(t.manifest.get("priority") or 0
                        for t in templates.load_all().values()
                        if intent in t.intents or not t.intents),
                )

    def test_a_missing_model_pushes_it_down_the_list_and_says_so(self):
        checkpoint = "qwen_image_edit_2511_int8_convrot.safetensors"
        fp8 = "qwen_image_edit_2511_fp8mixed.safetensors"
        template, fallback = templates.resolve_for_intent(
            "edit_pose", self.RIG, lambda f, n: n not in (checkpoint, fp8)
        )
        # Nothing installed can serve it: reported, with the file to fetch.
        self.assertIsNone(template)
        self.assertEqual(fallback["code"], "no_runnable_preset")
        self.assertEqual(fallback["blocked"][0]["code"], "missing_model")
        self.assertEqual(fallback["recommend"], "cloud")

    def test_a_card_below_the_tier_floor_blocks_the_preset(self):
        template, fallback = templates.resolve_for_intent(
            "edit_pose", {**self.RIG, "vramTotalMb": 8192}, lambda f, n: True
        )
        self.assertIsNone(template)
        self.assertEqual(fallback["blocked"][0]["code"], "insufficient_vram")

    def test_an_unserved_intent_is_named(self):
        template, fallback = templates.resolve_for_intent(
            "make_video", self.RIG, lambda f, n: True
        )
        self.assertIsNone(template)
        self.assertEqual(fallback["code"], "intent_not_supported")
        self.assertIn("edit_pose", fallback["available"])

    def test_an_explicit_template_id_still_wins(self):
        """Pinning one is how a test or a power user overrides the resolver."""
        self.assertEqual(templates.get("qwen-image-edit").id, "qwen-image-edit")

    def test_the_summary_carries_what_a_ui_needs_but_not_a_second_selector(self):
        summary = next(s for s in templates.summaries()
                       if s["id"] == "qwen-image-edit-2511-base")
        for field in ("intents", "priority", "tier", "roles", "images"):
            self.assertIn(field, summary)


class TestModelFilesOnDisk(TemplateTestCase):
    """ComfyUI listing a model is not proof the file exists.

    Seen on the development machine: qwen_image_edit_2511_bf16.safetensors was
    offered by UNETLoader and present in no models folder at all.
    """

    def test_a_listed_but_absent_model_is_still_reported(self):
        template = templates.get("qwen-image-edit-2511-base")
        values = {"$PROMPT": "x", "$IMAGE_1": "rb_0123abcd.png"}
        info = fake_object_info(template.graph)     # object_info lists everything

        report = validate.validate(template, values, info)
        self.assertTrue(report.ok, "without a disk check this passes -- that is the bug")

        report = validate.validate(
            template, values, info, file_exists=lambda folder, name: False
        )
        self.assertFalse(report.ok)
        codes = {i.code for i in report.errors}
        self.assertIn("missing_model", codes)

    def test_a_conditional_model_is_only_a_warning(self):
        template = templates.get("qwen-image-edit-2511-base")
        lightning = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
        report = validate.validate(
            template,
            {"$PROMPT": "x", "$IMAGE_1": "rb_0123abcd.png"},
            fake_object_info(template.graph),
            file_exists=lambda folder, name: name != lightning,
        )
        self.assertTrue(report.ok, [i.as_dict() for i in report.errors])
        warning = next(w for w in report.warnings if w.code == "missing_model")
        self.assertIn("preview", warning.message)

    def test_the_message_says_where_to_put_it(self):
        template = templates.get("qwen-image-edit-2511-base")
        report = validate.validate(
            template,
            {"$PROMPT": "x", "$IMAGE_1": "rb_0123abcd.png"},
            fake_object_info(template.graph),
            file_exists=lambda folder, name: False,
        )
        messages = " ".join(e.message for e in report.errors)
        for folder in ("models/text_encoders/", "models/vae/"):
            self.assertIn(folder, messages)
        self.assertIn("GB", messages)


class TestGeometry(BridgeTestCase):
    """Pad in, crop back out, lose nothing."""

    def test_the_documented_case(self):
        t = geometry.plan((848, 478))
        self.assertEqual(t.as_dict(), {
            "source": [848, 478], "padded": [885, 478], "model": [1392, 752],
            "output": [848, 478],
            "padding": {"left": 18, "right": 19, "top": 0, "bottom": 0},
            "mode": "fit_pad", "padMode": "black",
        })

    def test_padding_removes_the_crop_that_ate_the_frame(self):
        """The whole reason this exists. Unpadded, FluxKontextImageScale takes
        10 rows off the top and 10 off the bottom -- 4.2% of an 848x478 frame,
        gone before the model sees it."""
        t = geometry.plan((848, 478))
        self.assertEqual(geometry.comfy_would_crop(848, 478, *t.model), (0, 10))
        self.assertEqual(geometry.comfy_would_crop(*t.padded, *t.model), (0, 0))

    def test_no_bucket_ever_leaves_a_crop_after_padding(self):
        """Rounding to whole pixels could leave the aspect a hair off, and a
        hair is a whole row. Checked across shapes rather than assumed."""
        checked = 0
        for w in (320, 640, 848, 1024, 1920, 401):
            for h in (180, 240, 478, 1024, 1080, 337):
                with self.subTest(size=(w, h)):
                    try:
                        t = geometry.plan((w, h))   # raises 'still_crops' if not
                    except geometry.GeometryError as exc:
                        # Shapes refused outright never reach the scaler, so they
                        # have nothing to prove here.
                        self.assertEqual(exc.code, "aspect_unsupported")
                        continue
                    self.assertEqual(
                        geometry.comfy_would_crop(*t.padded, *t.model), (0, 0)
                    )
                    checked += 1
        self.assertGreater(checked, 20, "the sweep stopped checking anything")

    def test_padding_is_deterministic_and_the_odd_pixel_goes_right(self):
        t = geometry.plan((848, 478))
        self.assertEqual(t.padding["left"] + 1, t.padding["right"])
        self.assertEqual(geometry.plan((848, 478)).as_dict(), t.as_dict())

    def test_a_round_trip_restores_the_exact_source_size(self):
        from PIL import Image
        import io

        for size in ((848, 478), (1920, 1080), (512, 512), (401, 337)):
            with self.subTest(size=size):
                t = geometry.plan(size)
                raw = tiny_png(*size)
                padded = geometry.pad(raw, t)
                self.assertEqual(Image.open(io.BytesIO(padded)).size, t.padded)

                # what ComfyUI does to it: a plain resize, no crop
                scaled = Image.open(io.BytesIO(padded)).resize(t.model, Image.LANCZOS)
                buffer = io.BytesIO(); scaled.save(buffer, "PNG")

                back = geometry.unpad(buffer.getvalue(), t)
                self.assertEqual(Image.open(io.BytesIO(back)).size, size)

    def test_the_drawing_stays_where_it_was_drawn(self):
        """Alignment, not just dimensions. A marker near each corner has to come
        back near the same corner: this is what actually breaks when the clean
        and annotated frames are transformed even slightly differently."""
        from PIL import Image, ImageDraw
        import io

        size = (848, 478)
        image = Image.new("RGB", size, (20, 20, 20))
        draw = ImageDraw.Draw(image)
        marks = {"tl": (12, 12), "tr": (835, 12), "bl": (12, 465), "br": (835, 465)}
        for x, y in marks.values():
            draw.rectangle([x - 6, y - 6, x + 6, y + 6], fill=(255, 255, 255))
        buffer = io.BytesIO(); image.save(buffer, "PNG")

        t = geometry.plan(size)
        padded = Image.open(io.BytesIO(geometry.pad(buffer.getvalue(), t)))
        scaled = padded.resize(t.model, Image.LANCZOS)
        out = io.BytesIO(); scaled.save(out, "PNG")
        result = Image.open(io.BytesIO(geometry.unpad(out.getvalue(), t))).convert("L")

        for name, (x, y) in marks.items():
            with self.subTest(mark=name):
                patch = result.crop((x - 8, y - 8, x + 9, y + 9))
                self.assertGreater(
                    max(patch.getdata()), 180,
                    f"the {name} marker did not survive the round trip in place",
                )

    def test_black_bars_are_actually_black(self):
        from PIL import Image
        import io

        t = geometry.plan((848, 478))
        padded = Image.open(io.BytesIO(geometry.pad(tiny_png(848, 478), t)))
        self.assertEqual(padded.getpixel((0, 100)), (0, 0, 0))
        self.assertEqual(padded.getpixel((884, 100)), (0, 0, 0))
        self.assertNotEqual(padded.getpixel((400, 100)), (0, 0, 0))

    def test_pad_modes_are_selectable_without_changing_the_contract(self):
        for mode in geometry.PAD_MODES:
            with self.subTest(mode=mode):
                t = geometry.plan((848, 478), pad_mode=mode)
                self.assertEqual(t.as_dict()["padMode"], mode)
                self.assertEqual(t.padding, {"left": 18, "right": 19, "top": 0, "bottom": 0})
                geometry.pad(tiny_png(848, 478), t)     # must not raise

    def test_an_unknown_pad_mode_is_refused(self):
        with self.assertRaises(geometry.GeometryError) as ctx:
            geometry.plan((848, 478), pad_mode="blur")
        self.assertEqual(ctx.exception.code, "bad_pad_mode")

    def test_a_square_frame_needs_no_padding(self):
        t = geometry.plan((1024, 1024))
        self.assertEqual(t.mode, "passthrough")
        self.assertFalse(t.has_padding)

    def test_padding_refuses_a_frame_of_the_wrong_size(self):
        """A transform belongs to one frame size. Applying it to another would
        silently shift everything by the difference."""
        t = geometry.plan((848, 478))
        with self.assertRaises(geometry.GeometryError) as ctx:
            geometry.pad(tiny_png(640, 360), t)
        self.assertEqual(ctx.exception.code, "size_mismatch")

    def test_the_continuity_board_is_fitted_not_stretched(self):
        """It is a separate reference and may be any shape, so it gets its own
        letterbox into the canvas -- never the frame's transform, which would
        be arithmetic from one picture applied to another."""
        from PIL import Image
        import io

        fitted = geometry.fit_into(tiny_png(400, 400), (885, 478))
        image = Image.open(io.BytesIO(fitted))
        self.assertEqual(image.size, (885, 478))
        # A square fitted into a wide canvas: bars at the sides, content centred
        # and still square.
        self.assertEqual(image.getpixel((5, 239)), (0, 0, 0))
        self.assertNotEqual(image.getpixel((442, 239)), (0, 0, 0))

    def test_hand_typed_canvas_sizes_are_accepted(self):
        """Blank projects let the user type the canvas size, so the frame size
        no longer comes from a file and is no longer implicitly sane. These are
        the shapes a person actually types."""
        for size in ((1280, 720), (1920, 1080), (1000, 1000), (800, 600),
                     (1080, 1920), (1001, 563), (512, 512), (2560, 1440)):
            with self.subTest(size=size):
                t = geometry.plan(size)
                bars = 1 - (size[0] * size[1]) / (t.padded[0] * t.padded[1])
                self.assertLess(bars, 0.5)

    def test_absurd_canvas_sizes_are_refused_with_a_readable_reason(self):
        """Not a stack trace about tensor dimensions from inside ComfyUI."""
        for size, code in (
            ((1, 1), "source_too_small"),
            ((3, 2), "source_too_small"),
            ((4000, 10), "source_too_small"),
            ((30000, 30000), "source_too_large"),
            ((8000, 6000), "source_too_large"),
        ):
            with self.subTest(size=size):
                with self.assertRaises(geometry.GeometryError) as ctx:
                    geometry.plan(size)
                self.assertEqual(ctx.exception.code, code)
                self.assertIn(str(size[0]), str(ctx.exception))

    def test_an_aspect_no_bucket_can_serve_is_refused(self):
        """A shape padded to more than half bars does not fail -- it produces
        confident rubbish, with the model spending its capacity on black."""
        with self.assertRaises(geometry.GeometryError) as ctx:
            geometry.plan((4000, 400))
        self.assertEqual(ctx.exception.code, "aspect_unsupported")
        self.assertIn(":1", str(ctx.exception))     # names the usable range

    def test_a_huge_canvas_never_reaches_pillow(self):
        """30000x30000 in RGB is 2.7 GB. The guard is in plan(), before a single
        pixel is allocated."""
        with self.assertRaises(geometry.GeometryError):
            geometry.plan((30000, 30000))

    def test_a_non_image_upload_is_refused(self):
        with self.assertRaises(geometry.GeometryError) as ctx:
            geometry.measure(b"not an image at all")
        self.assertEqual(ctx.exception.code, "bad_image")

    def test_pillow_is_available(self):
        """Declared as a dependency, but the bridge runs inside ComfyUI's own
        interpreter -- so what matters is whether it is importable there."""
        self.assertTrue(geometry.pillow_available())


class TestRunnerOutputs(BridgeTestCase):
    """Port of serve.py's output parsing (lines 658-703)."""

    def test_images_are_parsed(self):
        entry = {"outputs": {"60": {"images": [
            {"filename": "ComfyUI_00001_.png", "subfolder": "", "type": "output"}
        ]}}}
        items = runner.parse_outputs(entry)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["kind"], "image")
        self.assertEqual(items[0]["mime"], "image/png")

    def test_video_key_is_parsed(self):
        entry = {"outputs": {"9": {"video": [
            {"filename": "out.mp4", "subfolder": "sub", "type": "output"}
        ]}}}
        items = runner.parse_outputs(entry)
        self.assertEqual(items[0]["kind"], "video")
        self.assertEqual(items[0]["subfolder"], "sub")

    def test_a_save_image_node_that_wrote_an_mp4_is_a_video(self):
        """serve.py checked this explicitly at line 684, and it was right to."""
        entry = {"outputs": {"9": {"images": [
            {"filename": "clip.mp4", "subfolder": "", "type": "output"}
        ]}}}
        self.assertEqual(runner.parse_outputs(entry)[0]["kind"], "video")

    def test_entries_without_a_filename_are_ignored(self):
        entry = {"outputs": {"9": {"images": [{"subfolder": ""}, "not-a-dict"]}}}
        self.assertEqual(runner.parse_outputs(entry), [])

    def test_execution_error_message_is_extracted(self):
        entry = {"status": {"status_str": "error", "completed": False, "messages": [
            ["execution_start", {}],
            ["execution_error", {"node_type": "UNETLoader",
                                 "exception_message": "value not in list"}],
        ]}}
        self.assertEqual(runner.entry_error(entry), "UNETLoader: value not in list")

    def test_successful_entry_has_no_error(self):
        entry = {"status": {"status_str": "success", "completed": True, "messages": []}}
        self.assertIsNone(runner.entry_error(entry))


class TestRunnerConfinement(BridgeTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.out = Path(self._tmp.name) / "output"
        (self.out / "sub").mkdir(parents=True)
        (self.out / "sub" / "result.png").write_bytes(b"png")
        self.secret = Path(self._tmp.name) / "secret.txt"
        self.secret.write_text("private")

        self.inputs = Path(self._tmp.name) / "input"
        self.inputs.mkdir()

        class FakeFolderPaths:
            get_input_directory = staticmethod(lambda: str(self.inputs))
            get_directory_by_type = staticmethod(
                lambda kind: str(self.out) if kind in ("output", "temp") else None
            )

        self._orig = runner._folder_paths
        runner._folder_paths = lambda: FakeFolderPaths  # type: ignore[assignment]

    def tearDown(self) -> None:
        runner._folder_paths = self._orig  # type: ignore[assignment]
        super().tearDown()

    def test_output_resolves_inside_comfyui(self):
        path = runner.resolve_output(
            {"type": "output", "subfolder": "sub", "filename": "result.png"}
        )
        self.assertEqual(path, (self.out / "sub" / "result.png").resolve())

    def test_output_traversal_is_refused(self):
        """filename and subfolder come from a node's own code, and a custom node
        is free to report any string it likes."""
        for item in (
            {"type": "output", "subfolder": "..", "filename": "secret.txt"},
            {"type": "output", "subfolder": "", "filename": "../secret.txt"},
            {"type": "output", "subfolder": "sub/../..", "filename": "secret.txt"},
            {"type": "output", "subfolder": "", "filename": str(self.secret)},
        ):
            with self.subTest(item=item), self.assertRaises(runner.RunError) as ctx:
                runner.resolve_output(item)
            self.assertIn(ctx.exception.code, ("bad_output", "output_missing"))

    def test_unknown_output_type_is_refused(self):
        with self.assertRaises(runner.RunError):
            runner.resolve_output({"type": "models", "subfolder": "", "filename": "x.png"})

    def test_write_input_names_the_file_itself(self):
        name = runner.write_input(b"\x89PNG data", ".png")
        self.assertTrue(name.startswith("rb_"))
        self.assertTrue(name.endswith(".png"))
        self.assertTrue((self.inputs / name).is_file())
        # And the name it invents is exactly what the validator will accept.
        self.assertRegex(name, validate.INPUT_FILENAME_RE)

    def test_write_input_refuses_anything_but_an_image(self):
        for suffix in (".py", ".safetensors", ".exe", "", ".png.py"):
            with self.subTest(suffix=suffix), self.assertRaises(runner.RunError) as ctx:
                runner.write_input(b"x", suffix)
            self.assertEqual(ctx.exception.code, "bad_input_type")

    def test_write_input_caps_the_size(self):
        with self.assertRaises(runner.RunError) as ctx:
            runner.write_input(b"x" * (runner.MAX_INPUT_BYTES + 1), ".png")
        self.assertEqual(ctx.exception.code, "input_too_large")

    def test_empty_upload_is_refused(self):
        with self.assertRaises(runner.RunError):
            runner.write_input(b"", ".png")

    def test_delete_input_only_touches_bridge_files(self):
        victim = self.inputs / "someone_elses_photo.png"
        victim.write_bytes(b"x")
        runner.delete_input("someone_elses_photo.png")
        self.assertTrue(victim.is_file())


class TestRunnerTransport(BridgeTestCase):
    """The bridge talks to ComfyUI in process, never over a port.

    ComfyUI's port is assigned per instance by the Desktop hub -- 8000 on the
    development machine, not the 8188 everybody assumes. Any code that guesses
    it works until it reaches somebody else's install.
    """

    def test_runner_never_reaches_for_a_port(self):
        # Comments and docstrings are stripped first: this module explains the
        # trap at length, and a test that fired on the explanation would be
        # asserting that nobody may write the word "8188" down.
        code = code_without_comments(Path(runner.__file__))
        for forbidden in ("8188", "8000", "urlopen", "urllib", "requests",
                          "ClientSession", "127.0.0.1", "localhost", "COMFYUI_URL"):
            self.assertTrue(
                forbidden not in code,
                f"runner.py talks to ComfyUI in process; it must never mention "
                f"{forbidden!r}",
            )

    def test_outside_comfyui_the_failure_is_named(self):
        """Not an ImportError traceback: the bridge reports 503 with a sentence
        the editor can show."""
        with self.assertRaises(runner.ComfyUnavailable):
            runner._prompt_server()
        with self.assertRaises(runner.ComfyUnavailable):
            runner._folder_paths()

    def _queue_one(self, graph=None, outputs=None):
        queued = []

        class FakeQueue:
            def put(self, item):
                # Assert the CONTRACT, not our own output. The first version of
                # this test unpacked exactly five elements, so it agreed with
                # the bug instead of catching it: ComfyUI reads item[5].
                assert isinstance(item, tuple), "the queue takes a tuple"
                expected = 6 if runner._queue_wants_sensitive() else 5
                assert len(item) == expected, (
                    f"prompt_worker will index item[{expected - 1}]; got {len(item)} elements"
                )
                queued.append(item)

        class FakeServer:
            number = 41
            prompt_queue = FakeQueue()

        server = FakeServer()
        orig = runner._prompt_server
        runner._prompt_server = lambda: server  # type: ignore[assignment]
        try:
            graph = graph or {"7": {"class_type": "SaveImage", "inputs": {}}}
            prompt_id = asyncio.run(runner.queue(graph, outputs or ["7"]))
        finally:
            runner._prompt_server = orig  # type: ignore[assignment]
        return queued, prompt_id, server, graph

    def test_queue_uses_the_prompt_queue_directly(self):
        queued, prompt_id, server, graph = self._queue_one()

        self.assertEqual(len(queued), 1)
        item = queued[0]
        self.assertEqual(item[0], 41)
        self.assertEqual(server.number, 42)
        self.assertEqual(item[1], prompt_id)
        self.assertEqual(item[2], graph)
        self.assertEqual(item[3], {})
        self.assertEqual(item[4], ["7"])

    def test_queue_tuple_carries_the_sensitive_element(self):
        """The regression. ComfyUI's prompt_worker does `sensitive = item[5]`
        unconditionally; a five-element tuple raises IndexError *inside the
        worker thread*, which kills it. After that the instance accepts prompts
        and runs none of them -- ours and the user's alike -- until restarted,
        and the only symptom in the editor is "Generating…" that never ends."""
        import sys
        import types

        fake = types.ModuleType("execution")
        fake.SENSITIVE_EXTRA_DATA_KEYS = ("auth_token_comfy_org", "api_key_comfy_org")
        original = sys.modules.get("execution")
        sys.modules["execution"] = fake
        try:
            self.assertTrue(runner._queue_wants_sensitive())
            queued, _, _, _ = self._queue_one()
            self.assertEqual(len(queued[0]), 6)
            self.assertEqual(queued[0][5], {}, "sensitive must be a dict the worker can iterate")
        finally:
            if original is None:
                sys.modules.pop("execution", None)
            else:
                sys.modules["execution"] = original

    def test_queue_tuple_falls_back_to_five_on_older_comfyui(self):
        """Older builds unpack five. Sending six there would be just as fatal
        in the other direction, so the arity is detected, never assumed."""
        import sys
        import types

        fake = types.ModuleType("execution")   # no SENSITIVE_EXTRA_DATA_KEYS
        original = sys.modules.get("execution")
        sys.modules["execution"] = fake
        try:
            self.assertFalse(runner._queue_wants_sensitive())
            queued, _, _, _ = self._queue_one()
            self.assertEqual(len(queued[0]), 5)
        finally:
            if original is None:
                sys.modules.pop("execution", None)
            else:
                sys.modules["execution"] = original

    def test_a_dead_queue_worker_fails_fast(self):
        """Otherwise the item sits in the queue, _in_queue keeps saying 'still
        there', and the run burns its whole timeout before saying nothing
        useful."""
        original = runner._worker_is_alive
        runner._worker_is_alive = lambda: False  # type: ignore[assignment]
        try:
            with self.assertRaises(runner.RunError) as ctx:
                self._queue_one()
            self.assertEqual(ctx.exception.code, "worker_dead")
        finally:
            runner._worker_is_alive = original  # type: ignore[assignment]

    def test_queue_refuses_a_graph_with_no_output(self):
        class FakeServer:
            number = 0
            prompt_queue = type("Q", (), {"put": lambda self, item: None})()

        orig = runner._prompt_server
        runner._prompt_server = lambda: FakeServer()  # type: ignore[assignment]
        try:
            with self.assertRaises(runner.RunError) as ctx:
                asyncio.run(runner.queue({"1": {"class_type": "KSampler", "inputs": {}}}, []))
            self.assertEqual(ctx.exception.code, "no_output")
        finally:
            runner._prompt_server = orig  # type: ignore[assignment]


class TestRunStore(BridgeTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.store = runner.RunStore()

    def test_unknown_run_is_rejected(self):
        with self.assertRaises(runner.RunError) as ctx:
            self.store.get("made-up")
        self.assertEqual(ctx.exception.code, "unknown_run")

    def test_run_id_is_opaque(self):
        run = self.store.create("qwen-image-edit", [])
        self.assertNotIn("qwen", run.id)
        self.assertGreaterEqual(len(run.id), 12)

    def test_concurrent_runs_are_capped(self):
        for _ in range(runner.MAX_ACTIVE_RUNS):
            self.store.create("t", [])
        with self.assertRaises(runner.RunError) as ctx:
            self.store.create("t", [])
        self.assertEqual(ctx.exception.code, "too_many_runs")

    def test_status_never_leaks_a_filesystem_path(self):
        run = self.store.create("t", [])
        run.state = "succeeded"
        run.files["ref1"] = {"filename": "C:\\ComfyUI\\output\\x.png", "subfolder": "s",
                             "type": "output", "kind": "image", "mime": "image/png",
                             "node": "60"}
        run.outputs = [{"ref": "ref1", "kind": "image", "mime": "image/png", "node": "60"}]
        payload = json.dumps(run.as_dict())
        self.assertNotIn("ComfyUI", payload)
        self.assertNotIn("x.png", payload)
        self.assertIn("ref1", payload)


class TestImageEditEndToEnd(TemplateTestCase, unittest.IsolatedAsyncioTestCase):
    """PUT the frame, POST the run, poll, read the result.

    The unit tests above each prove one piece. This is the only one that proves
    they fit: a slot that binds but is never queued, or a run that succeeds but
    whose output cannot be read, passes every test except this one.
    """

    def setUp(self) -> None:
        super().setUp()
        from reanimator import server

        self.server_module = server
        root = Path(self._tmp.name)
        self.inputs = root / "input"
        self.outputs = root / "output"
        self.inputs.mkdir()
        self.outputs.mkdir()
        self.queued: list[tuple] = []
        self.absent_models: set[str] = set()

        test = self

        class FakeFolderPaths:
            get_input_directory = staticmethod(lambda: str(test.inputs))
            get_directory_by_type = staticmethod(
                lambda kind: str(test.outputs) if kind in ("output", "temp") else None
            )
            # Every declared model present unless a test says otherwise. This
            # machine has no models at all, so without it the pre-flight would
            # correctly refuse every run and the end-to-end tests would be
            # testing the refusal path instead of the working one.
            get_full_path = staticmethod(
                lambda folder, name: None if name in test.absent_models
                else f"/fake/models/{folder}/{name}"
            )

        class FakeQueue:
            def __init__(self) -> None:
                self.histories: dict[str, dict] = {}

            def put(self, item):
                # Same contract check as TestRunnerTransport: a fake that
                # accepts any shape is how the five-element tuple reached a
                # real ComfyUI and killed its worker thread.
                expected = 6 if runner._queue_wants_sensitive() else 5
                assert len(item) == expected, (
                    f"prompt_worker indexes item[{expected - 1}]; got {len(item)}"
                )
                test.queued.append(item)
                # ComfyUI would run the graph here. Write what SaveImage would.
                (test.outputs / "ComfyUI_00001_.png").write_bytes(b"\x89PNG-result")
                self.histories[item[1]] = {
                    "outputs": {item[4][0]: {"images": [
                        {"filename": "ComfyUI_00001_.png", "subfolder": "", "type": "output"}
                    ]}},
                    "status": {"status_str": "success", "completed": True, "messages": []},
                }

            def get_history(self, prompt_id=None, **_):
                entry = self.histories.get(prompt_id)
                return {prompt_id: entry} if entry else {}

            def get_current_queue(self):
                return ([], [])

        class FakeServer:
            number = 0
            prompt_queue = FakeQueue()

        self._patches = []
        for module, name, value in (
            (runner, "_folder_paths", lambda: FakeFolderPaths),
            (runner, "_prompt_server", lambda: FakeServer),
            # Both installed templates: the security layer checks every node
            # class of whichever one the request names, so a fake built from
            # one template refuses the other for nodes it does have.
            (runner, "object_info", lambda refresh=False: with_checkpoints(
                merge_object_info(
                    fake_object_info(self.graph),
                    fake_object_info(templates.get("qwen-image-edit-2511-base").graph),
                ),
                templates.get("qwen-image-edit-2511-base"),
            )),
        ):
            self._patches.append((module, name, getattr(module, name)))
            setattr(module, name, value)

        self._orig_runs = runner.runs
        runner.runs = runner.RunStore()
        self.server_module._inputs.clear()
        self.token = self._issue_token()

    def tearDown(self) -> None:
        for module, name, original in self._patches:
            setattr(module, name, original)
        runner.runs = self._orig_runs
        self.server_module._inputs.clear()
        super().tearDown()

    def _issue_token(self) -> str:
        claims = self.claims()
        pairing.verify_assertion(make_jws(self.private, claims), self.nonces)
        approval = pairing.approvals.create(claims, "Test PC")
        pairing.approvals.resolve(approval.request_id, True)
        return pairing.tokens.issue(approval, "test browser").token

    async def client(self):
        from aiohttp.test_utils import TestClient, TestServer

        client = TestClient(TestServer(self.server_module.build_app()))
        await client.start_server()
        self.addAsyncCleanup(client.close)
        client.session.headers.update(
            {"Origin": config.ALLOWED_ORIGIN, "Authorization": f"Bearer {self.token}"}
        )
        return client

    async def test_a_frame_goes_in_and_an_image_comes_out(self):
        client = await self.client()

        listed = await (await client.get("/rb/v1/templates")).json()
        # By kind, not by position. This read templates[0] while there was only
        # one preset installed, and adding the first video preset made an image
        # test fail for a reason that had nothing to do with images.
        images = [t for t in listed["templates"] if t.get("kind") == "image"]
        template = next(t for t in images if t["id"] == "qwen-image-edit")

        response = await client.put(
            "/rb/v1/input/img1",
            data=tiny_png(),
            headers={"Content-Type": "image/png"},
        )
        self.assertEqual(response.status, 200)

        response = await client.post(
            "/rb/v1/run",
            json={
                "templateId": template["id"],
                "templateHash": template["hash"],
                "parameters": {"$PROMPT": "remove the UI text"},
                "inputIds": ["img1"],
            },
        )
        self.assertEqual(response.status, 202, await response.text())
        run = await response.json()

        for _ in range(200):
            status = await (await client.get(f"/rb/v1/run/{run['runId']}")).json()
            if status["state"] not in ("queued", "running"):
                break
            await asyncio.sleep(0.02)
        self.assertEqual(status["state"], "succeeded", status.get("error"))

        # The prompt and the frame really did reach the queue, bound by title.
        graph = self.queued[0][2]
        slots = binder.find_slots(self.graph, self.template.slots)
        self.assertEqual(
            graph[slots["$PROMPT"].node_id]["inputs"]["prompt"], "remove the UI text"
        )
        self.assertRegex(
            graph[slots["$IMAGE_1"].node_id]["inputs"]["image"], validate.INPUT_FILENAME_RE
        )

        ref = status["outputs"][0]["ref"]
        result = await client.get(f"/rb/v1/output/{run['runId']}/{ref}")
        self.assertEqual(result.status, 200)
        self.assertEqual(await result.read(), b"\x89PNG-result")

    async def test_the_uploaded_frame_is_deleted_when_the_run_ends(self):
        """Local mode promises the media stays on this machine -- it does not
        promise to quietly accumulate a copy of the footage in ComfyUI's
        input folder."""
        client = await self.client()
        await client.put(
            "/rb/v1/input/img1", data=tiny_png(), headers={"Content-Type": "image/png"}
        )
        self.assertEqual(len(list(self.inputs.glob("rb_*"))), 1)

        response = await client.post(
            "/rb/v1/run",
            json={"templateId": "qwen-image-edit", "parameters": {"$PROMPT": "x"},
                  "inputIds": ["img1"]},
        )
        run = await response.json()
        for _ in range(200):
            status = await (await client.get(f"/rb/v1/run/{run['runId']}")).json()
            if status["state"] not in ("queued", "running"):
                break
            await asyncio.sleep(0.02)
        self.assertEqual(status["state"], "succeeded")
        self.assertEqual(list(self.inputs.glob("rb_*")), [])

    async def test_a_seed_is_generated_per_run(self):
        """randomizePerRun in the manifest. A fixed seed would make 'generate
        again' -- the main way a user works around a bad result -- a no-op."""
        client = await self.client()
        seeds = set()
        for i in range(2):
            await client.put(
                f"/rb/v1/input/img{i}", data=tiny_png(),
                headers={"Content-Type": "image/png"},
            )
            await client.post(
                "/rb/v1/run",
                json={"templateId": "qwen-image-edit", "parameters": {"$PROMPT": "x"},
                      "inputIds": [f"img{i}"]},
            )
        slots = binder.find_slots(self.graph, self.template.slots)
        for item in self.queued:
            seeds.add(item[2][slots["$SEED"].node_id]["inputs"]["seed"])
        self.assertEqual(len(seeds), 2)

    async def test_the_cloud_cannot_send_a_graph(self):
        """The non-negotiable one (plan §5). ComfyUI custom nodes are arbitrary
        Python: a bridge that runs a pushed workflow is an RCE target for
        whoever compromises reanimator.app.

        A pushed graph is not rejected, it is *ignored* -- the run still uses
        the installed template. Asserting on the queued graph rather than on a
        4xx is deliberate: a body the bridge silently accepted and then executed
        would pass a status-code check just as happily.
        """
        client = await self.client()
        evil = {"1": {"class_type": "SaveImage", "inputs": {"filename_prefix": "pwned"}}}

        for key in ("prompt", "graph", "workflow", "nodes", "extra_data"):
            with self.subTest(key=key):
                response = await client.post(
                    "/rb/v1/run", json={"templateId": "qwen-image-edit", key: evil}
                )
                self.assertIn(response.status, (202, 400, 422))

        # No templateId now means "resolve one from the intent" rather than
        # "unknown template", so this is refused by validation instead of by
        # lookup. Either way the graph never reaches the queue, which is the
        # property under test.
        response = await client.post("/rb/v1/run", json={"prompt": evil})
        self.assertIn(response.status, (404, 422))

        self.assertNotIn(
            "pwned", json.dumps(self.queued), "a pushed graph reached the queue"
        )
        for item in self.queued:
            self.assertEqual(
                set(item[2]), set(self.graph), "the queued graph is not the template's"
            )

    async def test_a_parameter_cannot_rewire_the_graph(self):
        """Values only, never structure: a link written into a widget would
        detach a node and the workflow would still run, producing something the
        user never asked for."""
        client = await self.client()
        for value in ([["78", 0]], {"class_type": "Evil"}, ["78", 0]):
            with self.subTest(value=value):
                response = await client.post(
                    "/rb/v1/run",
                    json={"templateId": "qwen-image-edit", "parameters": {"$PROMPT": value}},
                )
                self.assertGreaterEqual(response.status, 400)
        self.assertEqual(self.queued, [])

    async def test_a_pinned_hash_that_does_not_match_is_refused(self):
        client = await self.client()
        response = await client.post(
            "/rb/v1/run",
            json={"templateId": "qwen-image-edit", "templateHash": "sha256:stale",
                  "parameters": {"$PROMPT": "x"}},
        )
        self.assertEqual(response.status, 404)
        self.assertEqual((await response.json())["error"]["code"], "template_mismatch")

    async def test_an_unregistered_input_id_cannot_name_a_file(self):
        client = await self.client()
        for input_id in ("../../../etc/passwd", "rb_deadbeef.png", "anything"):
            with self.subTest(input_id=input_id):
                response = await client.post(
                    "/rb/v1/run",
                    json={"templateId": "qwen-image-edit",
                          "parameters": {"$PROMPT": "x"}, "inputIds": [input_id]},
                )
                self.assertGreaterEqual(response.status, 400)
        self.assertEqual(self.queued, [])

    async def test_missing_model_is_reported_before_anything_is_queued(self):
        present = {
            v for node in self.graph.values() for v in node["inputs"].values()
            if isinstance(v, str)
        } - {"qwen_image_vae.safetensors"}
        runner.object_info = lambda refresh=False: fake_object_info(  # type: ignore[assignment]
            self.graph, files=present
        )
        client = await self.client()
        response = await client.post(
            "/rb/v1/run",
            json={"templateId": "qwen-image-edit", "parameters": {"$PROMPT": "x"}},
        )
        self.assertEqual(response.status, 422)
        body = await response.json()
        # The specific code, not a blanket one: "install this model" and "this
        # action needs an annotated frame" call for different reactions.
        self.assertEqual(body["error"]["code"], "missing_model")
        self.assertIn("qwen_image_vae.safetensors", body["error"]["message"])
        self.assertEqual(self.queued, [], "nothing may reach the GPU after a 422")

    async def test_a_keyframe_larger_than_aiohttps_default_still_uploads(self):
        """aiohttp caps a request body at 1 MB by default, and a 1024x1024
        keyframe goes past that. The 413 it raises comes from outside the
        handler, so it would also have escaped without CORS headers — reaching
        the editor as an unexplained "Failed to fetch"."""
        client = await self.client()
        response = await client.put(
            "/rb/v1/input/big",
            data=tiny_png(1400, 1400),
            headers={"Content-Type": "image/png"},
        )
        self.assertEqual(response.status, 200)

    async def test_an_error_from_outside_the_handler_still_carries_cors(self):
        # Shrunk so the test can overrun client_max_size without pushing 65 MB
        # through a socket. build_app() reads this, so it is patched before the
        # app is built.
        original = runner.MAX_INPUT_BYTES
        runner.MAX_INPUT_BYTES = 64 * 1024
        try:
            client = await self.client()
            response = await client.put(
                "/rb/v1/input/huge",
                data=b"x" * (2 * 1024 * 1024),
                headers={"Content-Type": "image/png"},
            )
        finally:
            runner.MAX_INPUT_BYTES = original
        self.assertGreaterEqual(response.status, 400)
        self.assertEqual(
            response.headers.get("Access-Control-Allow-Origin"), config.ALLOWED_ORIGIN
        )

    async def test_validating_does_not_consume_the_uploads(self):
        """The regression the end-to-end probe caught. /validate ran the same
        preparation as /run, which rewrote the uploaded frames as padded copies
        and deleted the originals -- so the /run that followed could not find
        the files validation had just declared fine."""
        client = await self.client()
        await client.put(
            "/rb/v1/input/img1", data=tiny_png(), headers={"Content-Type": "image/png"}
        )
        before = sorted(p.name for p in self.inputs.glob("rb_*"))

        response = await client.post(
            "/rb/v1/validate",
            json={"templateId": "qwen-image-edit", "parameters": {"$PROMPT": "x"},
                  "inputIds": ["img1"]},
        )
        self.assertEqual(response.status, 200)
        self.assertEqual(
            sorted(p.name for p in self.inputs.glob("rb_*")), before,
            "validating must leave the uploads exactly as it found them",
        )

        response = await client.post(
            "/rb/v1/run",
            json={"templateId": "qwen-image-edit", "parameters": {"$PROMPT": "x"},
                  "inputIds": ["img1"]},
        )
        self.assertEqual(response.status, 202, await response.text())

    async def test_a_canvas_the_user_typed_wrong_gets_a_422_not_a_500(self):
        """These sizes arrive from a form now, so the error is a message for a
        person -- not a Pillow traceback or a tensor complaint from ComfyUI."""
        client = await self.client()
        for size, code in (((10, 10), "source_too_small"),
                           ((4000, 400), "aspect_unsupported")):
            with self.subTest(size=size):
                for slot in ("a", "b"):
                    await client.put(
                        f"/rb/v1/input/{slot}{size[0]}",
                        data=tiny_png(*size), headers={"Content-Type": "image/png"},
                    )
                response = await client.post(
                    "/rb/v1/run",
                    json={
                        "templateId": "qwen-image-edit-2511-base",
                        "intent": "edit_pose", "quality": "preview",
                        "inputs": {"cleanFrame": f"a{size[0]}",
                                   "annotatedFrame": f"b{size[0]}"},
                        "parameters": {"instruction": "x"},
                    },
                )
                self.assertEqual(response.status, 422, await response.text())
                body = await response.json()
                self.assertEqual(body["error"]["code"], code)
                self.assertNotIn("Traceback", body["error"]["message"])
                self.assertIn(str(size[0]), body["error"]["message"])

    async def test_mismatched_frame_sizes_are_refused_before_generating(self):
        client = await self.client()
        await client.put("/rb/v1/input/c", data=tiny_png(848, 478),
                         headers={"Content-Type": "image/png"})
        await client.put("/rb/v1/input/a", data=tiny_png(640, 360),
                         headers={"Content-Type": "image/png"})
        response = await client.post(
            "/rb/v1/run",
            json={"templateId": "qwen-image-edit-2511-base", "intent": "edit_pose",
                  "quality": "preview",
                  "inputs": {"cleanFrame": "c", "annotatedFrame": "a"},
                  "parameters": {"instruction": "x"}},
        )
        self.assertEqual(response.status, 400)
        self.assertEqual((await response.json())["error"]["code"], "frame_size_mismatch")
        self.assertEqual(self.queued, [])

    async def test_more_context_frames_than_slots_is_refused_not_truncated(self):
        """This template has three image slots: the clean frame, the annotated
        frame and ONE context frame. A fourth image has nowhere to go.

        It used to be accepted with a 202 and no warning, because the binder
        zipped the given list against the slots and zip() stops at the shorter
        side. The caller was told the run was fine while the model never saw
        the image -- and then blamed the model for ignoring it. Silent
        truncation is the failure this refuses; the editor already declines to
        do it, and doing it here instead just hid it one layer further down.
        """
        client = await self.client()
        for name in ("c", "a", "r1", "r2"):
            await client.put(f"/rb/v1/input/{name}", data=tiny_png(848, 478),
                             headers={"Content-Type": "image/png"})

        response = await client.post(
            "/rb/v1/run",
            json={"templateId": "qwen-image-edit-2511-base", "intent": "edit_pose",
                  "quality": "preview",
                  "inputs": {"cleanFrame": "c", "annotatedFrame": "a",
                             "contextFrames": ["r1", "r2"]},
                  "parameters": {"instruction": "x"}},
        )
        self.assertEqual(response.status, 400)
        body = await response.json()
        self.assertEqual(body["error"]["code"], "too_many_inputs")
        # The message has to say how many fit, or the caller cannot fix it.
        self.assertIn("1", body["error"]["message"])
        self.assertIn("contextFrames", body["error"]["message"])
        self.assertEqual(self.queued, [], "nothing may reach the GPU after a refusal")

    async def test_a_second_context_frame_is_refused_even_without_an_annotated_frame(self):
        """The tempting inference is 'three image slots minus the two I am
        using leaves one free, so a second reference fits'. It does not.

        Slots are fixed per role: $IMAGE_2 is the annotated frame's, and when
        that frame is absent the manifest DETACHES it rather than handing it to
        somebody else. Proven against a real GPU before this was written: with
        no drawing, one reference and two references produced byte-identical
        output, so the second never reached the model.
        """
        client = await self.client()
        for name in ("c", "r1", "r2"):
            await client.put(f"/rb/v1/input/{name}", data=tiny_png(848, 478),
                             headers={"Content-Type": "image/png"})

        response = await client.post(
            "/rb/v1/run",
            json={"templateId": "qwen-image-edit-2511-base", "intent": "edit_keyframe",
                  "quality": "preview",
                  "inputs": {"cleanFrame": "c", "contextFrames": ["r1", "r2"]},
                  "parameters": {"instruction": "x"}},
        )
        self.assertEqual(response.status, 400)
        self.assertEqual((await response.json())["error"]["code"], "too_many_inputs")
        self.assertEqual(self.queued, [])

    async def test_the_listing_states_how_many_images_each_role_takes(self):
        """The editor must not have to infer the limit. It was computing
        images-minus-the-ones-in-use, which promises a second reference the
        model never receives."""
        client = await self.client()
        listed = await (await client.get("/rb/v1/templates")).json()
        by_id = {t["id"]: t for t in listed["templates"]}
        tpl = by_id["qwen-image-edit-2511-base"]
        self.assertEqual(tpl["roleMax"]["contextFrames"], 1)
        self.assertEqual(tpl["roleMax"]["cleanFrame"], 1)
        self.assertEqual(tpl["roleMax"]["annotatedFrame"], 1)
        # Three image slots in total, and still only one reference: the
        # difference between the two numbers is the whole point.
        self.assertEqual(tpl["images"], 3)

    def test_the_declared_max_is_honoured_not_just_the_slot_count(self):
        """`max` sits in the manifest next to the slots. Nothing read it, so a
        template could declare a limit and hand out more anyway — the same
        silence this refuses, one level up."""
        template = templates.get("qwen-image-edit-2511-base")
        self.assertEqual(template.role_max("contextFrames"), 1)
        self.assertEqual(template.role_max("cleanFrame"), 1)

        # A declaration cannot conjure a slot that does not exist...
        greedy = templates.Template(
            id="stub", graph={}, hash="sha256:x",
            manifest={"id": "stub", "roles": {"refs": {"slots": ["$A"], "max": 5}}},
        )
        self.assertEqual(greedy.role_max("refs"), 1)
        # ...and a slot cannot be used past a limit the template states.
        capped = templates.Template(
            id="stub", graph={}, hash="sha256:x",
            manifest={"id": "stub", "roles": {"refs": {"slots": ["$A", "$B"], "max": 1}}},
        )
        self.assertEqual(capped.role_max("refs"), 1)

    async def test_one_context_frame_still_binds_to_its_slot(self):
        """The refusal above must not become 'context frames are broken': the
        one that fits has to reach the model, in its own slot."""
        client = await self.client()
        for name in ("c", "a", "r1"):
            await client.put(f"/rb/v1/input/{name}", data=tiny_png(848, 478),
                             headers={"Content-Type": "image/png"})

        response = await client.post(
            "/rb/v1/run",
            json={"templateId": "qwen-image-edit-2511-base", "intent": "edit_pose",
                  "quality": "preview",
                  "inputs": {"cleanFrame": "c", "annotatedFrame": "a",
                             "contextFrames": ["r1"]},
                  "parameters": {"instruction": "x"}},
        )
        self.assertEqual(response.status, 202, await response.text())
        # La plantilla de esta clase es la de una imagen; el run pidió la de
        # tres, así que el slot se busca en ESA.
        tpl = templates.get("qwen-image-edit-2511-base")
        graph = self.queued[0][2]
        slots = binder.find_slots(tpl.graph, tpl.slots)
        bound = graph[slots["$IMAGE_3"].node_id]["inputs"]["image"]
        self.assertRegex(bound, validate.INPUT_FILENAME_RE)

    async def test_the_padded_copies_are_cleaned_up_not_the_originals(self):
        """The graph runs on the padded files, so those are what must be deleted
        afterwards. Listing the originals left the real files behind and tried
        to remove names that no longer existed."""
        client = await self.client()
        for slot in ("c", "a"):
            await client.put(f"/rb/v1/input/{slot}", data=tiny_png(848, 478),
                             headers={"Content-Type": "image/png"})
        response = await client.post(
            "/rb/v1/run",
            json={"templateId": "qwen-image-edit-2511-base", "intent": "edit_pose",
                  "quality": "preview",
                  "inputs": {"cleanFrame": "c", "annotatedFrame": "a"},
                  "parameters": {"instruction": "x"}},
        )
        self.assertEqual(response.status, 202, await response.text())
        run = await response.json()
        self.assertEqual(run["resolution"]["padding"],
                         {"left": 18, "right": 19, "top": 0, "bottom": 0})
        for _ in range(200):
            status = await (await client.get(f"/rb/v1/run/{run['runId']}")).json()
            if status["state"] not in ("queued", "running"):
                break
            await asyncio.sleep(0.02)
        self.assertEqual(status["state"], "succeeded", status.get("error"))
        self.assertEqual(list(self.inputs.glob("rb_*")), [])

    async def test_a_streamed_result_carries_cors_headers(self):
        """Only a browser catches this. prepare() writes the headers to the
        socket, so anything the middleware sets afterwards changes an object
        whose headers have already gone out. A Python client never notices; the
        browser refuses the response and reports "Failed to fetch" — making a
        generation that had already finished on the GPU look like the bridge
        being unreachable."""
        client = await self.client()
        await client.put("/rb/v1/input/img1", data=tiny_png(),
                         headers={"Content-Type": "image/png"})
        run = await (await client.post(
            "/rb/v1/run",
            json={"templateId": "qwen-image-edit", "parameters": {"$PROMPT": "x"},
                  "inputIds": ["img1"]},
        )).json()
        for _ in range(200):
            status = await (await client.get(f"/rb/v1/run/{run['runId']}")).json()
            if status["state"] not in ("queued", "running"):
                break
            await asyncio.sleep(0.02)
        self.assertEqual(status["state"], "succeeded")

        result = await client.get(
            f"/rb/v1/output/{run['runId']}/{status['outputs'][0]['ref']}"
        )
        self.assertEqual(result.status, 200)
        self.assertEqual(
            result.headers.get("Access-Control-Allow-Origin"), config.ALLOWED_ORIGIN,
            "a streamed result without CORS is invisible to the page that asked for it",
        )
        self.assertIn("Content-Length", result.headers.get("Access-Control-Expose-Headers", ""))

    async def test_an_unreadable_checkpoint_is_refused_before_anything_is_queued(self):
        """The whole point of screening: ComfyUI would find this out while
        loading 20 GB of weights and report a KeyError from UNETLoader. Here it
        costs one request and says which format and what to do."""
        client = await self.client()
        original = runner.quantization_formats
        runner.quantization_formats = lambda: {"nvfp4"}      # knows neither variant
        try:
            await client.put("/rb/v1/input/img1", data=tiny_png(),
                             headers={"Content-Type": "image/png"})
            response = await client.post("/rb/v1/run", json={
                "templateId": "qwen-image-edit-2511-base",
                "intent": "edit_keyframe",
                "inputs": {"cleanFrame": "img1"},
                "parameters": {"instruction": "x"},
            })
        finally:
            runner.quantization_formats = original

        self.assertEqual(response.status, 422)
        body = await response.json()
        self.assertEqual(body["error"]["code"], "incompatible_checkpoint")
        self.assertIn("int8_tensorwise", body["error"]["message"])
        self.assertIn("cannot read", body["error"]["message"])
        self.assertEqual(self.queued, [], "nothing may reach ComfyUI's queue")

    async def test_the_fallback_it_chose_is_visible_in_the_response(self):
        """Running something other than the manifest's first choice has to be
        legible from the outside, or nobody can tell why it got slower."""
        client = await self.client()
        original = runner.quantization_formats
        runner.quantization_formats = lambda: {"float8_e4m3fn", "float8_e5m2"}
        try:
            await client.put("/rb/v1/input/img1", data=tiny_png(),
                             headers={"Content-Type": "image/png"})
            response = await client.post("/rb/v1/run", json={
                "templateId": "qwen-image-edit-2511-base",
                "intent": "edit_keyframe",
                "inputs": {"cleanFrame": "img1"},
                "parameters": {"instruction": "x"},
            })
        finally:
            runner.quantization_formats = original

        self.assertEqual(response.status, 202, await response.text())
        checkpoint = (await response.json())["resolved"]["checkpoint"]
        self.assertEqual(checkpoint["variant"], "fp8mixed")
        self.assertEqual(checkpoint["selectedBy"], "alternative")
        self.assertTrue(checkpoint["compatible"])
        self.assertEqual(checkpoint["fallbackFrom"]["variant"], "int8_convrot")
        self.assertEqual(checkpoint["fallbackFrom"]["code"], "unsupported_quantization")
        self.assertIn("int8_tensorwise", checkpoint["fallbackFrom"]["missingOps"])

    async def test_generation_routes_reject_a_missing_token(self):
        client = await self.client()
        del client.session.headers["Authorization"]
        for method, path in (
            ("get", "/rb/v1/templates"),
            ("post", "/rb/v1/run"),
            ("post", "/rb/v1/validate"),
            ("get", "/rb/v1/run/anything"),
            ("get", "/rb/v1/output/anything/anything"),
        ):
            with self.subTest(path=path):
                response = await getattr(client, method)(path, json={})
                self.assertEqual(response.status, 401)

    async def test_an_output_ref_belongs_to_exactly_one_run(self):
        client = await self.client()
        await client.put(
            "/rb/v1/input/img1", data=tiny_png(), headers={"Content-Type": "image/png"}
        )
        run = await (await client.post(
            "/rb/v1/run",
            json={"templateId": "qwen-image-edit", "parameters": {"$PROMPT": "x"},
                  "inputIds": ["img1"]},
        )).json()
        for _ in range(200):
            status = await (await client.get(f"/rb/v1/run/{run['runId']}")).json()
            if status["state"] not in ("queued", "running"):
                break
            await asyncio.sleep(0.02)
        ref = status["outputs"][0]["ref"]

        other = runner.runs.create("qwen-image-edit", [])
        response = await client.get(f"/rb/v1/output/{other.id}/{ref}")
        self.assertEqual(response.status, 404)


class TestLocalProjectStorage(BridgeTestCase):
    """docs/contract-local-project-storage.md, the half the bridge owes.

    The editor already refuses to call anything saved until this side confirms
    bytes and hash. These tests are about the confirmation being worth
    something: a declared capability that was never tried, or a write that
    reports success for truncated bytes, would turn "saved" back into "sent".
    """

    PROJECT = "c3556d02-1111-2222-3333-444455556666"
    ASSET = "ho8el6cannotation01"

    def test_capability_reports_a_root_and_a_real_write(self):
        block = projects.capability()
        self.assertEqual(block["contract"], projects.CONTRACT)
        self.assertTrue(block["available"])
        self.assertTrue(block["writable"])
        self.assertTrue(block["root"])
        self.assertIsNone(block["reason"])
        self.assertGreater(block["freeBytes"], 0)

    def test_the_write_test_leaves_nothing_behind(self):
        projects.capability()
        projects.capability()
        leftovers = [p.name for p in projects.root().iterdir()]
        self.assertEqual(leftovers, [], "the probe file must not survive itself")

    def test_a_root_that_cannot_exist_is_reported_before_any_user_write(self):
        """A disk problem has to appear in the capability the user reads, not
        in the first save that loses their drawing."""
        blocker = Path(self._tmp.name) / "not-a-folder"
        blocker.write_text("I am a file")
        config.update(local_projects_root=str(blocker))
        block = projects.capability()
        self.assertFalse(block["available"])
        self.assertFalse(block["writable"])
        self.assertIsNotNone(block["reason"])

    def test_writable_comes_from_writing_not_from_the_folder_existing(self):
        """The whole point of the probe. A folder that is there, and looks
        perfectly fine, and refuses the write -- a full disk, a locked profile,
        a syncing client -- must come back writable: false. Checking that the
        directory exists would pass this and lose the user's next drawing."""
        original = projects.secrets.token_hex
        projects.secrets.token_hex = lambda _n=6: "deadbeef"
        try:
            # Something the probe cannot overwrite, sitting on its exact name.
            (projects.root() / ".writetest-deadbeef").mkdir(parents=True)
            block = projects.capability()
        finally:
            projects.secrets.token_hex = original
        self.assertTrue(block["available"], "the folder is there and readable")
        self.assertFalse(block["writable"])
        self.assertIn("not writable", block["reason"])

    def test_capabilities_endpoint_declares_the_block(self):
        from reanimator import capabilities as caps

        block = caps.capabilities()["localProjectStorage"]
        self.assertEqual(block["contract"], projects.CONTRACT)
        self.assertTrue(block["writable"])

    def test_capabilities_survive_a_broken_storage_subsystem(self):
        """BRIDGE OFFLINE and STORAGE NOT READY are fixed in completely
        different ways, so a storage fault must never be reported as the
        device being unreachable."""
        from reanimator import capabilities as caps

        original = projects.capability
        projects.capability = lambda: (_ for _ in ()).throw(RuntimeError("disk on fire"))
        try:
            block = caps.capabilities()["localProjectStorage"]
        finally:
            projects.capability = original
        self.assertFalse(block["available"])
        self.assertIn("disk on fire", block["reason"])

    def test_a_write_round_trips_and_reports_its_hash(self):
        data = tiny_png()
        digest = hashlib.sha256(data).hexdigest()
        stored = projects.write(self.PROJECT, self.ASSET, "image/png", data, digest)
        self.assertEqual(stored["sha256"], digest)
        self.assertEqual(stored["bytes"], len(data))
        self.assertEqual(stored["ref"], f"{self.PROJECT}/{self.ASSET}.png")
        self.assertEqual(projects.find_asset(self.PROJECT, self.ASSET).read_bytes(), data)

    def test_a_reference_never_carries_an_absolute_path(self):
        data = tiny_png()
        stored = projects.write(
            self.PROJECT, self.ASSET, "image/png", data,
            hashlib.sha256(data).hexdigest(),
        )
        self.assertNotIn(str(projects.root()), json.dumps(stored))
        self.assertFalse(Path(stored["ref"]).is_absolute())

    def test_truncated_bytes_are_refused_not_stored(self):
        data = tiny_png()
        promised = hashlib.sha256(data).hexdigest()
        with self.assertRaises(projects.StorageError) as ctx:
            projects.write(self.PROJECT, self.ASSET, "image/png", data[:-5], promised)
        self.assertEqual(ctx.exception.status, 409)
        self.assertIsNone(
            projects.find_asset(self.PROJECT, self.ASSET),
            "a rejected write must leave nothing readable behind",
        )

    def test_a_failed_write_leaves_no_partial_file(self):
        with self.assertRaises(projects.StorageError):
            projects.write(self.PROJECT, self.ASSET, "image/png", b"x", "deadbeef")
        directory = projects.root() / self.PROJECT
        if directory.exists():
            self.assertEqual(list(directory.iterdir()), [])

    def test_rewriting_the_same_bytes_is_a_no_op(self):
        data = tiny_png()
        digest = hashlib.sha256(data).hexdigest()
        first = projects.write(self.PROJECT, self.ASSET, "image/png", data, digest)
        second = projects.write(self.PROJECT, self.ASSET, "image/png", data, digest)
        self.assertEqual(first, second)

    def test_an_id_can_never_name_a_path(self):
        data = tiny_png()
        digest = hashlib.sha256(data).hexdigest()
        for project_id, asset_id in (
            ("..", "goodassetid1"),
            ("../../etc", "goodassetid1"),
            (self.PROJECT, "../escape"),
            (self.PROJECT, "a"),                       # too short
            (self.PROJECT, "has.a.dot.png"),
            (self.PROJECT, "C:\\Windows\\win.ini"),
        ):
            with self.subTest(project=project_id, asset=asset_id):
                with self.assertRaises(projects.StorageError) as ctx:
                    projects.write(project_id, asset_id, "image/png", data, digest)
                self.assertEqual(ctx.exception.status, 422)

    def test_the_extension_comes_from_the_content_type_only(self):
        data = tiny_png()
        digest = hashlib.sha256(data).hexdigest()
        stored = projects.write(self.PROJECT, self.ASSET, "image/png", data, digest)
        self.assertTrue(stored["ref"].endswith(".png"))
        with self.assertRaises(projects.StorageError) as ctx:
            projects.write(self.PROJECT, "scriptasset01", "text/x-python", b"x", None)
        self.assertEqual(ctx.exception.status, 415)

    def test_verify_tells_present_from_correct(self):
        data = tiny_png()
        digest = hashlib.sha256(data).hexdigest()
        projects.write(self.PROJECT, self.ASSET, "image/png", data, digest)
        results = projects.verify(self.PROJECT, [
            {"assetId": self.ASSET, "sha256": digest},
            {"assetId": self.ASSET, "sha256": "0" * 64},
            {"assetId": "missingasset1", "sha256": digest},
        ])
        self.assertEqual([r["present"] for r in results], [True, True, False])
        self.assertEqual([r["sha256Matches"] for r in results], [True, False, False])

    def test_deleting_is_explicit_and_scoped(self):
        data = tiny_png()
        projects.write(self.PROJECT, self.ASSET, "image/png", data,
                       hashlib.sha256(data).hexdigest())
        self.assertTrue(projects.delete(self.PROJECT, self.ASSET))
        self.assertIsNone(projects.find_asset(self.PROJECT, self.ASSET))
        self.assertFalse(projects.delete(self.PROJECT, self.ASSET))

    def test_one_project_cannot_read_anothers_assets(self):
        data = tiny_png()
        projects.write(self.PROJECT, self.ASSET, "image/png", data,
                       hashlib.sha256(data).hexdigest())
        other = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        self.assertIsNone(projects.find_asset(other, self.ASSET))


class TestOrphanCollectorGroundwork(BridgeTestCase):
    """What the collector needs, and the reason it is about FILES.

    One asset id can have two files on disk -- write() picks the extension from
    the content type, so the same id written twice as different types leaves
    both. An inventory keyed by asset id would report one live thing where
    there are two, and a delete keyed by asset id would take whichever
    extension came first. The collector's whole promise is that it may leave
    garbage but must never remove something still needed, so both halves here
    speak in file names.
    """

    PROJECT = "c3556d02-1111-2222-3333-444455556666"
    ASSET = "ho8el6cannotation01"

    def _write(self, asset_id, mime="image/png"):
        data = tiny_png() if mime == "image/png" else b"RIFF0000WEBPVP8 "
        return projects.write(self.PROJECT, asset_id, mime, data,
                              hashlib.sha256(data).hexdigest())

    def test_the_inventory_is_one_entry_per_file(self):
        self._write(self.ASSET, "image/png")
        self._write(self.ASSET, "image/webp")          # mismo id, otro fichero
        items = projects.list_files(self.PROJECT)
        self.assertEqual([i["filename"] for i in items],
                         [f"{self.ASSET}.png", f"{self.ASSET}.webp"])
        self.assertEqual({i["assetId"] for i in items}, {self.ASSET})
        self.assertEqual([i["mime"] for i in items], ["image/png", "image/webp"])
        self.assertTrue(all(i["bytes"] > 0 for i in items))

    def test_what_is_not_ours_is_not_listed(self):
        self._write(self.ASSET)
        directory = projects.project_dir(self.PROJECT, create=True)
        for name in ("notes.txt", ".hidden.png", f".{self.ASSET}.abcd1234.part",
                     "short.png", "sub"):
            if name == "sub":
                (directory / name).mkdir()
            else:
                (directory / name).write_bytes(b"x")
        names = [i["filename"] for i in projects.list_files(self.PROJECT)]
        self.assertEqual(names, [f"{self.ASSET}.png"],
                         "only <known id><known ext> files are ours to describe")

    def test_an_unknown_project_lists_nothing_instead_of_failing(self):
        self.assertEqual(projects.list_files("ffffffff-0000-1111-2222-333344445555"), [])
        with self.assertRaises(projects.StorageError) as ctx:
            projects.list_files("../etc")
        self.assertEqual(ctx.exception.status, 422)

    def test_deleting_names_the_file_it_means(self):
        self._write(self.ASSET, "image/png")
        self._write(self.ASSET, "image/webp")
        self.assertTrue(projects.delete_file(self.PROJECT, f"{self.ASSET}.webp"))
        left = [i["filename"] for i in projects.list_files(self.PROJECT)]
        self.assertEqual(left, [f"{self.ASSET}.png"],
                         "the other extension must survive untouched")
        # Segunda vez: ya no está, y eso no es un error.
        self.assertFalse(projects.delete_file(self.PROJECT, f"{self.ASSET}.webp"))

    def test_a_name_that_is_not_a_project_file_is_refused(self):
        self._write(self.ASSET)
        directory = projects.project_dir(self.PROJECT, create=True)
        (directory / "notes.txt").write_bytes(b"keep me")
        for name in ("notes.txt", "../../etc/passwd", f"../{self.PROJECT}/x.png",
                     "sh.png", f"{self.ASSET}.py", f"{self.ASSET}", "",
                     f"{self.ASSET}.png.bak"):
            with self.subTest(name=name):
                with self.assertRaises(projects.StorageError) as ctx:
                    projects.delete_file(self.PROJECT, name)
                self.assertEqual(ctx.exception.status, 422)
        self.assertTrue((directory / "notes.txt").is_file())
        self.assertTrue((directory / f"{self.ASSET}.png").is_file())

    def test_one_project_cannot_delete_anothers_files(self):
        self._write(self.ASSET)
        other = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        self.assertFalse(projects.delete_file(other, f"{self.ASSET}.png"))
        self.assertIsNotNone(projects.find_asset(self.PROJECT, self.ASSET))


class TestLocalProjectStorageOverHttp(BridgeTestCase, unittest.IsolatedAsyncioTestCase):
    """The routes, the token gate and the CORS the browser actually needs."""

    PROJECT = "c3556d02-1111-2222-3333-444455556666"
    ASSET = "ho8el6cannotation01"

    def setUp(self) -> None:
        super().setUp()
        from reanimator import server

        self.server_module = server
        claims = self.claims()
        pairing.verify_assertion(make_jws(self.private, claims), self.nonces)
        approval = pairing.approvals.create(claims, "Test PC")
        pairing.approvals.resolve(approval.request_id, True)
        self.token = pairing.tokens.issue(approval, "test browser").token

    async def client(self):
        from aiohttp.test_utils import TestClient, TestServer

        client = TestClient(TestServer(self.server_module.build_app()))
        await client.start_server()
        self.addAsyncCleanup(client.close)
        client.session.headers.update(
            {"Origin": config.ALLOWED_ORIGIN, "Authorization": f"Bearer {self.token}"}
        )
        return client

    def _path(self, asset_id: str | None = None) -> str:
        return f"/rb/v1/projects/{self.PROJECT}/assets/{asset_id or self.ASSET}"

    async def test_put_then_get_returns_the_same_bytes(self):
        client = await self.client()
        data = tiny_png()
        digest = hashlib.sha256(data).hexdigest()

        response = await client.put(
            self._path(), data=data,
            headers={"Content-Type": "image/png", "X-Sha256": digest},
        )
        self.assertEqual(response.status, 200, await response.text())
        body = await response.json()
        self.assertEqual(body["sha256"], digest)

        read = await client.get(self._path())
        self.assertEqual(read.status, 200)
        self.assertEqual(await read.read(), data)
        self.assertEqual(read.headers["Content-Type"], "image/png")

    async def test_a_mismatched_hash_is_a_409(self):
        client = await self.client()
        response = await client.put(
            self._path(), data=tiny_png(),
            headers={"Content-Type": "image/png", "X-Sha256": "0" * 64},
        )
        self.assertEqual(response.status, 409)
        self.assertEqual((await response.json())["error"]["code"], "sha256_mismatch")
        self.assertEqual((await client.get(self._path())).status, 404)

    async def test_a_missing_asset_is_404_not_an_empty_200(self):
        client = await self.client()
        self.assertEqual((await client.get(self._path("neverwrittenid"))).status, 404)

    async def test_verify_answers_for_the_whole_project_at_once(self):
        client = await self.client()
        data = tiny_png()
        digest = hashlib.sha256(data).hexdigest()
        await client.put(self._path(), data=data,
                         headers={"Content-Type": "image/png", "X-Sha256": digest})

        response = await client.post(
            f"/rb/v1/projects/{self.PROJECT}/assets/verify",
            json={"refs": [{"assetId": self.ASSET, "sha256": digest},
                           {"assetId": "goneassetid1", "sha256": digest}]},
        )
        self.assertEqual(response.status, 200, await response.text())
        results = (await response.json())["results"]
        self.assertTrue(results[0]["present"] and results[0]["sha256Matches"])
        self.assertFalse(results[1]["present"])

    async def test_the_inventory_and_the_per_file_delete_over_http(self):
        client = await self.client()
        png, webp = tiny_png(), b"RIFF0000WEBPVP8 "
        for data, mime in ((png, "image/png"), (webp, "image/webp")):
            await client.put(self._path(), data=data, headers={
                "Content-Type": mime, "X-Sha256": hashlib.sha256(data).hexdigest()})

        listed = await client.get(f"/rb/v1/projects/{self.PROJECT}/files")
        self.assertEqual(listed.status, 200, await listed.text())
        items = (await listed.json())["items"]
        self.assertEqual([i["filename"] for i in items],
                         [f"{self.ASSET}.png", f"{self.ASSET}.webp"])

        gone = await client.delete(
            f"/rb/v1/projects/{self.PROJECT}/files/{self.ASSET}.webp")
        self.assertEqual(gone.status, 204)
        left = (await (await client.get(
            f"/rb/v1/projects/{self.PROJECT}/files")).json())["items"]
        self.assertEqual([i["filename"] for i in left], [f"{self.ASSET}.png"])
        # Y el que quedó se sigue pudiendo leer: no se ha tocado.
        self.assertEqual((await client.get(self._path())).status, 200)

    async def test_a_file_name_that_is_not_ours_is_refused_over_http(self):
        client = await self.client()
        response = await client.delete(
            f"/rb/v1/projects/{self.PROJECT}/files/notes.txt")
        self.assertEqual(response.status, 422)
        self.assertEqual((await response.json())["error"]["code"], "bad_file_name")

    async def test_the_inventory_needs_the_token_like_everything_else(self):
        client = await self.client()
        client.session.headers.pop("Authorization")
        self.assertEqual(
            (await client.get(f"/rb/v1/projects/{self.PROJECT}/files")).status, 401)

    async def test_verify_is_not_swallowed_by_the_asset_id_route(self):
        """'verify' sits where an asset id would. It must reach its own
        handler, not be stored as a file called verify."""
        client = await self.client()
        response = await client.post(
            f"/rb/v1/projects/{self.PROJECT}/assets/verify", json={"refs": []}
        )
        self.assertEqual(response.status, 200)
        self.assertEqual((await response.json())["results"], [])

    async def test_delete_is_204_and_removes_the_file(self):
        client = await self.client()
        data = tiny_png()
        await client.put(self._path(), data=data, headers={
            "Content-Type": "image/png", "X-Sha256": hashlib.sha256(data).hexdigest()})
        self.assertEqual((await client.delete(self._path())).status, 204)
        self.assertEqual((await client.get(self._path())).status, 404)

    async def test_storage_routes_reject_a_missing_token(self):
        client = await self.client()
        del client.session.headers["Authorization"]
        for method, path in (
            ("put", self._path()),
            ("get", self._path()),
            ("delete", self._path()),
            ("post", f"/rb/v1/projects/{self.PROJECT}/assets/verify"),
        ):
            with self.subTest(path=f"{method} {path}"):
                response = await getattr(client, method)(path)
                self.assertEqual(response.status, 401)

    async def test_the_preflight_allows_the_hash_header_and_delete(self):
        """Without both of these the browser never sends the request at all,
        and the page sees an opaque failure instead of a verified write."""
        client = await self.client()
        response = await client.options(
            self._path(), headers={"Origin": config.ALLOWED_ORIGIN}
        )
        self.assertEqual(response.status, 204)
        self.assertIn("X-Sha256", response.headers["Access-Control-Allow-Headers"])
        self.assertIn("DELETE", response.headers["Access-Control-Allow-Methods"])
        self.assertIn("PUT", response.headers["Access-Control-Allow-Methods"])

    async def test_a_read_carries_cors_so_the_canvas_is_not_tainted(self):
        client = await self.client()
        data = tiny_png()
        await client.put(self._path(), data=data, headers={
            "Content-Type": "image/png", "X-Sha256": hashlib.sha256(data).hexdigest()})
        read = await client.get(self._path())
        self.assertEqual(
            read.headers.get("Access-Control-Allow-Origin"), config.ALLOWED_ORIGIN
        )

    async def test_an_invalid_id_is_422_and_touches_no_disk(self):
        client = await self.client()
        response = await client.put(
            "/rb/v1/projects/short/assets/alsoshort", data=b"x",
            headers={"Content-Type": "image/png"},
        )
        self.assertEqual(response.status, 422)
        self.assertFalse((projects.root() / "short").exists())


class TestAppBuilds(BridgeTestCase):
    """Route registration only fails at startup, which is far too late.

    A duplicate route (e.g. declaring @routes.head next to @routes.get, which
    aiohttp already adds for you) raises RuntimeError inside add_routes and the
    whole bridge silently never starts.
    """

    def test_app_builds_without_route_conflicts(self):
        from reanimator import server

        app = server.build_app()
        paths = {
            r.resource.canonical
            for r in app.router.routes()
            if r.resource is not None
        }
        for expected in (
            "/rb/v1/hello",
            "/rb/v1/pair",
            "/rb/v1/capabilities",
            "/rb/v1/media/{capability_id}",
            "/rb/v1/templates",
            "/rb/v1/validate",
            "/rb/v1/input/{input_id}",
            "/rb/v1/run",
            "/rb/v1/run/{run_id}",
            "/rb/v1/run/{run_id}/interrupt",
            "/rb/v1/output/{run_id}/{ref}",
            "/rb/v1/projects/{project_id}/assets/{asset_id}",
            "/rb/v1/projects/{project_id}/assets/verify",
        ):
            self.assertIn(expected, paths)

    def test_project_asset_route_takes_put_get_and_delete(self):
        from reanimator import server

        app = server.build_app()
        methods = {
            r.method
            for r in app.router.routes()
            if r.resource is not None
            and r.resource.canonical == "/rb/v1/projects/{project_id}/assets/{asset_id}"
        }
        self.assertEqual(methods, {"PUT", "GET", "HEAD", "DELETE"})

    def test_input_route_takes_put_and_delete(self):
        """Two methods on one path is fine. The startup-killing duplicate is
        specifically @routes.head next to @routes.get, because aiohttp already
        registers HEAD for every GET."""
        from reanimator import server

        app = server.build_app()
        methods = {
            r.method
            for r in app.router.routes()
            if r.resource is not None and r.resource.canonical == "/rb/v1/input/{input_id}"
        }
        self.assertEqual(methods, {"PUT", "DELETE"})

    def test_generation_routes_require_a_token(self):
        """None of them may sit in the public set, and none may be swept up by
        the /rb/v1/pair/ or /rb/v1/media/ prefix exemptions in auth_middleware."""
        from reanimator import server

        for path in (
            "/rb/v1/templates", "/rb/v1/validate", "/rb/v1/run",
            "/rb/v1/run/abc", "/rb/v1/input/abc", "/rb/v1/output/abc/def",
        ):
            with self.subTest(path=path):
                self.assertNotIn(path, server.PUBLIC_ROUTES)
                self.assertFalse(path.startswith("/rb/v1/pair/"))
                self.assertFalse(path.startswith("/rb/v1/media/"))

    def test_cors_middleware_is_outermost(self):
        """aiohttp applies middlewares in reverse, so the first entry wraps the
        rest. If error_middleware wrapped cors_middleware instead, every error
        response would lose its CORS headers and reach the browser as an opaque
        "Failed to fetch" with the status and body stripped."""
        from reanimator import server

        app = server.build_app()
        names = [m.__name__ for m in app.middlewares]
        self.assertEqual(names[0], "cors_middleware")
        self.assertLess(names.index("cors_middleware"), names.index("error_middleware"))

    def test_rejected_origin_is_readable_by_javascript(self):
        """A 403 without CORS headers reaches the page as "Failed to fetch",
        which is indistinguishable from four other failures and sends users off
        debugging the wrong thing."""
        from reanimator import server

        response = server._rejected_origin("https://evil.example")
        self.assertEqual(response.status, 403)
        self.assertEqual(
            response.headers["Access-Control-Allow-Origin"], "https://evil.example"
        )

    def test_media_route_answers_head(self):
        from reanimator import server

        app = server.build_app()
        methods = {
            r.method
            for r in app.router.routes()
            if r.resource is not None
            and r.resource.canonical == "/rb/v1/media/{capability_id}"
        }
        self.assertIn("GET", methods)
        self.assertIn("HEAD", methods)  # aiohttp adds it for us


if __name__ == "__main__":
    unittest.main(verbosity=2)
