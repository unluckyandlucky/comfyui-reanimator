"""Template loading, slot binding and pre-flight validation.

The cloud sends a template id and validated parameters. It never sends a graph:
ComfyUI custom nodes are arbitrary Python, so a bridge that executes what it is
told turns every paired machine into an RCE target the moment reanimator.app is
compromised (docs/plan-local-bridge.md D5/D6, §5).
"""
