"""Everything that touches ComfyUI itself.

Nothing in here opens a socket to ComfyUI. Its port is assigned per instance by
the Desktop hub -- it was 8000 on the development machine, not 8188 -- so any
code that guesses a port works until it does not, on somebody else's install,
with no useful error. See docs/local-bridge-handoff.md §3.
"""
