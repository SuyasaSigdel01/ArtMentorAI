"""HTTP layer. Thin on purpose.

The one architectural rule this project has: `artmentor.core` is pure Python
with no framework in it. This package is the other side of that line — it parses
requests, calls into core, and serializes what comes back. It must never contain
image processing or color science of its own.

A concrete test of whether the boundary still holds: nothing under `api/` should
ever `import cv2`, `import numpy` for real work, or compute a delta-E. If it
does, logic has leaked out of the part that is unit-tested without a server.
"""