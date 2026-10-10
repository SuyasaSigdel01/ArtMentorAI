"""ArtMentorAI — an AI painting tutor.

Package layout follows one hard rule: everything under `artmentor.core` is pure
Python with no web framework, no database, and no HTTP. It must be runnable from
a plain script with no server running. If anything in an API layer ever needs to
`import cv2` directly, a boundary has been broken.
"""

__version__ = "0.1.0"
