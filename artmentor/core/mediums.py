"""What changes when the paint changes, and when the painter gets better.

Feature 3 of the original brief: skill levels and medium-specific guidance.
The temptation with a feature like this is to write six paragraphs of advice
per medium and call it done. That is a blog post, not a product — nothing in
the software would behave differently, and a learner painting in watercolour
would still be told to add titanium white to a mix.

So everything here is a constraint the rest of the pipeline obeys:

* **Which paints are available.** Watercolour has no white in the mix — the
  white is the paper. Asking for one part titanium white is not a near miss,
  it is an instruction that cannot be carried out.
* **How a mix is worded.** "Two parts ultramarine, one part white" in acrylic
  is "ultramarine, thinned to about a third strength" in watercolour. Same
  target colour, different physical act.
* **How many pigments to combine.** Three-pigment mixes stay clean in oil,
  where you have hours to adjust. In watercolour they go to mud on the paper.
* **How fine the pattern may be.** A shape a brush cannot hold an edge on is
  not a shape worth numbering.
* **How many colours to start with.** A first-timer given twenty-two numbered
  regions puts the brush down.

Two things are deliberately approximate and marked as such. `dry_shift_L` is
the value change on drying — real, well known to painters, and not measured by
us; it is an estimate from the literature and from how these paints are
generally described, in the same category as the eyeballed paint catalogue in
`paints.py`. And `target_delta_e` per level is a judgement call about what
"good enough to move on" means, not a discovered constant.

What this does not do
---------------------
It does not grade technique, and it does not claim a learner is at a level.
`should_advance` answers a narrow question — has this person been hitting the
fidelity their current level aims at — and the answer is always a suggestion
the learner can ignore.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median

__all__ = [
    "LEVELS",
    "MEDIUMS",
    "Guidance",
    "Level",
    "Medium",
    "describe_mix",
    "describe_value",
    "get_level",
    "get_medium",
    "guidance_for",
    "next_level",
    "should_advance",
]


# --------------------------------------------------------------------------
# Mediums
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Medium:
    """A painting medium, described by the things that change the maths."""

    id: str
    name: str

    #: How a colour is made lighter. "white" means a white pigment goes into
    #: the mix; "water" means dilution and the paper doing the work; "pressure"
    #: means less of it on the surface.
    lightens_with: str

    #: Whether a layer hides what is under it. Drives whether a learner can
    #: correct a mistake by painting over it.
    opaque: bool

    #: Approximate L* change between wet and dry. Negative is darker. These are
    #: estimates, not measurements — see the module docstring.
    dry_shift_L: float

    #: How many pigments combine before a mix turns muddy in this medium.
    max_components: int

    #: Smallest region worth numbering, in pixels at the working resolution.
    #: A wet brush cannot hold an edge the way a pencil can.
    min_region_area: int

    #: Sensible number of colours for a pattern before the level adjusts it.
    base_colors: int

    #: Can a dried passage be reworked or lifted?
    reworkable: bool

    #: One sentence a learner acts on, specific to this medium.
    note: str

    #: No hue at all. A pattern for this medium is a map of values, and naming
    #: a pigment in its recipes would be meaningless — graphite is graphite.
    monochrome: bool = False

    #: The lightest and darkest L* this medium can actually reach on paper.
    #: Graphite is reflective and tops out well short of black, which is why a
    #: beginner's "darkest" is usually nowhere near dark enough.
    value_floor: float = 0.0

    @property
    def mixes_white(self) -> bool:
        return self.lightens_with == "white"

    @property
    def dries(self) -> str:
        if self.dry_shift_L <= -1.0:
            return "darker"
        if self.dry_shift_L >= 1.0:
            return "lighter"
        return "much the same"


MEDIUMS: tuple[Medium, ...] = (
    Medium(
        id="acrylic",
        name="Acrylic",
        lightens_with="white",
        opaque=True,
        # The binder is milky white when wet and clear when dry, so the colour
        # you mixed is not quite the colour that ends up on the canvas.
        dry_shift_L=-2.5,
        max_components=3,
        min_region_area=90,
        base_colors=12,
        reworkable=True,
        note=(
            "Acrylic dries a shade darker than it looks wet, and it dries fast. "
            "Mix a little lighter than you think, and keep a wet palette."
        ),
    ),
    Medium(
        id="oil",
        name="Oil",
        lightens_with="white",
        opaque=True,
        dry_shift_L=-0.5,
        # Hours of working time means a third pigment can still be pulled back.
        max_components=3,
        min_region_area=110,
        base_colors=12,
        reworkable=True,
        note=(
            "Oil stays open for hours, so blend on the canvas rather than on the "
            "palette. Keep later layers oilier than earlier ones or they will crack."
        ),
    ),
    Medium(
        id="watercolour",
        name="Watercolour",
        lightens_with="water",
        opaque=False,
        # Water leaves, pigment settles, and the paper shows through more.
        dry_shift_L=+6.0,
        # Three pigments on wet paper is how mud is made.
        max_components=2,
        min_region_area=260,
        base_colors=8,
        reworkable=False,
        note=(
            "Your white is the paper, so plan what you will leave untouched before "
            "you start. Everything dries lighter than it looks wet."
        ),
    ),
    Medium(
        id="gouache",
        name="Gouache",
        lightens_with="white",
        opaque=True,
        # Light mixes dry down and dark ones dry up; the average is close to nil
        # but the variance is what catches people out.
        dry_shift_L=-1.5,
        max_components=2,
        min_region_area=170,
        base_colors=10,
        reworkable=True,
        note=(
            "Gouache shifts as it dries — pale mixes dry darker, dark ones dry "
            "lighter. Test a stroke on scrap and let it dry before committing."
        ),
    ),
    Medium(
        id="pencil",
        name="Graphite pencil",
        lightens_with="pressure",
        opaque=False,
        dry_shift_L=0.0,
        # Nothing is mixed at all; a "recipe" here is a value, not a blend.
        max_components=1,
        min_region_area=45,
        base_colors=6,
        reworkable=True,
        note=(
            "There is no colour to mix here, only value. Graphite cannot reach a "
            "true black, so save your darkest pressure for the very darkest shape."
        ),
        monochrome=True,
        value_floor=22.0,
    ),
    Medium(
        id="crayon",
        name="Coloured pencil or crayon",
        lightens_with="pressure",
        opaque=False,
        dry_shift_L=0.0,
        # Layering is optical, not physical: the eye mixes, the pigments do not.
        max_components=2,
        min_region_area=70,
        base_colors=10,
        reworkable=False,
        note=(
            "Colours here are layered, not mixed — lay one lightly over another and "
            "let your eye do the blending. Press hard too early and nothing more "
            "will stick."
        ),
    ),
)

_MEDIUM_BY_ID = {m.id: m for m in MEDIUMS}


# --------------------------------------------------------------------------
# Skill levels
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Level:
    """How much to put in front of someone, and what counts as on target."""

    id: str
    name: str

    #: Multiplier on the medium's colour count. A beginner gets fewer shapes.
    colors_scale: float

    #: Ceiling on pigments per mix, applied on top of the medium's own limit.
    max_components: int

    #: Average Lab distance this level is aiming to come within. Not a
    #: standard — a judgement about what is a fair target at each stage.
    target_delta_e: float

    #: What moving up actually asks of them.
    next_step: str | None


LEVELS: tuple[Level, ...] = (
    Level(
        id="beginner",
        name="Just starting",
        colors_scale=0.6,
        # Two pigments plus white is already a lot to hold in your head.
        max_components=2,
        target_delta_e=16.0,
        next_step="Work from a fuller palette and let the mixes get to three pigments.",
    ),
    Level(
        id="improving",
        name="Getting the hang of it",
        colors_scale=1.0,
        max_components=3,
        target_delta_e=11.0,
        next_step="Try the same subject without the numbered pattern, matching by eye.",
    ),
    Level(
        id="confident",
        name="Comfortable",
        colors_scale=1.5,
        max_components=3,
        target_delta_e=8.0,
        next_step=None,
    ),
)

_LEVEL_BY_ID = {level.id: level for level in LEVELS}

#: The order used when suggesting a move up.
_LADDER = [level.id for level in LEVELS]


def get_medium(medium_id: str) -> Medium:
    try:
        return _MEDIUM_BY_ID[medium_id]
    except KeyError:
        raise KeyError(
            f"unknown medium {medium_id!r}; known: {', '.join(sorted(_MEDIUM_BY_ID))}"
        ) from None


def get_level(level_id: str) -> Level:
    try:
        return _LEVEL_BY_ID[level_id]
    except KeyError:
        raise KeyError(
            f"unknown level {level_id!r}; known: {', '.join(sorted(_LEVEL_BY_ID))}"
        ) from None


# --------------------------------------------------------------------------
# The resolved combination
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Guidance:
    """Medium and level resolved into numbers the pipeline can use."""

    medium: Medium
    level: Level
    colors: int
    max_components: int
    min_region_area: int
    excluded_paint_ids: tuple[str, ...]
    notes: tuple[str, ...]

    @property
    def target_delta_e(self) -> float:
        return self.level.target_delta_e


#: Paints that exist in the catalogue but not in every medium's mixing.
#: Watercolour's lightener is dilution, so a white pigment in a recipe is an
#: instruction the learner physically cannot follow.
_WHITES = ("ti-white",)


def guidance_for(medium_id: str = "acrylic", level_id: str = "improving") -> Guidance:
    """Combine a medium and a skill level into concrete pipeline settings.

    Both limits are respected by taking the smaller: a beginner in oil gets the
    beginner's two-pigment ceiling, and a confident painter in watercolour
    still gets watercolour's. Neither should be able to raise the other.
    """
    medium = get_medium(medium_id)
    level = get_level(level_id)

    colors = max(4, min(24, round(medium.base_colors * level.colors_scale)))
    notes = [medium.note]

    if not medium.mixes_white:
        if medium.lightens_with == "water":
            notes.append(
                "Recipes leave white out — lighten by adding water and letting the "
                "paper show through."
            )
        else:
            notes.append(
                "Recipes leave white out — lighten by easing off the pressure."
            )
    if not medium.reworkable:
        notes.append("Mistakes here are hard to cover, so work from light to dark.")
    if level.next_step:
        notes.append(f"When this feels easy: {level.next_step}")

    return Guidance(
        medium=medium,
        level=level,
        colors=colors,
        max_components=min(medium.max_components, level.max_components),
        min_region_area=medium.min_region_area,
        excluded_paint_ids=() if medium.mixes_white else _WHITES,
        notes=tuple(notes),
    )


# --------------------------------------------------------------------------
# Wording a recipe for the medium in hand
# --------------------------------------------------------------------------
def describe_mix(components: list[tuple[str, float]], medium: Medium) -> str:
    """Turn (paint name, parts) pairs into an instruction for this medium.

    The same ratio is a different physical act depending on what is in your
    hand. In acrylic you measure out parts; in watercolour you dilute; with a
    pencil you press harder or softer. A single phrasing for all three would be
    wrong in two of them.
    """
    if not components:
        raise ValueError("a mix needs at least one component")

    named = [(name, parts) for name, parts in components if parts > 0]
    if not named:
        raise ValueError("every component has zero parts")

    if medium.lightens_with == "pressure":
        if len(named) == 1:
            return f"{named[0][0]}, worked up to the right depth"
        first, second = named[0], named[1]
        heavier = "evenly" if first[1] == second[1] else f"{first[0]} the heavier of the two"
        return f"{first[0]} layered over {second[0]} — {heavier}"

    body = " + ".join(
        f"{parts:g} part{'s' if parts != 1 else ''} {name}" for name, parts in named
    )
    if medium.lightens_with == "water":
        return f"{body}, thinned with water until it matches"
    return body


# --------------------------------------------------------------------------
# "Should I move up?"
# --------------------------------------------------------------------------
def should_advance(level: Level, recent_delta_e: list[float], *, needed: int = 3) -> bool:
    """Has this painter been hitting what their level aims at?

    The median rather than the mean, and a run of attempts rather than one:
    a single good photograph under good light should not promote anybody, and
    one bad session should not hold them back. `needed` attempts is the floor
    because two points are a line, not a trend.

    Returns a suggestion. Nothing in the product acts on it without being told.
    """
    if level.next_step is None:
        return False  # already at the top of the ladder
    usable = [d for d in recent_delta_e if d >= 0]
    if len(usable) < needed:
        return False
    return median(usable[-needed:]) <= level.target_delta_e


def next_level(level: Level) -> Level | None:
    index = _LADDER.index(level.id)
    if index + 1 >= len(_LADDER):
        return None
    return _LEVEL_BY_ID[_LADDER[index + 1]]


def describe_value(lightness: float, medium: Medium) -> str:
    """Express a target lightness as something a pencil can be pressed to.

    Scaled against what the medium can physically reach, not against pure
    black. Telling someone to make a shape "90% dark" when their pencil bottoms
    out at 22 L* sets them chasing a tone that does not exist in graphite, and
    the usual result is a flattened drawing and a torn sheet of paper.
    """
    span = max(1.0, 100.0 - medium.value_floor)
    depth = (100.0 - max(0.0, min(100.0, lightness))) / span
    steps = max(0, min(10, round(depth * 10)))

    if steps == 0:
        return "Leave this as bare paper"
    if steps <= 2:
        phrase = "A barely-there tone"
    elif steps <= 4:
        phrase = "A light tone"
    elif steps <= 6:
        phrase = "A mid tone"
    elif steps <= 8:
        phrase = "A dark tone"
    else:
        phrase = "Your very darkest"
    return f"{phrase} — about {steps} out of 10 of the darkest you can press"