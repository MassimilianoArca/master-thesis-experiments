import enum


class GeneratorType(enum.Enum):
    """
    This enum describes type of data produced by the corresponding
    generator
    """

    SYNTH = enum.auto()
    REAL = enum.auto()
