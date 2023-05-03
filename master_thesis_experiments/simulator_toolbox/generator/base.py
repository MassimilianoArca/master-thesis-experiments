from master_thesis_experiments.simulator_toolbox.enums import GeneratorType


class Generator:
    """
    Base class for data generation
    """

    def __init__(self, generator_type: GeneratorType, name):
        self.generator_type = generator_type
        self.name = name
        self.is_generating = False

    def generate(self, generate):
        """
        base method to define a common interface
        """
        raise NotImplementedError


