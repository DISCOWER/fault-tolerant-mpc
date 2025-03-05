class BrokenThruster:
    def __init__(self, index, intensity):
        """
        Storage class for a broken thruster.

        Args:
            index (int): Index of the broken thruster
            intensity (float): Intensity of the broken thruster, between 0 and 1
        """
        self.index = index
        self.intensity = intensity