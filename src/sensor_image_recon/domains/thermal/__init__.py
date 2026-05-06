class ThermalDomainAdapter:
    """Reserved adapter boundary for thermal S-parameter image reconstruction."""

    name = "thermal"

    def __init__(self, dataset_config: dict):
        self.dataset_config = dataset_config

    def _not_implemented(self):
        raise NotImplementedError("Thermal domain adapter is reserved for a later phase")

    def parse_condition(self, row, channels):
        self._not_implemented()

    def target_path(self, row):
        self._not_implemented()

    def target_to_tensor(self, image, image_size):
        self._not_implemented()

    def sample_id(self, row):
        self._not_implemented()
