"""Domain adapter registry."""

from sensor_image_recon.domains.corrosion import CorrosionDomainAdapter
from sensor_image_recon.domains.thermal import ThermalDomainAdapter


def get_domain_adapter(name: str, dataset_config: dict):
    adapters = {
        "corrosion": CorrosionDomainAdapter,
        "thermal": ThermalDomainAdapter,
    }
    try:
        return adapters[name](dataset_config)
    except KeyError as exc:
        raise KeyError(f"Unknown domain: {name}") from exc
