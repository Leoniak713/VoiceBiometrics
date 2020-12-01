import copy
import random
import typing as t


class HPSVariant:
    def __init__(self, variants_list: t.Sequence, param_name: str = '') -> None:
        self.variants_list = variants_list
        self.param_name = param_name
        
    def draw_variant(self) -> t.Any:
        return random.choice(self.variants_list)

    def get_param_name(self, key_name: str) -> str:
        return self.param_name if self.param_name else key_name

class HPSParser:
    def __init__(self, source_config: t.Dict) -> None:
        self.source_config = source_config

    def parse_config(self, subelement: t.Any) -> None:
        subelement_iterator = None
        if isinstance(subelement, dict):
            subelement_iterator = subelement.items()
        elif isinstance(subelement, list):
            subelement_iterator = enumerate(subelement)
        if subelement_iterator:
            for key, value in subelement_iterator:
                if isinstance(value, HPSVariant):
                    variant = value.draw_variant()
                    subelement[key] = variant
                    self.drawn_variants[value.get_param_name(key)] = variant
                else:
                    self.parse_config(value)

    def draw_config(self) -> t.Tuple[t.Dict, t.List[str]]:
        config = copy.deepcopy(self.source_config)
        self.drawn_variants = {}
        self.parse_config(config)
        return config, self.drawn_variants