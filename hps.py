import random
import copy


class HPSVariant:
    def __init__(self, variants_list, param_name=''):
        self.variants_list = variants_list
        self.param_name = param_name
        
    def draw_variant(self):
        return random.choice(self.variants_list)

    def get_param_name(self, key_name):
        return self.param_name if self.param_name else key_name

class HPSParser:
    def __init__(self, source_config):
        self.source_config = source_config

    def parse_config(self, subelement):
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

    def draw_config(self):
        config = copy.deepcopy(self.source_config)
        self.drawn_variants = {}
        self.parse_config(config)
        return config, self.drawn_variants