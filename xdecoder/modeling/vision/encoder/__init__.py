from .transformer_encoder_fpn import *
# try:
#     from .transformer_encoder_deform import *
# except:
#     print(' Encoder is not available.1')
try:
    from .transformer_encoder_deform import *
except Exception as e:
    print(f'Encoder is not available.1: {e}')

from .build import *


def build_encoder(config, *args, **kwargs):
    model_name = config['MODEL']['ENCODER']['NAME']

    if not is_model(model_name):
        raise ValueError(f'Unkown model: {model_name}')

    return model_entrypoints(model_name)(config, *args, **kwargs)