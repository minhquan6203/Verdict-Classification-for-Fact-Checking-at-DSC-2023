from encoder_module.multi_modal_encoder import CoAttentionEncoder,CrossModalityEncoder,GuidedAttentionEncoder
from encoder_module.uni_modal_encoder import UniModalEncoder
def build_multi_modal_encoder(config):
    if config['multi_encoder']['type']=='cross':
        return CrossModalityEncoder(config)
    if config['multi_encoder']['type']=='co':
        return CoAttentionEncoder(config)
    if config['multi_encoder']['type']=='guide':
        return GuidedAttentionEncoder(config)

def build_uni_modal_encoder(config):
    return UniModalEncoder(config)

