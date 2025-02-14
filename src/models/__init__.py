from . import ppat, MATIG

def make(config):
    if config.model.name == "MATIG":
        model = MATIG.make(config)
    elif config.model.name == "PointBERT":
        model = ppat.make(config)
    elif config.model.name == "DGCNN":
        from . import dgcnn
        model = dgcnn.make(config)
    elif config.model.name == "PointNeXt":
        from . import pointnext
        model = pointnext.make(config)
    elif config.model.name == "PointMLP":
        from . import pointmlp
        model = pointmlp.make(config)
    elif config.model.name == "PointNet":
        from . import pointnet
        model = pointnet.make(config)
    elif config.model.name == "ViTGuidanceMAE":
        from . import vitmae
        model = vitmae.make(config)
    else:
        raise NotImplementedError("Model %s not supported." % config.model.name)
    return model
