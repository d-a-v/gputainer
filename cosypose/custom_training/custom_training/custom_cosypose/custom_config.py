CUSTOM_CONFIG = dict(
    input_resize=(720, 540),
    urdf_ds_name='own_dataset',
    obj_ds_name='custom',
    train_pbr_ds_name=['custom.pbr'],
    test_ds_name=[],
    # Number of points for the loss, that should be smaller than the number of points in the CAD models
    n_points_loss=8,
    max_sampled_npoints=8,
)
