class CIFAR10_SpVit():
    vit_batch_size = 64
    vit_lr_factor = 1e-5
    RL_batch_size = 64
    ReplayBufferCapacity = 10000
    actor_lr_factor = 1e-5
    critic_lr_factor = 1e-6
    training_data_dir = ""
    test_data_dir = ""
    num_classes = 10
    