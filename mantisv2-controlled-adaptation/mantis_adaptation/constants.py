"""Protocol constants pinned for the primary experiment."""

MANTIS_GIT_COMMIT = "0c94f8ceb9f1d1421dd292ed917090df8c31605b"
MANTIS_VERSION = "1.0.0"
CHECKPOINT = "paris-noah/MantisV2"
CHECKPOINT_REVISION = "8f6ca35cb54ab14b120618943c6fca5ddf5a76a6"

SEQ_LEN = 512
EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.05
ADAM_BETAS = (0.9, 0.999)
SEEDS = (0, 1, 2)
MODES = ("frozen", "full", "scratch")
