import tensorflow as tf

from tensorflow.keras import layers, Model


def build_detector(
    image_size=224,
    grid_size=7,
    num_classes=6
):

    inputs = layers.Input(
        shape=(image_size, image_size, 3)
    )

    backbone = tf.keras.applications.MobileNetV2(

        input_shape=(image_size, image_size, 3),

        include_top=False,

        weights="imagenet"

    )

    backbone.trainable = False

    x = tf.keras.applications.mobilenet_v2.preprocess_input(
        inputs
    )

    x = backbone(
        x,
        training=False
    )

    x = layers.Conv2D(
        256,
        3,
        padding="same",
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(
        128,
        3,
        padding="same",
        activation="relu"
    )(x)

    if x.shape[1] != grid_size:
        x = layers.Resizing(
            grid_size,
            grid_size
        )(x)

    outputs = layers.Conv2D(
        11,
        1,
        padding="same"
    )(x)

    model = Model(
        inputs,
        outputs
    )

    return model, backbone