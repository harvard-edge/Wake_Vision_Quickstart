import os
os.environ["KERAS_BACKEND"] = "jax"

import keras
import tensorflow as tf
from pp_ops import preprocessing

def train(model_path,
          dataset_dir,
          training_set="quality",
          dataset_type="tfds",
          input_shape=(224, 224, 3),
          batch_size=128,
          model_name="mobilenetv2.keras"
          ):
    
    
    if dataset_type == "hf":
        from datasets import load_dataset
        ds = load_dataset("Harvard-Edge/Wake-Vision", data_dir=dataset_dir)
        ds = ds.to_tf_dataset()
    elif dataset_type == "tfds":
        import tensorflow_datasets as tfds 
        ds = tfds.load("wake_vision",
                       data_dir=dataset_dir,
                       shuffle_files=False)
    if training_set == "large":
        train = ds["train_large"]
    else:
        train = ds["train_quality"]
    train = preprocessing(train, batch_size=batch_size, input_shape=input_shape, train=True)
    val = preprocessing(ds["validation"], batch_size=batch_size, input_shape=input_shape, train=False)
    test = preprocessing(ds["test"], batch_size=batch_size, input_shape=input_shape, train=False)
        
    model = keras.applications.MobileNetV2(
            input_shape=input_shape,
            alpha=0.25,
            weights=None,
            classes=2,
        )
    
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        0.00001,
        decay_steps=199000,
        alpha=0.0,
        warmup_target=0.002,
        warmup_steps=1000,
    )
    
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=keras.optimizers.AdamW(
        learning_rate=lr_schedule, weight_decay=0.000004
        ),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc"),],
    )
    
    # We use a fixed number of steps (rather than a set number of epochs) 
    # because we want to train for an equal ammount of steps regardless of
    # the training set we pick (quality or large). This allows us to compare
    # the training sets fairly.
    model.fit(
        train,
        epochs=(200000 // 10000), #Total Steps // Steps per epoch sets how often we want to eval
        steps_per_epoch=10000,
        validation_data=val,
    )
    score = model.evaluate(test, verbose=1)
    print(score)
    
    model.save(model_path)
    
    
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", type=str, required=True)
    parser.add_argument("-d", "--dataset_dir", type=str, required=True)
    parser.add_argument("--training_set", type=str, default="quality")
    parser.add_argument("--dataset_type", type=str, default="tfds")
    parser.add_argument("--input_shape", type=str, default="224,224,3")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--model_name", type=str, default="mobilenetv2.keras")
    args = parser.parse_args()
    train(
        model_path=args.model_path,
        dataset_dir=args.dataset_dir,
        training_set=args.training_set,
        dataset_type=args.dataset_type,
        input_shape=tuple(map(int, args.input_shape.split(","))),
        batch_size=args.batch_size,
        model_name=args.model_name
          )