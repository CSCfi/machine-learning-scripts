import argparse
import os

from filelock import FileLock
from tensorflow.keras.datasets import mnist

import ray
from ray import tune
#from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.keras import TuneReportCallback

import os
if 'SLURM_CPUS_PER_TASK' in os.environ:
    ray.init(num_cpus=int(os.environ['SLURM_CPUS_PER_TASK']))

def train_mnist(config):
    # https://github.com/tensorflow/tensorflow/issues/32159
    import tensorflow as tf
    batch_size = 128
    num_classes = 10
    epochs = 10

    with FileLock(os.path.expanduser("~/.data.lock")):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(config["hidden"], activation="relu"),
        tf.keras.layers.Dropout(config["dropout"]),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.SGD(
            lr=config["lr"], momentum=config["momentum"]),
        metrics=["accuracy"])

    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=0,
        validation_data=(x_test, y_test),
        callbacks=[TuneReportCallback({
            "mean_accuracy": "accuracy",
            "mean_loss": "val_loss"
        })])


def tune_mnist():
    sched = ASHAScheduler(
        time_attr="training_iteration")

    metric="mean_accuracy"

    analysis = tune.run(
        train_mnist,
        name="foo",
        scheduler=sched,
        metric=metric,
        mode="max",
        #stop={
        #    "mean_accuracy": 0.99,
        #    "training_iteration": num_training_iterations
        #},
        num_samples=50,
        resources_per_trial={
            "cpu": 1,
            "gpu": 0
        },
        config={
            "dropout": tune.uniform(0.05, 0.5),
            "lr": tune.uniform(0.001, 0.1),
            "momentum": tune.uniform(0.1, 0.9),
            "hidden": tune.randint(32, 512),
        })
    print("Best hyperparameters found were: ", analysis.best_config)
    print("Best value for", metric, ':', analysis.best_result[metric])


if __name__ == "__main__":
    tune_mnist()
