#!/usr/bin/env python

import os
# Set thread usage BEFORE importing tensorflow
os.environ["OMP_NUM_THREADS"] = "48"
os.environ["TF_NUM_INTRAOP_THREADS"] = "48"
os.environ["TF_NUM_INTEROP_THREADS"] = "4"
os.environ["KMP_BLOCKTIME"] = "1"
os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"

import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(48)
tf.config.threading.set_inter_op_parallelism_threads(4)

# Confirm thread settings
print("Intra-op threads:", tf.config.threading.get_intra_op_parallelism_threads())
print("Inter-op threads:", tf.config.threading.get_inter_op_parallelism_threads())

from tensorflow.keras import layers, Model, Input, activations, callbacks
from tensorflow.keras.regularizers import l2
from tensorflow.data import Dataset
import tensorflow_probability as tfp
import random, pickle, joblib, numpy as np, wandb, pandas as pd
from wandb.integration.keras import WandbCallback
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K
tfd = tfp.distributions

# Check if GPU is available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("Num GPUs Available: ", len(gpus))
    print("First GPU Device:", gpus[0])
else:
    print("No GPUs found, running on CPU.")

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# CONFIG
VERSION = "v4"
SUBVERSION = "v6"
RUN_ID = "1"
SEED = 42

EPOCHS = 50
BATCH_SIZE = 512
LSTM_UNITS = 64
LEARNING_RATE = 1e-3
NUM_MIXTURES = 1
LAMBDA_MSE = 0.2
DECAY_fACTOR = 0.0
CLIPNORM = 1.0

random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# LOAD DATA
base = f"/cluster/home/fusg/VT1/training"
train_var = np.load(f"{base}/train_in_var_{SUBVERSION}.npy",  mmap_mode="r")
train_con = np.load(f"{base}/train_in_con_{SUBVERSION}.npy",  mmap_mode="r")
train_out = np.load(f"{base}/train_out_{SUBVERSION}.npy",     mmap_mode="r")

val_var   = np.load(f"{base}/val_in_var_{SUBVERSION}.npy",    mmap_mode="r")
val_con   = np.load(f"{base}/val_in_con_{SUBVERSION}.npy",    mmap_mode="r")
val_out   = np.load(f"{base}/val_out_{SUBVERSION}.npy",       mmap_mode="r")

train_idx = np.random.choice(train_var.shape[0], int(train_var.shape[0]), replace=False)
val_idx = np.random.choice(val_var.shape[0], int(val_var.shape[0]), replace=False)

train_var = train_var[train_idx]
train_con = train_con[train_idx]
train_out = train_out[train_idx, :, :3]

val_var = val_var[val_idx]
val_con = val_con[val_idx]
val_out = val_out[val_idx, :, :3]

TIME_STEPS_IN  = train_var.shape[1]
VARIABLE_FEATURES = train_var.shape[2]
CONSTANT_FEATURES = train_con.shape[2]

TIME_STEPS_OUT = train_out.shape[1]
OUTPUT_FEATURES = train_out.shape[2]

print(f"Input features shape: {train_var.shape}")
print(f"Constant features shape: {train_con.shape}")
print(f"Output features shape: {train_out.shape}")

## MODEL ##
class MixtureDensityOutput(layers.Layer):
    def __init__(self, output_dimension, num_mixtures, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dimension
        self.num_mix = num_mixtures
        self.mdn_mus = layers.Dense(
            self.num_mix * self.output_dim,
            name="mdn_mus"
        ) # mix*output vals, no activation
        self.mdn_sigmas = layers.Dense(
            self.num_mix * self.output_dim,
            activation='softplus', #'exponential', #elu_plus_one_plus_epsilon,,
            name="mdn_sigmas",
        ) # mix*output vals exp activation
        self.mdn_pi = layers.Dense(self.num_mix, name="mdn_pi") # mix vals, logits

    def build(self, input_shape):
        self.mdn_mus.build(input_shape)
        self.mdn_sigmas.build(input_shape)
        self.mdn_pi.build(input_shape)
        super().build(input_shape)

    @property
    def trainable_weights(self):
        return (
            self.mdn_mus.trainable_weights
            + self.mdn_sigmas.trainable_weights
            + self.mdn_pi.trainable_weights
        )

    @property
    def non_trainable_weights(self):
        return (
            self.mdn_mus.non_trainable_weights
            + self.mdn_sigmas.non_trainable_weights
            + self.mdn_pi.non_trainable_weights
        )
    
    def call(self, x):
        return tf.concat(
            [self.mdn_mus(x), self.mdn_sigmas(x), self.mdn_pi(x)], axis=-1, name="mdn_outputs"
        )
    
    def compute_output_shape(self, input_shape):
        # input_shape is (batch_size, dense_units)
        batch_size = input_shape[0]
        total_dim  = (2 * self.num_mix * self.output_dim) + self.num_mix
        return (batch_size, total_dim)

lambda_mse_var = tf.Variable(LAMBDA_MSE)
def get_mixture_loss_func(output_dim, num_mixes):
    def mdn_loss_func(y_true, y_pred):
        y_pred = tf.reshape(y_pred, [-1, (2 * num_mixes * output_dim) + num_mixes])
        y_true = tf.reshape(y_true, [-1, output_dim])

        out_mu, out_sigma, out_pi = tf.split(y_pred, [num_mixes * output_dim, num_mixes * output_dim, num_mixes], axis=-1)

        out_mu = tf.reshape(out_mu, [-1, num_mixes, output_dim])
        out_sigma = tf.reshape(out_sigma, [-1, num_mixes, output_dim])

        cat = tfd.Categorical(logits=out_pi)
        mvn = tfd.MultivariateNormalDiag(loc=out_mu, scale_diag=out_sigma)
        mixture = tfd.MixtureSameFamily(mixture_distribution=cat,
                                        components_distribution=mvn)

        nll = - mixture.log_prob(y_true)
        mdn_nll = tf.reduce_mean(nll)

        pi = tf.nn.softmax(out_pi, axis=-1)
        pi_exp = tf.expand_dims(pi, axis=-1)
        mean_weighted = tf.reduce_sum(pi_exp * out_mu, axis=1)
        
        mse = tf.reduce_mean(tf.square(mean_weighted - y_true))
        
        return mdn_nll + lambda_mse_var * mse
    return mdn_loss_func

class LogValMSE(tf.keras.callbacks.Callback):
    def __init__(self, val_data, output_dim, num_mixtures, **kwargs):
        super().__init__(**kwargs)
        self.val_var, self.val_con, self.val_out = val_data
        self.output_dim = output_dim
        self.num_mix = num_mixtures

    def on_epoch_end(self, epoch, logs=None):
        subset_size = min(5000, len(self.val_var))
        idx = np.random.choice(len(self.val_var), size=subset_size, replace=False)
        val_var = self.val_var[idx]
        val_con = self.val_con[idx]
        val_out = self.val_out[idx]

        with tf.device('/CPU:0'):
            y_pred = self.model.predict([val_var, val_con],
                                        batch_size=BATCH_SIZE, verbose=0)

        total_dim = (2 * self.num_mix * self.output_dim) + self.num_mix
        y_pred = y_pred.reshape(-1, total_dim)

        mu = y_pred[:, :self.num_mix * self.output_dim]
        pi_logits = y_pred[:, 2 * self.num_mix * self.output_dim:]
        mu = mu.reshape(-1, self.num_mix, self.output_dim)

        pi = np.exp(pi_logits - np.logaddexp.reduce(pi_logits, axis=1, keepdims=True))
        pred_mean = (pi[..., None] * mu).sum(axis=1)

        true_flat = val_out.reshape(-1, self.output_dim)
        mse = float(np.mean((pred_mean - true_flat) ** 2))

        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        wandb.log({
            "train_loss": logs.get("loss"),
            "val_loss": logs.get("val_loss"),
            "val_mse": mse,
            "learning_rate": lr,
            "lambda_mse": float(lambda_mse_var.numpy())
        }, step=epoch)

        # Decay lambda_mse
        #decay_factor = DECAY_fACTOR
        #lambda_mse_var.assign(float(lambda_mse_var.numpy()) * decay_factor)

def build_model():
    inp_var = layers.Input(shape=(TIME_STEPS_IN, VARIABLE_FEATURES), name="var_seq")
    inp_con = layers.Input(shape=(1, CONSTANT_FEATURES), name="const_seq")
    flat_con = layers.Reshape((CONSTANT_FEATURES,))(inp_con)
    rep_con = layers.RepeatVector(TIME_STEPS_IN)(flat_con)
    enc_in = layers.Concatenate(axis=-1)([inp_var, rep_con])

    bi_out = layers.Bidirectional(
        layers.LSTM(LSTM_UNITS, return_state=True),
        merge_mode="concat",
        name="bidirectional_encoder"
    )(enc_in)

    h_fwd, c_fwd, h_bwd, c_bwd = bi_out[1], bi_out[2], bi_out[3], bi_out[4]
    h_enc = layers.Concatenate(name="encoder_hidden_concat")([h_fwd, h_bwd])
    c_enc = layers.Concatenate(name="encoder_cell_concat")([c_fwd, c_bwd])

    dec_rep = layers.RepeatVector(TIME_STEPS_OUT, name="repeat_hidden")(h_enc)
    rep_con2 = layers.RepeatVector(TIME_STEPS_OUT, name="repeat_const")(flat_con)
    dec_in = layers.Concatenate(axis=-1, name="decoder_input_concat")([dec_rep, rep_con2])

    dec_out = layers.LSTM(
        2 * LSTM_UNITS,
        return_sequences=True,
        name="decoder_lstm"
    )(dec_in, initial_state=[h_enc, c_enc])

    mdn_out = layers.TimeDistributed(
        MixtureDensityOutput(
            output_dimension=OUTPUT_FEATURES,
            num_mixtures=NUM_MIXTURES
        ),
        name="mdn_time_distributed"
    )(dec_out)

    return Model([inp_var, inp_con], mdn_out, name=f"VT1_LAST_Run_{RUN_ID}")

model = build_model()
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=CLIPNORM),
    loss=get_mixture_loss_func(OUTPUT_FEATURES, NUM_MIXTURES)
)

wandb.init(
    project = "VT1_LAST_EXPERIMENT",
    name    = f"Run_{RUN_ID}",
    config = dict(
        epochs            = EPOCHS,
        batch_size        = BATCH_SIZE,
        lstm_units       = LSTM_UNITS,
        num_mixtures      = NUM_MIXTURES,
        output_features   = OUTPUT_FEATURES,
        loss              = f"MDN (out={OUTPUT_FEATURES}, mix={NUM_MIXTURES}) + MSE",
        model             = "BLSTM-MDN with overlap",
        input_seq_len     = TIME_STEPS_IN,
        output_seq_len    = TIME_STEPS_OUT,
        input_dim_var     = VARIABLE_FEATURES,
        input_dim_con     = CONSTANT_FEATURES,
        lambda_mse        = LAMBDA_MSE,
        clipnorm          = CLIPNORM,
        decay_factor      = DECAY_fACTOR,
        seed              = SEED
    )
)

cbs = [
    callbacks.ModelCheckpoint(f"/cluster/home/fusg/VT1/last_models/B_LSTM_MDN_model_{RUN_ID}.keras",
                                     save_best_only=True,
                                     monitor="val_loss",
                                     mode="min"),
    callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=10, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, mode="min"),
    LogValMSE(
        val_data=(val_var, val_con, val_out),
        output_dim=OUTPUT_FEATURES,
        num_mixtures=NUM_MIXTURES
    )
]

history = model.fit(
    [train_var, train_con],
    train_out,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=([val_var, val_con], val_out),
    callbacks=cbs
)

# Save model in multiple formats for future compatibility
base_path = f"/cluster/home/fusg/VT1/last_models/VT1_LAST_Run_{RUN_ID}"

# 1. Save as Keras format (current standard)
keras_path = f"{base_path}.keras"
model.save(keras_path)
print(f"Keras model saved to {keras_path}")

# 2. Save as TensorFlow SavedModel format (most portable)
savedmodel_path = f"{base_path}_savedmodel"
tf.saved_model.save(model, savedmodel_path)
print(f"SavedModel saved to {savedmodel_path}")

# 3. Save weights separately (most reliable for reconstruction)
weights_path = f"{base_path}_weights.h5"
model.save_weights(weights_path)
print(f"Weights saved to {weights_path}")

# 4. Save model architecture as JSON
architecture_path = f"{base_path}_architecture.json"
with open(architecture_path, 'w') as f:
    f.write(model.to_json())
print(f"Architecture saved to {architecture_path}")

# 5. Save model configuration and metadata
config_path = f"{base_path}_config.pkl"
model_config = {
    'model_config': model.get_config(),
    'optimizer_config': model.optimizer.get_config(),
    'loss_function': 'custom_mdn_loss',
    'hyperparameters': {
        'EPOCHS': EPOCHS,
        'BATCH_SIZE': BATCH_SIZE,
        'LSTM_UNITS': LSTM_UNITS,
        'LEARNING_RATE': LEARNING_RATE,
        'NUM_MIXTURES': NUM_MIXTURES,
        'LAMBDA_MSE': LAMBDA_MSE,
        'DECAY_fACTOR': DECAY_fACTOR,
        'CLIPNORM': CLIPNORM,
        'TIME_STEPS_IN': TIME_STEPS_IN,
        'TIME_STEPS_OUT': TIME_STEPS_OUT,
        'VARIABLE_FEATURES': VARIABLE_FEATURES,
        'CONSTANT_FEATURES': CONSTANT_FEATURES,
        'OUTPUT_FEATURES': OUTPUT_FEATURES,
        'SEED': SEED
    },
    'tensorflow_version': tf.__version__,
    'keras_version': tf.keras.__version__,
    'training_date': pd.Timestamp.now().isoformat()
}

with open(config_path, 'wb') as f:
    pickle.dump(model_config, f)
print(f"Configuration saved to {config_path}")

# Save training history
pd.DataFrame(history.history).to_csv(f"{base_path}_history.csv", index=False)
print(f"Training history saved to {base_path}_history.csv")

wandb.finish()