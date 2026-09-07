"""DeepSpCas9 on-target activity scoring for PE-DB standardization."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_BEST_MODEL = "PreTrain-Final-3-5-7-100-70-40-0.001-550-80-60"
_TEST_BATCH = 500
_GUIDE_LENGTH = 20
_TARGET30_LENGTH = 30


def _resolve_project_root() -> Path:
    import os

    env_root = os.getenv("PE_PROJECT_ROOT") or os.getenv("PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    current = Path(__file__).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "vendor" / "models" / "deepprime").exists():
            return candidate
    return current.parents[4]


def _resolve_model_dir() -> Path:
    model_root = _resolve_project_root() / "vendor" / "models"
    candidates = (
        model_root / "deepprime" / "models" / "DeepSpCas9",
        model_root / "DeepSpCas9",
    )
    for path in candidates:
        if path.is_dir():
            return path.resolve()
    raise FileNotFoundError(
        "DeepSpCas9 checkpoint directory not found. Expected one of: "
        + ", ".join(str(path) for path in candidates)
    )


def extract_deepspcas9_target30(
    wt_sequence: str,
    protospacer_location_l: int,
    *,
    guide_length: int = _GUIDE_LENGTH,
) -> Optional[str]:
    """Return the 30-nt DeepSpCas9 input: 4 bp upstream + guide + PAM + 3 bp downstream."""
    wt = str(wt_sequence).upper()
    start = int(protospacer_location_l) - 4
    end = int(protospacer_location_l) + guide_length + 6
    if start < 0 or end > len(wt) or (end - start) != _TARGET30_LENGTH:
        return None
    return wt[start:end]


def _preprocess_seq(sequences: list[str], seq_length: int = _TARGET30_LENGTH) -> np.ndarray:
    encoded = np.zeros((len(sequences), 1, seq_length, 4), dtype=float)
    for row_idx, sequence in enumerate(sequences):
        seq = str(sequence).upper()
        for col_idx in range(min(seq_length, len(seq))):
            base = seq[col_idx]
            if base in "Aa":
                encoded[row_idx, 0, col_idx, 0] = 1
            elif base in "Cc":
                encoded[row_idx, 0, col_idx, 1] = 1
            elif base in "Gg":
                encoded[row_idx, 0, col_idx, 2] = 1
            elif base in "TtUu":
                encoded[row_idx, 0, col_idx, 3] = 1
    return encoded


class _DeepSpCas9Scorer:
    """Load the Kim et al. DeepSpCas9 checkpoint once and score batches of 30-mers."""

    def __init__(self, model_dir: Path) -> None:
        import tensorflow as tf

        self._tf = tf
        checkpoint = model_dir / _BEST_MODEL
        if not (model_dir / f"{_BEST_MODEL}.index").exists():
            raise FileNotFoundError(f"DeepSpCas9 checkpoint not found at {checkpoint}")

        (
            filter_size,
            filter_num,
            learning_rate,
            _load_episode,
            node_1,
            node_2,
        ) = self._parse_model_name(_BEST_MODEL)

        conf = tf.compat.v1.ConfigProto()
        conf.gpu_options.allow_growth = True
        self._tf.compat.v1.reset_default_graph()
        self._session = tf.compat.v1.Session(config=conf)
        self._model = self._build_model(filter_size, filter_num, node_1, node_2, learning_rate)
        saver = tf.compat.v1.train.Saver()
        saver.restore(self._session, str(checkpoint))

    @staticmethod
    def _parse_model_name(model_name: str) -> tuple[list[int], list[int], float, int, int, int]:
        values: list[object] = []
        for token in model_name.split("-"):
            if token == "True":
                values.append(True)
            elif token == "False":
                values.append(False)
            else:
                try:
                    values.append(int(token))
                except ValueError:
                    try:
                        values.append(float(token))
                    except ValueError:
                        values.append(token)
        filter_size = [int(values[2]), int(values[3]), int(values[4])]
        filter_num = [int(values[5]), int(values[6]), int(values[7])]
        learning_rate = float(values[8])
        load_episode = int(values[9])
        node_1 = int(values[10])
        node_2 = int(values[11])
        return filter_size, filter_num, learning_rate, load_episode, node_1, node_2

    def _build_model(
        self,
        filter_size: list[int],
        filter_num: list[int],
        node_1: int,
        node_2: int,
        learning_rate: float,
    ):
        tf = self._tf
        length = _TARGET30_LENGTH

        class DeepSpCas9Model:
            def __init__(self) -> None:
                self.inputs = tf.compat.v1.placeholder(tf.float32, [None, 1, length, 4])
                self.targets = tf.compat.v1.placeholder(tf.float32, [None, 1])
                self.is_training = tf.compat.v1.placeholder(tf.bool)

                def conv_layer(input_data, in_channels, out_channels, kernel, pool, name):
                    weights = tf.compat.v1.Variable(
                        tf.compat.v1.truncated_normal(
                            [kernel[0], kernel[1], in_channels, out_channels],
                            stddev=0.03,
                        ),
                        name=f"{name}_W",
                    )
                    bias = tf.compat.v1.Variable(
                        tf.compat.v1.truncated_normal([out_channels]),
                        name=f"{name}_b",
                    )
                    out = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding="VALID") + bias
                    out = tf.keras.layers.Dropout(rate=0.3)(tf.nn.relu(out))
                    return tf.nn.avg_pool(
                        out,
                        ksize=[1, pool[0], pool[1], 1],
                        strides=[1, 1, 2, 1],
                        padding="SAME",
                    )

                pool_0 = conv_layer(self.inputs, 4, filter_num[0], [1, filter_size[0]], [1, 2], "conv1")
                pool_1 = conv_layer(self.inputs, 4, filter_num[1], [1, filter_size[1]], [1, 2], "conv2")
                pool_2 = conv_layer(self.inputs, 4, filter_num[2], [1, filter_size[2]], [1, 2], "conv3")

                with tf.compat.v1.variable_scope("Fully_Connected_Layer1"):
                    node_0 = int((length - filter_size[0]) / 2) + 1
                    num_0 = node_0 * filter_num[0]
                    node_1_size = int((length - filter_size[1]) / 2) + 1
                    num_1 = node_1_size * filter_num[1]
                    node_2_size = int((length - filter_size[2]) / 2) + 1
                    num_2 = node_2_size * filter_num[2]
                    flat = tf.concat(
                        [
                            tf.reshape(pool_0, [-1, num_0]),
                            tf.reshape(pool_1, [-1, num_1]),
                            tf.reshape(pool_2, [-1, num_2]),
                        ],
                        1,
                        name="concat",
                    )
                    hidden = tf.nn.relu(
                        tf.nn.bias_add(
                            tf.matmul(flat, tf.compat.v1.get_variable("W_fcl1", shape=[num_0 + num_1 + num_2, node_1])),
                            tf.compat.v1.get_variable("B_fcl1", shape=[node_1]),
                        )
                    )
                    hidden = tf.keras.layers.Dropout(rate=0.3)(hidden)

                with tf.compat.v1.variable_scope("Fully_Connected_Layer2"):
                    hidden = tf.nn.relu(
                        tf.nn.bias_add(
                            tf.matmul(hidden, tf.compat.v1.get_variable("W_fcl2", shape=[node_1, node_2])),
                            tf.compat.v1.get_variable("B_fcl2", shape=[node_2]),
                        )
                    )
                    hidden = tf.keras.layers.Dropout(rate=0.3)(hidden)

                with tf.compat.v1.variable_scope("Output_Layer"):
                    self.outputs = tf.nn.bias_add(
                        tf.matmul(hidden, tf.compat.v1.get_variable("W_out", shape=[node_2, 1])),
                        tf.compat.v1.get_variable("B_out", shape=[1]),
                    )

                self.obj_loss = tf.reduce_mean(tf.square(self.targets - self.outputs))
                self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(self.obj_loss)

        model = DeepSpCas9Model()
        self._session.run(tf.compat.v1.global_variables_initializer())
        return model

    def score(self, sequences: list[str]) -> list[float]:
        if not sequences:
            return []
        encoded = _preprocess_seq(sequences)
        outputs = np.zeros((encoded.shape[0], 1), dtype=float)
        for start in range(0, encoded.shape[0], _TEST_BATCH):
            stop = start + _TEST_BATCH
            batch = encoded[start:stop]
            feed = {
                self._model.inputs: batch,
                self._model.is_training: False,
            }
            outputs[start:stop] = self._session.run(self._model.outputs, feed_dict=feed)
        return [float(value) for value in outputs.reshape(-1).tolist()]


@lru_cache(maxsize=1)
def _get_scorer() -> _DeepSpCas9Scorer:
    return _DeepSpCas9Scorer(_resolve_model_dir())


def score_target30_sequences(sequences: list[str]) -> list[float]:
    """Score unique 30-nt target sequences with the bundled DeepSpCas9 model."""
    try:
        return _get_scorer().score(sequences)
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required to compute DeepSpCas9 scores. "
            "Install tensorflow or tensorflow-macos in the PE-DB environment."
        ) from exc


def fill_missing_spcas9_scores(
    df: pd.DataFrame,
    *,
    score_fn: Optional[Callable[[list[str]], list[float]]] = None,
) -> pd.DataFrame:
    """Fill NaN ``spcas9_score`` values from ``wt_sequence`` and protospacer bounds."""
    if "spcas9_score" not in df.columns:
        return df

    output = df.copy()
    missing_mask = output["spcas9_score"].isna()
    if not missing_mask.any():
        return output

    row_targets: dict[int, str] = {}
    for row_idx in output.index[missing_mask]:
        row = output.loc[row_idx]
        target30 = extract_deepspcas9_target30(
            str(row["wt_sequence"]),
            int(row["protospacer_location_l"]),
        )
        if target30 is not None:
            row_targets[int(row_idx)] = target30

    if not row_targets:
        logger.warning("No valid DeepSpCas9 30-mers extracted for rows with missing spcas9_score")
        return output

    unique_targets = sorted(set(row_targets.values()))
    scorer = score_fn or score_target30_sequences
    try:
        scores = scorer(unique_targets)
    except RuntimeError as exc:
        logger.warning("Skipping DeepSpCas9 scoring: %s", exc)
        return output
    except FileNotFoundError as exc:
        logger.warning("Skipping DeepSpCas9 scoring: %s", exc)
        return output

    score_by_target = dict(zip(unique_targets, scores))
    for row_idx, target30 in row_targets.items():
        output.at[row_idx, "spcas9_score"] = score_by_target[target30]

    filled = int(output.loc[list(row_targets.keys()), "spcas9_score"].notna().sum())
    logger.info(
        "Filled DeepSpCas9 scores for %s/%s rows (%s unique 30-mers)",
        filled,
        int(missing_mask.sum()),
        len(unique_targets),
    )
    return output
