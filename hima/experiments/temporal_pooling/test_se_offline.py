#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import numpy.typing as npt
from tqdm import tqdm, trange

from hima.common.config.base import TConfig
from hima.common.config.global_config import GlobalConfig
from hima.common.lazy_imports import lazy_import
from hima.common.run.wandb import get_logger
from hima.common.scheduler import Scheduler
from hima.common.sdr import OutputMode, wrap_as_rate_sdr, RateSdr
from hima.common.sdr_array import SdrArray
from hima.common.sds import Sds
from hima.common.timer import timer, print_with_timestamp
from hima.common.utils import isnone, prepend_dict_keys
from hima.experiments.temporal_pooling.resolvers.type_resolver import StpLazyTypeResolver
from hima.experiments.temporal_pooling.stats.sdr_tracker import SdrTracker
from hima.experiments.temporal_pooling.stp.mlp_torch import MlpClassifier
from hima.experiments.temporal_pooling.utils import resolve_random_seed

wandb = lazy_import('wandb')
sns = lazy_import('seaborn')
pd = lazy_import('pandas')


class TrainConfig:
    n_epochs: int
    batch_size: int

    def __init__(self, n_epochs: int, batch_size: int):
        self.n_epochs = n_epochs
        self.batch_size = batch_size


class TestConfig:
    eval_first: int
    eval_scheduler: Scheduler
    n_epochs: int
    noise: float

    def __init__(
            self, eval_first: int, eval_schedule: int, n_epochs: int, noise: float = 0.0
    ):
        self.eval_first = eval_first
        self.eval_scheduler = Scheduler(eval_schedule)
        self.n_epochs = n_epochs
        self.noise = noise

    def tick(self):
        return self.eval_scheduler.tick()


class EpochStats:
    def __init__(self):
        pass


class SpatialEncoderOfflineExperiment:
    training: TrainConfig
    testing: TestConfig

    dataset_sds: Sds
    encoding_sds: Sds

    stats: EpochStats

    def __init__(
            self, config: TConfig, config_path: Path,
            log: bool, seed: int, train: TConfig, test: TConfig,
            setup: TConfig, classifier: TConfig, data: str,
            enable_autoencoding_task: bool,
            sdr_tracker: TConfig, debug: bool,
            project: str = None,
            wandb_init: TConfig = None,
            **_
    ):
        self.init_time = timer()
        self.print_with_timestamp('==> Init')

        self.config = GlobalConfig(
            config=config, config_path=config_path, type_resolver=StpLazyTypeResolver()
        )
        self.log = log
        self.logger = self.config.resolve_object(
            isnone(wandb_init, {}),
            object_type_or_factory=get_logger,
            config=config, log=log, project=project
        )
        self.seed = resolve_random_seed(seed)
        self.rng = np.random.default_rng(self.seed)

        setup = self.config.config_resolver.resolve(setup, config_type=dict)
        (
            encoder, encoding_sds, input_mode, req_sdr_tracker,
            classifier_symexp_logits, ds_norm, grayscale, contrastive
        ) = self._get_setup(**setup)
        self.input_mode = OutputMode[input_mode.upper()]
        self.is_binary = self.input_mode == OutputMode.BINARY
        self.classifier_symexp_logits = classifier_symexp_logits

        if data in ['mnist', 'cifar']:
            from hima.experiments.temporal_pooling.data.mnist_ext import MnistDataset
            self.data = MnistDataset(
                seed=seed, binary=self.is_binary, ds=data, debug=debug,
                grayscale=grayscale, contrastive=contrastive
            )
            self.classification = True
        elif data == 'dvs':
            from hima.experiments.temporal_pooling.data.dvs_ext import DvsDataset
            ds_filepath = Path('~/data/outdoors_walking').expanduser()
            self.data = DvsDataset(seed=seed, filepath=ds_filepath, binary=self.is_binary)
            self.classification = False
        else:
            raise ValueError(f'Data type {data} is unsupported')

        self.dataset_sds = self.data.sds
        self.encoding_sds = Sds.make(encoding_sds)

        self.training = self.config.resolve_object(train, object_type_or_factory=TrainConfig)
        self.testing = self.config.resolve_object(test, object_type_or_factory=TestConfig)

        if encoder is not None:
            # spatial encoding layer + 1-layer linear ANN classifier/regressor
            self.encoder = self.config.resolve_object(
                encoder, feedforward_sds=self.dataset_sds, output_sds=self.encoding_sds
            )

            self.sdr_tracker = None
            if req_sdr_tracker:
                self.sdr_tracker: SdrTracker = self.config.resolve_object(
                    sdr_tracker, sds=self.encoding_sds
                )
            print(f'Encoder: {self.encoder.feedforward_sds} -> {self.encoder.output_sds}')

            normalizer = partial(
                normalize_ds, norm=ds_norm,
                p=getattr(self.encoder, 'lebesgue_p', None)
            )
        else:
            self.encoder = None
            normalizer = partial(normalize_ds, norm=ds_norm, p=None)

        self.data.train.normalize(normalizer)
        self.data.test.normalize(normalizer)

        self.n_classes = self.data.n_classes
        self.classifier: TConfig = classifier
        self.persistent_ann_classifier = self.make_ann_classifier()

        self.enable_autoencoding_task = enable_autoencoding_task
        if self.enable_autoencoding_task:
            self.persistent_ann_autoencoder = self.make_ann_autoencoder()
        self.i_train_epoch = 0
        self.set_metrics()

    def run(self):
        self.print_with_timestamp('==> Run')
        if self.encoder is None:
            self.run_ann()
        else:
            self.run_se_ann()

    def run_se_ann(self):
        """
        Train linear ANN classifier over a spatial encoding layer.
        There are three modes operating simultaneously:
        - we train SE in an epoch-based regime
        - [on testing schedule] we train ANN classifier for N epochs over
            a frozen K-epoch pretrained SE (K-N mode) and then test it
        - [for the first M epochs] we train ANN classifier batch-ONLINE mode
            without testing to report only the training losses (ONLINE mode).
            It is used to compare representation stability of SE and SP.
        """
        n_epochs = self.training.n_epochs
        self.i_train_epoch = 0
        while self.i_train_epoch < n_epochs:
            self.i_train_epoch += 1
            self.print_with_timestamp(f'Epoch {self.i_train_epoch}')

            self.train_epoch_se(self.data.train)

            # [on testing schedule] train and test ANN classifier in K-N mode
            self.test_epoch_se_ann_kn_mode(self.data.train, self.data.test)

    def train_epoch_se(self, data):
        n_samples = len(data)
        order = self.rng.permutation(n_samples)
        # Note that the learn is true -> we do not need results, only the random-order traversing
        # [= encoding] over the entire dataset with learning enabled.
        self.encode_array(data.sdrs, order=order, learn=True, track=False)
        self.print_encoding_speed('train')

    def test_epoch_se_ann_kn_mode(self, train_data, test_data):
        if not self.should_test():
            return

        print(f'==> Test after {self.i_train_epoch}')
        train_speed = self.get_encoding_speed()

        # ==> train and test epoch-specific ANN classifier
        kn_ann_classifier = self.make_ann_classifier()
        kn_ann_autoencoder = self.make_ann_autoencoder() if self.enable_autoencoding_task else None

        track_sdrs = self.sdr_tracker is not None and self.logger is not None
        n_train_samples = len(train_data)
        train_order = np.arange(n_train_samples)
        raw_train_sdrs, encoded_train_sdrs = self.encode_array(
            train_data.sdrs, order=train_order, learn=False, track=track_sdrs
        )
        eval_speed = self.get_encoding_speed()

        n_test_samples = len(test_data)
        test_order = np.arange(n_test_samples)
        raw_test_sdrs, encoded_test_sdrs = self.encode_array(
            test_data.sdrs, order=test_order, learn=False, track=track_sdrs,
            noise=self.testing.noise
        )
        entropy = None
        if track_sdrs:
            entropy = self.sdr_tracker.on_sequence_finished(None, ignore=False)['H']

        final_epoch_kn_losses = [0.], [0.]
        for ii in trange(self.testing.n_epochs):
            kn_epoch_losses = self.train_epoch_ann_classifier(
                kn_ann_classifier, encoded_train_sdrs, train_data.targets,
                autoencoder=kn_ann_autoencoder, ae_targets=raw_train_sdrs
            )
            # NB: is overwritten every epoch => stores the last epoch losses after the loop
            final_epoch_kn_losses = kn_epoch_losses

            if ii % 5 == 0:
                accuracy, ae_accuracy = self.evaluate_ann_classifier(
                    kn_ann_classifier, encoded_test_sdrs, self.data.test.targets,
                    autoencoder=kn_ann_autoencoder, ae_targets=raw_test_sdrs
                )
                self.print_decoder_quality(
                    kn_ann_classifier, accuracy, np.mean(kn_epoch_losses[0])
                )
                if kn_ann_autoencoder is not None:
                    self.print_decoder_quality(
                        kn_ann_autoencoder, ae_accuracy, np.mean(kn_epoch_losses[1]), prefix='AE'
                    )

        final_epoch_kn_losses, final_epoch_kn_ae_losses = final_epoch_kn_losses
        final_epoch_kn_loss = np.mean(final_epoch_kn_losses)

        accuracy, ae_accuracy = self.evaluate_ann_classifier(
            kn_ann_classifier, encoded_test_sdrs, self.data.test.targets,
            autoencoder=kn_ann_autoencoder, ae_targets=raw_test_sdrs
        )
        self.print_decoder_quality(kn_ann_classifier, accuracy, final_epoch_kn_loss)

        # add metrics
        epoch_metrics = {
            'kn_loss': final_epoch_kn_loss,
            'kn_accuracy': accuracy,
        }
        if train_speed is not None and eval_speed is not None:
            epoch_metrics |= {
                'train_speed_kcps': train_speed,
                'eval_speed_kcps': eval_speed,
            }
        if entropy is not None:
            epoch_metrics['se_entropy'] = entropy

        if kn_ann_autoencoder is not None:
            final_epoch_kn_ae_loss = np.mean(final_epoch_kn_ae_losses)
            self.print_decoder_quality(
                kn_ann_autoencoder, ae_accuracy, final_epoch_kn_ae_loss, prefix='AE'
            )
            epoch_metrics |= {
                'kn_ae_loss': final_epoch_kn_ae_loss,
                'kn_ae_accuracy': ae_accuracy,
            }

        self.log_progress(epoch_metrics)
        self.print_encoding_speed('eval')
        print('<== Test')

    def run_ann(self):
        """
        Train 2-layer ANN classifier for N epochs with Batch SGD. Every epoch, the train
        dataset is split into batches, and the classifier is updated with each batch.
        We also collect all losses and provide it to the logger.

        Testing schedule determines when to evaluate the classifier on the test dataset.
        """
        classifier = self.persistent_ann_classifier
        autoencoder = self.persistent_ann_autoencoder if self.enable_autoencoding_task else None

        train_sdrs, train_targets = self.data.train.sdrs, self.data.train.targets
        test_data = self.data.test

        self.i_train_epoch = 0
        while self.i_train_epoch < self.testing.n_epochs:
            self.i_train_epoch += 1
            self.print_with_timestamp(f'Epoch {self.i_train_epoch}')
            # NB: it is `nn_` instead of `kn` as both first and second layers trained for N epochs,
            # i.e. K-N mode for 2-layer ANN is N-N mode.
            nn_epoch_losses = self.train_epoch_ann_classifier(
                classifier, train_sdrs, train_targets, autoencoder=autoencoder
            )
            self.test_epoch_ann(classifier, test_data, nn_epoch_losses, autoencoder=autoencoder)

    def train_epoch_ann_classifier(
            self, classifier, sdrs, targets, *, autoencoder=None, ae_targets=None
    ):
        n_samples = len(sdrs)
        order = self.rng.permutation(n_samples)
        batched_indices = split_to_batches(order, self.training.batch_size)

        losses = []
        ae_losses = []
        for batch_ixs in batched_indices:
            batch = sdrs.get_batch_dense(batch_ixs)
            target_cls = targets[batch_ixs]
            classifier.learn(batch, target_cls)
            losses.append(classifier.losses[-1])

            if autoencoder is not None:
                ae_target_batch = (
                    batch if ae_targets is None
                    else ae_targets.get_batch_dense(batch_ixs)
                )
                autoencoder.learn(batch, ae_target_batch)
                ae_losses.append(autoencoder.losses[-1])

        return losses, ae_losses

    def test_epoch_ann(
            self, classifier, data, nn_epoch_losses, *, autoencoder=None
    ):
        if not self.should_test():
            return

        nn_epoch_losses, nn_epoch_ae_losses = nn_epoch_losses

        nn_epoch_loss = np.mean(nn_epoch_losses)
        accuracy, ae_accuracy = self.evaluate_ann_classifier(
            classifier, data.sdrs, data.targets,
            noise=self.testing.noise, autoencoder=autoencoder
        )
        self.print_decoder_quality(classifier, accuracy, nn_epoch_loss)

        epoch_metrics = {
            'kn_loss': nn_epoch_loss,
            'kn_accuracy': accuracy,
        }

        if autoencoder is not None:
            nn_epoch_ae_loss = np.mean(nn_epoch_ae_losses) ** 0.5
            self.print_decoder_quality(autoencoder, ae_accuracy, nn_epoch_ae_loss, prefix='AE')

            epoch_metrics |= {
                'kn_ae_loss': nn_epoch_ae_loss,
                'kn_ae_accuracy': ae_accuracy,
            }

        self.log_progress(epoch_metrics)

    def evaluate_ann_classifier(
            self, classifier, sdrs, targets, *, noise=0., autoencoder=None, ae_targets=None
    ):
        n_samples = len(sdrs)
        order = np.arange(n_samples)
        batched_indices = split_to_batches(order, self.training.batch_size)

        sum_accuracy = 0.0
        sum_ae_accuracy = 0.0
        for batch_ixs in batched_indices:
            if noise > 0.0:
                batch_sdrs = sdrs.create_slice(batch_ixs)
                batch_sdrs = self.apply_noise_to_input(batch_sdrs, noise)
                batch_sdrs = batch_sdrs.get_batch_dense(np.arange(len(batch_sdrs)))
            else:
                batch_sdrs = sdrs.get_batch_dense(batch_ixs)

            target_cls = targets[batch_ixs]

            prediction = classifier.predict(batch_sdrs)
            sum_accuracy += self.get_accuracy(classifier, prediction, target_cls)

            if autoencoder is not None:
                ae_prediction = autoencoder.predict(batch_sdrs)
                ae_target_batch = (
                    batch_sdrs if ae_targets is None
                    else ae_targets.get_batch_dense(batch_ixs)
                )
                if sum_ae_accuracy == 0.:
                    plot_img(
                        (32, 32), [ae_target_batch, ae_prediction], 0,
                        with_err=True
                    )

                sum_ae_accuracy += self.get_accuracy(autoencoder, ae_prediction, ae_target_batch)

        return sum_accuracy / n_samples, sum_ae_accuracy / n_samples

    def encode_array(
            self, sdrs: SdrArray, *, order, learn=False, track=False, noise=0.
    ):
        assert self.encoder is not None, 'Encoder is not defined'
        raw_sdrs = []
        encoded_sdrs = []

        if getattr(self.encoder, 'compute_batch', False):
            # I expect that batch computing is defined for dense SDRs, as only for this kind
            # of encoding batch computing is reasonable.
            batched_indices = split_to_batches(order, self.training.batch_size)
            for batch_ixs in tqdm(batched_indices):
                batch_sdrs = sdrs.create_slice(batch_ixs)
                batch_sdrs = self.apply_noise_to_input(batch_sdrs, noise)
                raw_sdrs.extend(batch_sdrs.sparse)

                encoded_batch: SdrArray = self.encoder.compute_batch(batch_sdrs, learn=learn)
                if track:
                    self.sdr_tracker.on_sdr_batch_updated(encoded_batch, ignore=False)
                encoded_sdrs.extend(encoded_batch.sparse)
        else:
            # for single SDR encoding, compute expects a sparse SDR.
            for ix in tqdm(order):
                obs_sdr = sdrs.get_sdr(ix, binary=self.is_binary)
                obs_sdr = self.apply_noise_to_input(obs_sdr, noise)
                raw_sdrs.append(wrap_as_rate_sdr(obs_sdr))

                enc_sdr = self.encoder.compute(obs_sdr, learn=learn)
                enc_sdr = wrap_as_rate_sdr(enc_sdr)
                if track:
                    self.sdr_tracker.on_sdr_updated(enc_sdr, ignore=False)
                encoded_sdrs.append(enc_sdr)

        raw_sdrs = SdrArray(sparse=raw_sdrs, sdr_size=self.dataset_sds.size)
        encoded_sdrs = SdrArray(sparse=encoded_sdrs, sdr_size=self.encoding_sds.size)
        return raw_sdrs, encoded_sdrs

    def apply_noise_to_input(self, sdr, noise):
        if noise == 0.0:
            return sdr

        from hima.common.sdr_sampling import (
            sample_noisy_rates_rate_sdr, sample_noisy_sdr_rate_sdr, sample_noisy_sdr
        )
        from functools import partial
        f_noise = (
            partial(sample_noisy_rates_rate_sdr, self.rng, frac=noise)
            if self.rng.random() > 0.5 else
            partial(sample_noisy_sdr_rate_sdr, self.rng, sds=self.dataset_sds, frac=noise)
        )
        if isinstance(sdr, SdrArray):
            batch_sdrs = sdr
            return SdrArray(
                sparse=[f_noise(rate_sdr=sdr) for sdr in batch_sdrs.sparse],
                sdr_size=self.dataset_sds.size
            )
        elif not self.is_binary:
            return f_noise(rate_sdr=sdr)
        else:
            return sample_noisy_sdr(self.rng, self.dataset_sds, sdr, noise)

    def log_progress(self, metrics: dict):
        if self.logger is None:
            return

        metrics['epoch'] = self.i_train_epoch
        self.logger.log(metrics)

    @staticmethod
    def get_accuracy(mlp: MlpClassifier, predictions, targets):
        if mlp.is_classifier:
            # number of correct predictions in a batch
            return np.count_nonzero(np.argmax(predictions, axis=-1) == targets)

        # MAE over each prediction coordinates => sum MAE over batch
        return np.sum(
            np.mean(np.abs(predictions - targets), axis=-1)
        )

    def make_ann_classifier(self) -> MlpClassifier:
        in_size, out_size = self.dataset_sds.size, self.n_classes
        enc_size = self.encoding_sds.size

        layers = [in_size] if self.encoder is None else []
        layers += [enc_size, out_size]

        return self.config.resolve_object(
            self.classifier, object_type_or_factory=MlpClassifier,
            layers=layers, classification=self.classification,
            symexp_logits=self.classifier_symexp_logits
        )

    def make_ann_autoencoder(self) -> MlpClassifier:
        # actually, we make a decoder part of the autoencoder if Hebbian encoder is provided,
        # or make a full autoencoder for the full-ANN baseline
        in_size = out_size = self.dataset_sds.size
        enc_size = self.encoding_sds.size

        layers = [in_size] if self.encoder is None else []
        layers += [enc_size, out_size]

        autoencoder_config = self.classifier.copy()
        # autoencoder_config['learning_rate'] = 0.002
        autoencoder_config['learning_rate'] /= 5

        # the only difference from `make_ann_classifier` are:
        #   - `out_size` for layers
        #   - `classification = False`, since it's a regressor
        return self.config.resolve_object(
            autoencoder_config, object_type_or_factory=MlpClassifier,
            layers=layers,
            classification=False,
            symexp_logits=self.classifier_symexp_logits
        )

    @staticmethod
    def print_decoder_quality(
            mlp: MlpClassifier, accuracy, nn_epoch_loss, prefix: str = ''
    ):
        if mlp.is_classifier:
            print(f'Accuracy: {accuracy:.3%} | KL Loss: {nn_epoch_loss:.4f}')
        else:
            if len(prefix) > 0:
                prefix = prefix + ' '
            print(f'{prefix}MAE: {accuracy:.7f} | sqrtMSE Loss: {nn_epoch_loss:.7f}')

    @staticmethod
    def _get_setup(
            input_mode: str, encoding_sds, encoder: TConfig = None, sdr_tracker: bool = True,
            classifier_symexp_logits: bool = False, ds_norm: str = None,
            grayscale: bool = True, contrastive: bool = False
    ):
        return (
            encoder, encoding_sds, input_mode, sdr_tracker,
            classifier_symexp_logits, ds_norm, grayscale, contrastive
        )

    def should_test(self):
        if self.i_train_epoch <= self.training.n_epochs:
            eval_scheduled = self.testing.tick()
            if eval_scheduled or self.i_train_epoch <= self.testing.eval_first:
                return True

        with_encoder = self.encoder is not None
        if with_encoder:
            # last training epoch (for encoder)
            return self.i_train_epoch == self.training.n_epochs

        # last training epoch (for ANN)
        return self.i_train_epoch == self.testing.n_epochs

    def print_with_timestamp(self, *args, cond: bool = True):
        if not cond:
            return
        print_with_timestamp(self.init_time, *args)

    def get_encoding_speed(self):
        if not hasattr(self.encoder, 'computation_speed'):
            return None

        return round(1.0 / self.encoder.computation_speed.get() / 1000.0, 2)

    def print_encoding_speed(self, kind):
        speed = self.get_encoding_speed()
        if speed is None:
            return
        print(f'{kind} speed_kcps: {speed:.2f}')

    def set_metrics(self):
        if self.logger is None:
            return

        self.logger.define_metric("epoch")
        self.logger.define_metric("se_entropy", step_metric="epoch")
        self.logger.define_metric("kn_loss", step_metric="epoch")
        self.logger.define_metric("kn_accuracy", step_metric="epoch")
        self.logger.define_metric("kn_ae_loss", step_metric="epoch")
        self.logger.define_metric("kn_ae_accuracy", step_metric="epoch")


def personalize_metrics(metrics: dict, prefix: str):
    return prepend_dict_keys(metrics, prefix, separator='/')


def split_to_batches(order, batch_size):
    n_samples = len(order)
    return np.array_split(order, n_samples // batch_size)


def normalize_ds(ds, norm, p=None):
    if norm is None:
        return ds
    if norm == 'l1':
        p = 1
    elif norm == 'l2':
        p = 2
    elif norm == 'lp':
        assert p is not None, 'p must be provided for lp norm'
    else:
        raise ValueError(f'Unknown normalization type: {norm}')

    r_p = np.linalg.norm(ds, ord=p, axis=-1)
    if np.ndim(r_p) > 0:
        r_p = r_p[:, np.newaxis]
    return ds / r_p


def plot_img(
        shape, img: list | RateSdr | SdrArray | npt.NDArray[float], ind: int = 0,
        with_err: bool = False
):
    images = to_images(shape, img, ind, with_err)
    n_images = len(images)

    import matplotlib.pyplot as plt
    _, axes = plt.subplots(1, n_images, sharey=True)

    for i in range(n_images):
        img = images[i]
        ax = axes[i] if n_images > 1 else axes
        ax.imshow(img)
        # print(img[16])

    plt.pause(2)
    plt.close()


def to_images(
        shape, img: list | RateSdr | SdrArray | npt.NDArray[float], ind: int = 0,
        with_err: bool = False
):
    sz = np.prod(shape)
    images = img
    if not isinstance(images, list):
        images = [images]
    n_images = len(images)

    result = []
    for i in range(n_images):
        img = images[i]

        if isinstance(img, SdrArray):
            img = img.get_sdr(ind).to_dense(sz)
        elif isinstance(img, RateSdr):
            img = img.to_dense(sz)
        elif isinstance(img, np.ndarray):
            if img.ndim > 1:
                img = img[ind]
        else:
            print(type(img))

        img = img.reshape(shape)
        result.append(img)

        if with_err and i % 2 == 1:
            result.append(
                np.abs(result[-1] - result[-2])
            )
    return result
