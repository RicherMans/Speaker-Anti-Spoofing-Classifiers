#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import datetime

import uuid
import glob
import fire

import pandas as pd
import torch
import numpy as np
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events, create_supervised_evaluator, create_supervised_trainer
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.metrics import Accuracy, Loss, RunningAverage, Precision, Recall
from tabulate import tabulate

import dataset
import models
import utils
import metrics

DEVICE = 'cpu'
if torch.cuda.is_available(
) and 'SLURM_JOB_PARTITION' in os.environ and 'gpu' in os.environ[
        'SLURM_JOB_PARTITION']:
    DEVICE = 'cuda'
    # Without results are slightly inconsistent
    torch.backends.cudnn.deterministic = True
DEVICE = torch.device(DEVICE)


class Runner(object):
    """Main class to run experiments with e.g., train and evaluate"""
    def __init__(self, seed=11):
        """__init__

        :param config: YAML config file
        :param **kwargs: Overwrite of yaml config
        """
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)

    @staticmethod
    def _forward(model, batch):
        inputs, targets, filenames = batch
        inputs, targets = inputs.float().to(DEVICE), targets.float().to(DEVICE)
        outputs = model(inputs)
        return outputs, targets

    def train(self, config, **kwargs):
        """Trains a given model specified in the config file or passed as the --model parameter.
        All options in the config file can be overwritten as needed by passing --PARAM
        Options with variable lengths ( e.g., kwargs can be passed by --PARAM '{"PARAM1":VAR1, "PARAM2":VAR2}'

        :param config: yaml config file
        :param **kwargs: parameters to overwrite yaml config
        """

        config_parameters = utils.parse_config_or_kwargs(config, **kwargs)
        outputdir = os.path.join(
            config_parameters['outputpath'], config_parameters['model'],
            "{}_{}".format(
                datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%m'),
                uuid.uuid1().hex))
        # Early init because of creating dir
        checkpoint_handler = ModelCheckpoint(
            outputdir,
            'run',
            n_saved=1,
            require_empty=False,
            create_dir=True,
            score_function=lambda engine: -engine.state.metrics['Loss'],
            save_as_state_dict=False,
            score_name='loss')
        logger = utils.getfile_outlogger(os.path.join(outputdir, 'train.log'))
        logger.info("Storing files in {}".format(outputdir))
        # utils.pprint_dict
        utils.pprint_dict(config_parameters, logger.info)
        logger.info("Running on device {}".format(DEVICE))
        labels_df = pd.read_csv(config_parameters['label'], sep='\t')
        # In case of ave dataset where index is int, we change the
        # absolute name to relname
        if not np.issubdtype(labels_df['filename'].dtype, np.number):
            # if not labels_df['filename'].isnumeric():
            labels_df.loc[:, 'filename'] = labels_df['filename'].apply(
                os.path.basename)
        labels_df['encoded'], encoder = utils.encode_labels(
            labels=labels_df['event_labels'])
        train_df, cv_df = utils.split_train_cv(
            labels_df, **config_parameters['data_args'])

        transform = utils.parse_transforms(config_parameters['transforms'])
        utils.pprint_dict({'Classes': encoder.classes_},
                          logger.info,
                          formatter='pretty')
        utils.pprint_dict(transform, logger.info, formatter='pretty')
        if 'sampler' in config_parameters and config_parameters[
                'sampler'] == 'MinimumOccupancySampler':
            # Asserts that each "batch" contains at least one instance
            train_sampler = dataset.MinimumOccupancySampler(np.stack(
                train_df['encoded'].values),
                                                            sampling_factor=1)
        else:
            train_sampler = None

        logger.info("Using Sampler {}".format(
            train_sampler.__class__.__name__))

        colname = config_parameters.get('colname', ('filename', 'encoded'))  #
        trainloader = dataset.getdataloader(
            train_df,
            config_parameters['data'],
            transform=transform,
            sampler=train_sampler,  # shuffle is mutually exclusive
            batch_size=config_parameters['batch_size'],
            colname=colname,  # For other datasets with different key names
            num_workers=config_parameters['num_workers'])
        cvdataloader = dataset.getdataloader(
            cv_df,
            config_parameters['data'],
            transform=None,
            shuffle=False,
            colname=colname,  # For other datasets with different key names
            batch_size=config_parameters['batch_size'],
            num_workers=config_parameters['num_workers'])
        if 'pretrained' in config_parameters and config_parameters[
                'pretrained'] is not None:
            model = models.load_pretrained(config_parameters['pretrained'],
                                           outputdim=len(encoder.classes_))
        else:
            model = getattr(models, config_parameters['model'],
                            'LightCNN')(inputdim=trainloader.dataset.datadim,
                                        outputdim=len(encoder.classes_),
                                        **config_parameters['model_args'])

        if config_parameters['optimizer'] == 'AdaBound':
            try:
                import adabound
                optimizer = adabound.AdaBound(
                    model.parameters(), **config_parameters['optimizer_args'])
            except ImportError:
                config_parameters['optimizer'] = 'Adam'
                config_parameters['optimizer_args'] = {}
        else:
            optimizer = getattr(
                torch.optim,
                config_parameters['optimizer'],
            )(model.parameters(), **config_parameters['optimizer_args'])

        utils.pprint_dict(optimizer, logger.info, formatter='pretty')
        utils.pprint_dict(model, logger.info, formatter='pretty')
        if DEVICE.type != 'cpu' and torch.cuda.device_count() > 1:
            logger.info("Using {} GPUs!".format(torch.cuda.device_count()))
            model = torch.nn.DataParallel(model)
        criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
        model = model.to(DEVICE)

        def _train_batch(_, batch):
            model.train()
            with torch.enable_grad():
                optimizer.zero_grad()
                clip_level_output, time_level_output, targets = self._forward(
                    model, batch)
                loss = criterion(clip_level_output, targets)
                loss.backward()
                optimizer.step()
                return loss.item()

        def _inference(_, batch):
            model.eval()
            with torch.no_grad():
                clip_level_output, _, targets = self._forward(model, batch)
                return clip_level_output, targets

        precision = Precision()
        recall = Recall()
        f1_score = (precision * recall * 2 / (precision + recall)).mean()
        metrics = {
            'Loss': Loss(criterion),
            'Precision': Precision(),
            'Recall': Recall(),
            'Accuracy': Accuracy(),
            'F1': f1_score,
        }
        train_engine = create_supervised_trainer(model,
                                                 optimizer=optimizer,
                                                 loss_fn=criterion,
                                                 device=DEVICE)
        inference_engine = create_supervised_evaluator(model,
                                                       metrics=metrics,
                                                       device=DEVICE)

        RunningAverage(output_transform=lambda x: x).attach(
            train_engine, 'run_loss')
        pbar = ProgressBar(persist=False)
        pbar.attach(train_engine, ['run_loss'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               patience=3,
                                                               factor=0.1)

        @inference_engine.on(Events.COMPLETED)
        def update_reduce_on_plateau(engine):
            val_loss = engine.state.metrics['Loss']
            if 'ReduceLROnPlateau' == scheduler.__class__.__name__:
                scheduler.step(val_loss)
            else:
                scheduler.step()

        early_stop_handler = EarlyStopping(
            patience=7,
            score_function=lambda engine: -engine.state.metrics['Loss'],
            trainer=train_engine)
        inference_engine.add_event_handler(Events.EPOCH_COMPLETED,
                                           early_stop_handler)
        inference_engine.add_event_handler(Events.EPOCH_COMPLETED,
                                           checkpoint_handler, {
                                               'model': model,
                                               'encoder': encoder,
                                               'config': config_parameters,
                                           })

        @train_engine.on(Events.EPOCH_COMPLETED)
        def compute_metrics(engine):
            inference_engine.run(cvdataloader)
            results = inference_engine.state.metrics
            output_str_list = [
                "Validation Results - Epoch : {:<5}".format(engine.state.epoch)
            ]
            for metric in metrics:
                output_str_list.append("{} {:<5.2f}".format(
                    metric, results[metric]))
            logger.info(" ".join(output_str_list))
            pbar.n = pbar.last_print_n = 0

        train_engine.run(trainloader, max_epochs=config_parameters['epochs'])
        return outputdir

    def evaluate(self, experiment_path: str, result_file: str, **kwargs):
        """evaluate

        :param experiment_path: Path to already trained model using train
        :type experiment_path: str
        """
        # Update config parameters with new kwargs

        config = torch.load(glob.glob(
            "{}/run_config*".format(experiment_path))[0],
                            map_location=lambda storage, loc: storage)
        config_parameters = dict(config, **kwargs)
        model = torch.load(glob.glob(
            "{}/run_model*".format(experiment_path))[0],
                           map_location=lambda storage, loc: storage)
        encoder = torch.load(glob.glob(
            '{}/run_encoder*'.format(experiment_path))[0],
                             map_location=lambda storage, loc: storage)
        strong_labels_df = pd.read_csv(config_parameters['label'], sep='\t')

        # Evaluation is done via the filenames, not full paths
        if not np.issubdtype(strong_labels_df['filename'].dtype, np.number):
            strong_labels_df['filename'] = strong_labels_df['filename'].apply(
                os.path.basename)
        if 'audiofilepath' in strong_labels_df.columns:  # In case of ave dataset, the audiofilepath column is the main column
            strong_labels_df['audiofilepath'] = strong_labels_df[
                'audiofilepath'].apply(os.path.basename)
            colname = 'audiofilepath'  # AVE
        else:
            colname = 'filename'  # Dcase etc.
        # Problem is that we iterate over the strong_labels_df, which is ambigious
        # In order to conserve some time and resources just reduce strong_label to weak_label format
        weak_labels_df = strong_labels_df.groupby(colname)[
            'event_label'].unique().apply(tuple).to_frame().reset_index()
        if "event_labels" in strong_labels_df.columns:
            assert False, "Data with the column event_labels are used to train not to evaluate"
        weak_labels_df['encoded'], encoder = utils.encode_labels(
            labels=weak_labels_df['event_label'], encoder=encoder)
        config_parameters.setdefault('colname', ('filename', 'encoded'))
        dataloader = dataset.getdataloader(
            weak_labels_df,
            config_parameters['data'],
            batch_size=1,
            colname=config_parameters[
                'colname']  # For other datasets with different key names
        )
        model = model.to(DEVICE).eval()
        time_predictions, clip_predictions = [], []
        sequences_to_save = []
        with torch.no_grad():
            for batch in dataloader:
                _, _, filenames = batch
                clip_pred, pred, _ = self._forward(model, batch)

    def train_evaluate(self, config, test_data, test_label, **kwargs):
        experiment_path = self.train(config, **kwargs)
        import h5py
        # Get the output time-ratio factor from the model
        model = torch.load(glob.glob(
            "{}/run_model*".format(experiment_path))[0],
                           map_location=lambda storage, loc: storage)
        # Dummy to calculate the pooling factor a bit dynamic
        with h5py.File(test_data, 'r') as store:
            data_dim = next(iter(store.values())).shape[-1]
        dummy = torch.randn(1, 501, data_dim)
        _, time_out = model(dummy)
        time_ratio = max(0.02, 0.02 * (dummy.shape[1] // time_out.shape[1]))
        # Parse for evaluation ( if any )
        config_parameters = utils.parse_config_or_kwargs(config, **kwargs)
        threshold = config_parameters.get('threshold', None)
        postprocessing = config_parameters.get('postprocessing', 'double')
        self.evaluate(experiment_path,
                      label=test_label,
                      data=test_data,
                      time_ratio=time_ratio,
                      postprocessing=postprocessing,
                      threshold=threshold)


if __name__ == "__main__":
    fire.Fire(Runner)
