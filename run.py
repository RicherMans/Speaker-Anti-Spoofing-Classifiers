#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv  # For writing the scores
import datetime
import glob
import uuid
from pathlib import Path
import os  # Just for os.environ

import fire
import numpy as np
import pandas as pd
import torch
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events, create_supervised_evaluator, create_supervised_trainer
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.metrics import Accuracy, Loss, Precision, Recall, RunningAverage
from ignite.utils import convert_tensor
from tqdm import tqdm

import dataset
import evaluation.eval_metrics as em
import models
import utils

DEVICE = 'cpu'
# Fix for cluster runs... some partitions support GPU even if you submit to CPU
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

    def train(self, config, **kwargs):
        """Trains a given model specified in the config file or passed as the --model parameter.
        All options in the config file can be overwritten as needed by passing --PARAM
        Options with variable lengths ( e.g., kwargs can be passed by --PARAM '{"PARAM1":VAR1, "PARAM2":VAR2}'

        :param config: yaml config file
        :param **kwargs: parameters to overwrite yaml config
        """

        config_parameters = utils.parse_config_or_kwargs(config, **kwargs)
        outputdir = Path(
            config_parameters['outputpath'], config_parameters['model'],
            "{}_{}".format(
                datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%m'),
                uuid.uuid1().hex[:8]))
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
        logger = utils.getfile_outlogger(Path(outputdir, 'train.log'))
        logger.info("Storing files in {}".format(outputdir))
        # utils.pprint_dict
        utils.pprint_dict(config_parameters, logger.info)
        logger.info("Running on device {}".format(DEVICE))
        labels_df = pd.read_csv(config_parameters['trainlabel'], sep=' ')
        labels_df['encoded'], encoder = utils.encode_labels(
            labels=labels_df['bintype'])
        train_df, cv_df = utils.split_train_cv(labels_df)

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
            sampling_kwargs = {"sampler": train_sampler, "shuffle": False}
        else:
            sampling_kwargs = {"shuffle": True}

        logger.info("Using Sampler {}".format(sampling_kwargs))

        colname = config_parameters.get('colname', ('filename', 'encoded'))  #
        trainloader = dataset.getdataloader(
            train_df,
            config_parameters['traindata'],
            transform=transform,
            batch_size=config_parameters['batch_size'],
            colname=colname,  # For other datasets with different key names
            num_workers=config_parameters['num_workers'],
            **sampling_kwargs)
        cvdataloader = dataset.getdataloader(
            cv_df,
            config_parameters['traindata'],
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
                logger.info(
                    "Adabound package not found, install via pip install adabound. Using Adam instead"
                )
                config_parameters['optimizer'] = 'Adam'
                config_parameters['optimizer_args'] = {
                }  # Default adam is adabount not found
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

        precision = Precision()
        recall = Recall()
        f1_score = (precision * recall * 2 / (precision + recall)).mean()
        metrics = {
            'Loss': Loss(criterion),
            'Precision': precision.mean(),
            'Recall': recall.mean(),
            'Accuracy': Accuracy(),
            'F1': f1_score,
        }

        # batch contains 3 elements, X,Y and filename. Filename is only used
        # during evaluation
        def _prep_batch(batch, device=DEVICE, non_blocking=False):
            x, y, _ = batch
            return (convert_tensor(x, device=device,
                                   non_blocking=non_blocking),
                    convert_tensor(y, device=device,
                                   non_blocking=non_blocking))

        train_engine = create_supervised_trainer(model,
                                                 optimizer=optimizer,
                                                 loss_fn=criterion,
                                                 prepare_batch=_prep_batch,
                                                 device=DEVICE)
        inference_engine = create_supervised_evaluator(
            model, metrics=metrics, prepare_batch=_prep_batch, device=DEVICE)

        RunningAverage(output_transform=lambda x: x).attach(
            train_engine, 'run_loss')  # Showing progressbar during training
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
            patience=5,
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
        def compute_validation_metrics(engine):
            inference_engine.run(cvdataloader)
            results = inference_engine.state.metrics
            output_str_list = [
                "Validation Results - Epoch : {:<5}".format(engine.state.epoch)
            ]
            for metric in metrics:
                output_str_list.append("{} {:<5.3f}".format(
                    metric, results[metric]))
            logger.info(" ".join(output_str_list))
            pbar.n = pbar.last_print_n = 0

        train_engine.run(trainloader, max_epochs=config_parameters['epochs'])
        return outputdir

    def score(self, experiment_path: str, result_file: str, **kwargs):
        """score
        Scores a given experiemnt path e.g., outputs probability scores
        for a given dataset passed as:
            --data features/hdf5/somedata.h5
            --label features/labels/somedata.csv


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
        testlabel = config_parameters['testlabel']
        testdata = config_parameters['testdata']
        # Only a single item to evaluate
        if isinstance(testlabel, list) and len(testlabel) == 1:
            testlabel = testlabel[0]
        if isinstance(testdata, list) and len(testdata) == 1:
            testdata = testdata[0]

        labels_df = pd.read_csv(testlabel, sep=' ')
        labels_df['encoded'], encoder = utils.encode_labels(
            labels=labels_df['bintype'], encoder=encoder)
        config_parameters.setdefault('colname', ('filename', 'encoded'))
        dataloader = dataset.getdataloader(
            data_frame=labels_df,
            data_file=testdata,
            num_workers=4,
            batch_size=1,  # do not apply any padding
            colname=config_parameters[
                'colname']  # For other datasets with different key names
        )
        model = model.to(DEVICE).eval()
        genuine_label_idx = encoder.transform(['genuine'])[0]

        with torch.no_grad(), open(result_file,
                                   'w') as wp, tqdm(total=len(dataloader),
                                                    unit='utts') as pbar:
            datawriter = csv.writer(wp, delimiter=' ')
            datawriter.writerow(['filename', 'score'])
            for batch in dataloader:
                inputs, _, filenames = batch
                inputs = inputs.float().to(DEVICE)
                preds = model(inputs)
                for pred, filename in zip(preds, filenames):
                    # Single batchsize
                    datawriter.writerow([filename, pred[0].item()])
                pbar.update()
        print("Score file can be found at {}".format(result_file))

    def run(self, config, **kwargs):
        """run

        Trains and evaluates a given config

        :param config: Config for training and evaluation
            :param data: pass --data for trainingdata (HDF5)
            :param label: pass --label for training labels
        :param test_data: Data to use for testing (HDF5)
        :param test_label: According labels for testing
        :param **kwargs:
        """
        config_parameters = utils.parse_config_or_kwargs(config, **kwargs)
        experiment_path = self.train(config, **kwargs)
        evaluation_logger = utils.getfile_outlogger(
            Path(experiment_path, 'evaluation.log'))
        for testdata, testlabel in zip(config_parameters['testdata'],
                                       config_parameters['testlabel']):
            evaluation_logger.info(
                f'Evaluting {testdata} with {testlabel} in {experiment_path}')
            # Scores for later evaluation
            scores_file = Path(experiment_path,
                               'scores_' + Path(testdata).stem + '.tsv')
            evaluation_result_file = Path(
                experiment_path) / 'evaluation_{}.txt'.format(
                    Path(testdata).stem)
            self.score(experiment_path,
                       result_file=scores_file,
                       label=testlabel,
                       data=testdata)
            self.evaluate_eer(scores_file,
                              ground_truth_file=testlabel,
                              evaluation_res_file=evaluation_result_file)

    def evaluate_eer(self,
                     scores_file,
                     ground_truth_file,
                     evaluation_res_file: str = None,
                     return_cm=False):
        # Directly run the evaluation
        gt_df = pd.read_csv(ground_truth_file, sep=' ')
        pred_df = pd.read_csv(scores_file, sep=' ')
        df = pred_df.merge(gt_df, on='filename')
        assert len(pred_df) == len(df) == len(
            gt_df
        ), "Merge was uncessful, some utterances (filenames) do not match"

        spoof_cm = df[df['bintype'] == 'spoof']['score']
        bona_cm = df[df['bintype'] != 'spoof'][
            'score']  # In any case its not "genuine"
        eer_cm = em.compute_eer(bona_cm, spoof_cm)[0]
        result_string = "EER = {:8.5f} % (Equal error rate for Spoofing detection)".format(
            eer_cm * 100)
        print(result_string.format(eer_cm * 100))
        if evaluation_res_file:  #Save to file
            with open(evaluation_res_file, 'w') as fp:
                print(
                    "EER = {:8.5f} % (Equal error rate for Spoofing detection)"
                    .format(eer_cm * 100),
                    file=fp)
            print(f"Evaluation results are at {evaluation_res_file}")
        # For evaluate_tDCF in order to avoid too many prints
        if return_cm:
            return spoof_cm, bona_cm, eer_cm

    def evaluate_tDCF(self, cm_scores_file: str, asv_scores_file: str,
                      evaluation_res_file: str):
        """evaluate_tDCF
        
        !! untested and unused

        :param cm_scores_file: Spoofing results 
        :type cm_scores_file: str
        :param asv_scores_file: Given by the challenge, asv17
        :type asv_scores_file: str
        :param evaluation_res_file:
        :type evaluation_res_file: str
        """

        # Spoofing related EER
        bona_cm, spoof_cm, eer_cm = self.evaluate_eer(cm_scores_file,
                                                      return_cm=True)

        asv_df = pd.read_csv(asv_scores_file)
        tar_asv = asv_df[asv_df['target'] == 'target']
        non_tar_asv = asv_df[asv_df['target'] == 'nontarget']
        spoof_asv = asv_df[asv_df['target'] == 'spoof']

        eer_asv, asv_threshold = em.compute_eer(tar_asv, non_tar_asv)
        [Pfa_asv, Pmiss_asv,
         Pmiss_spoof_asv] = em.obtain_asv_error_rates(tar_asv, non_tar_asv,
                                                      spoof_asv, asv_threshold)
        # Default values from ASVspoof2019
        Pspoof = 0.05
        cost_model = {
            'Pspoof': Pspoof,  # Prior probability of a spoofing attack
            'Ptar': (1 - Pspoof) * 0.99,  # Prior probability of target speaker
            'Pnon':
            (1 - Pspoof) * 0.01,  # Prior probability of nontarget speaker
            'Cmiss_asv':
            1,  # Cost of ASV system falsely rejecting target speaker
            'Cfa_asv':
            10,  # Cost of ASV system falsely accepting nontarget speaker
            'Cmiss_cm':
            1,  # Cost of CM system falsely rejecting target speaker
            'Cfa_cm': 10,  # Cost of CM system falsely accepting spoof
        }
        tDCF_curve, CM_thresholds = em.compute_tDCF(bona_cm, spoof_cm, Pfa_asv,
                                                    Pmiss_asv, Pmiss_spoof_asv,
                                                    cost_model, True)
        min_tDCF_index = np.argmin(tDCF_curve)
        min_tDCF = tDCF_curve[min_tDCF_index]

        result_string = f"""
        ASV System
            EER = {eer_asv*100:<8.5f} (Equal error rate (target vs. nontarget)
            Pfa = {Pfa_asv*100:<8.5f} (False acceptance rate)
            Pmiss = {Pmiss_asv*100:<8.5f} (False rejection rate) 
            1-Pmiss, spoof = {(1-Pmiss_asv)*100:<8.5f} (Spoof false acceptance rate)
        
        CM System
            EER = {eer_cm*100:<8.5f} (Equal error rate for counter measure)

        Tandem
            min-tDCF = {min_tDCF:<8.f}
        """

        print(result_string)
        if evaluation_res_file:
            with open(evaluation_res_file, 'w') as wp:
                print(result_string, file=wp)


if __name__ == "__main__":
    fire.Fire(Runner)
