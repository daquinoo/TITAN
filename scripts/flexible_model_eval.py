#!/usr/bin/env python3
"""Train Affinity predictor model."""
import argparse
import json
import logging
import os
import sys

import numpy as np
import torch
from paccmann_predictor.models import MODEL_FACTORY
from paccmann_predictor.utils.utils import get_device
from pytoda.datasets import (
    DrugAffinityDataset, ProteinProteinInteractionDataset
)
from pytoda.proteins import ProteinFeatureLanguage, ProteinLanguage
from pytoda.smiles.smiles_language import SMILESTokenizer
from sklearn.metrics import (
    auc, average_precision_score, precision_recall_curve, roc_curve
)

torch.manual_seed(123456)

# setup logging
logging.basicConfig(stream=sys.stdout)

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument(
    'test_affinity_filepath', type=str,
    help='Path to the affinity data.'
)
parser.add_argument(
    'receptor_filepath', type=str,
    help='Path to the receptor aa data. (.csv)'
)
parser.add_argument(
    'ligand_filepath', type=str,
    help='Path to the ligand data. (SMILES .smi or aa .csv)'
)
parser.add_argument(
    'model_path', type=str,
    help='Directory from where the model will be loaded.'
)
parser.add_argument(
    'model_type', type=str,
    help='Name model type you want to use: bimodal_mca, bimodal_mca_multiscale, context_encoding_mca.'
)
parser.add_argument(
    'save_name', type=str,
    help='Name you want to save results under.'
)
# yapf: enable


def main(
    test_affinity_filepath, receptor_filepath, ligand_filepath, model_path,
    model_type, save_name
):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Process parameter file:
    params_filepath = os.path.join(model_path, 'model_params.json')
    params = {}
    with open(params_filepath) as fp:
        params.update(json.load(fp))

    device = get_device()

    # Load languages

    if params.get('receptor_embedding', 'learned') == 'predefined':
        custom_vocab_path = os.path.join("data", "train_vocab.txt")
        with open(custom_vocab_path, 'r') as f:
            vocab = [line.strip() for line in f]
        
        protein_language = ProteinFeatureLanguage(
            features='blosum',
            amino_acid_dict=vocab,
            add_special_tokens=False
        )
    else:
        protein_language = ProteinLanguage()

    # Prepare the dataset
    logger.info("Start data preprocessing...")

    test_dataset = ProteinProteinInteractionDataset(
        sequence_filepaths=[[receptor_filepath]],  # Single sequence input
        entity_names=['sequence_id'],
        labels_filepath=test_affinity_filepath,
        annotations_column_names=['label'],
        protein_languages=protein_language,
        padding_lengths=[params.get('receptor_padding_length', None)],
        paddings=params.get('receptor_padding', True),
        add_start_and_stops=params.get('receptor_start_stop_token', True),
        iterate_datasets=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=params['batch_size'],
        shuffle=False,
        drop_last=True,
        num_workers=params.get('num_workers', 0)
    )
    
    logger.info(f'Test dataset has {len(test_dataset)} samples.')

    model_fn = params.get('model_fn', model_type)
    model = MODEL_FACTORY[model_fn](params).to(device)
    model._associate_language(protein_language)

    model_file = os.path.join(
        model_path, 'weights', 'best_ROC-AUC_{model_type}.pt'
    )

    logger.info(f'looking for model in {model_file}')

    if os.path.isfile(model_file):
        logger.info('Found existing model, restoring now...')
        model.load(model_file, map_location=device)

        logger.info(f'model loaded: {model_file}')

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Number of parameters: {num_params}')

    # Measure validation performance
    loss_validation = []
    model.eval()
    with torch.no_grad():
        test_loss = 0
        predictions = []
        labels = []
        for ind, (ligand, receptors, y) in enumerate(test_loader):
            torch.cuda.empty_cache()
            y_hat, pred_dict = model(ligand.to(device), receptors.to(device))
            predictions.append(y_hat)
            labels.append(y.clone())
            loss = model.loss(y_hat, y.to(device))
            test_loss += loss.item()

    predictions = torch.cat(predictions, dim=0).flatten().cpu().numpy()
    labels = torch.cat(labels, dim=0).flatten().cpu().numpy()
    loss_validation.append(test_loss / len(test_loader))

    test_loss = test_loss / len(test_loader)
    fpr, tpr, _ = roc_curve(labels, predictions)
    test_roc_auc = auc(fpr, tpr)

    # calculations for visualization plot
    precision, recall, _ = precision_recall_curve(labels, predictions)
    avg_precision = average_precision_score(labels, predictions)

    logger.info(
        f"\t **** TESTING **** loss: {test_loss:.5f}, "
        f"ROC-AUC: {test_roc_auc:.3f}, Average precision: {avg_precision:.3f}."
    )

    np.save(
        os.path.join(model_path, 'results', save_name + '.npy'),
        np.vstack([predictions, labels])
    )


if __name__ == '__main__':
    # parse arguments
    args = parser.parse_args()
    # run the training
    main(
        args.test_affinity_filepath, args.receptor_filepath,
        args.ligand_filepath, args.model_path, args.model_type, args.save_name
    )
