#!/usr/bin/env python3
"""Train Affinity predictor model."""
import argparse
import json
import logging
import os
import sys
import pandas as pd

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

def trim_filepaths(*filepaths):
    """Trim whitespace from file paths."""
    return [filepath.strip() for filepath in filepaths]

def preprocess_to_tab_delimited(filepath):
    """Convert a CSV file to tab-delimited format."""
    try:
        df = pd.read_csv(filepath, sep='\t', header=None)  # Ensure tab-delimited format
        tab_filepath = filepath.replace('.csv', '.smi')   # Change to .smi extension
        df.to_csv(tab_filepath, sep='\t', index=False, header=False)
        print(f"Converted {filepath} to tab-delimited format: {tab_filepath}")
        return tab_filepath
    except Exception as e:
        print(f"Error preprocessing {filepath}: {e}")
        raise

torch.manual_seed(123456)

# setup logging
logging.basicConfig(stream=sys.stdout)

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument(
    'tcr_test_filepath', type=str,
    help='Path to the TCR test data.'
)
parser.add_argument(
    'epi_test_filepath', type=str,
    help='Path to the epitope test data.'
)
parser.add_argument(
    'negative_samples_filepath', type=str,
    help='Path to the negative samples file (TCRrepertoires.csv).'
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

def read_split_data(tcr_filepath, epi_filepath, negative_samples_path):
    # Read positive examples from tcr and epi splits
    tcr_df = pd.read_csv(tcr_filepath, sep='\t', header=None, 
                        names=['epitope', 'tcr', 'label'])
    epi_df = pd.read_csv(epi_filepath, sep='\t', header=None,
                        names=['epitope', 'tcr', 'label'])
    
    # Merge positive examples
    positive_df = pd.concat([tcr_df, epi_df]).drop_duplicates()
    
    # Read and process negative examples
    negative_df = pd.read_csv(negative_samples_path, header=None,
                            names=['epitope', 'tcr', 'label'])
    negative_df = negative_df[negative_df['label'] == 0]
    
    # Combine positive and negative examples
    combined_df = pd.concat([positive_df, negative_df])
    
    # Create temporary files
    epitopes_file = tcr_filepath.replace('.csv', '_epitopes.csv')
    tcrs_file = tcr_filepath.replace('.csv', '_tcrs.csv')
    labels_file = tcr_filepath.replace('.csv', '_labels.csv')
    
    combined_df['epitope'].to_csv(epitopes_file, index=False, header=False)
    combined_df['tcr'].to_csv(tcrs_file, index=False, header=False)
    combined_df[['label']].to_csv(labels_file, index=False, header=False)
    
    return epitopes_file, tcrs_file, labels_file
    
def main(
    tcr_test_filepath, epi_test_filepath, negative_samples_filepath, model_path,
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
            features='blosum'
        )
        protein_language.add_vocabulary(vocab)
    else:
        protein_language = ProteinLanguage()

    # Prepare the dataset
    logger.info("Start data preprocessing...")

    # Process test data
    test_epitopes, test_tcrs, test_labels = read_split_data(
        tcr_test_filepath, epi_test_filepath, negative_samples_filepath
    )

    test_epitopes = preprocess_to_tab_delimited(test_epitopes)
    test_tcrs = preprocess_to_tab_delimited(test_tcrs)
    test_labels = preprocess_to_tab_delimited(test_labels)
    
    test_dataset = ProteinProteinInteractionDataset(
        sequence_filepaths=[[test_epitopes], [test_tcrs]],
        entity_names=['epitope', 'tcr'],
        labels_filepath=test_labels,
        annotations_column_names=['label'],
        protein_languages=protein_language,
        padding_lengths=[params.get('receptor_padding_length', None)],
        paddings=params.get('receptor_padding', True),
        add_start_and_stops=params.get('receptor_start_stop_token', True),
        iterate_datasets=True,
        delimiter='\t',
        file_format='csv'
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
        model_path, 'weights', f'best_ROC-AUC_{model_type}.pt'
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

    (
        tcr_test_filepath, epi_test_filepath, 
        negative_samples_filepath, model_path, 
        model_type, save_name
    ) = trim_filepaths(
        args.tcr_test_filepath, args.epi_test_filepath,
        args.negative_samples_filepath, args.model_path, 
        args.model_type, args.save_name
    )
    
    # run the training
    main(
        args.tcr_test_filepath, args.epi_test_filepath,
        args.negative_samples_filepath, args.model_path, 
        args.model_type, args.save_name
    )
