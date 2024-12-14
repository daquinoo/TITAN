#!/usr/bin/env python3
"""Train Affinity predictor model."""
import argparse
import json
import logging
import os
import sys
from time import time
import pandas as pd

import numpy as np
import torch
from paccmann_predictor.models import MODEL_FACTORY
from paccmann_predictor.utils.hyperparams import OPTIMIZER_FACTORY
from paccmann_predictor.utils.utils import get_device
from pytoda.datasets import (
    DrugAffinityDataset, ProteinProteinInteractionDataset
)
from pytoda.proteins import ProteinFeatureLanguage, ProteinLanguage
from pytoda.smiles.smiles_language import SMILESTokenizer
from sklearn.metrics import (
    auc, average_precision_score, precision_recall_curve, roc_curve
)
from pytoda.smiles import metadata

torch.manual_seed(123456)

# setup logging
logging.basicConfig(stream=sys.stdout)

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument(
    'tcr_train_filepath', type=str,
    help='Path to the TCR training data.'
)
parser.add_argument(
    'tcr_test_filepath', type=str,
    help='Path to the TCR test data.'
)
parser.add_argument(
    'epi_train_filepath', type=str,
    help='Path to the epitope training data.'
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
    help='Directory where the model will be stored.'
)
parser.add_argument(
    'params_filepath', type=str,
    help='Path to the parameter file.'
)
parser.add_argument(
    'training_name', type=str,
    help='Name for the training.'
)
parser.add_argument(
    'model_type', type=str,
    help='Name model type you want to use: bimodal_mca, bimodal_mca_multiscale.'
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
    tcr_train_filepath, tcr_test_filepath, 
    epi_train_filepath, epi_test_filepath,
    negative_samples_filepath, model_path, 
    params_filepath, training_name, model_type
):

    logger = logging.getLogger(f'{training_name}')
    logger.setLevel(logging.DEBUG)
    # Process parameter file:
    params = {}
    with open(params_filepath) as fp:
        params.update(json.load(fp))

    # Create model directory and dump files
    model_dir = os.path.join(model_path, training_name)
    os.makedirs(os.path.join(model_dir, 'weights'), exist_ok=True)
    os.makedirs(os.path.join(model_dir, 'results'), exist_ok=True)
    with open(os.path.join(model_dir, 'model_params.json'), 'w') as fp:
        json.dump(params, fp, indent=4)

    device = get_device()
    # Load languages
    # Replace/modify the existing protein language setup
    if params.get('receptor_embedding', 'learned') == 'predefined':
        # Load custom vocabulary
        custom_vocab_path = os.path.join("data", "train_vocab.txt")  
        with open(custom_vocab_path, 'r') as f:
            vocab = [line.strip() for line in f]
        
        protein_language = ProteinFeatureLanguage(
            features='blosum',
            amino_acid_dict=vocab,  # Use custom vocabulary
            add_special_tokens=False 
        )
    else:
        protein_language = ProteinLanguage()

    if params.get('ligand_embedding', 'learned') == 'one_hot':
        logger.warning(
            'ligand_embedding_size parameter in param file is ignored in '
            'one_hot embedding setting, ligand_vocabulary_size used instead.'
        )
    if params.get('receptor_embedding', 'learned') == 'one_hot':
        logger.warning(
            'receptor_embedding_size parameter in param file is ignored in '
            'one_hot embedding setting, receptor_vocabulary_size used instead.'
        )

    # Update parameters for data format
    params.update({
        'receptor_padding': True,
        'receptor_padding_length': 50,  # Adjust based on  max sequence length
        'receptor_start_stop_token': True,
        'receptor_amino_acid_dict': 'custom',  # Mark using custom vocabulary
        'batch_size': 32,  # Adjust as needed
        'ligand_as': 'amino acids'
    })

    # Prepare the dataset
    logger.info("Start data preprocessing...")

    # Assemble datasets
    # Process training and test data
    train_epitopes, train_tcrs, train_labels = read_split_data(
        tcr_train_filepath, epi_train_filepath, negative_samples_filepath
    )
    test_epitopes, test_tcrs, test_labels = read_split_data(
        tcr_test_filepath, epi_test_filepath, negative_samples_filepath
    )

    # Assemble datasets
    train_dataset = ProteinProteinInteractionDataset(
        sequence_filepaths=[[train_epitopes], [train_tcrs]],
        entity_names=['epitope', 'tcr'],
        labels_filepath=train_labels,
        annotations_column_names=['label'],
        protein_languages=protein_language,
        padding_lengths=[params.get('receptor_padding_length', None)],
        paddings=params.get('receptor_padding', True),
        add_start_and_stops=params.get('receptor_start_stop_token', True),
        iterate_datasets=True
    )
    
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=params['batch_size'],
        shuffle=True,
        drop_last=True,
        num_workers=params.get('num_workers', 0)
    )
    
    test_dataset = ProteinProteinInteractionDataset(
        sequence_filepaths=[[test_epitopes], [test_tcrs]],
        entity_names=['epitope', 'tcr'],
        labels_filepath=test_labels,
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
        shuffle=True,
        drop_last=True,
        num_workers=params.get('num_workers', 0)
    )
    
    params.update({
        'receptor_vocabulary_size': protein_language.number_of_tokens,
        'ligand_vocabulary_size': protein_language.number_of_tokens  # Same as receptor since both are proteins
    })
    
    logger.info(
        f'Training dataset has {len(train_dataset)} samples, test set has '
        f'{len(test_dataset)}.'
    )
    save_top_model = os.path.join(model_dir, 'weights/{}_{}_{}.pt')

    model_fn = params.get('model_fn', model_type)
    model = MODEL_FACTORY[model_fn](params).to(device)
    model._associate_language(protein_language)

    protein_language.save(os.path.join(model_dir, 'protein_language.pkl'))

    # Define optimizer
    min_loss, max_roc_auc = 100, 0
    optimizer = (
        OPTIMIZER_FACTORY[params.get('optimizer', 'adam')](
            model.parameters(),
            lr=params.get('lr', 0.001),
            weight_decay=params.get('weight_decay', 0.001)
        )
    )
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params.update({'number_of_parameters': num_params})
    logger.info(f'Number of parameters: {num_params}')
    logger.info(f'Model: {model}')
    # Overwrite params.json file with updated parameters.
    with open(os.path.join(model_dir, 'model_params.json'), 'w') as fp:
        json.dump(params, fp)

    # Start training
    logger.info('Training about to start...\n')
    t = time()
    loss_training = []
    loss_validation = []

    model.save(save_top_model.format('epoch', '0', model_fn))

    for epoch in range(params['epochs']):

        model.train()
        logger.info(f"== Epoch [{epoch}/{params['epochs']}] ==")
        train_loss = 0

        for ind, (ligand, receptors, y) in enumerate(train_loader):

            torch.cuda.empty_cache()
            if ind % 10 == 0:
                logger.info(f'Batch {ind}/{len(train_loader)}')
            y_hat, pred_dict = model(ligand.to(device), receptors.to(device))
            loss = model.loss(y_hat, y.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        logger.info(
            "\t **** TRAINING ****   "
            f"Epoch [{epoch + 1}/{params['epochs']}], "
            f"loss: {train_loss / len(train_loader):.5f}. "
            f"This took {time() - t:.1f} secs."
        )

        t = time()

        # Measure validation performance
        model.eval()
        with torch.no_grad():
            test_loss = 0
            predictions = []
            labels = []
            for ind, (ligand, receptors, y) in enumerate(test_loader):
                torch.cuda.empty_cache()
                y_hat, pred_dict = model(
                    ligand.to(device), receptors.to(device)
                )
                predictions.append(y_hat)
                labels.append(y.clone())
                loss = model.loss(y_hat, y.to(device))
                test_loss += loss.item()

        predictions = torch.cat(predictions, dim=0).flatten().cpu().numpy()
        labels = torch.cat(labels, dim=0).flatten().cpu().numpy()
        loss_validation.append(test_loss / len(test_loader))
        loss_training.append(train_loss / len(train_loader))

        test_loss = test_loss / len(test_loader)
        fpr, tpr, _ = roc_curve(labels, predictions)
        test_roc_auc = auc(fpr, tpr)

        # calculations for visualization plot
        precision, recall, _ = precision_recall_curve(labels, predictions)
        avg_precision = average_precision_score(labels, predictions)

        logger.info(
            f"\t **** TESTING **** Epoch [{epoch + 1}/{params['epochs']}], "
            f"loss: {test_loss:.5f}, ROC-AUC: {test_roc_auc:.3f}, "
            f"Average precision: {avg_precision:.3f}."
        )

        def save(path, metric, typ, val=None):
            model.save(path.format(typ, metric, model_fn))
            info = {
                'best_roc_auc': str(max_roc_auc),
                'test_loss': str(min_loss)
            }
            with open(
                os.path.join(model_dir, 'results', metric + '.json'), 'w'
            ) as f:
                json.dump(info, f)
            np.save(
                os.path.join(model_dir, 'results', metric + '_preds.npy'),
                np.vstack([predictions, labels])
            )
            if typ == 'best':
                logger.info(
                    f'\t New best performance in "{metric}"'
                    f' with value : {val:.7f} in epoch: {epoch}'
                )

        if test_roc_auc > max_roc_auc:
            max_roc_auc = test_roc_auc
            save(save_top_model, 'ROC-AUC', 'best', max_roc_auc)
            ep_roc = epoch
            roc_auc_loss = test_loss
            roc_auc_pr = avg_precision

        if test_loss < min_loss:
            min_loss = test_loss
            save(save_top_model, 'loss', 'best', min_loss)
            ep_loss = epoch
            loss_roc_auc = test_roc_auc
        if (epoch + 1) % params.get('save_model', 100) == 0:
            save(save_top_model, 'epoch', str(epoch))

    logger.info(
        'Overall best performances are: \n \t'
        f'Loss = {min_loss:.4f} in epoch {ep_loss} '
        f'\t (ROC-AUC was {loss_roc_auc:4f}) \n \t'
        f'ROC-AUC = {max_roc_auc:.4f} in epoch {ep_roc} '
        f'\t (Loss was {roc_auc_loss:4f})'
    )
    save(save_top_model, 'training', 'done')
    logger.info('Done with training, models saved, shutting down.')

    np.save(
        os.path.join(model_dir, 'results', 'loss_training.npy'), loss_training
    )
    np.save(
        os.path.join(model_dir, 'results', 'loss_validation.npy'),
        loss_validation
    )

    # save best results
    result_file = os.path.join(model_path, 'results_overview.csv')
    with open(result_file, "a") as myfile:
        myfile.write(
            f'{training_name},{max_roc_auc:.4f},{roc_auc_pr:.4f},{roc_auc_loss:.4f},{ep_roc} \n \t'
        )


if __name__ == '__main__':
    # parse arguments
    args = parser.parse_args()
    # run the training
    main(
        args.tcr_train_filepath, args.tcr_test_filepath,
        args.epi_train_filepath, args.epi_test_filepath,
        args.negative_samples_filepath, args.model_path,
        args.params_filepath, args.training_name, args.model_type
    )
